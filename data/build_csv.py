"""
build_csv.py
============
Scans the ICBHI dataset directory and produces a single master CSV
(cycles.csv) that is the sole source of truth for all downstream
data loading and splitting logic.

Usage
-----
    python -m data.build_csv --data_dir /path/to/icbhi --out cycles.csv

Expected ICBHI directory layout (standard challenge release)
-------------------------------------------------------------
    <data_dir>/
        audio/                          (optional subfolder, or flat)
            101_1b1_Al_sc_Meditron.wav
            ...
        ICBHI_Challenge_train_test.txt
        ICBHI_Challenge_diagnosis.txt
        ICBHI_Challenge_demographic_info.txt
        *.txt                           (per-recording annotation files)

Output CSV columns
------------------
    PID             Patient ID (integer)
    Filename        Base filename without extension
    RecordingIndex  Middle part of filename, e.g. "1b1"
    CycleIndex      0-based cycle index within the recording
    CycleStart      Start time in seconds
    CycleEnd        End time in seconds
    CycleDuration   CycleEnd - CycleStart
    Crackles        0 or 1
    Wheezes         0 or 1
    Label           0=Normal 1=Crackle 2=Wheeze 3=Both
    LabelName       Normal / Crackle / Wheeze / Both
    Split           train / test  (official challenge split)
    Fold            1–4 patient-stratified fold (for cross-validation)
    Device          Recording device name
    AuscLoc         Auscultation location code (Al, Ar, Ll, Lr, Tc, ...)
    AcqMode         sc (single-channel) or mc (multi-channel)
    Age             Patient age (years, may be missing)
    Sex             M / F (may be missing)
    Weight          kg (may be missing)
    Height          cm (may be missing)
    BMI             Computed if W and H are available, else from file
    Disease         Diagnosis label string
    AudioPath       Absolute path to .wav file
    AnnotationPath  Absolute path to annotation .txt file
"""

import os
import re
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── Label helpers ────────────────────────────────────────────────────────────

LABEL_MAP = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}
LABEL_NAMES = {0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Both"}

# Official test patient split (patient IDs as strings).
# Used as fallback when ICBHI_Challenge_train_test.txt is absent.
OFFICIAL_TEST_PIDS = {
    "101", "102", "104", "109", "113", "118", "119", "123", "124", "126",
    "133", "136", "139", "143", "144", "145", "146", "147", "149", "150",
    "151", "156", "160", "165", "169", "170", "171", "173", "174", "175",
    "176", "177", "178", "182", "185", "187", "194", "195", "198", "202",
    "205", "207", "211", "214", "216", "218", "223", "224", "225",
}


# ── Filename parser ───────────────────────────────────────────────────────────

def parse_filename(stem: str) -> Optional[Dict]:
    """
    Parse ICBHI filename convention:
        <PID>_<RecordingIndex>_<AuscLoc>_<AcqMode>_<Device>

    Returns None if the filename does not match the expected pattern.
    """
    parts = stem.split("_")
    if len(parts) < 5:
        return None
    try:
        pid = str(int(parts[0]))  # validate that it is numeric
    except ValueError:
        return None

    return {
        "PID": pid,
        "RecordingIndex": parts[1],
        "AuscLoc": parts[2],
        "AcqMode": parts[3],
        "Device": parts[4],
    }


# ── Metadata file parsers ─────────────────────────────────────────────────────

def load_official_split(data_dir: Path) -> Dict[str, str]:
    """
    Load ICBHI_Challenge_train_test.txt.
    Returns dict mapping filename_stem → "train" | "test".
    Falls back to the hardcoded PID list if the file is absent.
    """
    candidates = [
        data_dir / "ICBHI_Challenge_train_test.txt",
        data_dir / "official_split.txt",
    ]
    for path in candidates:
        if path.exists():
            split_map = {}
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        split_map[parts[0]] = parts[1].lower()
            print(f"  Loaded official split from {path.name} "
                  f"({len(split_map)} entries)")
            return split_map
    print("  Official split file not found — using hardcoded PID list.")
    return {}  # will fall back to PID-based logic


def load_demographics(data_dir: Path) -> Dict[str, Dict]:
    """
    Load ICBHI_Challenge_demographic_info.txt.
    Format varies across dataset versions; handles both:
        PID  Age  Sex  BMI  Weight  Height
        PID  Age  Sex  BMI  (no W/H)
    Returns dict mapping PID string → {Age, Sex, BMI, Weight, Height}.
    """
    candidates = [
        data_dir / "ICBHI_Challenge_demographic_info.txt",
        data_dir / "demographic_info.txt",
        data_dir / "metadata.txt",
    ]
    demo = {}
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                pid = str(parts[0])
                record = {
                    "Age": _safe_float(parts[1]) if len(parts) > 1 else None,
                    "Sex": parts[2].strip() if len(parts) > 2 else None,
                    "BMI": _safe_float(parts[3]) if len(parts) > 3 else None,
                    "Weight": _safe_float(parts[4]) if len(parts) > 4 else None,
                    "Height": _safe_float(parts[5]) if len(parts) > 5 else None,
                }
                # Compute BMI if weight and height are present but BMI is missing
                if record["BMI"] is None and record["Weight"] and record["Height"]:
                    h_m = record["Height"] / 100.0
                    record["BMI"] = round(record["Weight"] / (h_m ** 2), 1)
                demo[pid] = record
        print(f"  Loaded demographics for {len(demo)} patients from {path.name}")
        return demo
    print("  Demographic file not found — demographic columns will be empty.")
    return {}


def load_diagnoses(data_dir: Path) -> Dict[str, str]:
    """
    Load ICBHI_Challenge_diagnosis.txt.
    Format: PID<tab>Disease
    Returns dict mapping PID string → disease string.
    """
    candidates = [
        data_dir / "ICBHI_Challenge_diagnosis.txt",
        data_dir / "diagnosis.txt",
        data_dir / "patient_diagnosis.txt",
    ]
    diag = {}
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    diag[str(parts[0])] = " ".join(parts[1:])
        print(f"  Loaded diagnoses for {len(diag)} patients from {path.name}")
        return diag
    print("  Diagnosis file not found — Disease column will be empty.")
    return {}


# ── Annotation parser ─────────────────────────────────────────────────────────

def parse_annotation(ann_path: Path) -> List[Tuple[float, float, int, int]]:
    """
    Parse a single ICBHI annotation file.
    Each line: start_time  end_time  crackle  wheeze
    Returns list of (start, end, crackle, wheeze).
    """
    cycles = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                crackle = int(parts[2])
                wheeze = int(parts[3])
                cycles.append((start, end, crackle, wheeze))
            except ValueError:
                continue
    return cycles


# ── Fold assignment ───────────────────────────────────────────────────────────

def assign_folds(pids: List[str], n_folds: int = 4, seed: int = 42) -> Dict[str, int]:
    """
    Assign patient-stratified folds.
    Each patient belongs to exactly one fold; no patient appears in two folds.
    Stratification is approximate — folds are balanced by number of patients,
    not by cycle count (patient-level stratification is the hard constraint).
    Returns dict mapping PID → fold index (1-based).
    """
    rng = np.random.RandomState(seed)
    pids_arr = np.array(sorted(set(pids)))
    rng.shuffle(pids_arr)
    fold_assignments = {}
    for i, pid in enumerate(pids_arr):
        fold_assignments[pid] = (i % n_folds) + 1  # 1-based
    return fold_assignments


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(s) -> Optional[float]:
    try:
        v = float(s)
        return v if not np.isnan(v) else None
    except (ValueError, TypeError):
        return None


def find_audio_files(data_dir: Path) -> Dict[str, Path]:
    """
    Recursively find all .wav files under data_dir.
    Returns dict mapping filename stem → absolute Path.
    """
    audio_map = {}
    for wav in data_dir.rglob("*.wav"):
        audio_map[wav.stem] = wav.resolve()
    return audio_map


def find_annotation_files(data_dir: Path, audio_stems: List[str]) -> Dict[str, Path]:
    """
    Find annotation .txt files corresponding to audio stems.
    Skips known metadata filenames.
    Returns dict mapping filename stem → absolute Path.
    """
    skip = {
        "ICBHI_Challenge_train_test",
        "ICBHI_Challenge_diagnosis",
        "ICBHI_Challenge_demographic_info",
        "train_test",
        "official_split",
        "diagnosis",
        "demographic_info",
        "metadata",
        "filename_differences",
        "filename_format",
        "patient_diagnosis",
        "patient_list_foldwise",
    }
    ann_map = {}
    for txt in data_dir.rglob("*.txt"):
        if txt.stem in skip:
            continue
        if txt.stem in audio_stems:
            ann_map[txt.stem] = txt.resolve()
    return ann_map


# ── Main builder ──────────────────────────────────────────────────────────────

def build_master_csv(
    data_dir: str,
    out_path: str = "cycles.csv",
    n_folds: int = 4,
    fold_seed: int = 42,
) -> pd.DataFrame:
    """
    Build the master cycles CSV from raw ICBHI files.

    Parameters
    ----------
    data_dir : str
        Root of the ICBHI dataset.
    out_path : str
        Where to write the resulting CSV.
    n_folds : int
        Number of cross-validation folds (patient-stratified).
    fold_seed : int
        Random seed for fold assignment.

    Returns
    -------
    pd.DataFrame with one row per respiratory cycle.
    """
    data_dir = Path(data_dir).resolve()
    print(f"\nScanning ICBHI dataset at: {data_dir}\n")

    # ── Load metadata files ──────────────────────────────────────────────────
    official_split = load_official_split(data_dir)
    demographics = load_demographics(data_dir)
    diagnoses = load_diagnoses(data_dir)

    # ── Discover audio and annotation files ──────────────────────────────────
    audio_map = find_audio_files(data_dir)
    print(f"  Found {len(audio_map)} .wav files")

    ann_map = find_annotation_files(data_dir, list(audio_map.keys()))
    print(f"  Found {len(ann_map)} annotation .txt files")

    matched = set(audio_map.keys()) & set(ann_map.keys())
    print(f"  Matched audio+annotation pairs: {len(matched)}\n")

    # ── Assign folds to all patients (both splits) ────────────────────────────
    all_pids = []
    for stem in matched:
        parsed = parse_filename(stem)
        if parsed:
            all_pids.append(parsed["PID"])
    fold_map = assign_folds(all_pids, n_folds=n_folds, seed=fold_seed)

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    skipped = 0

    for stem in sorted(matched):
        parsed = parse_filename(stem)
        if parsed is None:
            skipped += 1
            continue

        pid = parsed["PID"]
        audio_path = audio_map[stem]
        ann_path = ann_map[stem]

        # Determine split
        if official_split:
            split = official_split.get(stem, "train")
        else:
            split = "test" if pid in OFFICIAL_TEST_PIDS else "train"

        # Demographic info
        demo = demographics.get(pid, {})
        disease = diagnoses.get(pid, None)
        fold = fold_map.get(pid, 1)

        # Parse cycles
        cycles = parse_annotation(ann_path)
        for cycle_idx, (start, end, crackle, wheeze) in enumerate(cycles):
            label = LABEL_MAP.get((crackle, wheeze), 0)
            duration = round(end - start, 4)

            row = {
                "PID": int(pid),
                "Filename": stem,
                "RecordingIndex": parsed["RecordingIndex"],
                "CycleIndex": cycle_idx,
                "CycleStart": round(start, 4),
                "CycleEnd": round(end, 4),
                "CycleDuration": duration,
                "Crackles": crackle,
                "Wheezes": wheeze,
                "Label": label,
                "LabelName": LABEL_NAMES[label],
                "Split": split,
                "Fold": fold,
                "Device": parsed["Device"],
                "AuscLoc": parsed["AuscLoc"],
                "AcqMode": parsed["AcqMode"],
                "Age": demo.get("Age"),
                "Sex": demo.get("Sex"),
                "Weight": demo.get("Weight"),
                "Height": demo.get("Height"),
                "BMI": demo.get("BMI"),
                "Disease": disease,
                "AudioPath": str(audio_path),
                "AnnotationPath": str(ann_path),
            }
            rows.append(row)

    if skipped:
        print(f"  Warning: skipped {skipped} files with non-standard filenames.")

    df = pd.DataFrame(rows)

    # ── Sanity report ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("  MASTER CSV SUMMARY")
    print("=" * 55)
    print(f"  Total cycles     : {len(df)}")
    print(f"  Unique patients  : {df['PID'].nunique()}")
    print(f"  Unique recordings: {df['Filename'].nunique()}")
    print(f"  Train cycles     : {(df['Split']=='train').sum()}")
    print(f"  Test  cycles     : {(df['Split']=='test').sum()}")
    print()
    print("  Label distribution:")
    for label, name in LABEL_NAMES.items():
        n = (df["Label"] == label).sum()
        pct = 100 * n / max(len(df), 1)
        print(f"    {name:<10}: {n:>5}  ({pct:.1f}%)")
    print()
    print("  Devices:")
    for dev, cnt in df["Device"].value_counts().items():
        print(f"    {dev:<25}: {cnt}")
    print()
    print("  Auscultation locations:")
    for loc, cnt in df["AuscLoc"].value_counts().items():
        print(f"    {loc:<10}: {cnt}")
    print()
    print(f"  Cycle duration — mean: {df['CycleDuration'].mean():.2f}s  "
          f"std: {df['CycleDuration'].std():.2f}s  "
          f"min: {df['CycleDuration'].min():.2f}s  "
          f"max: {df['CycleDuration'].max():.2f}s")
    print("=" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}\n")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build ICBHI master CSV from raw dataset files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Root directory of the ICBHI dataset."
    )
    parser.add_argument(
        "--out", type=str, default="cycles.csv",
        help="Output CSV file path."
    )
    parser.add_argument(
        "--n_folds", type=int, default=4,
        help="Number of cross-validation folds (patient-stratified)."
    )
    parser.add_argument(
        "--fold_seed", type=int, default=42,
        help="Random seed for fold assignment."
    )
    args = parser.parse_args()

    build_master_csv(
        data_dir=args.data_dir,
        out_path=args.out,
        n_folds=args.n_folds,
        fold_seed=args.fold_seed,
    )


if __name__ == "__main__":
    main()
