"""
dataset.py
==========
CSV-driven ICBHI dataset.

The master CSV produced by build_csv.py is the single source of truth.
Three split strategies are supported, all enforcing the hard constraint
that no patient appears in both train and test partitions.

Split modes
-----------
"official"
    Uses the Split column in the CSV (official 60/40 challenge split).
    This is the default and what results should be reported against.

"fold"
    Uses the Fold column.  Pass `test_fold=<int>` (1-based) to hold out
    that fold; all other folds are used for training.  This enables
    proper k-fold cross-validation while respecting patient boundaries.

"random"
    Randomly assigns patients to train/test according to `test_ratio`.
    Patient IDs are shuffled (seeded) and split at the ratio boundary —
    guaranteeing no leakage.  Use for quick ablations / sanity checks.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import librosa
import warnings
warnings.filterwarnings("ignore")

LABEL_NAMES = {0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Both"}
CLASS_NAMES = ["Normal", "Crackle", "Wheeze", "Both"]


# ── Split helpers ─────────────────────────────────────────────────────────────

def _official_split(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """Filter by the 'Split' column (official challenge split)."""
    col = df["Split"].str.lower()
    return df[col == role.lower()].reset_index(drop=True)


def _fold_split(df: pd.DataFrame, role: str, test_fold: int) -> pd.DataFrame:
    """
    Hold out test_fold as the test set; all other folds are training.
    Operates at the patient level — all cycles of a patient go to the
    same partition.
    """
    if test_fold not in df["Fold"].unique():
        raise ValueError(
            f"test_fold={test_fold} not found in CSV. "
            f"Available folds: {sorted(df['Fold'].unique())}"
        )
    if role == "test":
        return df[df["Fold"] == test_fold].reset_index(drop=True)
    else:
        return df[df["Fold"] != test_fold].reset_index(drop=True)


def _random_split(
    df: pd.DataFrame,
    role: str,
    test_ratio: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random patient-stratified split.
    Patients are shuffled and divided at test_ratio — no cycle-level
    splitting that would allow the same patient in both partitions.
    """
    pids = np.array(sorted(df["PID"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n_test = max(1, int(len(pids) * test_ratio))
    test_pids = set(pids[:n_test].tolist())
    train_pids = set(pids[n_test:].tolist())

    if role == "test":
        return df[df["PID"].isin(test_pids)].reset_index(drop=True)
    else:
        return df[df["PID"].isin(train_pids)].reset_index(drop=True)


def get_split(
    df: pd.DataFrame,
    role: str,
    split_mode: str = "official",
    test_fold: int = 1,
    test_ratio: float = 0.4,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Dispatch to the correct split strategy.

    Parameters
    ----------
    df : master DataFrame (all cycles)
    role : "train" or "test"
    split_mode : "official" | "fold" | "random"
    test_fold : which fold to hold out (only used when split_mode="fold")
    test_ratio : fraction of patients for test (only used when split_mode="random")
    random_seed : seed for random split
    """
    mode = split_mode.lower()
    if mode == "official":
        return _official_split(df, role)
    elif mode == "fold":
        return _fold_split(df, role, test_fold)
    elif mode == "random":
        return _random_split(df, role, test_ratio, random_seed)
    else:
        raise ValueError(
            f"Unknown split_mode '{split_mode}'. "
            "Choose from: official, fold, random"
        )


# ── Optional filters ──────────────────────────────────────────────────────────

def filter_cycles(
    df: pd.DataFrame,
    # ── Cycle-level ───────────────────────────────────────────────────────
    min_duration: float = 0.5,
    max_duration: float = 16.0,
    labels: Optional[List[int]] = None,          # e.g. [0,1] → Normal+Crackle only
    # ── Recording metadata ────────────────────────────────────────────────
    devices: Optional[List[str]] = None,          # e.g. ["Meditron","LittC2SE"]
    locations: Optional[List[str]] = None,        # e.g. ["Al","Ar","Ll","Lr","Tc"]
    acq_modes: Optional[List[str]] = None,        # ["sc"] or ["mc"]
    # ── Patient demographics ──────────────────────────────────────────────
    sexes: Optional[List[str]] = None,            # ["M"], ["F"], or ["M","F"]
    min_age: Optional[float] = None,
    max_age: Optional[float] = None,
    min_bmi: Optional[float] = None,
    max_bmi: Optional[float] = None,
    min_weight: Optional[float] = None,
    max_weight: Optional[float] = None,
    min_height: Optional[float] = None,
    max_height: Optional[float] = None,
    # ── Clinical ─────────────────────────────────────────────────────────
    diseases: Optional[List[str]] = None,         # e.g. ["COPD","Pneumonia"]
    exclude_diseases: Optional[List[str]] = None, # blacklist specific diagnoses
    # ── Patient-level ────────────────────────────────────────────────────
    pids: Optional[List[int]] = None,             # restrict to specific patient IDs
    exclude_pids: Optional[List[int]] = None,     # exclude specific patient IDs
    # ── Fold ─────────────────────────────────────────────────────────────
    folds: Optional[List[int]] = None,            # restrict to specific fold numbers
) -> pd.DataFrame:
    """
    Apply filters to the cycle DataFrame for ablation studies.

    All list-valued filters are *inclusive whitelists* unless the parameter
    name starts with ``exclude_``, which makes them blacklists.
    Numeric range filters (min_*/max_*) operate on the corresponding column;
    rows where the column is NaN pass through unless both bounds are given,
    in which case NaN rows are dropped (conservative).

    Returns a fresh DataFrame with a reset index. Prints a one-line summary
    of how many cycles were dropped and why.
    """
    n_before = len(df)
    applied = []

    mask = pd.Series(True, index=df.index)

    # ── Cycle duration ────────────────────────────────────────────────────
    mask &= df["CycleDuration"] >= min_duration
    mask &= df["CycleDuration"] <= max_duration

    # ── Label whitelist ───────────────────────────────────────────────────
    if labels is not None:
        mask &= df["Label"].isin(labels)
        applied.append(f"labels={labels}")

    # ── Device ────────────────────────────────────────────────────────────
    if devices:
        mask &= df["Device"].isin(devices)
        applied.append(f"devices={devices}")

    # ── Auscultation location ─────────────────────────────────────────────
    if locations:
        mask &= df["AuscLoc"].isin(locations)
        applied.append(f"locations={locations}")

    # ── Acquisition mode ──────────────────────────────────────────────────
    if acq_modes:
        mask &= df["AcqMode"].isin(acq_modes)
        applied.append(f"acq_modes={acq_modes}")

    # ── Sex ───────────────────────────────────────────────────────────────
    if sexes:
        # Normalise to uppercase for safety
        sexes_norm = [s.upper() for s in sexes]
        mask &= df["Sex"].str.upper().isin(sexes_norm)
        applied.append(f"sex={sexes_norm}")

    # ── Age range ─────────────────────────────────────────────────────────
    if min_age is not None:
        valid = df["Age"].notna()
        mask &= (~valid) | (df["Age"] >= min_age)   # NaN rows pass unless max also set
        applied.append(f"age>={min_age}")
    if max_age is not None:
        valid = df["Age"].notna()
        mask &= (~valid) | (df["Age"] <= max_age)
        applied.append(f"age<={max_age}")
    if min_age is not None and max_age is not None:
        # Both bounds: drop NaN ages
        mask &= df["Age"].notna()

    # ── BMI range ─────────────────────────────────────────────────────────
    if min_bmi is not None:
        mask &= df["BMI"].isna() | (df["BMI"] >= min_bmi)
        applied.append(f"bmi>={min_bmi}")
    if max_bmi is not None:
        mask &= df["BMI"].isna() | (df["BMI"] <= max_bmi)
        applied.append(f"bmi<={max_bmi}")
    if min_bmi is not None and max_bmi is not None:
        mask &= df["BMI"].notna()

    # ── Weight range ──────────────────────────────────────────────────────
    if min_weight is not None:
        mask &= df["Weight"].isna() | (df["Weight"] >= min_weight)
        applied.append(f"weight>={min_weight}")
    if max_weight is not None:
        mask &= df["Weight"].isna() | (df["Weight"] <= max_weight)
        applied.append(f"weight<={max_weight}")
    if min_weight is not None and max_weight is not None:
        mask &= df["Weight"].notna()

    # ── Height range ──────────────────────────────────────────────────────
    if min_height is not None:
        mask &= df["Height"].isna() | (df["Height"] >= min_height)
        applied.append(f"height>={min_height}")
    if max_height is not None:
        mask &= df["Height"].isna() | (df["Height"] <= max_height)
        applied.append(f"height<={max_height}")
    if min_height is not None and max_height is not None:
        mask &= df["Height"].notna()

    # ── Disease whitelist ─────────────────────────────────────────────────
    if diseases:
        mask &= df["Disease"].isin(diseases)
        applied.append(f"diseases={diseases}")

    # ── Disease blacklist ─────────────────────────────────────────────────
    if exclude_diseases:
        mask &= ~df["Disease"].isin(exclude_diseases)
        applied.append(f"exclude_diseases={exclude_diseases}")

    # ── Patient whitelist ─────────────────────────────────────────────────
    if pids:
        mask &= df["PID"].isin(pids)
        applied.append(f"pids={pids}")

    # ── Patient blacklist ─────────────────────────────────────────────────
    if exclude_pids:
        mask &= ~df["PID"].isin(exclude_pids)
        applied.append(f"exclude_pids={exclude_pids}")

    # ── Fold whitelist ────────────────────────────────────────────────────
    if folds:
        mask &= df["Fold"].isin(folds)
        applied.append(f"folds={folds}")

    df = df[mask].reset_index(drop=True)
    n_dropped = n_before - len(df)

    if n_dropped or applied:
        filter_str = ", ".join(applied) if applied else "duration only"
        print(f"  [filter] {filter_str}")
        print(f"  [filter] {n_before} → {len(df)} cycles  "
              f"({n_dropped} dropped, {df['PID'].nunique()} patients remain)")

    return df


# ── Dataset ───────────────────────────────────────────────────────────────────

class ICBHIDataset(Dataset):
    """
    ICBHI respiratory cycle dataset backed by the master CSV.

    Split parameters
    ----------------
    csv_path      : path to cycles.csv from build_csv
    role          : "train" | "test"
    split_mode    : "official" | "fold" | "random"
    test_fold     : fold to hold out (split_mode="fold" only)
    test_ratio    : test fraction (split_mode="random" only)
    random_seed   : seed for random split

    Audio parameters
    ----------------
    sample_rate   : target Hz
    cycle_duration: fixed length in seconds (pad / truncate)
    transform     : waveform augmentation callable
    feature_extractor : feature extraction callable

    Cycle-level filters
    -------------------
    min_duration / max_duration : cycle length bounds (seconds)
    filter_labels               : keep only these label integers e.g. [0,1]

    Recording filters
    -----------------
    filter_devices    : whitelist of device names
    filter_locations  : whitelist of auscultation location codes
    filter_acq_modes  : whitelist of acquisition modes ("sc"/"mc")

    Demographic filters
    -------------------
    filter_sexes      : e.g. ["M"] or ["F"] or ["M","F"]
    min_age / max_age : patient age range (years)
    min_bmi / max_bmi : BMI range
    min_weight / max_weight : weight range (kg)
    min_height / max_height : height range (cm)

    Clinical filters
    ----------------
    filter_diseases         : whitelist of diagnosis strings
    filter_exclude_diseases : blacklist of diagnosis strings

    Patient filters
    ---------------
    filter_pids         : restrict to specific patient ID integers
    filter_exclude_pids : exclude specific patient ID integers
    filter_folds        : restrict to specific fold numbers
    """

    def __init__(
        self,
        csv_path: str,
        role: str = "train",
        split_mode: str = "official",
        test_fold: int = 1,
        test_ratio: float = 0.4,
        random_seed: int = 42,
        sample_rate: int = 22050,
        cycle_duration: float = 5.0,
        transform: Optional[Callable] = None,
        feature_extractor: Optional[Callable] = None,
        # Cycle-level
        min_duration: float = 0.5,
        max_duration: float = 16.0,
        filter_labels: Optional[List[int]] = None,
        # Recording
        filter_devices: Optional[List[str]] = None,
        filter_locations: Optional[List[str]] = None,
        filter_acq_modes: Optional[List[str]] = None,
        # Demographics
        filter_sexes: Optional[List[str]] = None,
        min_age: Optional[float] = None,
        max_age: Optional[float] = None,
        min_bmi: Optional[float] = None,
        max_bmi: Optional[float] = None,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        min_height: Optional[float] = None,
        max_height: Optional[float] = None,
        # Clinical
        filter_diseases: Optional[List[str]] = None,
        filter_exclude_diseases: Optional[List[str]] = None,
        # Patient
        filter_pids: Optional[List[int]] = None,
        filter_exclude_pids: Optional[List[int]] = None,
        filter_folds: Optional[List[int]] = None,
    ):
        self.role = role
        self.sample_rate = sample_rate
        self.cycle_length = int(cycle_duration * sample_rate)
        self.transform = transform
        self.feature_extractor = feature_extractor

        # Load and split
        df_all = pd.read_csv(csv_path)
        df = get_split(
            df_all, role, split_mode,
            test_fold=test_fold,
            test_ratio=test_ratio,
            random_seed=random_seed,
        )

        # Full filter pass
        df = filter_cycles(
            df,
            min_duration=min_duration,
            max_duration=max_duration,
            labels=filter_labels,
            devices=filter_devices,
            locations=filter_locations,
            acq_modes=filter_acq_modes,
            sexes=filter_sexes,
            min_age=min_age,
            max_age=max_age,
            min_bmi=min_bmi,
            max_bmi=max_bmi,
            min_weight=min_weight,
            max_weight=max_weight,
            min_height=min_height,
            max_height=max_height,
            diseases=filter_diseases,
            exclude_diseases=filter_exclude_diseases,
            pids=filter_pids,
            exclude_pids=filter_exclude_pids,
            folds=filter_folds,
        )

        self.df = df.reset_index(drop=True)
        self.labels: List[int] = self.df["Label"].tolist()

        self._validate_paths()
        self._compute_class_weights()
        self._print_summary(split_mode, test_fold)

    # ── Internal setup ────────────────────────────────────────────────────────

    def _validate_paths(self):
        missing = self.df[~self.df["AudioPath"].apply(os.path.exists)]
        if len(missing):
            print(f"  [warning] {len(missing)} audio files not found on disk.")

    def _compute_class_weights(self):
        counts = np.array(
            [self.labels.count(i) for i in range(4)], dtype=np.float32
        )
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        self.sample_weights = np.array(
            [weights[l] for l in self.labels], dtype=np.float32
        )
        self.class_weights = torch.from_numpy(
            weights / weights.sum() * 4
        )

    def _print_summary(self, split_mode: str, test_fold: int):
        dist = [self.labels.count(i) for i in range(4)]
        mode_str = split_mode
        if split_mode == "fold":
            mode_str = f"fold (test_fold={test_fold})"
        print(
            f"[{self.role}|{mode_str}] "
            f"{len(self.df)} cycles | "
            f"{self.df['PID'].nunique()} patients | "
            f"Normal={dist[0]} Crackle={dist[1]} Wheeze={dist[2]} Both={dist[3]}"
        )

    # ── Audio loading ─────────────────────────────────────────────────────────

    def _load_cycle(self, row: pd.Series) -> np.ndarray:
        audio, _ = librosa.load(
            row["AudioPath"],
            sr=self.sample_rate,
            offset=float(row["CycleStart"]),
            duration=float(row["CycleDuration"]),
            mono=True,
        )
        std = audio.std()
        if std > 1e-8:
            audio = (audio - audio.mean()) / std
        else:
            audio = audio - audio.mean()

        if len(audio) < self.cycle_length:
            audio = np.pad(
                audio, (0, self.cycle_length - len(audio)), mode="constant"
            )
        else:
            audio = audio[: self.cycle_length]

        return audio.astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        label = int(row["Label"])
        audio = self._load_cycle(row)

        if self.transform is not None:
            audio = self.transform(audio, self.sample_rate)

        if self.feature_extractor is not None:
            features = self.feature_extractor(audio, self.sample_rate)
            return torch.from_numpy(features), label

        return torch.from_numpy(audio), label

    def get_weighted_sampler(self):
        from torch.utils.data import WeightedRandomSampler
        return WeightedRandomSampler(
            weights=torch.from_numpy(self.sample_weights),
            num_samples=len(self.sample_weights),
            replacement=True,
        )

    def get_metadata(self, idx: int) -> dict:
        return self.df.iloc[idx].to_dict()

    def class_distribution(self) -> dict:
        return {LABEL_NAMES[i]: self.labels.count(i) for i in range(4)}

    def patient_ids(self) -> List[int]:
        return sorted(self.df["PID"].unique().tolist())

    @property
    def n_classes(self) -> int:
        return 4
