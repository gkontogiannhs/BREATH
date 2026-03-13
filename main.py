#!/usr/bin/env python3
"""
AuscultAI — ICBHI Respiratory Sound Classifier
===============================================
Main entry point with subcommands for every stage of the pipeline.

Workflow
--------
  Step 1 — build the master CSV (once per dataset):
    python main.py build_csv --data_dir /path/to/icbhi --out cycles.csv

  Step 2 — train:
    python main.py train --csv cycles.csv
    python main.py train --csv cycles.csv --split_mode fold --test_fold 2
    python main.py train --csv cycles.csv --split_mode random --test_ratio 0.4
    python main.py train --csv cycles.csv --lr 5e-4 --loss focal --gamma 3.0

  Step 3 — evaluate:
    python main.py evaluate --csv cycles.csv --checkpoint ./checkpoints/best_model.pt
    python main.py evaluate --csv cycles.csv --checkpoint ./checkpoints/best_model.pt --tta

  Step 4 — visualize attention maps:
    python main.py visualize --csv cycles.csv --checkpoint ./checkpoints/best_model.pt

  Smoke test (no dataset needed):
    python main.py test_run
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from typing import Optional

from data import (
    ICBHIDataset,
    MultiChannelFeatureExtractor,
    WaveformAugmentor,
    SpecAugment,
    CLASS_NAMES,
    build_master_csv,
)
from models import ICBHIClassifier, build_loss
from utils import (
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    load_checkpoint,
    build_optimizer,
    build_scheduler,
    MetricTracker,
    icbhi_score,
    format_metrics,
    run_inference,
    test_time_augmentation,
    plot_training_curves,
    plot_confusion_matrix,
    plot_attention_map,
    plot_per_class_metrics,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto", gpu_id: Optional[int] = None) -> torch.device:
    """
    Resolve the compute device.

    Priority order:
      1. If gpu_id is given, use cuda:<gpu_id> (errors if CUDA unavailable).
      2. If device_str is explicit (not "auto"), use it directly.
      3. Auto-detect: CUDA → MPS → CPU.
    """
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise RuntimeError(f"--gpu {gpu_id} requested but CUDA is not available.")
        n_gpus = torch.cuda.device_count()
        if gpu_id >= n_gpus:
            raise RuntimeError(
                f"--gpu {gpu_id} requested but only {n_gpus} GPU(s) found "
                f"(valid indices: 0–{n_gpus-1})."
            )
        device = torch.device(f"cuda:{gpu_id}")
    elif device_str != "auto":
        device = torch.device(device_str)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Friendly info line
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(f"Device: {device}  ({name}, {mem:.1f} GB)")
    else:
        print(f"Device: {device}")
    return device


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(args):
    sr = args.sample_rate
    cycle_samples = int(args.cycle_duration * sr)
    fixed_frames = cycle_samples // args.hop_length + 1

    feature_extractor = MultiChannelFeatureExtractor(
        sample_rate=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        hpss_kernel_size=args.hpss_kernel_size,
        fixed_length=fixed_frames,
    )

    train_augmentor = WaveformAugmentor(
        time_stretch_range=tuple(args.time_stretch_range),
        time_stretch_prob=args.time_stretch_prob,
        pitch_shift_range=tuple(args.pitch_shift_range),
        pitch_shift_prob=args.pitch_shift_prob,
        noise_snr_range=tuple(args.noise_snr_range),
        noise_prob=args.noise_prob,
        sample_rate=sr,
    ) if getattr(args, "augment", True) else None

    # Collect all filter kwargs from args (all are None by default → no-op)
    filter_kwargs = dict(
        min_duration           = args.min_duration,
        max_duration           = args.max_duration,
        filter_labels          = getattr(args, "filter_labels",          None),
        filter_devices         = getattr(args, "filter_devices",         None),
        filter_locations       = getattr(args, "filter_locations",       None),
        filter_acq_modes       = getattr(args, "filter_acq_modes",       None),
        filter_sexes           = getattr(args, "filter_sexes",           None),
        min_age                = getattr(args, "min_age",                None),
        max_age                = getattr(args, "max_age",                None),
        min_bmi                = getattr(args, "min_bmi",                None),
        max_bmi                = getattr(args, "max_bmi",                None),
        min_weight             = getattr(args, "min_weight",             None),
        max_weight             = getattr(args, "max_weight",             None),
        min_height             = getattr(args, "min_height",             None),
        max_height             = getattr(args, "max_height",             None),
        filter_diseases        = getattr(args, "filter_diseases",        None),
        filter_exclude_diseases= getattr(args, "filter_exclude_diseases",None),
        filter_pids            = getattr(args, "filter_pids",            None),
        filter_exclude_pids    = getattr(args, "filter_exclude_pids",    None),
        filter_folds           = getattr(args, "filter_folds",           None),
    )

    ds_kwargs = dict(
        csv_path     = args.csv,
        split_mode   = args.split_mode,
        test_fold    = getattr(args, "test_fold",  1),
        test_ratio   = getattr(args, "test_ratio", 0.4),
        random_seed  = args.seed,
        sample_rate  = sr,
        cycle_duration = args.cycle_duration,
        feature_extractor = feature_extractor,
        **filter_kwargs,
    )

    train_ds = ICBHIDataset(role="train", transform=train_augmentor, **ds_kwargs)
    test_ds  = ICBHIDataset(role="test",  transform=None,            **ds_kwargs)

    if getattr(args, "use_weighted_sampler", True):
        train_sampler = train_ds.get_weighted_sampler()
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    return train_loader, test_loader, train_ds, test_ds, feature_extractor


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(args, device):
    model = ICBHIClassifier(
        in_channels=args.in_channels,
        n_mels=args.n_mels,
        n_classes=args.n_classes,
        stem_channels=args.stem_channels,
        inception_channels=args.inception_channels,
        n_inception_blocks=args.n_inception_blocks,
        attn_heads=args.attn_heads,
        attn_dropout=args.attn_dropout,
        mlp_dropout=args.mlp_dropout,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")
    return model


# ── BUILD_CSV command ─────────────────────────────────────────────────────────

def cmd_build_csv(args):
    build_master_csv(
        data_dir=args.data_dir,
        out_path=args.out,
        n_folds=args.n_folds,
        fold_seed=args.fold_seed,
    )


# ── TRAIN command ─────────────────────────────────────────────────────────────

def cmd_train(args):
    set_seed(args.seed)
    device = get_device(args.device, gpu_id=getattr(args, "gpu", None))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.checkpoint_dir, run_id)
    res_dir  = os.path.join(args.results_dir, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    print(f"\n{'='*60}\n  AuscultAI Training  [{run_id}]\n{'='*60}")
    with open(os.path.join(res_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSplit mode : {args.split_mode}"
          + (f"  (fold={args.test_fold})" if args.split_mode == "fold" else "")
          + (f"  (ratio={args.test_ratio})" if args.split_mode == "random" else ""))

    train_loader, test_loader, train_ds, _, _ = build_dataloaders(args)

    spec_aug = None
    if args.spec_augment:
        spec_aug = SpecAugment(
            time_mask_param=args.spec_time_mask,
            freq_mask_param=args.spec_freq_mask,
            num_time_masks=args.spec_num_time_masks,
            num_freq_masks=args.spec_num_freq_masks,
            prob=args.spec_prob,
        )

    model     = build_model(args, device)
    cw        = train_ds.class_weights.to(device) if args.use_class_weights else None
    criterion = build_loss(args.loss, cw, args.gamma, args.smoothing, args.n_classes)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.scheduler, args.epochs,
                                args.cosine_t0, args.eta_min)

    start_epoch = 0
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = state["epoch"] + 1

    tracker = MetricTracker()
    val_metrics_history = []

    print(f"\nTraining for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  lr={lr_now:.2e}")

        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            spec_augment=spec_aug, grad_clip=args.grad_clip,
        )
        val_stats, val_metrics = validate_one_epoch(
            model, test_loader, criterion, device
        )
        val_metrics_history.append(val_metrics)

        if args.scheduler == "plateau":
            scheduler.step(val_metrics["icbhi_score"])
        else:
            scheduler.step()

        print(f"  train_loss={train_stats['loss']:.4f}  "
              f"acc={train_stats['accuracy']:.4f}  "
              f"val_loss={val_stats['loss']:.4f}")
        print(format_metrics(val_metrics))

        is_best = tracker.update(epoch, train_stats["loss"],
                                 val_stats["loss"], val_metrics)
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                        os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"),
                        is_best=is_best)

        if tracker.should_stop_early(args.early_stopping_patience):
            print(f"\nEarly stopping. {tracker.summary()}")
            break
        print()

    print(f"\n{'='*60}\n{tracker.summary()}\n{'='*60}")

    with open(os.path.join(res_dir, "history.json"), "w") as f:
        json.dump(tracker.history, f, indent=2)

    plot_training_curves(tracker.history,
                         os.path.join(res_dir, "training_curves.png"))
    plot_per_class_metrics(val_metrics_history, CLASS_NAMES,
                           os.path.join(res_dir, "per_class_f1.png"))

    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")
    if os.path.exists(best_ckpt):
        print("\nFinal evaluation on best model:")
        _run_full_eval(args, best_ckpt, test_loader, device, res_dir)


# ── EVALUATE command ──────────────────────────────────────────────────────────

def cmd_evaluate(args):
    set_seed(args.seed)
    device = get_device(args.device, gpu_id=getattr(args, "gpu", None))
    print(f"\n{'='*60}\n  AuscultAI Evaluation\n{'='*60}")
    _, test_loader, _, _, _ = build_dataloaders(args)
    os.makedirs(args.results_dir, exist_ok=True)
    _run_full_eval(args, args.checkpoint, test_loader, device, args.results_dir)


def _run_full_eval(args, ckpt_path, test_loader, device, results_dir):
    model = build_model(args, device)
    load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    if getattr(args, "tta", False):
        print(f"TTA inference (n={args.tta_n})...")
        all_preds, all_targets = [], []
        from tqdm import tqdm
        with torch.no_grad():
            for features, targets in tqdm(test_loader, desc="  TTA"):
                probs = test_time_augmentation(model, features, device, args.tta_n)
                all_preds.extend(probs.argmax(1).cpu().tolist())
                all_targets.extend(targets.tolist())
        all_preds   = np.array(all_preds)
        all_targets = np.array(all_targets)
    else:
        all_preds, all_targets, _ = run_inference(model, test_loader, device)

    metrics = icbhi_score(all_targets, all_preds)
    print(f"\nTest Results:\n{format_metrics(metrics)}")

    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(
        all_targets, all_preds, CLASS_NAMES,
        save_path=os.path.join(results_dir, "confusion_matrix.png"),
        title=f"ICBHI Score: {metrics['icbhi_score']:.4f}",
    )
    return metrics


# ── VISUALIZE command ─────────────────────────────────────────────────────────

def cmd_visualize(args):
    set_seed(args.seed)
    device = get_device(args.device, gpu_id=getattr(args, "gpu", None))
    print(f"\nGenerating {args.n_samples} attention visualizations...")

    _, _, _, test_ds, _ = build_dataloaders(args)
    model = build_model(args, device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    os.makedirs(args.results_dir, exist_ok=True)
    indices = random.sample(range(len(test_ds)), min(args.n_samples, len(test_ds)))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            feat, true_label = test_ds[idx]
            logits, attn = model(feat.unsqueeze(0).to(device))
            pred_label = logits.argmax(1).item()

            plot_attention_map(
                features=feat.numpy(),
                attn_weights=attn[0].cpu().numpy(),
                predicted_label=pred_label,
                true_label=true_label,
                class_names=CLASS_NAMES,
                save_path=os.path.join(
                    args.results_dir,
                    f"attn_{i:03d}_true{CLASS_NAMES[true_label]}"
                    f"_pred{CLASS_NAMES[pred_label]}.png"
                ),
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
            )
    print(f"Saved to {args.results_dir}")


# ── TEST_RUN command ──────────────────────────────────────────────────────────

def cmd_test_run(args):
    print("\nSmoke test with synthetic data...")
    set_seed(42)
    device = get_device(getattr(args, "device", "auto"), gpu_id=getattr(args, "gpu", None))

    n_mels, T, B = 128, 220, 4
    model = ICBHIClassifier(
        in_channels=4, n_mels=n_mels, n_classes=4,
        stem_channels=32, inception_channels=[64, 128, 256],
        n_inception_blocks=3, attn_heads=4,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    x = torch.randn(B, 4, n_mels, T).to(device)
    targets = torch.randint(0, 4, (B,)).to(device)
    logits, attn = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Attn:   {tuple(attn.shape)}")

    loss = build_loss("focal_smooth", gamma=2.0, smoothing=0.1)(logits, targets)
    print(f"Loss:   {loss.item():.4f}")
    loss.backward()
    print("Backward: OK")

    from data import MultiChannelFeatureExtractor
    ext = MultiChannelFeatureExtractor(sample_rate=22050, n_fft=512,
                                       hop_length=220, n_mels=128, fixed_length=T)
    feat = ext(np.random.randn(22050 * 5).astype(np.float32))
    print(f"Features: {feat.shape}  (expected (4, 128, {T}))")
    print("\nSmoke test passed!")


# ── Argument Parser ───────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="auscultai",
        description="AuscultAI — ICBHI respiratory sound classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # ── shared parent ──────────────────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--seed",        type=int,   default=42)
    shared.add_argument("--device",      type=str,   default="auto",
                        help="auto | cuda | cuda:0 | cpu | mps")
    shared.add_argument("--gpu",         type=int,   default=None,
                        help="GPU index to use (e.g. --gpu 0 or --gpu 2). "
                             "Overrides --device for CUDA selection.")
    shared.add_argument("--csv",         type=str,   default="cycles.csv",
                        help="Path to master CSV from build_csv command")
    shared.add_argument("--split_mode",  type=str,   default="official",
                        choices=["official", "fold", "random"])
    shared.add_argument("--test_fold",   type=int,   default=1)
    shared.add_argument("--test_ratio",  type=float, default=0.4)
    shared.add_argument("--sample_rate", type=int,   default=4000)
    shared.add_argument("--cycle_duration", type=float, default=5.0)
    shared.add_argument("--num_workers", type=int,   default=4)
    shared.add_argument("--pin_memory",  action="store_true", default=True)
    # features
    shared.add_argument("--n_fft",            type=int,   default=256)
    shared.add_argument("--hop_length",       type=int,   default=40)
    shared.add_argument("--n_mels",           type=int,   default=128)
    shared.add_argument("--fmin",             type=float, default=50.0)
    shared.add_argument("--fmax",             type=float, default=1999.0)
    shared.add_argument("--hpss_kernel_size", type=int,   default=15)
    # model
    shared.add_argument("--in_channels",         type=int,   default=3)
    shared.add_argument("--n_classes",           type=int,   default=4)
    shared.add_argument("--stem_channels",       type=int,   default=32)
    shared.add_argument("--inception_channels",  type=int,   nargs="+",
                        default=[64, 128, 256])
    shared.add_argument("--n_inception_blocks",  type=int,   default=3)
    shared.add_argument("--attn_heads",          type=int,   default=4)
    shared.add_argument("--attn_dropout",        type=float, default=0.1)
    shared.add_argument("--mlp_dropout",         type=float, default=0.4)
    # output
    shared.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    shared.add_argument("--results_dir",    type=str, default="./results")
    # evaluation
    shared.add_argument("--tta",         action="store_true", default=False)
    shared.add_argument("--tta_n",       type=int, default=5)
    shared.add_argument("--batch_size",  type=int, default=128)

    # ── Cycle-level filters ────────────────────────────────────────────────
    shared.add_argument("--min_duration",  type=float, default=0.5,
                        help="Drop cycles shorter than this (seconds)")
    shared.add_argument("--max_duration",  type=float, default=16.0,
                        help="Drop cycles longer than this (seconds)")
    shared.add_argument("--filter_labels", type=int, nargs="+", default=None,
                        help="Keep only these label ints: 0=Normal 1=Crackle 2=Wheeze 3=Both "
                             "e.g. --filter_labels 1 2 3")

    # ── Recording filters ──────────────────────────────────────────────────
    shared.add_argument("--filter_devices",   type=str, nargs="+", default=None,
                        help="Whitelist device names e.g. --filter_devices Meditron LittC2SE")
    shared.add_argument("--filter_locations", type=str, nargs="+", default=None,
                        help="Whitelist auscultation locations e.g. --filter_locations Al Ar Ll Lr Tc")
    shared.add_argument("--filter_acq_modes", type=str, nargs="+", default=None,
                        help="Whitelist acquisition modes: sc and/or mc")

    # ── Demographic filters ────────────────────────────────────────────────
    shared.add_argument("--filter_sexes", type=str, nargs="+", default=None,
                        help="Keep only these sexes: M and/or F  e.g. --filter_sexes F")
    shared.add_argument("--min_age",    type=float, default=None,
                        help="Minimum patient age (years)")
    shared.add_argument("--max_age",    type=float, default=None,
                        help="Maximum patient age (years)")
    shared.add_argument("--min_bmi",    type=float, default=None)
    shared.add_argument("--max_bmi",    type=float, default=None)
    shared.add_argument("--min_weight", type=float, default=None,
                        help="Minimum patient weight (kg)")
    shared.add_argument("--max_weight", type=float, default=None)
    shared.add_argument("--min_height", type=float, default=None,
                        help="Minimum patient height (cm)")
    shared.add_argument("--max_height", type=float, default=None)

    # ── Clinical filters ───────────────────────────────────────────────────
    shared.add_argument("--filter_diseases", type=str, nargs="+", default=None,
                        help="Whitelist diagnoses e.g. --filter_diseases COPD Pneumonia Healthy")
    shared.add_argument("--filter_exclude_diseases", type=str, nargs="+", default=None,
                        help="Blacklist diagnoses e.g. --filter_exclude_diseases URTI")

    # ── Patient filters ────────────────────────────────────────────────────
    shared.add_argument("--filter_pids", type=int, nargs="+", default=None,
                        help="Restrict to specific patient IDs")
    shared.add_argument("--filter_exclude_pids", type=int, nargs="+", default=None,
                        help="Exclude specific patient IDs")
    shared.add_argument("--filter_folds", type=int, nargs="+", default=None,
                        help="Restrict to specific fold numbers")

    # ── build_csv ──────────────────────────────────────────────────────────
    p_csv = subs.add_parser("build_csv", help="Build master cycles.csv from raw ICBHI files")
    p_csv.add_argument("--data_dir",  type=str, required=True,
                       help="Root directory of the ICBHI dataset")
    p_csv.add_argument("--out",       type=str, default="cycles.csv")
    p_csv.add_argument("--n_folds",   type=int, default=4)
    p_csv.add_argument("--fold_seed", type=int, default=42)

    # ── train ──────────────────────────────────────────────────────────────
    p_tr = subs.add_parser("train", parents=[shared], help="Train the model")
    p_tr.add_argument("--epochs",   type=int,   default=100)
    p_tr.add_argument("--optimizer",type=str,   default="adamw",
                      choices=["adamw", "adam", "sgd"])
    p_tr.add_argument("--lr",           type=float, default=1e-3)
    p_tr.add_argument("--weight_decay", type=float, default=1e-4)
    p_tr.add_argument("--grad_clip",    type=float, default=1.0)
    p_tr.add_argument("--scheduler",    type=str,   default="cosine_warm",
                      choices=["cosine_warm", "cosine", "plateau", "step"])
    p_tr.add_argument("--cosine_t0",    type=int,   default=10)
    p_tr.add_argument("--eta_min",      type=float, default=1e-6)
    p_tr.add_argument("--early_stopping_patience", type=int, default=15)
    p_tr.add_argument("--use_weighted_sampler", action="store_true",  default=True)
    p_tr.add_argument("--no_weighted_sampler",  dest="use_weighted_sampler",
                      action="store_false")
    # loss
    p_tr.add_argument("--loss",             type=str,   default="focal_smooth",
                      choices=["focal", "focal_smooth", "cross_entropy"])
    p_tr.add_argument("--gamma",            type=float, default=2.0)
    p_tr.add_argument("--smoothing",        type=float, default=0.1)
    p_tr.add_argument("--use_class_weights",action="store_true",  default=True)
    p_tr.add_argument("--no_class_weights", dest="use_class_weights",
                      action="store_false")
    # augmentation
    p_tr.add_argument("--time_stretch_range",  type=float, nargs=2, default=[0.8, 1.2])
    p_tr.add_argument("--time_stretch_prob",   type=float, default=0.5)
    p_tr.add_argument("--pitch_shift_range",   type=float, nargs=2, default=[-2, 2])
    p_tr.add_argument("--pitch_shift_prob",    type=float, default=0.4)
    p_tr.add_argument("--noise_snr_range",     type=float, nargs=2, default=[15, 40])
    p_tr.add_argument("--noise_prob",          type=float, default=0.6)
    p_tr.add_argument("--spec_augment",        action="store_true",  default=True)
    p_tr.add_argument("--no_spec_augment",     dest="spec_augment", action="store_false")
    p_tr.add_argument("--spec_time_mask",      type=int,   default=30)
    p_tr.add_argument("--spec_freq_mask",      type=int,   default=13)
    p_tr.add_argument("--spec_num_time_masks", type=int,   default=2)
    p_tr.add_argument("--spec_num_freq_masks", type=int,   default=2)
    p_tr.add_argument("--spec_prob",           type=float, default=0.8)
    p_tr.add_argument("--resume",              type=str,   default=None)
    p_tr.add_argument("--augment",             action="store_true",  default=True)
    p_tr.add_argument("--no_augment",          dest="augment", action="store_false")

    # ── evaluate ───────────────────────────────────────────────────────────
    p_ev = subs.add_parser("evaluate", parents=[shared], help="Evaluate a trained model")
    p_ev.add_argument("--checkpoint", type=str, required=True)
    p_ev.add_argument("--augment",    action="store_false", dest="augment", default=False)
    p_ev.add_argument("--time_stretch_range",  type=float, nargs=2, default=[0.8, 1.2])
    p_ev.add_argument("--time_stretch_prob",   type=float, default=0.0)
    p_ev.add_argument("--pitch_shift_range",   type=float, nargs=2, default=[-2, 2])
    p_ev.add_argument("--pitch_shift_prob",    type=float, default=0.0)
    p_ev.add_argument("--noise_snr_range",     type=float, nargs=2, default=[15, 40])
    p_ev.add_argument("--noise_prob",          type=float, default=0.0)
    p_ev.add_argument("--use_weighted_sampler",action="store_false", default=True)

    # ── visualize ──────────────────────────────────────────────────────────
    p_vi = subs.add_parser("visualize", parents=[shared], help="Visualize attention maps")
    p_vi.add_argument("--checkpoint", type=str, required=True)
    p_vi.add_argument("--n_samples",  type=int, default=10)
    p_vi.add_argument("--augment",    action="store_false", dest="augment", default=False)
    p_vi.add_argument("--time_stretch_range",  type=float, nargs=2, default=[0.8, 1.2])
    p_vi.add_argument("--time_stretch_prob",   type=float, default=0.0)
    p_vi.add_argument("--pitch_shift_range",   type=float, nargs=2, default=[-2, 2])
    p_vi.add_argument("--pitch_shift_prob",    type=float, default=0.0)
    p_vi.add_argument("--noise_snr_range",     type=float, nargs=2, default=[15, 40])
    p_vi.add_argument("--noise_prob",          type=float, default=0.0)
    p_vi.add_argument("--use_weighted_sampler",action="store_false", default=False)

    # ── test_run ───────────────────────────────────────────────────────────
    p_ts = subs.add_parser("test_run", help="Smoke test with synthetic data")
    p_ts.add_argument("--device", type=str, default="auto")
    p_ts.add_argument("--gpu",    type=int, default=None,
                      help="GPU index override")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "build_csv": cmd_build_csv,
        "train":     cmd_train,
        "evaluate":  cmd_evaluate,
        "visualize": cmd_visualize,
        "test_run":  cmd_test_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
