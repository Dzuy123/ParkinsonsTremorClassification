"""
Parkinson's Disease Motion Analysis Model
------------------------------------------
Feature engineering and classification pipeline for Parkinson's disease detection
based on hand motion data from different motor tasks.


"""

import os
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid
from tqdm.auto import tqdm

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = "data"   
TRAIN_CSV = os.path.join(ROOT, "train.csv")
TEST_CSV  = os.path.join(ROOT, "test.csv")
RAW_DIR   = os.path.join(ROOT, "data")

FEATURE_TRAIN_OUT = os.path.join(ROOT, "train_features_v2.csv")
FEATURE_TEST_OUT  = os.path.join(ROOT, "test_features_v2.csv")
SUBMISSION_OUT    = os.path.join(ROOT, "submit_motion_model.csv")

# ============================================================================
# HAND LANDMARKS AND ANATOMICAL CONSTANTS
# ============================================================================

LANDMARKS = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

TIPS = [
    "THUMB_TIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_TIP",
    "PINKY_TIP",
]

ADJ_TIP_PAIRS = [
    ("THUMB_TIP", "INDEX_FINGER_TIP"),
    ("INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP"),
    ("MIDDLE_FINGER_TIP", "RING_FINGER_TIP"),
    ("RING_FINGER_TIP", "PINKY_TIP"),
]

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def clean_metadata(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean and normalize metadata columns."""
    out = df.copy()

    # gender "0" -> Unknown
    out["gender"] = out["gender"].replace({0: "Unknown", "0": "Unknown"}).astype(str)

    # age 0 is almost certainly missing, not true age
    out["age_missing"] = (out["age"] == 0).astype(int)
    out["age"] = out["age"].replace({0: np.nan}).astype(float)

    out["patient_off_on"] = out["patient_off_on"].astype(str)
    out["doctor_diagnosis_0_5"] = out["doctor_diagnosis_0_5"].astype(float)

    if is_train:
        out["folder_path"] = out["folder_path"].astype(str)

    return out


# Load train and test metadata
train = clean_metadata(pd.read_csv(TRAIN_CSV), is_train=True)
test  = clean_metadata(pd.read_csv(TEST_CSV), is_train=False)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# ============================================================================
# SIGNAL SEGMENTATION
# ============================================================================

def split_into_segments(
    df: pd.DataFrame,
    time_col: str = "TIME",
    min_frames: int = 20,
    gap_abs_seconds: float = 0.50,
    gap_mult: float = 8.0,
):
    """
    Split a raw file into continuous segments using:
      - timestamp reversal / reset (dt <= 0)
      - giant time jump (dt > max(gap_abs_seconds, gap_mult * median_positive_dt))

    Returns:
      kept_segments: list[pd.DataFrame]
      qc: dict of quality-control stats
    """
    out_segments = []

    if time_col not in df.columns:
        seg = df.copy()
        seg["_segment_id"] = 0
        seg["_segment_local_time"] = np.arange(len(seg), dtype=float)
        qc = {
            "raw_n_rows": len(df),
            "has_time": 0,
            "has_reset": 0,
            "has_large_gap": 0,
            "n_negative_dt": 0,
            "n_large_gaps": 0,
            "median_dt_raw": np.nan,
            "gap_threshold": np.nan,
            "n_segments_total": 1,
            "n_segments_kept": 1,
            "largest_segment_frac": 1.0,
        }
        return [seg], qc

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)

    if len(t) <= 1:
        seg = df.copy()
        seg["_segment_id"] = 0
        seg["_segment_local_time"] = np.arange(len(seg), dtype=float)
        qc = {
            "raw_n_rows": len(df),
            "has_time": 1,
            "has_reset": 0,
            "has_large_gap": 0,
            "n_negative_dt": 0,
            "n_large_gaps": 0,
            "median_dt_raw": np.nan,
            "gap_threshold": np.nan,
            "n_segments_total": 1,
            "n_segments_kept": 1,
            "largest_segment_frac": 1.0,
        }
        return [seg], qc

    dt = np.diff(t)
    pos_dt = dt[np.isfinite(dt) & (dt > 0)]

    median_dt = np.median(pos_dt) if len(pos_dt) else (1.0 / 20.0)
    gap_threshold = max(gap_abs_seconds, gap_mult * median_dt)

    break_after = (~np.isfinite(dt)) | (dt <= 0) | (dt > gap_threshold)

    seg_id = np.zeros(len(df), dtype=int)
    seg_id[1:] = np.cumsum(break_after)

    all_lengths = []
    for sid in np.unique(seg_id):
        seg = df.loc[seg_id == sid].copy()
        all_lengths.append(len(seg))

        if len(seg) < min_frames:
            continue

        seg["_segment_id"] = sid
        seg["_segment_local_time"] = np.arange(len(seg), dtype=float) * median_dt
        out_segments.append(seg)

    largest_frac = (max(all_lengths) / len(df)) if len(all_lengths) else np.nan

    qc = {
        "raw_n_rows": len(df),
        "has_time": 1,
        "has_reset": int(np.any(dt <= 0)),
        "has_large_gap": int(np.any(dt > gap_threshold)),
        "n_negative_dt": int(np.sum(dt <= 0)),
        "n_large_gaps": int(np.sum(dt > gap_threshold)),
        "median_dt_raw": float(median_dt),
        "gap_threshold": float(gap_threshold),
        "n_segments_total": int(len(np.unique(seg_id))),
        "n_segments_kept": int(len(out_segments)),
        "largest_segment_frac": float(largest_frac),
        "raw_time_start": float(np.nanmin(t)) if np.isfinite(t).any() else np.nan,
        "raw_time_end": float(np.nanmax(t)) if np.isfinite(t).any() else np.nan,
        "raw_duration": float(np.nanmax(t) - np.nanmin(t)) if np.isfinite(t).any() else np.nan,
    }

    if len(out_segments) == 0:
        # fallback: keep everything in frame order with synthetic time
        seg = df.copy()
        seg["_segment_id"] = 0
        seg["_segment_local_time"] = np.arange(len(seg), dtype=float) * median_dt
        out_segments = [seg]
        qc["n_segments_kept"] = 1
        qc["largest_segment_frac"] = 1.0

    return out_segments, qc

# ============================================================================
# GEOMETRY HELPERS
# ============================================================================

def get_xyz(df: pd.DataFrame, name: str) -> np.ndarray:
    """Extract x, y, z coordinates for a landmark."""
    cols = [f"{name}.x", f"{name}.y", f"{name}.z"]
    return df[cols].to_numpy(dtype=float)


def safe_norm(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Numerically stable norm calculation."""
    return np.sqrt(np.sum(x * x, axis=axis) + eps)


def unit_vector(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Normalize vectors to unit length."""
    return x / np.expand_dims(safe_norm(x, axis=axis, eps=eps), axis=axis)


def angle_between_vectors(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Calculate angle between corresponding vectors in two arrays."""
    a_u = unit_vector(a, eps=eps)
    b_u = unit_vector(b, eps=eps)
    cosang = np.sum(a_u * b_u, axis=1)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.arccos(cosang)


def weighted_nanmean(values, weights):
    """Weighted mean ignoring NaN values."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    if mask.sum() == 0:
        return np.nan
    return np.average(values[mask], weights=weights[mask])

# ============================================================================
# 1D SIGNAL FEATURE EXTRACTION
# ============================================================================

def bandpower_from_psd(freqs, psd, lo, hi):
    """Calculate bandpower in a frequency range from power spectral density."""
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() == 0:
        return 0.0
    return float(trapezoid(psd[mask], freqs[mask]))


def summarize_1d_signal(x: np.ndarray, dt: float, prefix: str) -> dict:
    """
    Extract statistical and spectral features from a 1D time series.
    
    Returns:
      Dictionary with features like mean, std, spectral entropy, bandpowers, etc.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    out = {}
    if len(x) == 0:
        keys = [
            "mean", "std", "min", "max", "median", "iqr", "rms", "range",
            "madiff", "zcr", "dom_freq", "spec_entropy",
            "bp_0_3", "bp_3_6", "bp_6_9", "bp_9_12"
        ]
        return {f"{prefix}__{k}": np.nan for k in keys}

    centered = x - np.median(x)

    out[f"{prefix}__mean"] = float(np.mean(x))
    out[f"{prefix}__std"] = float(np.std(x))
    out[f"{prefix}__min"] = float(np.min(x))
    out[f"{prefix}__max"] = float(np.max(x))
    out[f"{prefix}__median"] = float(np.median(x))
    out[f"{prefix}__iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    out[f"{prefix}__rms"] = float(np.sqrt(np.mean(x**2)))
    out[f"{prefix}__range"] = float(np.max(x) - np.min(x))
    out[f"{prefix}__madiff"] = float(np.mean(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    out[f"{prefix}__zcr"] = float(np.mean(centered[:-1] * centered[1:] < 0)) if len(x) > 1 else 0.0

    if dt is not None and np.isfinite(dt) and dt > 0 and len(x) >= 16:
        fs = 1.0 / dt
        freqs, psd = signal.welch(centered, fs=fs, nperseg=min(256, len(centered)))
        psd = np.asarray(psd, dtype=float)

        if len(freqs) > 1 and np.sum(psd[1:]) > 0:
            idx = int(np.argmax(psd[1:]) + 1)
            out[f"{prefix}__dom_freq"] = float(freqs[idx])

            p = psd[1:] / np.sum(psd[1:])
            out[f"{prefix}__spec_entropy"] = float(-(p * np.log(p + 1e-12)).sum())

            out[f"{prefix}__bp_0_3"] = bandpower_from_psd(freqs, psd, 0.0, 3.0)
            out[f"{prefix}__bp_3_6"] = bandpower_from_psd(freqs, psd, 3.0, 6.0)
            out[f"{prefix}__bp_6_9"] = bandpower_from_psd(freqs, psd, 6.0, 9.0)
            out[f"{prefix}__bp_9_12"] = bandpower_from_psd(freqs, psd, 9.0, 12.0)
        else:
            out[f"{prefix}__dom_freq"] = np.nan
            out[f"{prefix}__spec_entropy"] = np.nan
            out[f"{prefix}__bp_0_3"] = np.nan
            out[f"{prefix}__bp_3_6"] = np.nan
            out[f"{prefix}__bp_6_9"] = np.nan
            out[f"{prefix}__bp_9_12"] = np.nan
    else:
        out[f"{prefix}__dom_freq"] = np.nan
        out[f"{prefix}__spec_entropy"] = np.nan
        out[f"{prefix}__bp_0_3"] = np.nan
        out[f"{prefix}__bp_3_6"] = np.nan
        out[f"{prefix}__bp_6_9"] = np.nan
        out[f"{prefix}__bp_9_12"] = np.nan

    return out

# ============================================================================
# CYCLIC MOTION ANALYSIS
# ============================================================================

def robust_smooth(x: np.ndarray) -> np.ndarray:
    """Smooth signal using Savitzky-Golay filter."""
    x = np.asarray(x, dtype=float)
    if len(x) < 9:
        return x.copy()
    win = min(len(x) if len(x) % 2 == 1 else len(x) - 1, 21)
    if win < 5:
        return x.copy()
    return signal.savgol_filter(x, window_length=win, polyorder=2)


def cycle_features(x: np.ndarray, dt: float, prefix: str) -> dict:
    """
    Extract features related to cyclic motion patterns.
    
    Returns:
      Dictionary with cycle duration, amplitude, interruptions, etc.
    """
    x = np.asarray(x, dtype=float)
    out = {
        f"{prefix}__n_peaks": np.nan,
        f"{prefix}__cycle_dur_mean": np.nan,
        f"{prefix}__cycle_dur_std": np.nan,
        f"{prefix}__cycle_dur_slope": np.nan,
        f"{prefix}__cycle_amp_mean": np.nan,
        f"{prefix}__cycle_amp_std": np.nan,
        f"{prefix}__cycle_amp_slope": np.nan,
        f"{prefix}__interruptions": np.nan,
    }

    if len(x) < 20 or dt is None or not np.isfinite(dt) or dt <= 0:
        return out

    xs = robust_smooth(x)
    fs = 1.0 / dt

    min_peak_dist = max(3, int(0.15 * fs))
    prominence = max(1e-4, 0.20 * np.std(xs))

    peaks, _ = signal.find_peaks(xs, distance=min_peak_dist, prominence=prominence)
    troughs, _ = signal.find_peaks(-xs, distance=min_peak_dist, prominence=prominence)

    out[f"{prefix}__n_peaks"] = int(len(peaks))

    if len(peaks) >= 2:
        cycle_durs = np.diff(peaks) * dt
        out[f"{prefix}__cycle_dur_mean"] = float(np.mean(cycle_durs))
        out[f"{prefix}__cycle_dur_std"] = float(np.std(cycle_durs))
        out[f"{prefix}__interruptions"] = int(np.sum(cycle_durs > 2.0 * np.median(cycle_durs)))

        if len(cycle_durs) >= 2:
            out[f"{prefix}__cycle_dur_slope"] = float(
                np.polyfit(np.arange(len(cycle_durs)), cycle_durs, 1)[0]
            )

        amps = []
        for lp, rp in zip(peaks[:-1], peaks[1:]):
            mid_troughs = troughs[(troughs > lp) & (troughs < rp)]
            if len(mid_troughs) == 0:
                continue
            trough_val = float(np.min(xs[mid_troughs]))
            peak_val = float(max(xs[lp], xs[rp]))
            amps.append(peak_val - trough_val)

        if len(amps) > 0:
            amps = np.asarray(amps, dtype=float)
            out[f"{prefix}__cycle_amp_mean"] = float(np.mean(amps))
            out[f"{prefix}__cycle_amp_std"] = float(np.std(amps))
            if len(amps) >= 2:
                out[f"{prefix}__cycle_amp_slope"] = float(
                    np.polyfit(np.arange(len(amps)), amps, 1)[0]
                )

    return out

# ============================================================================
# SEGMENT-LEVEL FEATURE EXTRACTION
# ============================================================================

def extract_segment_features(seg: pd.DataFrame) -> dict:
    """
    Extract motion features from a single continuous segment.
    
    Features include:
    - Hand geometry distances and angles
    - Spectral properties (frequency, entropy, bandpowers)
    - Cycle characteristics (duration, amplitude, interruptions)
    - Motion energy metrics
    """
    feats = {}

    t = seg["_segment_local_time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else np.nan

    # Load landmark positions
    pts = {lm: get_xyz(seg, lm) for lm in LANDMARKS}
    wrist = pts["WRIST"]

    # Translation invariance: wrist-centered coordinates
    rel = {lm: pts[lm] - wrist for lm in LANDMARKS}

    # Scale invariance: palm-relative scaling
    palm_scale = (
        safe_norm(rel["INDEX_FINGER_MCP"], axis=1) +
        safe_norm(rel["PINKY_MCP"], axis=1)
    ) / 2.0
    palm_scale = np.where(np.isfinite(palm_scale) & (palm_scale > 1e-6), palm_scale, np.nanmedian(palm_scale))
    palm_scale = np.where(np.isfinite(palm_scale) & (palm_scale > 1e-6), palm_scale, 1.0)

    rel_norm = {lm: rel[lm] / palm_scale[:, None] for lm in LANDMARKS}

    # Clinically relevant distance signals
    thumb_index_dist = safe_norm(rel_norm["THUMB_TIP"] - rel_norm["INDEX_FINGER_TIP"], axis=1)
    fingertip_wrist_dists = np.column_stack([safe_norm(rel_norm[lm], axis=1) for lm in TIPS])
    mean_fingertip_wrist = np.mean(fingertip_wrist_dists, axis=1)

    fingertip_spread = np.mean(
        np.column_stack([
            safe_norm(rel_norm[a] - rel_norm[b], axis=1) for a, b in ADJ_TIP_PAIRS
        ]),
        axis=1,
    )

    index_tip_mcp_dist = safe_norm(rel_norm["INDEX_FINGER_TIP"] - rel_norm["INDEX_FINGER_MCP"], axis=1)
    middle_tip_mcp_dist = safe_norm(rel_norm["MIDDLE_FINGER_TIP"] - rel_norm["MIDDLE_FINGER_MCP"], axis=1)

    # Palm orientation (pronation/supination)
    palm_v1 = rel_norm["INDEX_FINGER_MCP"]
    palm_v2 = rel_norm["PINKY_MCP"]
    palm_normal = np.cross(palm_v1, palm_v2)
    palm_normal = unit_vector(palm_normal)

    palm_nx = palm_normal[:, 0]
    palm_ny = palm_normal[:, 1]
    palm_nz = palm_normal[:, 2]

    palm_rot_speed = np.zeros(len(seg), dtype=float)
    if len(seg) > 1:
        palm_rot_speed[1:] = angle_between_vectors(palm_normal[:-1], palm_normal[1:]) / max(dt, 1e-8)

    # Pose-change and motion energy
    tip_stack = np.stack([rel_norm[lm] for lm in TIPS], axis=1)          # [T, 5, 3]
    all_stack = np.stack([rel_norm[lm] for lm in LANDMARKS], axis=1)     # [T, 21, 3]

    tip_motion_energy = np.zeros(len(seg), dtype=float)
    pose_motion_energy = np.zeros(len(seg), dtype=float)

    if len(seg) > 1:
        tip_motion_energy[1:] = np.mean(safe_norm(np.diff(tip_stack, axis=0), axis=2), axis=1) / max(dt, 1e-8)
        pose_motion_energy[1:] = np.mean(safe_norm(np.diff(all_stack, axis=0), axis=2), axis=1) / max(dt, 1e-8)

    # Thumb-index angle signal
    thumb_vec = rel_norm["THUMB_TIP"]
    index_vec = rel_norm["INDEX_FINGER_TIP"]
    thumb_index_angle = angle_between_vectors(thumb_vec, index_vec)

    # Compile all signals
    signals = {
        "thumb_index_dist": thumb_index_dist,
        "mean_fingertip_wrist": mean_fingertip_wrist,
        "fingertip_spread": fingertip_spread,
        "index_tip_mcp_dist": index_tip_mcp_dist,
        "middle_tip_mcp_dist": middle_tip_mcp_dist,
        "palm_nx": palm_nx,
        "palm_ny": palm_ny,
        "palm_nz": palm_nz,
        "palm_rot_speed": palm_rot_speed,
        "tip_motion_energy": tip_motion_energy,
        "pose_motion_energy": pose_motion_energy,
        "thumb_index_angle": thumb_index_angle,
    }

    # Extract statistical features for each signal
    for name, x in signals.items():
        feats.update(summarize_1d_signal(x, dt=dt, prefix=name))

    # Extract cyclic features from key signals
    feats.update(cycle_features(thumb_index_dist, dt=dt, prefix="thumb_index_dist"))
    feats.update(cycle_features(fingertip_spread, dt=dt, prefix="fingertip_spread"))
    feats.update(cycle_features(palm_rot_speed, dt=dt, prefix="palm_rot_speed"))

    # Segment quality control
    feats["segment_n_rows"] = len(seg)
    feats["segment_duration"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    feats["segment_dt"] = dt

    return feats

# ============================================================================
# MULTI-SEGMENT AGGREGATION
# ============================================================================

def aggregate_segment_features(segment_feature_dicts, segment_weights):
    """Aggregate features from multiple segments using weighted nanmean."""
    keys = sorted(set().union(*[d.keys() for d in segment_feature_dicts]))
    out = {}

    for k in keys:
        vals = [d.get(k, np.nan) for d in segment_feature_dicts]
        out[k] = weighted_nanmean(vals, segment_weights)

    return out

# ============================================================================
# FILE-LEVEL FEATURIZATION
# ============================================================================

def featurize_file(file_path: str) -> dict:
    """
    Extract features from a complete raw data file.
    
    Process:
    1. Load CSV
    2. Split into continuous segments
    3. Extract features from each segment
    4. Aggregate back to one row per file
    """
    df = pd.read_csv(file_path)

    segments, qc = split_into_segments(df)

    seg_feats = []
    seg_weights = []

    for seg in segments:
        try:
            f = extract_segment_features(seg)
            seg_feats.append(f)
            seg_weights.append(len(seg))
        except Exception as e:
            warnings.warn(
                f"Feature extraction failed for segment in {os.path.basename(file_path)}: {e}"
            )

    if len(seg_feats) == 0:
        base = {}
        base["feature_extraction_failed"] = 1
    else:
        base = aggregate_segment_features(seg_feats, seg_weights)
        base["feature_extraction_failed"] = 0

    base.update(qc)
    base["data_file_name"] = os.path.basename(file_path)
    return base


def build_feature_table(meta_df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Build complete feature table from metadata and raw files."""
    rows = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        fname = row["data_file_name"]
        fpath = os.path.join(RAW_DIR, fname)

        feats = featurize_file(fpath)

        # Append metadata
        feats["gender"] = row["gender"]
        feats["age"] = row["age"]
        feats["age_missing"] = row["age_missing"]
        feats["patient_off_on"] = row["patient_off_on"]
        feats["doctor_diagnosis_0_5"] = row["doctor_diagnosis_0_5"]

        if is_train:
            feats["folder_path"] = row["folder_path"]

        rows.append(feats)

    return pd.DataFrame(rows)


# Build feature tables
print("Building feature tables...")
train_feat = build_feature_table(train, is_train=True)
test_feat  = build_feature_table(test, is_train=False)

print(f"Train features: {train_feat.shape}, Test features: {test_feat.shape}")
print(f"Feature extraction failures:\n{train_feat['feature_extraction_failed'].value_counts(dropna=False)}")

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

# Prepare training data
target_col = "folder_path"
drop_cols = ["data_file_name", target_col]

X = train_feat.drop(columns=drop_cols, errors="ignore").copy()
y = train_feat[target_col].copy()

# Enforce metadata dtypes
for c in ["gender", "patient_off_on"]:
    if c in X.columns:
        X[c] = X[c].astype("string")

for c in ["age", "age_missing", "doctor_diagnosis_0_5"]:
    if c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

# Coerce accidentally string numeric columns back to numeric
known_cat = {"gender", "patient_off_on"}
for c in X.columns:
    if c not in known_cat:
        if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_string_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="ignore")

# Identify categorical and numeric columns
cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print(f"Categorical columns: {cat_cols}")
print(f"Number of numeric columns: {len(num_cols)}")

# Build preprocessing pipeline
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]), cat_cols),
    ],
    remainder="drop",
)

# Define models
logreg_pipe = Pipeline([
    ("prep", preprocess),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )),
])

rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=700,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
])

et_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", ExtraTreesClassifier(
        n_estimators=900,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
])

models = {
    "LogReg": logreg_pipe,
    "RF": rf_pipe,
    "ExtraTrees": et_pipe,
}

# Try to add CatBoost if available
try:
    from catboost import CatBoostClassifier

    cb_pipe = Pipeline([
        ("prep", preprocess),
        ("clf", CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            depth=6,
            learning_rate=0.05,
            iterations=800,
            l2_leaf_reg=5,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )),
    ])
    models["CatBoost"] = cb_pipe
except Exception as e:
    print(f"CatBoost not available, skipping it: {e}")

# Cross-validation setup
GROUP_COL = None
groups = train_feat[GROUP_COL] if (GROUP_COL is not None and GROUP_COL in train_feat.columns) else None

if groups is not None:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fit_kwargs = {"groups": groups}
else:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fit_kwargs = {}

scoring = {
    "accuracy": "accuracy",
    "balanced_acc": "balanced_accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}

# Run model benchmarking
print("\nBenchmarking models...")
rows = []
for name, model in models.items():
    print(f"  Running {name}...")
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
        **fit_kwargs,
    )
    rows.append({
        "model": name,
        "acc_mean": scores["test_accuracy"].mean(),
        "acc_std": scores["test_accuracy"].std(),
        "bal_acc_mean": scores["test_balanced_acc"].mean(),
        "bal_acc_std": scores["test_balanced_acc"].std(),
        "macro_f1_mean": scores["test_macro_f1"].mean(),
        "macro_f1_std": scores["test_macro_f1"].std(),
        "weighted_f1_mean": scores["test_weighted_f1"].mean(),
        "weighted_f1_std": scores["test_weighted_f1"].std(),
    })

results_df = pd.DataFrame(rows).sort_values(
    ["macro_f1_mean", "bal_acc_mean", "acc_mean"],
    ascending=False
).reset_index(drop=True)

print("\nModel Benchmark Results:")
print(results_df.to_string())

# Analyze best model
best_name = results_df.iloc[0]["model"]
best_model = models[best_name]

y_pred = cross_val_predict(
    best_model,
    X,
    y,
    cv=cv,
    n_jobs=1,
    **fit_kwargs,
)

print(f"\nBest Model: {best_name}")
print(f"Accuracy        : {accuracy_score(y, y_pred):.4f}")
print(f"Balanced Acc    : {balanced_accuracy_score(y, y_pred):.4f}")
print(f"Macro F1        : {f1_score(y, y_pred, average='macro'):.4f}")
print(f"Weighted F1     : {f1_score(y, y_pred, average='weighted'):.4f}")
print()
print(classification_report(y, y_pred, digits=4))

cm = confusion_matrix(y, y_pred, labels=sorted(y.unique()), normalize="true")
cm_df = pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique()))
print("\nConfusion Matrix (Normalized):")
print(cm_df.to_string())

# ============================================================================
# GENERATE TEST PREDICTIONS
# ============================================================================

print("\nGenerating test predictions...")
X_test = test_feat[X.columns].copy()

test_pred = best_model.predict(X_test)
test_pred = np.asarray(test_pred).reshape(-1)

submission = pd.DataFrame({
    "path": test_feat["data_file_name"].to_numpy(),
    "pred": test_pred,
})

submission.to_csv(SUBMISSION_OUT, index=False)
print(f"Saved submission to: {SUBMISSION_OUT}")
print(f"Submission shape: {submission.shape}")
print(f"\nPrediction distribution:")
print(submission["pred"].value_counts())