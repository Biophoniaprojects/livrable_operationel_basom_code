#!/usr/bin/env python3
"""
Puffin abundance estimation workflow.

This script:
- loads validated annotation tables
- extracts E1 / I1 / E2 / I2 syllables from candidate audio clips
- computes the retained feature set
- performs dimensionality reduction and clustering per station
- generates 2D clustering figures
- estimates nest counts per station from a pre-calibrated Negative Binomial model

Refactored for a simple operational workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hdbscan
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from scipy.signal import butter, filtfilt, find_peaks, medfilt, savgol_filter
from sklearn.preprocessing import StandardScaler
from umap import UMAP


# ============================================================
# --- CONSTANTS ---
# ============================================================

MFCC_ORDER = 5

DURATION_FEATURES = ["DE1", "DE2", "DINS1", "DI1", "DISS", "DINS2", "DI2"]
MFCC_FEATURES = [
    "mfcc_E1_1",
    "mfcc_E1_2",
    "mfcc_E1_3",
    "mfcc_E1_4",
    "mfcc_E1_5",
    "mfcc_E2_1",
    "mfcc_E2_2",
    "mfcc_E2_3",
    "mfcc_E2_4",
    "mfcc_E2_5",
]
PITCH_FEATURES = ["mean", "median", "std", "range"]

FINAL_FEATURE_COLUMNS = DURATION_FEATURES + MFCC_FEATURES + PITCH_FEATURES

PARSELMOUTH_COLUMNS = [
    "Median pitch",
    "Mean pitch",
    "Standard deviation",
    "Minimum pitch",
    "Maximum pitch",
    "Number of pulses",
    "Number of periods",
    "Mean period",
    "Standard deviation of period",
    "Fraction of locally unvoiced frames",
    "Number of voice breaks",
    "Degree of voice breaks",
    "Jitter (local)",
    "Jitter (local, absolute)",
    "Jitter (rap)",
    "Jitter (ppq5)",
    "Jitter (ddp)",
    "Shimmer (local)",
    "Shimmer (local, dB)",
    "Shimmer (apq3)",
    "Shimmer (apq5)",
    "Shimmer (apq11)",
    "Shimmer (dda)",
    "Mean autocorrelation",
    "Mean noise-to-harmonics ratio",
    "Mean harmonics-to-noise ratio",
]

# Parselmouth parameters
TIME_STEP = 0.0
PITCH_FLOOR = 190
PITCH_CEILING = 620
MAX_CANDIDATES = 15
VERY_ACCURATE = "off"
SILENCE_THRESHOLD = 0.03
VOICING_THRESHOLD = 0.45
OCTAVE_COST = 0.1
OCTAVE_JUMP_COST = 0.35
VOICED_UNVOICED_COST = 0.14

VOICE_REPORT_START_TIME = 0.0
VOICE_REPORT_END_TIME = 0.0
MAX_PERIOD_FACTOR = 1.3
MAX_AMP_FACTOR = 1.6

DEFAULT_FULL_FEATURES_CSV = "puffin_features_full.csv"
DEFAULT_SELECTED_FEATURES_CSV = "puffin_features_selected.csv"
DEFAULT_CLUSTERED_SAMPLES_CSV = "puffin_clustered_samples.csv"
DEFAULT_CLUSTER_SUMMARY_CSV = "puffin_cluster_summary_by_station.csv"
DEFAULT_NEST_ESTIMATION_CSV = "puffin_nest_estimation_by_station.csv"

# Default DR / clustering params
UMAP_CLUSTER_PARAMS = dict(
    n_neighbors=5,
    min_dist=0.0,
    n_components=5,
    metric="euclidean",
    random_state=42,
)
UMAP_PLOT_PARAMS = dict(
    n_neighbors=5,
    min_dist=0.0,
    n_components=2,
    metric="euclidean",
    random_state=42,
)

# Default NB parameters
# Calibrated Negative Binomial model coefficients.
# Mean prediction under log-link:
#   mu = exp(intercept + coef_clusters * n_clusters + coef_outliers * n_outliers)
#
# Best fitted model:
#   const      = 0.6884652688315651
#   n_clusters = 0.12891930346471553
#   n_outliers = 0.1311595166409341
#   alpha      = 1.9361873882105654e-08
#
# Note:
# alpha is kept here as metadata only. The current script uses the mean prediction mu
# and does not simulate the full Negative Binomial distribution.
NB_INTERCEPT = 0.6884652688315651
NB_COEF_CLUSTERS = 0.12891930346471553
NB_COEF_OUTLIERS = 0.1311595166409341
NB_ALPHA = 1.9361873882105654e-08


# ============================================================
# --- CLI ---
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrait les features utiles pour le workflow Puffin, puis réalise "
            "la réduction de dimension, le clustering et l'estimation du nombre de nids."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dossier racine contenant les fichiers audio candidats annotés.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        required=True,
        help="CSV d'annotations validées issu de l'interface graphique.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dossier où écrire les syllabes, features, clustering et figures.",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Liste optionnelle de stations à traiter.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode test : limite le nombre d'annotations traitées.",
    )
    parser.add_argument(
        "--debug-max-rows",
        type=int,
        default=20,
        help="Nombre maximal de lignes à traiter en mode debug.",
    )

    # Clustering
    parser.add_argument("--umap-n-neighbors", type=int, default=5)
    parser.add_argument("--umap-min-dist", type=float, default=0.0)
    parser.add_argument("--umap-cluster-dim", type=int, default=5)
    parser.add_argument("--umap-plot-dim", type=int, default=2)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=6)
    parser.add_argument("--hdbscan-min-samples", type=int, default=3)

    # Negative binomial estimation
    parser.add_argument("--nb-intercept", type=float, default=NB_INTERCEPT)
    parser.add_argument("--nb-coef-clusters", type=float, default=NB_COEF_CLUSTERS)
    parser.add_argument("--nb-coef-outliers", type=float, default=NB_COEF_OUTLIERS)

    return parser.parse_args()


# ============================================================
# --- HELPERS ---
# ============================================================


def load_annotations(features_file: Path) -> pd.DataFrame:
    if not features_file.exists():
        raise FileNotFoundError(f"Fichier d'annotations introuvable : {features_file}")
    return pd.read_csv(features_file, sep=None, engine="python", encoding="utf-8")


def filter_annotation_rows(
    df: pd.DataFrame,
    stations: list[str] | None,
    debug: bool,
    debug_max_rows: int,
) -> pd.DataFrame:
    df = df.copy()

    if "SE_valid" in df.columns:
        df["SE_valid"] = df["SE_valid"].astype(str).str.strip().str.lower().eq("true")

    if stations is not None and "station" in df.columns:
        df = df[df["station"].isin(stations)].copy()

    if debug:
        df = df.head(debug_max_rows).copy()

    return df.reset_index(drop=True)


def resolve_audio_path(input_dir: Path, row: pd.Series) -> Path:
    if "wav_path" in row and pd.notna(row["wav_path"]):
        return Path(str(row["wav_path"])).resolve()
    if "fileName" in row and pd.notna(row["fileName"]):
        return (input_dir / str(row["fileName"])).resolve()
    raise ValueError("Impossible de résoudre le chemin audio pour cette ligne.")


def ensure_syllable_dirs(output_dir: Path) -> dict[str, Path]:
    syllable_dirs = {}
    for syllable in ["E1", "I1", "E2", "I2"]:
        path = output_dir / "syllables" / syllable
        path.mkdir(parents=True, exist_ok=True)
        syllable_dirs[syllable] = path
    return syllable_dirs


def export_single_syllable(
    audio_path: Path, start_time: float, end_time: float, out_path: Path
) -> None:
    if np.isnan(start_time) or np.isnan(end_time) or end_time <= start_time:
        return
    y, sr = librosa.load(
        audio_path,
        offset=float(start_time),
        duration=float(end_time - start_time),
        sr=None,
    )
    sf.write(out_path, y, sr)


def extract_syllables(
    df: pd.DataFrame, input_dir: Path, output_dir: Path
) -> pd.DataFrame:
    syllable_dirs = ensure_syllable_dirs(output_dir)

    for idx, row in df.iterrows():
        if "SE_valid" in row and not bool(row["SE_valid"]):
            continue

        try:
            audio_path = resolve_audio_path(input_dir, row)
        except Exception as exc:
            print(f"[WARNING] Ligne {idx} ignorée, chemin audio introuvable : {exc}")
            continue

        if not audio_path.exists():
            print(f"[WARNING] Fichier audio introuvable : {audio_path}")
            continue

        file_stem = Path(audio_path).stem
        segments = [
            ("E1", row.get("t0", np.nan), row.get("tE1", np.nan)),
            ("I1", row.get("tINS1", np.nan), row.get("tI1", np.nan)),
            ("E2", row.get("tISS", np.nan), row.get("tE2", np.nan)),
            ("I2", row.get("tINS2", np.nan), row.get("t_end", np.nan)),
        ]

        for syllable, t_start, t_end in segments:
            if pd.isna(t_start) or pd.isna(t_end):
                continue
            out_path = syllable_dirs[syllable] / f"{file_stem}_{syllable}.wav"
            try:
                export_single_syllable(
                    audio_path, float(t_start), float(t_end), out_path
                )
            except Exception as exc:
                print(
                    f"[WARNING] Extraction impossible pour {file_stem} / {syllable}: {exc}"
                )

    return df


# ============================================================
# --- FEATURE COMPUTATION ---
# ============================================================


def parse_voice_report(voice_report_str: str, name: str) -> pd.Series:
    lines = voice_report_str.split("\n")[1:]
    data = {}

    for line in lines:
        if ":" not in line or line.endswith(":"):
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if "undefined" in value:
            value = np.nan
        elif value.endswith("%"):
            value = float(value[:-1])
        elif value.endswith("seconds"):
            value = float(value[:-7].replace("E", "e"))
        elif key in {"Fraction of locally unvoiced frames", "Degree of voice breaks"}:
            value = float(value.split("%", 1)[0]) if "%" in value else 0.0
        else:
            value = float(value.split()[0])

        data[key] = value

    series = pd.Series(data)
    series.name = name
    return series


def calculate_parselmouth_features(
    row: pd.Series, syllable: str, syllable_root: Path
) -> pd.Series:
    if not bool(row.get("SE_valid", False)) or pd.isna(row.get("DE1", np.nan)):
        s = pd.Series([np.nan] * len(PARSELMOUTH_COLUMNS), index=PARSELMOUTH_COLUMNS)
        s.index = [f"{syllable}_{col}" for col in s.index]
        return s

    audio_name = Path(str(row["resolved_file_name"])).stem + f"_{syllable}.wav"
    input_file = syllable_root / syllable / audio_name

    try:
        praat_sound = parselmouth.Sound(str(input_file))
    except Exception:
        s = pd.Series([np.nan] * len(PARSELMOUTH_COLUMNS), index=PARSELMOUTH_COLUMNS)
        s.index = [f"{syllable}_{col}" for col in s.index]
        return s

    if praat_sound.duration < 3 / PITCH_FLOOR:
        s = pd.Series([np.nan] * len(PARSELMOUTH_COLUMNS), index=PARSELMOUTH_COLUMNS)
        s.index = [f"{syllable}_{col}" for col in s.index]
        return s

    pitch = parselmouth.praat.call(
        praat_sound,
        "To Pitch (ac)",
        TIME_STEP,
        PITCH_FLOOR,
        MAX_CANDIDATES,
        VERY_ACCURATE,
        SILENCE_THRESHOLD,
        VOICING_THRESHOLD,
        OCTAVE_COST,
        OCTAVE_JUMP_COST,
        VOICED_UNVOICED_COST,
        PITCH_CEILING,
    )

    pulses = parselmouth.praat.call([praat_sound, pitch], "To PointProcess (cc)")
    voice_report_str = parselmouth.praat.call(
        [praat_sound, pitch, pulses],
        "Voice report",
        VOICE_REPORT_START_TIME,
        VOICE_REPORT_END_TIME,
        PITCH_FLOOR,
        PITCH_CEILING,
        MAX_PERIOD_FACTOR,
        MAX_AMP_FACTOR,
        SILENCE_THRESHOLD,
        VOICING_THRESHOLD,
    )

    s = parse_voice_report(voice_report_str, audio_name)
    s.index = [f"{syllable}_{col}" for col in s.index]
    return s


def calculate_mfcc_features(df: pd.DataFrame, syllable_root: Path) -> pd.DataFrame:
    mfcc_rows = []

    for _, row in df.iterrows():
        if not bool(row.get("SE_valid", False)) or pd.isna(row.get("DE1", np.nan)):
            mfcc_rows.append(np.full(10, np.nan))
            continue

        stem = Path(str(row["resolved_file_name"])).stem

        try:
            y, sr = librosa.load(syllable_root / "E1" / f"{stem}_E1.wav", sr=None)
            mfcc_e1 = librosa.feature.mfcc(y=y, sr=sr)[:MFCC_ORDER, :]
            y, sr = librosa.load(syllable_root / "E2" / f"{stem}_E2.wav", sr=None)
            mfcc_e2 = librosa.feature.mfcc(y=y, sr=sr)[:MFCC_ORDER, :]
        except Exception:
            mfcc_rows.append(np.full(10, np.nan))
            continue

        mfcc_rows.append(
            np.concatenate([np.mean(mfcc_e1, axis=1), np.mean(mfcc_e2, axis=1)])
        )

    mfcc_array = np.vstack(mfcc_rows)
    df[
        [f"mfcc_E1_{i + 1}" for i in range(MFCC_ORDER)]
        + [f"mfcc_E2_{i + 1}" for i in range(MFCC_ORDER)]
    ] = mfcc_array
    return df


# ============================================================
# --- PITCH COMPUTATION WITH HARMONIC CORRECTION ---
# ============================================================


def highpass_filter(y, sr, cutoff=200.0, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, y)


def track_f0_parselmouth(
    y,
    sr,
    time_step=0.0,
    pitch_floor=350,
    pitch_ceiling=650,
    max_candidates=15,
    very_accurate="off",
    silence_threshold=0.03,
    voicing_threshold=0.45,
    octave_cost=0.01,
    octave_jump_cost=0.35,
    voiced_unvoiced_cost=0.14,
):
    """
    Pitch tracking with Parselmouth / Praat.
    """
    pm_sound = parselmouth.Sound(y, sampling_frequency=sr)

    pitch = parselmouth.praat.call(
        pm_sound,
        "To Pitch (ac)",
        time_step,
        pitch_floor,
        max_candidates,
        very_accurate,
        silence_threshold,
        voicing_threshold,
        octave_cost,
        octave_jump_cost,
        voiced_unvoiced_cost,
        pitch_ceiling,
    )

    f0 = pitch.selected_array["frequency"]
    f0[f0 == 0] = np.nan
    times = pitch.xs()

    invalid = (f0 < pitch_floor) | (f0 > pitch_ceiling)
    f0[invalid] = np.nan

    return times, f0


def track_f0_harmonic_coherent(
    y,
    sr,
    f0_min,
    f0_max,
    search_fmin,
    search_fmax,
    n_fft=2048,
    hop_length=None,
    peak_min_distance_hz=300.0,
    peak_min_prominence=0.02,
    spec_smooth_window_hz=50.0,
    spec_smooth_poly=2,
    max_harmonic_order=15,
    harmonic_tol_hz=8.0,
    min_harmonic_match=4,
    median_kernel=5,
    sg_window=9,
    sg_poly=2,
):
    """
    Pitch tracking using:
    - wide-band peak detection
    - direct f0 candidate if dominant peak is in band
    - harmonic coherence fallback
    - temporal smoothing
    """
    if hop_length is None:
        hop_length = n_fft // 8

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    band_mask = (freqs >= search_fmin) & (freqs <= search_fmax)
    freqs_band = freqs[band_mask]
    S_band = S[band_mask]

    f0_raw = np.full(S.shape[1], np.nan)

    hz_per_bin = freqs[1] - freqs[0]
    min_dist_bins = int(peak_min_distance_hz / hz_per_bin)

    win_bins = int(spec_smooth_window_hz / hz_per_bin)
    if win_bins % 2 == 0:
        win_bins += 1
    win_bins = max(win_bins, 3)

    for t in range(S.shape[1]):
        spectrum = S_band[:, t]
        if np.all(spectrum == 0):
            continue

        spectrum_s = savgol_filter(
            spectrum,
            window_length=win_bins,
            polyorder=spec_smooth_poly,
        )

        peaks, _ = find_peaks(
            spectrum_s,
            distance=min_dist_bins,
            prominence=peak_min_prominence,
        )

        if len(peaks) == 0:
            continue

        peak_freqs = freqs_band[peaks]
        peak_amps = spectrum_s[peaks]

        idx_dom = np.argmax(peak_amps)
        f_peak = peak_freqs[idx_dom]

        if f0_min <= f_peak <= f0_max:
            f0_raw[t] = f_peak
            continue

        best_f0 = np.nan
        best_support = 0

        for f in peak_freqs:
            for k in range(2, max_harmonic_order + 1):
                f0 = f / k
                if not (f0_min <= f0 <= f0_max):
                    continue

                ks = np.round(peak_freqs / f0).astype(int)
                recon = ks * f0
                err = np.abs(recon - peak_freqs)

                support = np.sum(err < harmonic_tol_hz)

                if support >= min_harmonic_match and support > best_support:
                    best_support = support
                    best_f0 = f0

        if best_support >= min_harmonic_match:
            f0_raw[t] = best_f0

    f0_smooth = f0_raw.copy()

    if median_kernel >= 3:
        f0_smooth = medfilt(f0_smooth, kernel_size=median_kernel)

    valid = ~np.isnan(f0_smooth)
    if np.sum(valid) > sg_window:
        f0_smooth[valid] = savgol_filter(
            f0_smooth[valid],
            window_length=sg_window,
            polyorder=sg_poly,
        )

    invalid = (f0_smooth < f0_min) | (f0_smooth > f0_max)
    f0_smooth[invalid] = np.nan

    invalid_raw = (f0_raw < f0_min) | (f0_raw > f0_max)
    f0_raw[invalid_raw] = np.nan

    return times, f0_raw, f0_smooth


def correct_parselmouth_with_harmonics(
    times_pm,
    f0_pm,
    times_h,
    f0_h,
    duration,
    bin_ratio=0.01,
    min_valid_ratio=0.3,
):
    """
    Fill missing Parselmouth pitch regions with harmonic-coherent pitch.
    """
    bin_size = duration * bin_ratio
    bin_edges = np.arange(0, duration + bin_size, bin_size)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def bin_pitch(times, f0):
        f0_bin = np.full(len(bin_centers), np.nan)

        for i in range(len(bin_centers)):
            mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
            vals = f0[mask]

            if len(vals) == 0:
                continue

            valid = vals[~np.isnan(vals)]
            if len(valid) / len(vals) < min_valid_ratio:
                continue

            f0_bin[i] = np.median(valid)

        return f0_bin

    f0_pm_bin = bin_pitch(times_pm, f0_pm)
    f0_h_bin = bin_pitch(times_h, f0_h)

    f0_corrected = f0_pm_bin.copy()
    missing = np.isnan(f0_corrected)
    f0_corrected[missing] = f0_h_bin[missing]

    return bin_centers, f0_corrected, f0_pm_bin, f0_h_bin


def crop_binned_pitch(
    bin_centers,
    f0,
    crop_start_ratio=0.1,
    crop_end_ratio=0.1,
):
    assert len(bin_centers) == len(f0)

    n = len(bin_centers)
    i_start = int(np.floor(crop_start_ratio * n))
    i_end = int(np.ceil((1.0 - crop_end_ratio) * n))

    return bin_centers[i_start:i_end], f0[i_start:i_end]


def smooth_binned_pitch(
    f0,
    median_kernel=5,
    sg_window=9,
    sg_poly=2,
):
    f0_smooth = f0.copy()

    if median_kernel >= 3 and median_kernel % 2 == 1:
        valid = ~np.isnan(f0_smooth)
        if np.sum(valid) >= median_kernel:
            tmp = f0_smooth.copy()
            tmp[~valid] = 0.0
            filtered = medfilt(tmp, kernel_size=median_kernel)
            f0_smooth[valid] = filtered[valid]

    valid = ~np.isnan(f0_smooth)
    if np.sum(valid) >= sg_window and sg_window > sg_poly:
        f0_valid = f0_smooth[valid]

        if sg_window % 2 == 0:
            sg_window += 1

        f0_smooth[valid] = savgol_filter(
            f0_valid,
            window_length=sg_window,
            polyorder=sg_poly,
        )

    return f0_smooth


def pitch_statistics(f0):
    valid = ~np.isnan(f0)
    if np.sum(valid) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "range": np.nan,
        }

    f = f0[valid]
    return {
        "mean": float(np.mean(f)),
        "median": float(np.median(f)),
        "std": float(np.std(f)),
        "range": float(np.max(f) - np.min(f)),
    }


def coverage_status(cov):
    if np.isnan(cov):
        return "invalid"
    elif cov < 0.30:
        return "low"
    elif cov < 0.80:
        return "medium"
    else:
        return "high"


def process_single_wav(
    wav_path,
    hp_freq=250,
    f0_min=350,
    f0_max=500,
    nfft=2048,
    peak_min_pro=0.02,
    bin_ratio=0.01,
):
    """
    Full pitch pipeline:
    1) Parselmouth pitch
    2) harmonic-coherent fallback
    3) correction / binning
    4) crop + smoothing
    5) simple summary features only
    """
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    duration = len(y) / sr

    y_hp = highpass_filter(y, sr, cutoff=hp_freq)
    times_pm, f0_pm = track_f0_parselmouth(
        y_hp,
        sr,
        pitch_floor=f0_min,
        pitch_ceiling=f0_max,
    )

    times_h, _, f0_h = track_f0_harmonic_coherent(
        y,
        sr,
        f0_min=f0_min,
        f0_max=f0_max,
        search_fmin=100,
        search_fmax=5000,
        n_fft=nfft,
        peak_min_prominence=peak_min_pro,
    )

    bin_centers, f0_corr, _, _ = correct_parselmouth_with_harmonics(
        times_pm=times_pm,
        f0_pm=f0_pm,
        times_h=times_h,
        f0_h=f0_h,
        duration=duration,
        bin_ratio=bin_ratio,
        min_valid_ratio=0.3,
    )

    cov_corr = np.sum(~np.isnan(f0_corr)) / len(f0_corr) if len(f0_corr) > 0 else np.nan

    bin_centers, f0_corr = crop_binned_pitch(
        bin_centers,
        f0_corr,
        crop_start_ratio=0.1,
        crop_end_ratio=0.1,
    )

    f0_corr = smooth_binned_pitch(f0_corr)

    invalid = (f0_corr < f0_min) | (f0_corr > f0_max)
    f0_corr[invalid] = np.nan

    return {
        "coverage_ratio": cov_corr,
        "coverage_status": coverage_status(cov_corr),
        **pitch_statistics(f0_corr),
    }


def calculate_custom_pitch_features(
    df: pd.DataFrame, syllable_root: Path
) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        if not bool(row.get("SE_valid", False)):
            rows.append(
                {
                    "coverage_ratio": np.nan,
                    "coverage_status": "invalid",
                    "mean": np.nan,
                    "median": np.nan,
                    "std": np.nan,
                    "range": np.nan,
                }
            )
            continue

        stem = Path(str(row["resolved_file_name"])).stem
        wav_path = syllable_root / "E1" / f"{stem}_E1.wav"

        try:
            feats = process_single_wav(
                wav_path=wav_path,
                hp_freq=250,
                f0_min=350,
                f0_max=500,
                nfft=2048,
                peak_min_pro=0.02,
                bin_ratio=0.01,
            )
        except Exception as exc:
            print(f"[WARNING] Pitch features impossibles pour {wav_path}: {exc}")
            feats = {
                "coverage_ratio": np.nan,
                "coverage_status": "invalid",
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "range": np.nan,
            }

        rows.append(feats)

    return pd.concat([df, pd.DataFrame(rows, index=df.index)], axis=1)


# ============================================================
# --- PIPELINE: FEATURES ---
# ============================================================


def compute_duration_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = df["t_end"] - df["t0"]
    df["DE1"] = df["tE1"] - df["t0"]
    df["DINS1"] = df["tINS1"] - df["tE1"]
    df["DI1"] = df["tI1"] - df["tINS1"]
    df["DISS"] = df["tISS"] - df["tI1"]
    df["DE2"] = df["tE2"] - df["tISS"]
    df["DINS2"] = df["tINS2"] - df["tE2"]
    df["DI2"] = df["t_end"] - df["tINS2"]
    return df


def add_resolved_file_info(df: pd.DataFrame, input_dir: Path) -> pd.DataFrame:
    df = df.copy()
    resolved_paths = []
    resolved_names = []

    for _, row in df.iterrows():
        audio_path = resolve_audio_path(input_dir, row)
        resolved_paths.append(str(audio_path))
        resolved_names.append(audio_path.name)

    df["resolved_wav_path"] = resolved_paths
    df["resolved_file_name"] = resolved_names
    return df


def compute_all_features(
    df: pd.DataFrame, input_dir: Path, output_dir: Path
) -> pd.DataFrame:
    df = add_resolved_file_info(df, input_dir)
    df = compute_duration_columns(df)
    df = extract_syllables(df, input_dir, output_dir)

    syllable_root = output_dir / "syllables"

    df = calculate_mfcc_features(df, syllable_root)

    # Conservé dans la table complète si besoin d'inspection
    for syllable in ["E1", "I1", "E2", "I2"]:
        new_features = df.apply(
            calculate_parselmouth_features,
            axis=1,
            args=(syllable, syllable_root),
        )
        df = pd.concat([df, new_features], axis=1)

    df = calculate_custom_pitch_features(df, syllable_root)
    return df


def build_selected_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "SE_valid" in df.columns:
        df = df[df["SE_valid"]].copy()

    if "DE1" in df.columns:
        df = df[~df["DE1"].isna()].copy()

    if "coverage_status" in df.columns:
        df = df[df["coverage_status"].isin({"medium", "high"})].copy()

    available_features = [col for col in FINAL_FEATURE_COLUMNS if col in df.columns]
    missing = sorted(set(FINAL_FEATURE_COLUMNS) - set(available_features))
    if missing:
        print(f"[WARNING] Features manquantes ignorées : {missing}")

    keep_cols = ["resolved_file_name"] + available_features
    if "station" in df.columns:
        keep_cols.insert(1, "station")

    selected = df[keep_cols].copy()
    selected = selected.dropna(axis=0, how="any").reset_index(drop=True)
    selected = selected.rename(columns={"resolved_file_name": "fileName"})
    return selected


# ============================================================
# --- DIMENSIONALITY REDUCTION / CLUSTERING ---
# ============================================================


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    meta_cols = {"fileName", "station"}
    numeric_cols = df.select_dtypes(include="number").columns
    return [c for c in numeric_cols if c not in meta_cols]


def cluster_station(
    df_station: pd.DataFrame,
    umap_cluster_params: dict,
    umap_plot_params: dict,
    min_cluster_size: int,
    min_samples: int,
):
    feat_cols = get_feature_cols(df_station)
    X = df_station[feat_cols].dropna()
    if len(X) < 10:
        return None, None, None, None

    Xs = StandardScaler().fit_transform(X.values).astype(np.float64)

    Xu_cluster = UMAP(**umap_cluster_params).fit_transform(Xs)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
    ).fit_predict(Xu_cluster)

    Xu_plot = UMAP(**umap_plot_params).fit_transform(Xs)

    df_out = df_station.loc[X.index].copy()
    df_out["cluster_label"] = labels
    df_out["umap1"] = Xu_plot[:, 0]
    df_out["umap2"] = Xu_plot[:, 1]

    n_outliers = int(np.sum(labels == -1))
    unique_labels = set(labels.tolist())
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    noise_ratio = float(n_outliers / len(labels)) if len(labels) else np.nan

    summary = {
        "n_samples": int(len(labels)),
        "n_clusters": int(n_clusters),
        "n_outliers": int(n_outliers),
        "noise_ratio": float(noise_ratio),
    }

    return df_out, Xu_plot, labels, summary


def plot_station_clusters(
    df_station_clustered: pd.DataFrame, station: str, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    labels = sorted(df_station_clustered["cluster_label"].dropna().unique())

    for lab in labels:
        mask = df_station_clustered["cluster_label"] == lab
        if lab == -1:
            ax.scatter(
                df_station_clustered.loc[mask, "umap1"],
                df_station_clustered.loc[mask, "umap2"],
                s=18,
                alpha=0.6,
                label="Outliers",
            )
        else:
            ax.scatter(
                df_station_clustered.loc[mask, "umap1"],
                df_station_clustered.loc[mask, "umap2"],
                s=24,
                alpha=0.85,
                label=f"Cluster {lab}",
            )

    ax.set_title(f"Clustering Puffin – {station}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"puffin_clusters_{station}.png", dpi=200)
    plt.close(fig)


# ============================================================
# --- NEST ESTIMATION ---
# ============================================================


def estimate_nests_negative_binomial(
    n_clusters: int,
    n_outliers: int,
    intercept: float,
    coef_clusters: float,
    coef_outliers: float,
) -> float:
    """
    Mean prediction under a log-link Negative Binomial model.
    """
    eta = (
        intercept
        + coef_clusters * float(n_clusters)
        + coef_outliers * float(n_outliers)
    )
    mu = float(np.exp(eta))
    return mu


# ============================================================
# --- MAIN ---
# ============================================================


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    features_file = args.features_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Input dir      : {input_dir}")
    print(f"[INFO] Features file : {features_file}")
    print(f"[INFO] Output dir    : {output_dir}")

    df = load_annotations(features_file)
    df = filter_annotation_rows(
        df=df,
        stations=args.stations,
        debug=args.debug,
        debug_max_rows=args.debug_max_rows,
    )

    if df.empty:
        print("[INFO] Aucune annotation à traiter.")
        return

    print(f"[INFO] {len(df)} annotation(s) retenue(s).")

    # ----------------------------------------
    # 1. Feature extraction
    # ----------------------------------------
    full_df = compute_all_features(df, input_dir=input_dir, output_dir=output_dir)
    selected_df = build_selected_feature_table(full_df)

    if selected_df.empty:
        print("[INFO] Aucun échantillon exploitable après sélection des features.")
        return

    full_csv = output_dir / DEFAULT_FULL_FEATURES_CSV
    selected_csv = output_dir / DEFAULT_SELECTED_FEATURES_CSV
    full_df.to_csv(full_csv, index=False)
    selected_df.to_csv(selected_csv, index=False)

    print(f"[OK] Features complètes : {full_csv}")
    print(f"[OK] Features retenues  : {selected_csv}")

    if "station" not in selected_df.columns:
        raise ValueError(
            "La colonne 'station' est nécessaire pour le clustering par station."
        )

    # ----------------------------------------
    # 2. Clustering
    # ----------------------------------------
    umap_cluster_params = dict(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_cluster_dim,
        metric="euclidean",
        random_state=42,
    )
    umap_plot_params = dict(
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        n_components=args.umap_plot_dim,
        metric="euclidean",
        random_state=42,
    )

    cluster_tables = []
    cluster_summary_rows = []
    nest_rows = []

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for station, df_station in selected_df.groupby("station"):
        print(f"\n=== CLUSTERING STATION {station} ===")

        clustered_df, _, _, summary = cluster_station(
            df_station=df_station,
            umap_cluster_params=umap_cluster_params,
            umap_plot_params=umap_plot_params,
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
        )

        if clustered_df is None or summary is None:
            print(f"[INFO] Station {station} ignorée : pas assez de données.")
            continue

        cluster_tables.append(clustered_df)

        summary_row = {"station": station, **summary}
        cluster_summary_rows.append(summary_row)

        n_nests_estimated = estimate_nests_negative_binomial(
            n_clusters=summary["n_clusters"],
            n_outliers=summary["n_outliers"],
            intercept=args.nb_intercept,
            coef_clusters=args.nb_coef_clusters,
            coef_outliers=args.nb_coef_outliers,
        )

        nest_rows.append(
            {
                "station": station,
                "n_samples": summary["n_samples"],
                "n_clusters": summary["n_clusters"],
                "n_outliers": summary["n_outliers"],
                "noise_ratio": summary["noise_ratio"],
                "n_nests_estimated": n_nests_estimated,
                "nb_intercept": args.nb_intercept,
                "nb_coef_clusters": args.nb_coef_clusters,
                "nb_coef_outliers": args.nb_coef_outliers,
            }
        )

        plot_station_clusters(
            df_station_clustered=clustered_df,
            station=station,
            output_dir=figures_dir / station,
        )

    # ----------------------------------------
    # 3. Exports
    # ----------------------------------------
    if cluster_tables:
        df_clusters = pd.concat(cluster_tables, ignore_index=True)
        df_clusters.to_csv(output_dir / DEFAULT_CLUSTERED_SAMPLES_CSV, index=False)

    df_cluster_summary = pd.DataFrame(cluster_summary_rows)
    df_cluster_summary.to_csv(output_dir / DEFAULT_CLUSTER_SUMMARY_CSV, index=False)

    df_nests = pd.DataFrame(nest_rows)
    df_nests.to_csv(output_dir / DEFAULT_NEST_ESTIMATION_CSV, index=False)

    print("\n=== RÉSUMÉ ===")
    print(
        f"[OK] Clustering par échantillon : {output_dir / DEFAULT_CLUSTERED_SAMPLES_CSV}"
    )
    print(
        f"[OK] Résumé clustering         : {output_dir / DEFAULT_CLUSTER_SUMMARY_CSV}"
    )
    print(
        f"[OK] Estimation des nids       : {output_dir / DEFAULT_NEST_ESTIMATION_CSV}"
    )
    print(f"[OK] Figures                  : {figures_dir}")


if __name__ == "__main__":
    main()
