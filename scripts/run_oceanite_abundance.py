#!/usr/bin/env python3
"""
Oceanite abundance detection and angle estimation pipeline
from a local year/station folder structure.

This script:
- loads audio files from a local folder tree organized as year/stations
- runs a classifier on selected audio files
- computes angle estimates from stereo recordings
- exports detailed per-detection tables
- estimates the number of nests per station from angular sectors
- generates angle rose figures

Yanis Basso-Bert – 2025
Refactored local version
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from opensoundscape.ml.cnn import load_model
from scipy.signal import butter, correlate, filtfilt


# ============================================================
# --- CONFIGURATION ---
# ============================================================

AUDIO_PARAMS = {
    "fs": 48000,
    "duration": 600,
    "mic_distance": 0.158,
    "sound_speed": 343,
}

MODEL_PARAMS = {
    "batch_size": 8,
    "activation_layer": "sigmoid",
    "overlap_fraction": 0.0,
}

FILTER_BAND = (800, 2000)
DEFAULT_AUDIO_EXTENSIONS = {".wav"}
DEFAULT_GLOBAL_CSV_NAME = "oceanite_abundance_all_stations.csv"
DEFAULT_NEST_ESTIMATION_CSV_NAME = "oceanite_nest_estimation_by_station.csv"

# Linear model fitted on 2023 + 2024 calibration data.
# Model form: y = a * x + b
# where:
#   - x = number of angular sectors with more than a threshold number of detections
#   - y = estimated number of occupied nests within the calibration radius
LINEAR_MODEL_A = 0.29180426
LINEAR_MODEL_B = 1.72020414


# ============================================================
# --- CLI ---
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance l'analyse d'abondance Océanite sur une arborescence locale "
            "organisée par année/stations, puis exporte les résultats détaillés, "
            "une table d'estimation du nombre de nids par station, et des figures "
            "de roses d'angle."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dossier année contenant un sous-dossier par station.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Dossier où écrire les résultats par station, le CSV global et les figures.",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        required=True,
        help="Chemin vers le classifieur OpenSoundscape à utiliser.",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Liste optionnelle des stations à traiter. Exemple : --stations RO1 RO2 MA1",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recherche récursive des fichiers audio dans chaque dossier station.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode test : copie un sous-ensemble de fichiers dans un dossier temporaire.",
    )
    parser.add_argument(
        "--debug-max-files",
        type=int,
        default=5,
        help="Nombre maximum de fichiers audio par station en mode debug (défaut : 5).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Nombre de workers pour l'inférence modèle (défaut : 4).",
    )
    parser.add_argument(
        "--date-start",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="Date de début incluse au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--date-end",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        required=True,
        help="Date de fin exclue au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--start-hour",
        type=int,
        default=0,
        help="Heure de début de la fenêtre quotidienne (0-23).",
    )
    parser.add_argument(
        "--end-hour",
        type=int,
        default=3,
        help="Heure de fin de la fenêtre quotidienne (0-23).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.8,
        help="Seuil minimal du score de détection pour l'estimation des nids (défaut : 0.8).",
    )
    parser.add_argument(
        "--wind-threshold",
        type=float,
        default=0.2,
        help="Seuil maximum sur wind_metric_mean pour garder une détection (défaut : 0.2).",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="purring inspi",
        help='Nom de la colonne de score utilisée pour le calcul des angles et l’estimation (défaut : "purring inspi").',
    )
    parser.add_argument(
        "--sector-deg",
        type=int,
        default=4,
        help="Largeur des secteurs angulaires en degrés (défaut : 4).",
    )
    parser.add_argument(
        "--sector-threshold",
        type=int,
        default=60,
        help="Nombre minimal de détections par secteur pour qu'il soit compté (défaut : 60).",
    )
    parser.add_argument(
        "--linear-a",
        type=float,
        default=LINEAR_MODEL_A,
        help=f"Coefficient a du modèle linéaire y = a*x + b (défaut : {LINEAR_MODEL_A}).",
    )
    parser.add_argument(
        "--linear-b",
        type=float,
        default=LINEAR_MODEL_B,
        help=f"Coefficient b du modèle linéaire y = a*x + b (défaut : {LINEAR_MODEL_B}).",
    )
    parser.add_argument(
        "--global-csv-name",
        type=str,
        default=DEFAULT_GLOBAL_CSV_NAME,
        help=f"Nom du CSV global concaténé (défaut : {DEFAULT_GLOBAL_CSV_NAME}).",
    )
    parser.add_argument(
        "--nest-estimation-csv-name",
        type=str,
        default=DEFAULT_NEST_ESTIMATION_CSV_NAME,
        help=f"Nom du CSV d'estimation des nids (défaut : {DEFAULT_NEST_ESTIMATION_CSV_NAME}).",
    )

    return parser.parse_args()


# ============================================================
# --- HELPERS ---
# ============================================================

def resolve_classifier_path(classifier_path: Path) -> Path:
    classifier_path = classifier_path.resolve()
    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifieur introuvable : {classifier_path}")
    return classifier_path


def list_station_dirs(input_dir: Path, selected_stations: list[str] | None) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Le dossier d'entrée n'existe pas : {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Le chemin d'entrée n'est pas un dossier : {input_dir}")

    station_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()])

    if selected_stations is None:
        return station_dirs

    selected_set = set(selected_stations)
    filtered = [p for p in station_dirs if p.name in selected_set]

    missing = sorted(selected_set - {p.name for p in filtered})
    if missing:
        print(
            f"[WARNING] Stations demandées introuvables : {', '.join(missing)}",
            file=sys.stderr,
        )

    return filtered


def find_audio_files(station_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        files = [
            p for p in station_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in DEFAULT_AUDIO_EXTENSIONS
        ]
    else:
        files = [
            p for p in station_dir.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_AUDIO_EXTENSIONS
        ]
    return sorted(files)


def get_file_datetime(filename: str) -> datetime | None:
    filename = Path(filename).name
    patterns = {
        r"\d{8}_\d{6}": "%Y%m%d_%H%M%S",
        r"\d{8}T\d{6}": "%Y%m%dT%H%M%S",
        r"\d{4}-\d{2}-\d{2}_\d{6}": "%Y-%m-%d_%H%M%S",
    }
    for pattern, fmt in patterns.items():
        match = re.search(pattern, filename)
        if match:
            try:
                return datetime.strptime(match.group(), fmt)
            except ValueError:
                continue
    return None


def parse_wav_stem(stem: str) -> tuple[str, str, str]:
    """
    Attend un nom du type : <serial>_<date>_<time>
    Retourne : serial, date, time
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def hour_in_window(dt: datetime, start_hour: int, end_hour: int) -> bool:
    """
    Gère aussi les fenêtres qui traversent minuit.
    Exemple :
      22 -> 4   => heures >= 22 ou < 4
      0  -> 3   => heures entre 0 et 3
    """
    h = dt.hour
    if start_hour <= end_hour:
        return start_hour <= h < end_hour
    return h >= start_hour or h < end_hour


def build_audio_metadata_df(
    audio_files: list[Path],
    station: str,
    date_start: datetime,
    date_end: datetime,
    start_hour: int,
    end_hour: int,
) -> pd.DataFrame:
    rows = []

    for audio_path in audio_files:
        dt = get_file_datetime(audio_path.name)
        if dt is None:
            continue

        if not (date_start <= dt < date_end):
            continue

        if not hour_in_window(dt, start_hour, end_hour):
            continue

        stem = audio_path.stem
        serial, date_str, time_str = parse_wav_stem(stem)

        rows.append(
            {
                "station": station,
                "wav_path": str(audio_path.resolve()),
                "filename": audio_path.name,
                "file_stem": stem,
                "serial": serial,
                "date": date_str,
                "time": time_str,
                "dt": dt,
            }
        )

    return pd.DataFrame(rows)


def copy_audio_files(audio_files: list[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for src in audio_files:
        dst = destination_dir / src.name
        shutil.copy2(src, dst)


# ============================================================
# --- MODEL INFERENCE ---
# ============================================================

def run_model_inference(
    audio_dir: Path,
    classifier_path: Path,
    num_workers: int,
) -> pd.DataFrame | None:
    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        print("[WARNING] Aucun fichier audio trouvé pour l'inférence.")
        return None

    print(f"[INFO] Lancement du classifieur sur {len(audio_files)} fichier(s)...")

    model = load_model(classifier_path)
    scores = model.predict(
        audio_files,
        overlap_fraction=MODEL_PARAMS["overlap_fraction"],
        activation_layer=MODEL_PARAMS["activation_layer"],
        num_workers=num_workers,
        batch_size=MODEL_PARAMS["batch_size"],
    )

    print("[INFO] Prédictions terminées.")
    return scores


# ============================================================
# --- ANGLE ESTIMATION ---
# ============================================================

def estimate_angles(
    df_pred: pd.DataFrame,
    target_column: str,
    score_threshold: float,
) -> pd.DataFrame:
    """
    Compute:
      - low-frequency metrics for all segments
      - angle estimation only for valid detections
    """

    fs = AUDIO_PARAMS["fs"]
    d = AUDIO_PARAMS["mic_distance"]
    c_sound = AUDIO_PARAMS["sound_speed"]
    window = 5
    fmin, fmax = FILTER_BAND

    if target_column not in df_pred.columns:
        raise ValueError(
            f'Colonne cible introuvable dans les prédictions : "{target_column}". '
            f"Colonnes disponibles : {list(df_pred.columns)}"
        )

    bp_order = 4
    b, a = butter(bp_order, [fmin / (0.5 * fs), fmax / (0.5 * fs)], btype="band")
    n_max = int(fs * d / c_sound)

    df = df_pred.copy().reset_index(drop=True)

    df["angle_estimated"] = np.nan
    df["angle_failed"] = False
    df["processing_version"] = "v3-local-nest-estimation"

    df["wind_metric_mean"] = np.nan
    df["wind_metric_max"] = np.nan
    df["wind_metric_p90"] = np.nan
    df["wind_metric_std"] = np.nan

    audio_cache_mono = {}
    audio_cache_stereo = {}

    # 1) Metrics for all segments
    for idx, row in df.iterrows():
        file_path = row["file"]

        if file_path not in audio_cache_mono:
            y, sr = librosa.load(file_path, sr=AUDIO_PARAMS["fs"], mono=True)
            audio_cache_mono[file_path] = (y, sr)

        y, sr = audio_cache_mono[file_path]

        i0 = int(row["start_time"] * sr)
        i1 = int(row["end_time"] * sr)
        y_seg = y[i0:i1]

        if len(y_seg) == 0:
            df.at[idx, "wind_metric_mean"] = 0.0
            df.at[idx, "wind_metric_max"] = 0.0
            df.at[idx, "wind_metric_p90"] = 0.0
            df.at[idx, "wind_metric_std"] = 0.0
            continue

        s_power = np.abs(librosa.stft(y_seg, n_fft=2048, hop_length=512)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        low_mask = (freqs >= 20) & (freqs <= 200)

        low_energy = s_power[low_mask].sum(axis=0)
        total_energy = s_power.sum(axis=0) + 1e-12
        ratio = low_energy / total_energy

        df.at[idx, "wind_metric_mean"] = float(np.mean(ratio))
        df.at[idx, "wind_metric_max"] = float(np.max(ratio))
        df.at[idx, "wind_metric_p90"] = float(np.percentile(ratio, 90))
        df.at[idx, "wind_metric_std"] = float(np.std(ratio))

    print(f"[INFO] Métriques basses fréquences calculées pour {len(df)} ligne(s).")

    # 2) Angle estimation only for confident detections
    mask = (
        (df[target_column] > score_threshold)
        & (df["start_time"] > window / 2)
        & (df["end_time"] < AUDIO_PARAMS["duration"] - window / 2)
    )
    df_valid = df[mask]

    print(f"[INFO] Détections valides pour le calcul d'angle : {len(df_valid)}")

    for idx, row in df_valid.iterrows():
        file_path = row["file"]

        if file_path not in audio_cache_stereo:
            y, sr_file = librosa.load(file_path, sr=None, mono=False)
            audio_cache_stereo[file_path] = (y, sr_file)

        s_full, sr_file = audio_cache_stereo[file_path]

        try:
            offset = (row["start_time"] + row["end_time"]) / 2 - window / 2

            i0 = int(offset * sr_file)
            i1 = int((offset + window) * sr_file)
            s = s_full[:, i0:i1]

            if s.ndim != 2 or s.shape[0] != 2:
                raise ValueError("Le fichier audio n'est pas stéréo.")

            s = filtfilt(b, a, s)

            intercorrelation = correlate(s[1, :], s[0, :])
            center = len(s[0])
            search = intercorrelation[center - n_max - 1 : center + n_max]

            n_delay = -(np.argmax(search) - n_max)
            ratio = np.clip((c_sound * n_delay / fs) / d, -1, 1)
            alpha = np.degrees(np.arcsin(ratio))

            df.at[idx, "angle_estimated"] = alpha

        except Exception:
            df.at[idx, "angle_failed"] = True

    print(f"[INFO] Angles calculés pour {df['angle_estimated'].notna().sum()} ligne(s).")
    return df


# ============================================================
# --- NEST ESTIMATION ---
# ============================================================

def metric_angle_sectors(
    df_serial_angles: pd.DataFrame,
    sector_deg: int = 4,
    threshold: int = 60,
) -> int:
    """
    Nombre de bins angulaires contenant plus de `threshold` détections.
    Hypothèse: angle_estimated en degrés relatifs dans [-90, 90].
    """
    angles = df_serial_angles["angle_estimated"].dropna().astype(float).values
    if angles.size == 0:
        return 0

    bin_edges = np.arange(-90, 90 + sector_deg, sector_deg)
    hist, _ = np.histogram(angles, bins=bin_edges)
    return int(np.sum(hist > threshold))


def filter_angles_for_nest_estimation(
    df_angles: pd.DataFrame,
    score_col: str,
    score_threshold: float,
    wind_threshold: float | None,
) -> pd.DataFrame:
    df = df_angles.copy()

    df = df.dropna(subset=["angle_estimated"]).copy()
    df = df[df["station"].notna()].copy()

    if wind_threshold is not None and "wind_metric_mean" in df.columns:
        df = df[
            df["wind_metric_mean"].notna()
            & (df["wind_metric_mean"].astype(float) <= float(wind_threshold))
        ].copy()

    if score_col in df.columns:
        df = df[
            df[score_col].notna()
            & (df[score_col].astype(float) > float(score_threshold))
        ].copy()

    return df


def estimate_nests_by_station(
    df_all: pd.DataFrame,
    score_col: str,
    score_threshold: float,
    wind_threshold: float | None,
    sector_deg: int,
    sector_threshold: int,
    linear_a: float,
    linear_b: float,
) -> pd.DataFrame:
    df_filtered = filter_angles_for_nest_estimation(
        df_angles=df_all,
        score_col=score_col,
        score_threshold=score_threshold,
        wind_threshold=wind_threshold,
    )

    rows = []

    for station, df_station in df_filtered.groupby("station"):
        n_sectors = metric_angle_sectors(
            df_serial_angles=df_station,
            sector_deg=sector_deg,
            threshold=sector_threshold,
        )
        n_nests_estimated = linear_a * n_sectors + linear_b

        rows.append(
            {
                "station": station,
                "n_detections_used": int(len(df_station)),
                "n_angle_sectors": int(n_sectors),
                "n_nests_estimated": float(n_nests_estimated),
                "sector_deg": int(sector_deg),
                "sector_threshold": int(sector_threshold),
                "score_col": score_col,
                "score_threshold": float(score_threshold),
                "wind_threshold": None if wind_threshold is None else float(wind_threshold),
                "linear_a": float(linear_a),
                "linear_b": float(linear_b),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "station",
                "n_detections_used",
                "n_angle_sectors",
                "n_nests_estimated",
                "sector_deg",
                "sector_threshold",
                "score_col",
                "score_threshold",
                "wind_threshold",
                "linear_a",
                "linear_b",
            ]
        )

    return pd.DataFrame(rows).sort_values("station").reset_index(drop=True)


# ============================================================
# --- FIGURES ---
# ============================================================

def plot_angle_roses_for_station(
    df_station: pd.DataFrame,
    station: str,
    output_dir: Path,
    sector_deg: int,
    sector_threshold: int,
    linear_a: float,
    linear_b: float,
) -> None:
    angles = df_station["angle_estimated"].dropna().astype(float).values
    if angles.size == 0:
        print(f"[INFO] Pas d'angles exploitables pour la station {station}, figures ignorées.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges = np.arange(-90, 90 + sector_deg, sector_deg)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist, _ = np.histogram(angles, bins=bin_edges)

    theta = np.deg2rad(bin_centers)
    widths = np.deg2rad(np.diff(bin_edges))

    x_sectors = int(np.sum(hist > sector_threshold))
    y_nests = linear_a * x_sectors + linear_b

    # Figure polaire
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 5))
    ax.bar(
        theta,
        hist,
        width=widths,
        edgecolor="black",
        align="center",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-90)
    ax.set_thetamax(90)
    ax.set_title(
        f"Rose des angles d'incidence – {station}\n"
        f"{x_sectors} secteurs > {sector_threshold} détections | "
        f"nids estimés = {y_nests:.1f}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"angles_histo_polar_{station}.png", dpi=200)
    plt.close(fig)

    # Figure cartésienne
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        bin_centers,
        hist,
        width=np.diff(bin_edges),
        edgecolor="black",
        align="center",
        alpha=0.8,
    )
    ax.set_xlim(-90, 90)
    ax.set_xlabel("Angle d'arrivée (°)")
    ax.set_ylabel("Nombre de détections")
    ax.set_title(
        f"Histogramme des angles d'incidence – {station}\n"
        f"{x_sectors} secteurs > {sector_threshold} détections | "
        f"nids estimés = {y_nests:.1f}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"angles_histo_{station}.png", dpi=200)
    plt.close(fig)


# ============================================================
# --- EXPORT ---
# ============================================================

def write_station_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_global_csv(df_list: list[pd.DataFrame], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not df_list:
        pd.DataFrame().to_csv(output_path, index=False)
        return

    df = pd.concat(df_list, ignore_index=True)
    df.to_csv(output_path, index=False)


# ============================================================
# --- MAIN ---
# ============================================================

def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = resolve_classifier_path(args.classifier)
    station_dirs = list_station_dirs(input_dir, args.stations)

    if not station_dirs:
        print(f"Aucun dossier station trouvé dans : {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Dossier année : {input_dir}")
    print(f"Dossier de sortie : {output_dir}")
    print(f"Classifieur : {classifier_path}")
    print(f"Nombre de stations à traiter : {len(station_dirs)}")

    all_station_results: list[pd.DataFrame] = []

    for station_dir in station_dirs:
        station = station_dir.name
        print(f"\n=== TRAITEMENT STATION {station} ===")

        audio_files = find_audio_files(station_dir, recursive=args.recursive)
        if not audio_files:
            print(f"[INFO] Aucun fichier audio trouvé dans {station_dir}")
            continue

        df_audio = build_audio_metadata_df(
            audio_files=audio_files,
            station=station,
            date_start=args.date_start,
            date_end=args.date_end,
            start_hour=args.start_hour,
            end_hour=args.end_hour,
        )

        if df_audio.empty:
            print("[INFO] Aucun fichier audio ne correspond aux filtres date/heure.")
            continue

        print(f"[INFO] {len(df_audio)} fichier(s) retenu(s) après filtrage date/heure.")

        if args.debug:
            df_audio = df_audio.head(args.debug_max_files).reset_index(drop=True)
            print(f"[DEBUG] Mode test : {len(df_audio)} fichier(s) conservé(s).")

        station_output_dir = output_dir / station
        station_output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"oceanite_abundance_{station}_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            selected_audio_paths = [Path(p) for p in df_audio["wav_path"].tolist()]
            copy_audio_files(selected_audio_paths, tmp_dir)

            scores = run_model_inference(
                audio_dir=tmp_dir,
                classifier_path=classifier_path,
                num_workers=args.num_workers,
            )

            if scores is None:
                continue

            scores = estimate_angles(
                df_pred=scores,
                target_column=args.target_column,
                score_threshold=args.score_threshold,
            )

            scores["file_stem"] = scores["file"].apply(lambda p: Path(p).stem)

            scores = scores.merge(
                df_audio[
                    [
                        "station",
                        "wav_path",
                        "filename",
                        "file_stem",
                        "serial",
                        "date",
                        "time",
                        "dt",
                    ]
                ],
                on="file_stem",
                how="left",
            )

            station_csv = station_output_dir / f"oceanite_abundance_{station}.csv"
            write_station_csv(scores, station_csv)

            all_station_results.append(scores)
            print(f"[OK] Résultats écrits : {station_csv}")

    # CSV global détaillé
    global_csv = output_dir / args.global_csv_name
    write_global_csv(all_station_results, global_csv)
    print(f"\n[OK] CSV global écrit : {global_csv}")

    # Estimation de nids + figures
    if all_station_results:
        df_all_results = pd.concat(all_station_results, ignore_index=True)

        nest_estimation_df = estimate_nests_by_station(
            df_all=df_all_results,
            score_col=args.target_column,
            score_threshold=args.score_threshold,
            wind_threshold=args.wind_threshold,
            sector_deg=args.sector_deg,
            sector_threshold=args.sector_threshold,
            linear_a=args.linear_a,
            linear_b=args.linear_b,
        )

        nest_estimation_csv = output_dir / args.nest_estimation_csv_name
        nest_estimation_df.to_csv(nest_estimation_csv, index=False)
        print(f"[OK] Table d'estimation des nids écrite : {nest_estimation_csv}")

        figures_root = output_dir / "figures"

        df_for_plots = filter_angles_for_nest_estimation(
            df_angles=df_all_results,
            score_col=args.target_column,
            score_threshold=args.score_threshold,
            wind_threshold=args.wind_threshold,
        )

        for station, df_station in df_for_plots.groupby("station"):
            station_fig_dir = figures_root / station
            plot_angle_roses_for_station(
                df_station=df_station,
                station=station,
                output_dir=station_fig_dir,
                sector_deg=args.sector_deg,
                sector_threshold=args.sector_threshold,
                linear_a=args.linear_a,
                linear_b=args.linear_b,
            )

        print(f"[OK] Figures de roses d'angle écrites dans : {figures_root}")

    print("\nAnalyse terminée sans erreur.")


if __name__ == "__main__":
    main()