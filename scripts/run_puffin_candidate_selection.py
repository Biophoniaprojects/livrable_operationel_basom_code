#!/usr/bin/env python3
"""
Puffin candidate selection workflow.

This script:
- ensures a species presence detection table exists by calling run_species_presence.py if needed
- loads local detection results
- applies candidate selection rules
- extracts audio clips around retained detections
- exports candidate tables and clips by station

Refactored local version
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import soundfile as sf
import numpy as np


DEFAULT_DETECTION_CSV_NAME = "presence_detection_all_stations.csv"
DEFAULT_CLIPS_DIRNAME = "candidate_clips"
DEFAULT_CANDIDATES_CSV_NAME = "puffin_candidates_all_stations.csv"
DEFAULT_PER_STATION_CSV_DIRNAME = "candidate_tables"


# ============================================================
# --- CLI ---
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance la présélection des vocalisations candidates du Puffin de Scopoli "
            "à partir d'une arborescence locale organisée par année/stations. "
            "Si le CSV global de détection n'existe pas, le script peut lancer "
            "run_species_presence.py automatiquement."
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
        help="Dossier racine de sortie pour les détections, tables candidates et clips.",
    )

    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help="Liste optionnelle des stations à traiter. Exemple : --stations ST01 ST02",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recherche récursive des fichiers audio dans chaque dossier station.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode test : limite le nombre de fichiers audio analysés / candidats exportés.",
    )
    parser.add_argument(
        "--debug-max-files",
        type=int,
        default=3,
        help="Nombre maximum de fichiers audio par station en mode debug (défaut : 3).",
    )

    # Détection préalable
    parser.add_argument(
        "--presence-script",
        type=Path,
        default=None,
        help=(
            "Chemin vers run_species_presence.py. "
            "Par défaut, il est recherché dans le même dossier que ce script."
        ),
    )
    parser.add_argument(
        "--presence-classifier",
        type=Path,
        default=None,
        help=(
            "Classifieur à utiliser pour la détection préalable si le CSV n'existe pas. "
            "Obligatoire si le CSV global de détection est absent."
        ),
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="birdnet:v1.3.1_v2",
        help="Image Docker utilisée par run_species_presence.py.",
    )
    parser.add_argument(
        "--presence-csv-name",
        type=str,
        default=DEFAULT_DETECTION_CSV_NAME,
        help=f"Nom du CSV global de détection (défaut : {DEFAULT_DETECTION_CSV_NAME}).",
    )

    # Filtres de sélection
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
        help="Date de fin incluse au format YYYY-MM-DD.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.9,
        help="Seuil minimal de score pour garder une détection (défaut : 0.9).",
    )
    parser.add_argument(
        "--min-detections-per-file",
        type=int,
        default=5,
        help="Nombre minimal de détections par fichier pour garder un candidat (défaut : 5).",
    )
    parser.add_argument(
        "--max-detections-per-file",
        type=int,
        default=50,
        help="Nombre maximal de détections par fichier pour garder un candidat (défaut : 50).",
    )
    parser.add_argument(
        "--min-run",
        type=int,
        default=5,
        help="Nombre minimal de détections rapprochées pour former un bloc valide (défaut : 5).",
    )
    parser.add_argument(
        "--max-gap-s",
        type=float,
        default=4.0,
        help="Écart maximal entre deux détections consécutives dans un bloc valide (défaut : 4.0 s).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=350,
        help="Nombre maximal de détections candidates retenues après filtrage (défaut : 350).",
    )
    parser.add_argument(
        "--pad-s",
        type=float,
        default=1.0,
        help="Marge temporelle ajoutée avant et après le segment extrait (défaut : 1.0 s).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Graine aléatoire pour l'échantillonnage des candidats (défaut : 42).",
    )

    return parser.parse_args()


# ============================================================
# --- HELPERS ---
# ============================================================


def resolve_presence_script(user_path: Path | None) -> Path:
    if user_path is not None:
        script_path = user_path.resolve()
    else:
        script_path = (
            Path(__file__).resolve().parent / "run_species_presence.py"
        ).resolve()

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script run_species_presence.py introuvable : {script_path}"
        )
    return script_path


def ensure_detection_csv(args: argparse.Namespace, detection_output_dir: Path) -> Path:
    """
    Vérifie la présence du CSV global de détection.
    Si absent, lance run_species_presence.py.
    """
    detection_output_dir.mkdir(parents=True, exist_ok=True)
    detection_csv = detection_output_dir / args.presence_csv_name

    if detection_csv.exists():
        print(f"[INFO] CSV de détection trouvé : {detection_csv}")
        return detection_csv

    if args.presence_classifier is None:
        raise FileNotFoundError(
            f"CSV de détection introuvable : {detection_csv}\n"
            "Fournir --presence-classifier pour lancer run_species_presence.py automatiquement."
        )

    presence_script = resolve_presence_script(args.presence_script)

    cmd = [
        sys.executable,
        str(presence_script),
        "--input-dir",
        str(args.input_dir.resolve()),
        "--output-dir",
        str(detection_output_dir.resolve()),
        "--classifier",
        str(args.presence_classifier.resolve()),
        "--docker-image",
        args.docker_image,
    ]

    if args.stations:
        cmd.extend(["--stations", *args.stations])

    if args.recursive:
        cmd.append("--recursive")

    if args.debug:
        cmd.append("--debug")
        cmd.extend(["--debug-max-files", str(args.debug_max_files)])

    print("[INFO] CSV de détection absent. Lancement de run_species_presence.py ...")
    print("[INFO] Commande :", " ".join(cmd))

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError("run_species_presence.py a échoué.")

    if not detection_csv.exists():
        raise FileNotFoundError(
            f"Le CSV de détection attendu n'a pas été produit : {detection_csv}"
        )

    return detection_csv


def load_audio_segment(
    path: Path, start_s: float, end_s: float, pad_s: float = 1.0
) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        duration = len(f) / sr

        start_s = max(0.0, float(start_s) - pad_s)
        end_s = min(duration, float(end_s) + pad_s)

        f.seek(int(start_s * sr))
        audio = f.read(int((end_s - start_s) * sr))

    return audio.astype(np.float32), sr


def has_consecutive_detections(
    df_file: pd.DataFrame, min_run: int = 5, max_gap_s: float = 4.0
) -> bool:
    """
    Retourne True si le fichier contient au moins une série de détections
    rapprochées satisfaisant les critères.
    """
    times = df_file["start_time"].astype(float).sort_values().values
    if len(times) == 0:
        return False

    run = 1
    for i in range(1, len(times)):
        if times[i] - times[i - 1] <= max_gap_s:
            run += 1
            if run >= min_run:
                return True
        else:
            run = 1

    return False


def prepare_detection_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonise les colonnes issues de run_species_presence.py.
    """
    expected_cols = {
        "station",
        "wav_path",
        "serial",
        "date",
        "time",
        "start_time",
        "end_time",
        "species",
        "score",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le CSV de détection : {sorted(missing)}"
        )

    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["annotation_file"] = df["wav_path"].apply(
        lambda p: str(Path(p).with_suffix(".BirdNET.results"))
    )
    df["confidence"] = df["score"].astype(float)

    return df


def filter_detection_rows(
    df: pd.DataFrame,
    date_start: datetime,
    date_end: datetime,
    score_threshold: float,
    stations: list[str] | None,
) -> pd.DataFrame:
    df = df.copy()

    df = df[df["date_dt"].notna()]
    df = df[(df["date_dt"] >= date_start) & (df["date_dt"] <= date_end)]
    df = df[df["confidence"] >= score_threshold]

    if stations is not None:
        df = df[df["station"].isin(stations)]

    return df.reset_index(drop=True)


def select_candidate_rows(
    df: pd.DataFrame,
    min_detections_per_file: int,
    max_detections_per_file: int,
    min_run: int,
    max_gap_s: float,
    max_candidates: int,
    random_seed: int,
) -> pd.DataFrame:
    valid_annotation_files = []

    for ann_file, df_file in df.groupby("annotation_file"):
        n = len(df_file)

        if not (min_detections_per_file <= n <= max_detections_per_file):
            continue

        if not has_consecutive_detections(
            df_file=df_file,
            min_run=min_run,
            max_gap_s=max_gap_s,
        ):
            continue

        valid_annotation_files.append(ann_file)

    df = df[df["annotation_file"].isin(valid_annotation_files)].copy()

    if len(df) > max_candidates:
        df = df.sample(n=max_candidates, random_state=random_seed)

    return df.sort_values(["station", "wav_path", "start_time"]).reset_index(drop=True)


def export_candidate_clips(
    df_candidates: pd.DataFrame,
    clips_root: Path,
    pad_s: float,
) -> None:
    clips_root.mkdir(parents=True, exist_ok=True)

    for idx, row in df_candidates.iterrows():
        wav_path = Path(row["wav_path"])
        if not wav_path.exists():
            print(f"[WARNING] Fichier audio introuvable, clip ignoré : {wav_path}")
            continue

        try:
            audio, sr = load_audio_segment(
                path=wav_path,
                start_s=float(row["start_time"]),
                end_s=float(row["end_time"]),
                pad_s=pad_s,
            )
        except Exception as exc:
            print(f"[WARNING] Échec extraction clip {wav_path.name}: {exc}")
            continue

        station = str(row["station"])
        station_dir = clips_root / station
        station_dir.mkdir(parents=True, exist_ok=True)

        wav_stem = wav_path.stem
        out_name = (
            f"{wav_stem}"
            f"_t{int(float(row['start_time']))}-{int(float(row['end_time']))}"
            f"_{idx}.wav"
        )

        sf.write(station_dir / out_name, audio, sr)


def write_candidate_tables(
    df_candidates: pd.DataFrame,
    output_dir: Path,
    global_csv_name: str,
    per_station_dirname: str,
) -> tuple[Path, Path]:
    global_csv = output_dir / global_csv_name
    per_station_dir = output_dir / per_station_dirname
    per_station_dir.mkdir(parents=True, exist_ok=True)

    df_candidates.to_csv(global_csv, index=False)

    for station, df_station in df_candidates.groupby("station"):
        station_csv = per_station_dir / f"puffin_candidates_{station}.csv"
        df_station.to_csv(station_csv, index=False)

    return global_csv, per_station_dir


# ============================================================
# --- MAIN ---
# ============================================================


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_output_dir = output_dir / "detections"
    clips_root = output_dir / DEFAULT_CLIPS_DIRNAME

    detection_csv = ensure_detection_csv(args, detection_output_dir)

    print(f"[INFO] Chargement des détections : {detection_csv}")
    df = pd.read_csv(detection_csv)
    df = prepare_detection_dataframe(df)

    df = filter_detection_rows(
        df=df,
        date_start=args.date_start,
        date_end=args.date_end,
        score_threshold=args.score_threshold,
        stations=args.stations,
    )

    if df.empty:
        print("[INFO] Aucune détection après filtrage date / score / stations.")
        return

    print(f"[INFO] {len(df)} détection(s) après filtrage initial.")

    df_candidates = select_candidate_rows(
        df=df,
        min_detections_per_file=args.min_detections_per_file,
        max_detections_per_file=args.max_detections_per_file,
        min_run=args.min_run,
        max_gap_s=args.max_gap_s,
        max_candidates=args.max_candidates
        if not args.debug
        else min(args.max_candidates, 30),
        random_seed=args.random_seed,
    )

    if df_candidates.empty:
        print("[INFO] Aucun candidat retenu après application des règles de sélection.")
        return

    print(f"[INFO] {len(df_candidates)} détection(s) candidate(s) retenue(s).")

    global_csv, per_station_dir = write_candidate_tables(
        df_candidates=df_candidates,
        output_dir=output_dir,
        global_csv_name=DEFAULT_CANDIDATES_CSV_NAME,
        per_station_dirname=DEFAULT_PER_STATION_CSV_DIRNAME,
    )

    export_candidate_clips(
        df_candidates=df_candidates,
        clips_root=clips_root,
        pad_s=args.pad_s,
    )

    print("\n=== RÉSUMÉ ===")
    print(f"[OK] Table globale des candidats : {global_csv}")
    print(f"[OK] Tables par station : {per_station_dir}")
    print(f"[OK] Clips extraits : {clips_root}")


if __name__ == "__main__":
    main()
