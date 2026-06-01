from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


DEFAULT_AUDIO_EXTENSIONS = {".wav"}
DEFAULT_DOCKER_IMAGE = "birdnet:v1.3.1_v2"
DEFAULT_OUTPUT_CSV_NAME = "presence_detection_all_stations.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lance une analyse de détection de présence sur une arborescence locale "
            "organisée par année/stations, exécute BirdNET dans Docker et concatène "
            "les résultats dans un CSV global."
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
        help="Dossier où écrire les résultats par station et le CSV global.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=DEFAULT_DOCKER_IMAGE,
        help=f"Nom de l'image Docker à utiliser (défaut : {DEFAULT_DOCKER_IMAGE}).",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        required=True,
        help="Chemin vers le classifieur .tflite à utiliser pour l'analyse.",
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
        "--threads",
        type=int,
        default=32,
        help="Nombre de threads à passer à BirdNET (défaut : 32).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Canal audio utilisé par BirdNET (défaut : 0).",
    )
    parser.add_argument(
        "--fmin",
        type=int,
        default=600,
        help="Fréquence minimale en Hz (défaut : 600).",
    )
    parser.add_argument(
        "--fmax",
        type=int,
        default=2500,
        help="Fréquence maximale en Hz (défaut : 2500).",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default=DEFAULT_OUTPUT_CSV_NAME,
        help=(f"Nom du CSV global concaténé (défaut : {DEFAULT_OUTPUT_CSV_NAME})."),
    )

    return parser.parse_args()


def resolve_classifier_path(classifier: Path) -> Path:
    classifier_path = classifier.resolve()

    if not classifier_path.exists():
        raise FileNotFoundError(f"Classifieur introuvable : {classifier_path}")

    if classifier_path.suffix.lower() != ".tflite":
        print(
            f"[WARNING] Le fichier classifieur n'a pas l'extension .tflite : {classifier_path}",
            file=sys.stderr,
        )

    return classifier_path


def list_station_dirs(
    input_dir: Path, selected_stations: list[str] | None
) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Le dossier d'entrée n'existe pas : {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Le chemin d'entrée n'est pas un dossier : {input_dir}"
        )

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
            p
            for p in station_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in DEFAULT_AUDIO_EXTENSIONS
        ]
    else:
        files = [
            p
            for p in station_dir.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_AUDIO_EXTENSIONS
        ]
    return sorted(files)


def copy_audio_files(audio_files: list[Path], destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for src in audio_files:
        dst = destination_dir / src.name
        shutil.copy2(src, dst)


def build_docker_command(
    data_dir: Path,
    classifier_path: Path,
    docker_image: str,
    threads: int,
    channel: int,
    fmin: int,
    fmax: int,
) -> list[str]:
    classifier_dir = classifier_path.parent
    classifier_name = classifier_path.name

    return [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{data_dir.resolve()}:/mes_data",
        "-v",
        f"{classifier_dir.resolve()}:/classifier",
        docker_image,
        "analyze.py",
        "--i",
        "mes_data",
        "--o",
        "mes_data",
        "--channel",
        str(channel),
        "--rtype",
        "audacity",
        "--classifier",
        f"classifier/{classifier_name}",
        "--fmin",
        str(fmin),
        "--fmax",
        str(fmax),
        "--skip_existing_results",
        "--threads",
        str(threads),
    ]


def run_command_streaming(cmd: list[str]) -> int:
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
        return proc.wait()


def find_result_files(result_dir: Path) -> list[Path]:
    candidates = []
    for pattern in ("*.BirdNET.results.txt", "*.BirdNET.results"):
        candidates.extend(result_dir.glob(pattern))
    return sorted(candidates)


def parse_wav_stem(stem: str) -> tuple[str, str, str]:
    """
    Attend un nom du type : <serial>_<date>_<time>
    Retourne : serial, date, time
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return "", "", ""
    return parts[0], parts[1], parts[2]


def infer_wav_name_from_result(result_file: Path) -> str:
    name = result_file.name
    if name.endswith(".BirdNET.results.txt"):
        return name[: -len(".BirdNET.results.txt")] + ".wav"
    if name.endswith(".BirdNET.results"):
        return name[: -len(".BirdNET.results")] + ".wav"
    return result_file.stem + ".wav"


def parse_result_file(
    result_file: Path,
    station: str,
    source_station_dir: Path,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []

    wav_name = infer_wav_name_from_result(result_file)
    wav_stem = Path(wav_name).stem
    serial, date, time = parse_wav_stem(wav_stem)

    wav_path = source_station_dir / wav_name

    with result_file.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 4:
                print(
                    f"[WARNING] Ligne ignorée dans {result_file.name}: {line}",
                    file=sys.stderr,
                )
                continue

            start_time, end_time, species, score = parts

            try:
                row = {
                    "station": station,
                    "wav_path": str(wav_path.resolve()),
                    "serial": serial,
                    "date": date,
                    "time": time,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "species": species,
                    "score": float(score),
                }
            except ValueError:
                print(
                    f"[WARNING] Valeur numérique invalide dans {result_file.name}: {line}",
                    file=sys.stderr,
                )
                continue

            rows.append(row)

    return rows


def write_global_csv(rows: list[dict[str, str | float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "station",
        "wav_path",
        "serial",
        "date",
        "time",
        "start_time",
        "end_time",
        "species",
        "score",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    print(f"Nombre de stations à traiter : {len(station_dirs)}")

    all_rows: list[dict[str, str | float]] = []
    failures: list[str] = []

    for station_dir in station_dirs:
        station = station_dir.name
        print(f"\n=== TRAITEMENT STATION {station} ===")

        audio_files = find_audio_files(station_dir, recursive=args.recursive)

        if not audio_files:
            print(f"[INFO] Aucun fichier audio trouvé dans {station_dir}")
            continue

        print(f"[INFO] {len(audio_files)} fichier(s) audio trouvé(s)")

        station_output_dir = output_dir / station
        station_output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"birdnet_{station}_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            if args.debug:
                selected_audio_files = audio_files[: args.debug_max_files]
                print(
                    f"[DEBUG] Mode test : {len(selected_audio_files)} fichier(s) "
                    f"copié(s) dans un dossier temporaire."
                )
            else:
                selected_audio_files = audio_files

            copy_audio_files(selected_audio_files, tmp_dir)

            docker_cmd = build_docker_command(
                data_dir=tmp_dir,
                classifier_path=classifier_path,
                docker_image=args.docker_image,
                threads=args.threads,
                channel=args.channel,
                fmin=args.fmin,
                fmax=args.fmax,
            )

            if args.debug:
                print("[DEBUG] Commande Docker :")
                print(" ".join(docker_cmd))

            exit_code = run_command_streaming(docker_cmd)

            if exit_code != 0:
                print(
                    f"[ERROR] Analyse échouée pour {station} (code retour {exit_code})",
                    file=sys.stderr,
                )
                failures.append(station)
                continue

            result_files = find_result_files(tmp_dir)
            print(f"[INFO] {len(result_files)} fichier(s) de résultat trouvé(s)")

            for result_file in result_files:
                shutil.copy2(result_file, station_output_dir / result_file.name)

            station_rows: list[dict[str, str | float]] = []
            for result_file in result_files:
                station_rows.extend(
                    parse_result_file(
                        result_file=result_file,
                        station=station,
                        source_station_dir=station_dir,
                    )
                )

            print(f"[INFO] {len(station_rows)} ligne(s) de détection parsée(s)")
            all_rows.extend(station_rows)

    output_csv = output_dir / args.csv_name
    write_global_csv(all_rows, output_csv)

    print("\n=== RÉSUMÉ ===")
    print(f"CSV global écrit : {output_csv}")
    print(f"Nombre total de lignes : {len(all_rows)}")

    if failures:
        print(f"Stations en échec : {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)

    print("Analyse terminée sans erreur.")


if __name__ == "__main__":
    main()
