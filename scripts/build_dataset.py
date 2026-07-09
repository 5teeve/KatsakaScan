# scripts/build_dataset.py
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PATHS
from features.extractor import extract_features


def build_dataset() -> None:
    """Parcourt dataset/saines et dataset/malades, extrait les features et crée le CSV."""
    print("Début de l'extraction des caractéristiques...")

    headers = ["id_image", "pct_rouille", "rugosite", "pct_jaunissement", "label_malade"]

    success_count = 0
    error_count = 0

    categories = [
        (PATHS.raw_healthy, 0),
        (PATHS.raw_diseased, 1),
    ]

    PATHS.features_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(PATHS.features_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for dossier, label in categories:
            print(f"📁 Analyse du dossier : {dossier.name} (Label: {label})")

            for img_path in dossier.glob("*.jpg"):
                try:
                    fv = extract_features(img_path)
                    row = [
                        img_path.stem,
                        fv.pct_rouille,
                        fv.rugosite,
                        fv.pct_jaunissement,
                        label,
                    ]
                    writer.writerow(row)
                    success_count += 1

                except ValueError as e:
                    print(f"⚠️ Image ignorée : {e}")
                    error_count += 1

    print(f"\n✅ Extraction terminée !")
    print(f"📊 Images traitées avec succès : {success_count}")
    print(f"❌ Images en échec (ignorées) : {error_count}")
    print(f"💾 Fichier sauvegardé sous : {PATHS.features_csv}")


if __name__ == "__main__":
    build_dataset()
