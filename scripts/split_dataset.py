# scripts/split_dataset.py
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PATHS, TRAIN


def split_dataset() -> None:
    print("📦 Chargement du dataset global...")
    df = pd.read_csv(PATHS.features_csv)

    PATHS.train_csv.parent.mkdir(parents=True, exist_ok=True)

    print("✂️ Split 80/20 (Train / Test)...")
    df_train, df_test = train_test_split(
        df,
        test_size=TRAIN.test_size,
        random_state=TRAIN.random_state,
        stratify=df["label_malade"],  # Garde l'équilibre saines/malades
    )

    print("💾 Sauvegarde des fichiers découpés...")
    df_train.to_csv(PATHS.train_csv, index=False)
    df_test.to_csv(PATHS.test_csv, index=False)

    print("\n✅ Répartition terminée avec succès !")
    print(f"📊 Données d'Entraînement (Train) : {len(df_train)} images")
    print(f"📊 Données de Test (Test)         : {len(df_test)} images")


if __name__ == "__main__":
    split_dataset()
