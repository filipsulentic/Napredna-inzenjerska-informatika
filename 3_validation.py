"""
03_validate_results.py

Validacija K-means klasteriranja za ModelNet10 projekt.

Workflow:
    1. Učitava K-means rezultate za train i test skup
    2. Računa validacijske metrike:
        - Silhouette score
        - Adjusted Rand Index, ARI
        - Normalized Mutual Information, NMI
    3. Izrađuje cluster × class tablice za train i test
    4. Sprema rezultate u outputs_final/05_validation

Važno:
    - Train skup služi za analizu / izgradnju modela.
    - Test skup služi za validaciju.
    - Silhouette score računa se u istom prostoru u kojem je rađen K-means,
      dakle na svih 10 PCA komponenti.
    - ARI i NMI uspoređuju dobivene klastere sa stvarnim oznakama klasa.
    - Cluster × class tablica pokazuje koje stvarne klase završavaju u kojim klasterima.

Ulaz:
    outputs_final\\04_kmeans\\kmeans_train_results.csv
    outputs_final\\04_kmeans\\kmeans_test_results.csv

Izlaz:
    outputs_final\\05_validation\\validation_metrics.csv
    outputs_final\\05_validation\\cluster_class_crosstab_train.csv
    outputs_final\\05_validation\\cluster_class_crosstab_test.csv
    outputs_final\\05_validation\\cluster_summary_train.csv
    outputs_final\\05_validation\\cluster_summary_test.csv
    outputs_final\\logs\\03_validation_report.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


# ============================================================
# 1. POSTAVKE
# ============================================================

BASE_DIR = Path(r"C:\Users\filip\Desktop\NII")
OUTPUT_DIR = BASE_DIR / "outputs_final"

KMEANS_DIR = OUTPUT_DIR / "04_kmeans"
VALIDATION_DIR = OUTPUT_DIR / "05_validation"
LOGS_DIR = OUTPUT_DIR / "logs"

VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

KMEANS_TRAIN_RESULTS_CSV = KMEANS_DIR / "kmeans_train_results.csv"
KMEANS_TEST_RESULTS_CSV = KMEANS_DIR / "kmeans_test_results.csv"

VALIDATION_METRICS_CSV = VALIDATION_DIR / "validation_metrics.csv"

CROSSTAB_TRAIN_CSV = VALIDATION_DIR / "cluster_class_crosstab_train.csv"
CROSSTAB_TEST_CSV = VALIDATION_DIR / "cluster_class_crosstab_test.csv"

CLUSTER_SUMMARY_TRAIN_CSV = VALIDATION_DIR / "cluster_summary_train.csv"
CLUSTER_SUMMARY_TEST_CSV = VALIDATION_DIR / "cluster_summary_test.csv"

REPORT_TXT = LOGS_DIR / "03_validation_report.txt"


# ============================================================
# 2. POMOĆNE FUNKCIJE
# ============================================================

def get_pc_columns(df: pd.DataFrame) -> List[str]:
    """
    Pronalazi PCA stupce PC1, PC2, ..., PC10 i sortira ih po broju.
    """

    pc_columns = [
        col for col in df.columns
        if col.startswith("PC") and col[2:].isdigit()
    ]

    pc_columns = sorted(
        pc_columns,
        key=lambda x: int(x.replace("PC", ""))
    )

    if len(pc_columns) == 0:
        raise ValueError("Nisu pronađeni PCA stupci PC1, PC2, ...")

    return pc_columns


def validate_required_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """
    Provjerava osnovne stupce potrebne za validaciju.
    """

    required_columns = [
        "object_id",
        "class_label",
        "split",
        "cluster",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"U {dataset_name} nedostaju potrebni stupci: {missing}"
        )


def compute_validation_metrics(
    df: pd.DataFrame,
    pc_columns: List[str],
    dataset_name: str
) -> Dict[str, object]:
    """
    Računa Silhouette, ARI i NMI za zadani skup podataka.
    """

    X = df[pc_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~X.isna().any(axis=1)

    df_valid = df.loc[valid_mask].copy()
    X_valid = X.loc[valid_mask].copy()

    if len(df_valid) == 0:
        raise RuntimeError(f"Nema valjanih redaka za validaciju skupa: {dataset_name}")

    labels_true = df_valid["class_label"].astype(str).values
    labels_cluster = df_valid["cluster"].astype(int).values

    label_encoder = LabelEncoder()
    y_true_encoded = label_encoder.fit_transform(labels_true)

    n_models = len(df_valid)
    n_classes = len(np.unique(labels_true))
    n_clusters = len(np.unique(labels_cluster))

    ari = adjusted_rand_score(
        y_true_encoded,
        labels_cluster
    )

    nmi = normalized_mutual_info_score(
        y_true_encoded,
        labels_cluster
    )

    # Silhouette ima smisla samo ako postoji više od jednog klastera
    # i ako broj klastera nije jednak broju uzoraka.
    if n_clusters > 1 and n_clusters < n_models:
        silhouette = silhouette_score(
            X_valid,
            labels_cluster
        )
    else:
        silhouette = np.nan

    metrics = {
        "dataset": dataset_name,
        "n_models": n_models,
        "n_classes": n_classes,
        "n_clusters_present": n_clusters,
        "n_pca_components_used": len(pc_columns),
        "pca_components_used": ", ".join(pc_columns),
        "silhouette_score": silhouette,
        "adjusted_rand_index": ari,
        "normalized_mutual_information": nmi,
        "n_removed_due_to_nan_inf": len(df) - len(df_valid),
    }

    return metrics


def make_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """
    Izrađuje cluster × class tablicu.
    """

    crosstab = pd.crosstab(
        df["cluster"],
        df["class_label"],
        rownames=["cluster"],
        colnames=["class_label"]
    )

    return crosstab


def make_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Izrađuje sažetak klastera:
        - broj modela u klasteru
        - najzastupljenija klasa
        - broj modela najzastupljenije klase
        - udio najzastupljenije klase
    """

    rows = []

    for cluster_id in sorted(df["cluster"].astype(int).unique()):
        cluster_df = df[df["cluster"].astype(int) == cluster_id]

        class_counts = cluster_df["class_label"].astype(str).value_counts()

        majority_class = class_counts.index[0]
        majority_count = int(class_counts.iloc[0])
        total_count = int(len(cluster_df))
        majority_share = majority_count / total_count if total_count > 0 else np.nan

        rows.append({
            "cluster": cluster_id,
            "n_models": total_count,
            "majority_class": majority_class,
            "majority_class_count": majority_count,
            "majority_class_share": majority_share,
        })

    summary_df = pd.DataFrame(rows)

    return summary_df


def format_metrics_for_report(metrics_df: pd.DataFrame) -> str:
    """
    Formatira metrike za tekstualni report.
    """

    lines = []

    for _, row in metrics_df.iterrows():
        lines.append(f"Dataset: {row['dataset']}")
        lines.append(f"Broj modela: {row['n_models']}")
        lines.append(f"Broj stvarnih klasa: {row['n_classes']}")
        lines.append(f"Broj prisutnih klastera: {row['n_clusters_present']}")
        lines.append(f"Broj PCA komponenti korištenih za validaciju: {row['n_pca_components_used']}")
        lines.append(f"PCA komponente: {row['pca_components_used']}")
        lines.append(f"Silhouette score: {row['silhouette_score']:.6f}")
        lines.append(f"Adjusted Rand Index, ARI: {row['adjusted_rand_index']:.6f}")
        lines.append(f"Normalized Mutual Information, NMI: {row['normalized_mutual_information']:.6f}")
        lines.append(f"Uklonjeno zbog NaN/inf: {row['n_removed_due_to_nan_inf']}")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# 3. GLAVNI PROGRAM
# ============================================================

def main() -> None:
    print("=" * 80)
    print("03_validate_results.py")
    print("Validacija K-means rezultata")
    print("=" * 80)

    if not KMEANS_TRAIN_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Ne postoji train K-means CSV: {KMEANS_TRAIN_RESULTS_CSV}")

    if not KMEANS_TEST_RESULTS_CSV.exists():
        raise FileNotFoundError(f"Ne postoji test K-means CSV: {KMEANS_TEST_RESULTS_CSV}")

    train_df = pd.read_csv(KMEANS_TRAIN_RESULTS_CSV)
    test_df = pd.read_csv(KMEANS_TEST_RESULTS_CSV)

    validate_required_columns(train_df, "train_df")
    validate_required_columns(test_df, "test_df")

    pc_columns_train = get_pc_columns(train_df)
    pc_columns_test = get_pc_columns(test_df)

    if pc_columns_train != pc_columns_test:
        raise ValueError(
            "Train i test nemaju isti skup PCA stupaca.\n"
            f"Train: {pc_columns_train}\n"
            f"Test: {pc_columns_test}"
        )

    pc_columns = pc_columns_train

    print(f"Train input: {KMEANS_TRAIN_RESULTS_CSV}")
    print(f"Test input: {KMEANS_TEST_RESULTS_CSV}")
    print(f"PCA stupci za validaciju: {pc_columns}")
    print()

    # ------------------------------------------------------------
    # 3.1 Metrike
    # ------------------------------------------------------------

    train_metrics = compute_validation_metrics(
        train_df,
        pc_columns,
        "train"
    )

    test_metrics = compute_validation_metrics(
        test_df,
        pc_columns,
        "test"
    )

    metrics_df = pd.DataFrame([
        train_metrics,
        test_metrics,
    ])

    metrics_df.to_csv(
        VALIDATION_METRICS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3.2 Cluster × class tablice
    # ------------------------------------------------------------

    crosstab_train = make_crosstab(train_df)
    crosstab_test = make_crosstab(test_df)

    crosstab_train.to_csv(
        CROSSTAB_TRAIN_CSV,
        encoding="utf-8-sig"
    )

    crosstab_test.to_csv(
        CROSSTAB_TEST_CSV,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3.3 Cluster summary
    # ------------------------------------------------------------

    cluster_summary_train = make_cluster_summary(train_df)
    cluster_summary_test = make_cluster_summary(test_df)

    cluster_summary_train.to_csv(
        CLUSTER_SUMMARY_TRAIN_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    cluster_summary_test.to_csv(
        CLUSTER_SUMMARY_TEST_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3.4 Report
    # ------------------------------------------------------------

    metrics_text = format_metrics_for_report(metrics_df)

    report = f"""
03_validate_results.py report

Ulazi:
{KMEANS_TRAIN_RESULTS_CSV}
{KMEANS_TEST_RESULTS_CSV}

Validacija:
Validacija je provedena zasebno za train i test skup.
Glavni skup za interpretaciju validacije je test skup.

Korištene PCA komponente:
{chr(10).join(pc_columns)}

Metrike:
Silhouette score:
- unutarnja mjera kvalitete klasteriranja
- računa se u PCA prostoru na istim komponentama koje su korištene za K-means
- veća vrijednost znači kompaktnije i bolje odvojene klastere

Adjusted Rand Index, ARI:
- vanjska mjera
- uspoređuje K-means klastere sa stvarnim ModelNet10 klasama
- 1 znači savršeno slaganje
- vrijednost oko 0 znači slaganje slično slučajnom

Normalized Mutual Information, NMI:
- vanjska mjera
- mjeri koliko informacija o stvarnim klasama nose dobiveni klasteri
- vrijednost je između 0 i 1

Cluster × class tablica:
- pokazuje koliko modela svake stvarne klase završava u svakom klasteru

Rezultati:
{metrics_text}

Izlazi:
{VALIDATION_METRICS_CSV}
{CROSSTAB_TRAIN_CSV}
{CROSSTAB_TEST_CSV}
{CLUSTER_SUMMARY_TRAIN_CSV}
{CLUSTER_SUMMARY_TEST_CSV}
""".strip()

    REPORT_TXT.write_text(report, encoding="utf-8")

    # ------------------------------------------------------------
    # 3.5 Ispis
    # ------------------------------------------------------------

    print("Validacijske metrike:")
    print(metrics_df)
    print()

    print("Train cluster × class tablica:")
    print(crosstab_train)
    print()

    print("Test cluster × class tablica:")
    print(crosstab_test)
    print()

    print("=" * 80)
    print("Gotovo.")
    print("=" * 80)
    print(f"Validation metrics: {VALIDATION_METRICS_CSV}")
    print(f"Train crosstab: {CROSSTAB_TRAIN_CSV}")
    print(f"Test crosstab: {CROSSTAB_TEST_CSV}")
    print(f"Train cluster summary: {CLUSTER_SUMMARY_TRAIN_CSV}")
    print(f"Test cluster summary: {CLUSTER_SUMMARY_TEST_CSV}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()