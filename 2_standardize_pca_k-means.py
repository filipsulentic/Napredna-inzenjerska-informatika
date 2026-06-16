"""
02_standardize_pca_kmeans.py

Standardizacija, PCA i K-means za ModelNet10 projekt.

Workflow:
    1. Učitava geometrijske značajke iz features_modelnet10.csv
    2. Odvaja train i test skup prema stupcu split
    3. StandardScaler se fita samo na train skupu
    4. Isti scaler se primjenjuje na train i test
    5. PCA se fita samo na train skupu
    6. Isti PCA model transformira train i test
    7. K-means se fita na train PCA rezultatima
    8. Test objekti se dodjeljuju najbližim centroidima pomoću predict()

Važno:
    - Standardizacija se ne radi po klasama.
    - Test skup ne sudjeluje u fitanju StandardScalera, PCA ni K-meansa.
    - K-means koristi svih 10 PCA komponenti.
    - Broj klastera je k = 10 jer ModelNet10 ima 10 klasa.

Ulaz:
    outputs_final\\01_features\\features_modelnet10.csv
    outputs_final\\01_features\\feature_columns.txt

Izlaz:
    outputs_final\\02_standardized\\...
    outputs_final\\03_pca\\...
    outputs_final\\04_kmeans\\...
    outputs_final\\logs\\02_standardize_pca_kmeans_report.txt
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ============================================================
# 1. POSTAVKE
# ============================================================

BASE_DIR = Path(r"C:\Users\filip\Desktop\NII")

OUTPUT_DIR = BASE_DIR / "outputs_final"

FEATURES_DIR = OUTPUT_DIR / "01_features"
STANDARDIZED_DIR = OUTPUT_DIR / "02_standardized"
PCA_DIR = OUTPUT_DIR / "03_pca"
KMEANS_DIR = OUTPUT_DIR / "04_kmeans"
LOGS_DIR = OUTPUT_DIR / "logs"

STANDARDIZED_DIR.mkdir(parents=True, exist_ok=True)
PCA_DIR.mkdir(parents=True, exist_ok=True)
KMEANS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = FEATURES_DIR / "features_modelnet10_iqr_filtered.csv"
FEATURE_COLUMNS_TXT = FEATURES_DIR / "feature_columns.txt"

TRAIN_STANDARDIZED_CSV = STANDARDIZED_DIR / "features_train_standardized.csv"
TEST_STANDARDIZED_CSV = STANDARDIZED_DIR / "features_test_standardized.csv"
ALL_STANDARDIZED_CSV = STANDARDIZED_DIR / "features_all_standardized.csv"
STANDARDIZATION_PARAMETERS_CSV = STANDARDIZED_DIR / "standardization_parameters.csv"

PCA_TRAIN_RESULTS_CSV = PCA_DIR / "pca_train_results.csv"
PCA_TEST_RESULTS_CSV = PCA_DIR / "pca_test_results.csv"
PCA_ALL_RESULTS_CSV = PCA_DIR / "pca_all_results.csv"
PCA_EXPLAINED_VARIANCE_CSV = PCA_DIR / "pca_explained_variance.csv"
PCA_LOADINGS_CSV = PCA_DIR / "pca_loadings.csv"

KMEANS_TRAIN_RESULTS_CSV = KMEANS_DIR / "kmeans_train_results.csv"
KMEANS_TEST_RESULTS_CSV = KMEANS_DIR / "kmeans_test_results.csv"
KMEANS_ALL_RESULTS_CSV = KMEANS_DIR / "kmeans_all_results.csv"
KMEANS_CENTROIDS_CSV = KMEANS_DIR / "kmeans_centroids.csv"

MODEL_OBJECTS_PKL = KMEANS_DIR / "standardizer_pca_kmeans_models.pkl"

REMOVED_ROWS_CSV = STANDARDIZED_DIR / "removed_rows_nan_inf.csv"

REPORT_TXT = LOGS_DIR / "02_standardize_pca_kmeans_report.txt"

RANDOM_STATE = 42
N_CLUSTERS = 10
N_INIT = 50


# ============================================================
# 2. POMOĆNE FUNKCIJE
# ============================================================

def read_feature_columns(path: Path) -> List[str]:
    """
    Čita popis značajki iz feature_columns.txt.
    """

    if not path.exists():
        raise FileNotFoundError(f"Ne postoji feature_columns.txt: {path}")

    with open(path, "r", encoding="utf-8") as f:
        feature_columns = [line.strip() for line in f if line.strip()]

    if len(feature_columns) == 0:
        raise ValueError("feature_columns.txt je prazan.")

    return feature_columns


def validate_input_dataframe(df: pd.DataFrame, feature_columns: List[str]) -> None:
    """
    Provjerava potrebne stupce.
    """

    required_columns = [
        "object_id",
        "class_label",
        "split",
        "object_path",
        "resolved_path",
    ]

    missing_required = [col for col in required_columns if col not in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]

    if missing_required:
        raise ValueError(f"Nedostaju obavezni metadata stupci: {missing_required}")

    if missing_features:
        raise ValueError(f"Nedostaju feature stupci: {missing_features}")


def clean_invalid_rows(df: pd.DataFrame, feature_columns: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uklanja retke koji imaju NaN ili inf vrijednosti u feature stupcima.

    Outlieri se ne uklanjaju.
    """

    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    valid_mask = ~X.isna().any(axis=1)

    clean_df = df.loc[valid_mask].copy()
    removed_df = df.loc[~valid_mask].copy()

    return clean_df, removed_df


def make_standardized_dataframe(
    metadata_df: pd.DataFrame,
    X_scaled: np.ndarray,
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Stvara DataFrame sa standardiziranim značajkama.
    """

    standardized_columns = [f"{col}_std" for col in feature_columns]

    X_scaled_df = pd.DataFrame(
        X_scaled,
        columns=standardized_columns,
        index=metadata_df.index
    )

    keep_columns = [
        "object_id",
        "class_label",
        "split",
        "object_path",
        "resolved_path",
    ]

    out_df = pd.concat(
        [
            metadata_df[keep_columns],
            X_scaled_df
        ],
        axis=1
    )

    return out_df


def make_pca_dataframe(
    metadata_df: pd.DataFrame,
    X_pca: np.ndarray,
    pc_columns: List[str]
) -> pd.DataFrame:
    """
    Stvara DataFrame s PCA koordinatama.
    """

    pca_scores_df = pd.DataFrame(
        X_pca,
        columns=pc_columns,
        index=metadata_df.index
    )

    keep_columns = [
        "object_id",
        "class_label",
        "split",
        "object_path",
        "resolved_path",
    ]

    out_df = pd.concat(
        [
            metadata_df[keep_columns],
            pca_scores_df
        ],
        axis=1
    )

    return out_df


def make_kmeans_dataframe(
    pca_df: pd.DataFrame,
    clusters: np.ndarray
) -> pd.DataFrame:
    """
    Dodaje K-means cluster oznaku na PCA rezultate.
    """

    out_df = pca_df.copy()
    out_df["cluster"] = clusters.astype(int)

    return out_df


# ============================================================
# 3. GLAVNI PROGRAM
# ============================================================

def main() -> None:
    print("=" * 80)
    print("02_standardize_pca_kmeans.py")
    print("Standardizacija, PCA i K-means")
    print("=" * 80)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Ne postoji features CSV: {FEATURES_CSV}")

    df = pd.read_csv(FEATURES_CSV)
    feature_columns = read_feature_columns(FEATURE_COLUMNS_TXT)

    validate_input_dataframe(df, feature_columns)

    df["split"] = df["split"].astype(str).str.lower()

    print(f"Ulazni features CSV: {FEATURES_CSV}")
    print(f"Broj redaka prije čišćenja: {len(df)}")
    print(f"Značajke: {feature_columns}")
    print()

    # ------------------------------------------------------------
    # 3.1 Čišćenje NaN/inf vrijednosti
    # ------------------------------------------------------------

    df_clean, removed_df = clean_invalid_rows(df, feature_columns)

    if len(removed_df) > 0:
        removed_df.to_csv(
            REMOVED_ROWS_CSV,
            index=False,
            encoding="utf-8-sig"
        )

    print(f"Broj redaka nakon uklanjanja NaN/inf: {len(df_clean)}")
    print(f"Uklonjeno redaka: {len(removed_df)}")
    print()

    if len(df_clean) == 0:
        raise RuntimeError("Nema valjanih redaka nakon uklanjanja NaN/inf vrijednosti.")

    # ------------------------------------------------------------
    # 3.2 Train/test split
    # ------------------------------------------------------------

    train_df = df_clean[df_clean["split"] == "train"].copy()
    test_df = df_clean[df_clean["split"] == "test"].copy()

    if len(train_df) == 0:
        raise RuntimeError("Train skup je prazan. Provjeri stupac split.")

    if len(test_df) == 0:
        raise RuntimeError("Test skup je prazan. Provjeri stupac split.")

    print(f"Broj train modela: {len(train_df)}")
    print(f"Broj test modela: {len(test_df)}")
    print()

    print("Train klase:")
    print(train_df["class_label"].value_counts().sort_index())
    print()

    print("Test klase:")
    print(test_df["class_label"].value_counts().sort_index())
    print()

    # ------------------------------------------------------------
    # 3.3 Standardizacija
    # ------------------------------------------------------------

    X_train = train_df[feature_columns].copy()
    X_test = test_df[feature_columns].copy()

    scaler = StandardScaler()

    # Fit samo na train
    X_train_scaled = scaler.fit_transform(X_train)

    # Test se samo transformira istim scalerom
    X_test_scaled = scaler.transform(X_test)

    standardized_columns = [f"{col}_std" for col in feature_columns]

    train_standardized_df = make_standardized_dataframe(
        train_df,
        X_train_scaled,
        feature_columns
    )

    test_standardized_df = make_standardized_dataframe(
        test_df,
        X_test_scaled,
        feature_columns
    )

    all_standardized_df = pd.concat(
        [train_standardized_df, test_standardized_df],
        axis=0
    ).sort_index()

    train_standardized_df.to_csv(
        TRAIN_STANDARDIZED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    test_standardized_df.to_csv(
        TEST_STANDARDIZED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    all_standardized_df.to_csv(
        ALL_STANDARDIZED_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    standardization_parameters_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_train": scaler.mean_,
        "std_train": scaler.scale_,
        "variance_train": scaler.var_,
        "mean_train_after_standardization": X_train_scaled.mean(axis=0),
        "std_train_after_standardization": X_train_scaled.std(axis=0, ddof=0),
        "mean_test_after_standardization": X_test_scaled.mean(axis=0),
        "std_test_after_standardization": X_test_scaled.std(axis=0, ddof=0),
    })

    standardization_parameters_df.to_csv(
        STANDARDIZATION_PARAMETERS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("Standardizacija završena.")
    print(f"Scaler fitan na train skupu.")
    print(f"Standardizirani train CSV: {TRAIN_STANDARDIZED_CSV}")
    print(f"Standardizirani test CSV: {TEST_STANDARDIZED_CSV}")
    print()

    # ------------------------------------------------------------
    # 3.4 PCA
    # ------------------------------------------------------------

    n_components = len(feature_columns)

    pca = PCA(n_components=n_components)

    # Fit samo na train
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Test se samo transformira istim PCA modelom
    X_test_pca = pca.transform(X_test_scaled)

    pc_columns = [f"PC{i + 1}" for i in range(n_components)]

    train_pca_df = make_pca_dataframe(
        train_df,
        X_train_pca,
        pc_columns
    )

    test_pca_df = make_pca_dataframe(
        test_df,
        X_test_pca,
        pc_columns
    )

    all_pca_df = pd.concat(
        [train_pca_df, test_pca_df],
        axis=0
    ).sort_index()

    train_pca_df.to_csv(
        PCA_TRAIN_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    test_pca_df.to_csv(
        PCA_TEST_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    all_pca_df.to_csv(
        PCA_ALL_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    pca_explained_variance_df = pd.DataFrame({
        "component": pc_columns,
        "explained_variance_ratio": explained_variance,
        "explained_variance_percent": explained_variance * 100,
        "cumulative_variance_ratio": cumulative_variance,
        "cumulative_variance_percent": cumulative_variance * 100,
    })

    pca_explained_variance_df.to_csv(
        PCA_EXPLAINED_VARIANCE_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    pca_loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=pc_columns,
        index=feature_columns
    )

    pca_loadings_df.to_csv(
        PCA_LOADINGS_CSV,
        encoding="utf-8-sig"
    )

    print("PCA završena.")
    print(f"PCA fitana na train skupu.")
    print(f"Broj PCA komponenti: {n_components}")
    print(f"PCA train CSV: {PCA_TRAIN_RESULTS_CSV}")
    print(f"PCA test CSV: {PCA_TEST_RESULTS_CSV}")
    print()
    print("Objašnjena varijanca:")
    print(pca_explained_variance_df)
    print()

    # ------------------------------------------------------------
    # 3.5 K-means
    # ------------------------------------------------------------

    pca_columns_for_kmeans = pc_columns

    X_train_kmeans = train_pca_df[pca_columns_for_kmeans].copy()
    X_test_kmeans = test_pca_df[pca_columns_for_kmeans].copy()

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT
    )

    # Fit samo na train PCA rezultatima
    train_clusters = kmeans.fit_predict(X_train_kmeans)

    # Test se samo dodjeljuje najbližim train centroidima
    test_clusters = kmeans.predict(X_test_kmeans)

    train_kmeans_df = make_kmeans_dataframe(
        train_pca_df,
        train_clusters
    )

    test_kmeans_df = make_kmeans_dataframe(
        test_pca_df,
        test_clusters
    )

    all_kmeans_df = pd.concat(
        [train_kmeans_df, test_kmeans_df],
        axis=0
    ).sort_index()

    train_kmeans_df.to_csv(
        KMEANS_TRAIN_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    test_kmeans_df.to_csv(
        KMEANS_TEST_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    all_kmeans_df.to_csv(
        KMEANS_ALL_RESULTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    kmeans_centroids_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=pca_columns_for_kmeans
    )

    kmeans_centroids_df.insert(
        0,
        "cluster",
        np.arange(N_CLUSTERS)
    )

    kmeans_centroids_df.to_csv(
        KMEANS_CENTROIDS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    print("K-means završen.")
    print(f"K-means fitan na train PCA rezultatima.")
    print(f"Broj klastera: {N_CLUSTERS}")
    print(f"K-means koristi PCA stupce: {pca_columns_for_kmeans}")
    print(f"K-means train CSV: {KMEANS_TRAIN_RESULTS_CSV}")
    print(f"K-means test CSV: {KMEANS_TEST_RESULTS_CSV}")
    print()

    # ------------------------------------------------------------
    # 3.6 Spremanje model objekata
    # ------------------------------------------------------------

    model_objects = {
        "feature_columns": feature_columns,
        "standardized_columns": standardized_columns,
        "pc_columns": pc_columns,
        "pca_columns_for_kmeans": pca_columns_for_kmeans,
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "random_state": RANDOM_STATE,
        "n_clusters": N_CLUSTERS,
        "n_init": N_INIT,
    }

    with open(MODEL_OBJECTS_PKL, "wb") as f:
        pickle.dump(model_objects, f)

    # ------------------------------------------------------------
    # 3.7 Report
    # ------------------------------------------------------------

    train_cluster_counts = train_kmeans_df["cluster"].value_counts().sort_index().to_dict()
    test_cluster_counts = test_kmeans_df["cluster"].value_counts().sort_index().to_dict()

    report = f"""
02_standardize_pca_kmeans.py report

Ulaz:
{FEATURES_CSV}
{FEATURE_COLUMNS_TXT}

Broj redaka prije čišćenja:
{len(df)}

Broj redaka nakon uklanjanja NaN/inf:
{len(df_clean)}

Broj uklonjenih redaka:
{len(removed_df)}

Broj train modela:
{len(train_df)}

Broj test modela:
{len(test_df)}

Korištene značajke:
{chr(10).join(feature_columns)}

Standardizacija:
StandardScaler je fitan samo na train skupu.
Train i test su transformirani istim scalerom.
Standardizacija je globalna po svim klasama, nije provedena po pojedinim klasama.

Standardizirani izlazi:
{TRAIN_STANDARDIZED_CSV}
{TEST_STANDARDIZED_CSV}
{ALL_STANDARDIZED_CSV}
{STANDARDIZATION_PARAMETERS_CSV}

PCA:
PCA je fitana samo na train skupu.
Train i test su transformirani istim PCA modelom.
Broj PCA komponenti: {n_components}

PCA izlazi:
{PCA_TRAIN_RESULTS_CSV}
{PCA_TEST_RESULTS_CSV}
{PCA_ALL_RESULTS_CSV}
{PCA_EXPLAINED_VARIANCE_CSV}
{PCA_LOADINGS_CSV}

PCA objašnjena varijanca:
{pca_explained_variance_df.to_string(index=False)}

K-means:
K-means je fitan samo na train PCA rezultatima.
Test modeli su dodijeljeni klasterima pomoću predict().
Broj klastera: {N_CLUSTERS}
Korištene PCA komponente za K-means:
{chr(10).join(pca_columns_for_kmeans)}

Train cluster counts:
{train_cluster_counts}

Test cluster counts:
{test_cluster_counts}

K-means inertia na train skupu:
{kmeans.inertia_}

K-means izlazi:
{KMEANS_TRAIN_RESULTS_CSV}
{KMEANS_TEST_RESULTS_CSV}
{KMEANS_ALL_RESULTS_CSV}
{KMEANS_CENTROIDS_CSV}

Spremljeni model objekti:
{MODEL_OBJECTS_PKL}

Uklonjeni redci zbog NaN/inf:
{REMOVED_ROWS_CSV if len(removed_df) > 0 else "Nema uklonjenih redaka."}
""".strip()

    REPORT_TXT.write_text(report, encoding="utf-8")

    print("=" * 80)
    print("Gotovo.")
    print("=" * 80)
    print(f"Report: {REPORT_TXT}")
    print(f"Model objects: {MODEL_OBJECTS_PKL}")
    print()
    print("Train cluster counts:")
    print(train_kmeans_df["cluster"].value_counts().sort_index())
    print()
    print("Test cluster counts:")
    print(test_kmeans_df["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()