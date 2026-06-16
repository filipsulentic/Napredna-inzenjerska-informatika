"""
04_generate_plots.py

Generiranje grafova za ModelNet10 projekt.

Workflow:
    1. Učitava rezultate prethodnih skripti
    2. Generira grafove standardizacije
    3. Generira PCA grafove
    4. Generira K-means grafove
    5. Generira validacijske grafove

Važno:
    - Outlieri se uklanjaju samo iz 2D vizualizacija.
    - Outlieri se NE uklanjaju iz analize, PCA, K-meansa ni validacije.
    - K-means je proveden na svih 10 PCA komponenti.
    - Vizualizacije se rade u PC1-PC2 prostoru.

Ulaz:
    outputs_final\\01_features\\features_modelnet10.csv
    outputs_final\\01_features\\feature_columns.txt
    outputs_final\\02_standardized\\features_train_standardized.csv
    outputs_final\\02_standardized\\features_test_standardized.csv
    outputs_final\\03_pca\\pca_train_results.csv
    outputs_final\\03_pca\\pca_test_results.csv
    outputs_final\\03_pca\\pca_explained_variance.csv
    outputs_final\\03_pca\\pca_loadings.csv
    outputs_final\\04_kmeans\\kmeans_train_results.csv
    outputs_final\\04_kmeans\\kmeans_test_results.csv
    outputs_final\\05_validation\\validation_metrics.csv
    outputs_final\\05_validation\\cluster_class_crosstab_train.csv
    outputs_final\\05_validation\\cluster_class_crosstab_test.csv

Izlaz:
    outputs_final\\06_plots\\standardization\\...
    outputs_final\\06_plots\\pca\\...
    outputs_final\\06_plots\\kmeans\\...
    outputs_final\\06_plots\\validation\\...
    outputs_final\\logs\\04_generate_plots_report.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. POSTAVKE
# ============================================================

BASE_DIR = Path(r"C:\Users\filip\Desktop\NII")
OUTPUT_DIR = BASE_DIR / "outputs_final"

FEATURES_DIR = OUTPUT_DIR / "01_features"
STANDARDIZED_DIR = OUTPUT_DIR / "02_standardized"
PCA_DIR = OUTPUT_DIR / "03_pca"
KMEANS_DIR = OUTPUT_DIR / "04_kmeans"
VALIDATION_DIR = OUTPUT_DIR / "05_validation"
PLOTS_DIR = OUTPUT_DIR / "06_plots"
LOGS_DIR = OUTPUT_DIR / "logs"

PLOTS_STANDARDIZATION_DIR = PLOTS_DIR / "standardization"
PLOTS_PCA_DIR = PLOTS_DIR / "pca"
PLOTS_KMEANS_DIR = PLOTS_DIR / "kmeans"
PLOTS_VALIDATION_DIR = PLOTS_DIR / "validation"

for d in [
    PLOTS_STANDARDIZATION_DIR,
    PLOTS_PCA_DIR,
    PLOTS_KMEANS_DIR,
    PLOTS_VALIDATION_DIR,
    LOGS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = FEATURES_DIR / "features_modelnet10.csv"
FEATURE_COLUMNS_TXT = FEATURES_DIR / "feature_columns.txt"

TRAIN_STANDARDIZED_CSV = STANDARDIZED_DIR / "features_train_standardized.csv"
TEST_STANDARDIZED_CSV = STANDARDIZED_DIR / "features_test_standardized.csv"
STANDARDIZATION_PARAMETERS_CSV = STANDARDIZED_DIR / "standardization_parameters.csv"

PCA_TRAIN_RESULTS_CSV = PCA_DIR / "pca_train_results.csv"
PCA_TEST_RESULTS_CSV = PCA_DIR / "pca_test_results.csv"
PCA_EXPLAINED_VARIANCE_CSV = PCA_DIR / "pca_explained_variance.csv"
PCA_LOADINGS_CSV = PCA_DIR / "pca_loadings.csv"

KMEANS_TRAIN_RESULTS_CSV = KMEANS_DIR / "kmeans_train_results.csv"
KMEANS_TEST_RESULTS_CSV = KMEANS_DIR / "kmeans_test_results.csv"

VALIDATION_METRICS_CSV = VALIDATION_DIR / "validation_metrics.csv"
CROSSTAB_TRAIN_CSV = VALIDATION_DIR / "cluster_class_crosstab_train.csv"
CROSSTAB_TEST_CSV = VALIDATION_DIR / "cluster_class_crosstab_test.csv"

REPORT_TXT = LOGS_DIR / "04_generate_plots_report.txt"

# Outlieri se uklanjaju samo iz 2D plotova.
# 0.01 i 0.99 znači da se iz prikaza uklanja ekstremnih 1 % s obje strane.
OUTLIER_LOW_Q = 0.01
OUTLIER_HIGH_Q = 0.99

DPI = 300


# ============================================================
# 2. POMOĆNE FUNKCIJE
# ============================================================

def read_feature_columns(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Ne postoji feature_columns.txt: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cols = [line.strip() for line in f if line.strip()]

    if len(cols) == 0:
        raise ValueError("feature_columns.txt je prazan.")

    return cols


def check_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Ne postoji datoteka: {path}")


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def filter_pc_outliers_for_plot(
    df: pd.DataFrame,
    x_col: str = "PC1",
    y_col: str = "PC2",
    low_q: float = OUTLIER_LOW_Q,
    high_q: float = OUTLIER_HIGH_Q,
) -> Tuple[pd.DataFrame, int]:
    """
    Uklanja outliere samo za potrebe 2D prikaza.

    Filtriranje se radi prema kvantilima PC1 i PC2.
    Originalni podaci i analitički rezultati se ne mijenjaju.
    """

    x_low = df[x_col].quantile(low_q)
    x_high = df[x_col].quantile(high_q)
    y_low = df[y_col].quantile(low_q)
    y_high = df[y_col].quantile(high_q)

    mask = (
        (df[x_col] >= x_low) &
        (df[x_col] <= x_high) &
        (df[y_col] >= y_low) &
        (df[y_col] <= y_high)
    )

    filtered = df.loc[mask].copy()
    n_removed = len(df) - len(filtered)

    return filtered, n_removed


def set_pc_axis_limits(ax, df: pd.DataFrame, x_col: str = "PC1", y_col: str = "PC2") -> None:
    """
    Postavlja granice osi s malom marginom kako bi se graf bolje vidio.
    """

    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0:
        x_range = 1.0

    if y_range == 0:
        y_range = 1.0

    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)


def get_std_columns(feature_columns: List[str]) -> List[str]:
    return [f"{col}_std" for col in feature_columns]


def plot_categorical_scatter(
    df: pd.DataFrame,
    category_col: str,
    title: str,
    output_path: Path,
    x_col: str = "PC1",
    y_col: str = "PC2",
    point_size: float = 12,
) -> int:
    """
    PC1-PC2 scatter plot s kategorijama.

    Outlieri se uklanjaju samo iz prikaza.
    """

    plot_df, n_removed = filter_pc_outliers_for_plot(df, x_col=x_col, y_col=y_col)

    fig, ax = plt.subplots(figsize=(12, 8))

    categories = sorted(plot_df[category_col].astype(str).unique())

    for category in categories:
        sub = plot_df[plot_df[category_col].astype(str) == category]
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=point_size,
            alpha=0.75,
            label=str(category),
        )

    ax.set_title(
        f"{title}\n"
        f"Outlieri uklonjeni samo iz prikaza: {n_removed}/{len(df)}",
        fontsize=13,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    set_pc_axis_limits(ax, plot_df, x_col=x_col, y_col=y_col)

    ax.legend(
        title=category_col,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
    )

    save_figure(fig, output_path)

    return n_removed


def plot_heatmap(
    matrix_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """
    Jednostavan heatmap bez seaborn biblioteke.
    """

    data = matrix_df.values

    fig_width = max(10, 0.8 * len(matrix_df.columns) + 4)
    fig_height = max(7, 0.6 * len(matrix_df.index) + 3)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, aspect="auto")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Stvarna klasa")
    ax.set_ylabel("K-means klaster")

    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_yticklabels(matrix_df.index)

    # Upis brojeva u ćelije
    max_value = data.max() if data.size > 0 else 0

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if max_value > 0 and value > 0.5 * max_value else "black"
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, output_path)


def sort_pc_columns(columns: List[str]) -> List[str]:
    pc_cols = [
        c for c in columns
        if c.startswith("PC") and c[2:].isdigit()
    ]

    return sorted(pc_cols, key=lambda c: int(c.replace("PC", "")))


# ============================================================
# 3. GRAFOVI STANDARDIZACIJE
# ============================================================

def generate_standardization_plots(
    features_df: pd.DataFrame,
    train_std_df: pd.DataFrame,
    test_std_df: pd.DataFrame,
    standardization_parameters_df: pd.DataFrame,
    feature_columns: List[str],
) -> List[str]:

    created = []

    train_raw_df = features_df[features_df["split"].astype(str).str.lower() == "train"].copy()
    test_raw_df = features_df[features_df["split"].astype(str).str.lower() == "test"].copy()

    std_columns = get_std_columns(feature_columns)

    # ------------------------------------------------------------
    # 1. Boxplot prije standardizacije, train
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(
        [train_raw_df[col].dropna().values for col in feature_columns],
        labels=feature_columns,
        showfliers=False,
    )
    ax.set_title("Raspon geometrijskih značajki prije standardizacije - train skup")
    ax.set_ylabel("Izvorna vrijednost")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    path = PLOTS_STANDARDIZATION_DIR / "01_boxplot_before_standardization_train.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 2. Boxplot poslije standardizacije, train
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(
        [train_std_df[col].dropna().values for col in std_columns],
        labels=feature_columns,
        showfliers=False,
    )
    ax.set_title("Značajke nakon z-score standardizacije - train skup")
    ax.set_ylabel("Standardizirana vrijednost")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    path = PLOTS_STANDARDIZATION_DIR / "02_boxplot_after_standardization_train.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 3. Boxplot poslije standardizacije, test
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.boxplot(
        [test_std_df[col].dropna().values for col in std_columns],
        labels=feature_columns,
        showfliers=False,
    )
    ax.set_title("Značajke nakon z-score standardizacije - test skup")
    ax.set_ylabel("Standardizirana vrijednost")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)

    path = PLOTS_STANDARDIZATION_DIR / "03_boxplot_after_standardization_test.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 4. Mean i std prije standardizacije
    # ------------------------------------------------------------

    x = np.arange(len(feature_columns))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, standardization_parameters_df["mean_train"].values)
    ax.set_title("Srednje vrijednosti značajki na train skupu prije standardizacije")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_columns, rotation=45, ha="right")
    ax.set_ylabel("Mean train")
    ax.grid(True, axis="y", alpha=0.3)

    path = PLOTS_STANDARDIZATION_DIR / "04_mean_train_before_standardization.png"
    save_figure(fig, path)
    created.append(str(path))

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, standardization_parameters_df["std_train"].values)
    ax.set_title("Standardne devijacije značajki na train skupu prije standardizacije")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_columns, rotation=45, ha="right")
    ax.set_ylabel("Std train")
    ax.grid(True, axis="y", alpha=0.3)

    path = PLOTS_STANDARDIZATION_DIR / "05_std_train_before_standardization.png"
    save_figure(fig, path)
    created.append(str(path))

    return created


# ============================================================
# 4. PCA GRAFOVI
# ============================================================

def generate_pca_plots(
    pca_train_df: pd.DataFrame,
    pca_test_df: pd.DataFrame,
    pca_explained_df: pd.DataFrame,
    pca_loadings_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:

    created = []
    outlier_messages = []

    # ------------------------------------------------------------
    # 1. Scree plot
    # ------------------------------------------------------------

    components = pca_explained_df["component"].astype(str).values
    explained_percent = pca_explained_df["explained_variance_percent"].values
    x = np.arange(len(components))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, explained_percent)
    ax.set_title("PCA scree plot - objašnjena varijanca po komponenti")
    ax.set_xlabel("PCA komponenta")
    ax.set_ylabel("Objašnjena varijanca (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.grid(True, axis="y", alpha=0.3)

    for i, value in enumerate(explained_percent):
        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    path = PLOTS_PCA_DIR / "01_pca_scree_plot.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 2. Kumulativna varijanca
    # ------------------------------------------------------------

    cumulative_percent = pca_explained_df["cumulative_variance_percent"].values

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(x, cumulative_percent, marker="o")
    ax.set_title("Kumulativna objašnjena varijanca PCA komponenti")
    ax.set_xlabel("PCA komponenta")
    ax.set_ylabel("Kumulativna objašnjena varijanca (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(components)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    for i, value in enumerate(cumulative_percent):
        ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)

    path = PLOTS_PCA_DIR / "02_pca_cumulative_variance.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 3. PC1-PC2 stvarne klase - train
    # ------------------------------------------------------------

    path = PLOTS_PCA_DIR / "03_pca_true_classes_train.png"
    n_removed = plot_categorical_scatter(
        pca_train_df,
        category_col="class_label",
        title="PCA prikaz stvarnih klasa - train skup",
        output_path=path,
    )
    created.append(str(path))
    outlier_messages.append(f"03_pca_true_classes_train: uklonjeno iz prikaza {n_removed}/{len(pca_train_df)}")

    # ------------------------------------------------------------
    # 4. PC1-PC2 stvarne klase - test
    # ------------------------------------------------------------

    path = PLOTS_PCA_DIR / "04_pca_true_classes_test.png"
    n_removed = plot_categorical_scatter(
        pca_test_df,
        category_col="class_label",
        title="PCA prikaz stvarnih klasa - test skup",
        output_path=path,
    )
    created.append(str(path))
    outlier_messages.append(f"04_pca_true_classes_test: uklonjeno iz prikaza {n_removed}/{len(pca_test_df)}")

    # ------------------------------------------------------------
    # 5. Loadings PC1
    # ------------------------------------------------------------

    pc1_loadings = pca_loadings_df["PC1"].copy()
    pc1_sorted = pc1_loadings.reindex(pc1_loadings.abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(pc1_sorted.index, pc1_sorted.values)
    ax.set_title("PCA loadings - PC1")
    ax.set_xlabel("Loading vrijednost")
    ax.grid(True, axis="x", alpha=0.3)

    path = PLOTS_PCA_DIR / "05_pca_loadings_PC1.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 6. Loadings PC2
    # ------------------------------------------------------------

    pc2_loadings = pca_loadings_df["PC2"].copy()
    pc2_sorted = pc2_loadings.reindex(pc2_loadings.abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(pc2_sorted.index, pc2_sorted.values)
    ax.set_title("PCA loadings - PC2")
    ax.set_xlabel("Loading vrijednost")
    ax.grid(True, axis="x", alpha=0.3)

    path = PLOTS_PCA_DIR / "06_pca_loadings_PC2.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 7. PCA biplot train
    # ------------------------------------------------------------

    path = PLOTS_PCA_DIR / "07_pca_biplot_train.png"
    n_removed = plot_pca_biplot(
        pca_train_df,
        pca_loadings_df,
        title="PCA biplot - train skup",
        output_path=path,
    )
    created.append(str(path))
    outlier_messages.append(f"07_pca_biplot_train: uklonjeno iz prikaza {n_removed}/{len(pca_train_df)}")

    # ------------------------------------------------------------
    # 8. PCA biplot test
    # ------------------------------------------------------------

    path = PLOTS_PCA_DIR / "08_pca_biplot_test.png"
    n_removed = plot_pca_biplot(
        pca_test_df,
        pca_loadings_df,
        title="PCA biplot - test skup",
        output_path=path,
    )
    created.append(str(path))
    outlier_messages.append(f"08_pca_biplot_test: uklonjeno iz prikaza {n_removed}/{len(pca_test_df)}")

    return created, outlier_messages


def plot_pca_biplot(
    pca_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> int:
    """
    PCA biplot za PC1-PC2.

    Točke se prikazuju u PC1-PC2 prostoru.
    Strelice prikazuju smjerove originalnih značajki prema loadings vrijednostima.
    """

    plot_df, n_removed = filter_pc_outliers_for_plot(pca_df, "PC1", "PC2")

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        s=10,
        alpha=0.35,
    )

    x_range = plot_df["PC1"].max() - plot_df["PC1"].min()
    y_range = plot_df["PC2"].max() - plot_df["PC2"].min()

    if x_range == 0:
        x_range = 1.0

    if y_range == 0:
        y_range = 1.0

    max_loading = np.sqrt(
        loadings_df["PC1"].values ** 2 +
        loadings_df["PC2"].values ** 2
    ).max()

    if max_loading == 0:
        max_loading = 1.0

    arrow_scale = 0.35 * min(x_range, y_range) / max_loading

    for feature_name, row in loadings_df.iterrows():
        x_arrow = row["PC1"] * arrow_scale
        y_arrow = row["PC2"] * arrow_scale

        ax.arrow(
            0,
            0,
            x_arrow,
            y_arrow,
            head_width=0.03 * min(x_range, y_range),
            length_includes_head=True,
            alpha=0.8,
        )

        ax.text(
            x_arrow * 1.08,
            y_arrow * 1.08,
            str(feature_name),
            fontsize=9,
        )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    ax.set_title(
        f"{title}\n"
        f"Outlieri uklonjeni samo iz prikaza: {n_removed}/{len(pca_df)}",
        fontsize=13,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)
    set_pc_axis_limits(ax, plot_df, "PC1", "PC2")

    save_figure(fig, output_path)

    return n_removed


# ============================================================
# 5. K-MEANS GRAFOVI
# ============================================================

def generate_kmeans_plots(
    kmeans_train_df: pd.DataFrame,
    kmeans_test_df: pd.DataFrame,
) -> Tuple[List[str], List[str]]:

    created = []
    outlier_messages = []

    # ------------------------------------------------------------
    # 1. K-means klasteri - train
    # ------------------------------------------------------------

    train_plot_df = kmeans_train_df.copy()
    train_plot_df["cluster"] = train_plot_df["cluster"].astype(str)

    path = PLOTS_KMEANS_DIR / "01_kmeans_clusters_train.png"
    n_removed = plot_categorical_scatter(
        train_plot_df,
        category_col="cluster",
        title="K-means klasteri u PC1-PC2 prostoru - train skup",
        output_path=path,
    )
    created.append(str(path))
    outlier_messages.append(f"01_kmeans_clusters_train: uklonjeno iz prikaza {n_removed}/{len(kmeans_train_df)}")

    # ------------------------------------------------------------
    # 2. K-means klasteri - test
    # ------------------------------------------------------------

    test_plot_df = kmeans_test_df.copy()
    test_plot_df["cluster"] = test_plot_df["cluster"].astype(str)

    path = PLOTS_KMEANS_DIR / "02_kmeans_clusters_test.png"
    n_removed = plot_categorical_scatter(
        test_plot_df,
        category_col="cluster",
        title="K-means klasteri u PC1-PC2 prostoru - test skup",
        output_path=path,
    )
    created.append(str(path))
    outlier_messages.append(f"02_kmeans_clusters_test: uklonjeno iz prikaza {n_removed}/{len(kmeans_test_df)}")

    # ------------------------------------------------------------
    # 3. Veličine klastera - train
    # ------------------------------------------------------------

    cluster_counts_train = kmeans_train_df["cluster"].astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(cluster_counts_train.index.astype(str), cluster_counts_train.values)
    ax.set_title("Broj modela po K-means klasteru - train skup")
    ax.set_xlabel("Klaster")
    ax.set_ylabel("Broj modela")
    ax.grid(True, axis="y", alpha=0.3)

    path = PLOTS_KMEANS_DIR / "03_cluster_sizes_train.png"
    save_figure(fig, path)
    created.append(str(path))

    # ------------------------------------------------------------
    # 4. Veličine klastera - test
    # ------------------------------------------------------------

    cluster_counts_test = kmeans_test_df["cluster"].astype(int).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(cluster_counts_test.index.astype(str), cluster_counts_test.values)
    ax.set_title("Broj modela po K-means klasteru - test skup")
    ax.set_xlabel("Klaster")
    ax.set_ylabel("Broj modela")
    ax.grid(True, axis="y", alpha=0.3)

    path = PLOTS_KMEANS_DIR / "04_cluster_sizes_test.png"
    save_figure(fig, path)
    created.append(str(path))

    return created, outlier_messages


# ============================================================
# 6. VALIDACIJSKI GRAFOVI
# ============================================================

def generate_validation_plots(
    validation_metrics_df: pd.DataFrame,
    crosstab_train_df: pd.DataFrame,
    crosstab_test_df: pd.DataFrame,
) -> List[str]:

    created = []

    # ------------------------------------------------------------
    # 1. Cluster x class heatmap train
    # ------------------------------------------------------------

    path = PLOTS_VALIDATION_DIR / "01_cluster_class_crosstab_train.png"
    plot_heatmap(
        crosstab_train_df,
        "Cluster × class tablica - train skup",
        path,
    )
    created.append(str(path))

    # ------------------------------------------------------------
    # 2. Cluster x class heatmap test
    # ------------------------------------------------------------

    path = PLOTS_VALIDATION_DIR / "02_cluster_class_crosstab_test.png"
    plot_heatmap(
        crosstab_test_df,
        "Cluster × class tablica - test skup",
        path,
    )
    created.append(str(path))

    # ------------------------------------------------------------
    # 3. Validacijske metrike train/test
    # ------------------------------------------------------------

    metric_cols = [
        "silhouette_score",
        "adjusted_rand_index",
        "normalized_mutual_information",
    ]

    metric_labels = [
        "Silhouette",
        "ARI",
        "NMI",
    ]

    available_metrics = [m for m in metric_cols if m in validation_metrics_df.columns]

    if len(available_metrics) > 0:
        datasets = validation_metrics_df["dataset"].astype(str).values

        x = np.arange(len(metric_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, dataset in enumerate(datasets):
            row = validation_metrics_df[validation_metrics_df["dataset"].astype(str) == dataset].iloc[0]
            values = [row[col] for col in metric_cols]

            offset = (idx - (len(datasets) - 1) / 2) * width

            ax.bar(
                x + offset,
                values,
                width,
                label=dataset,
            )

        ax.set_title("Validacijske metrike K-means klasteriranja")
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylabel("Vrijednost")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Skup")

        path = PLOTS_VALIDATION_DIR / "03_validation_metrics_train_test.png"
        save_figure(fig, path)
        created.append(str(path))

    return created


# ============================================================
# 7. GLAVNI PROGRAM
# ============================================================

def main() -> None:
    print("=" * 80)
    print("04_generate_plots.py")
    print("Generiranje grafova")
    print("=" * 80)

    input_files = [
        FEATURES_CSV,
        FEATURE_COLUMNS_TXT,
        TRAIN_STANDARDIZED_CSV,
        TEST_STANDARDIZED_CSV,
        STANDARDIZATION_PARAMETERS_CSV,
        PCA_TRAIN_RESULTS_CSV,
        PCA_TEST_RESULTS_CSV,
        PCA_EXPLAINED_VARIANCE_CSV,
        PCA_LOADINGS_CSV,
        KMEANS_TRAIN_RESULTS_CSV,
        KMEANS_TEST_RESULTS_CSV,
        VALIDATION_METRICS_CSV,
        CROSSTAB_TRAIN_CSV,
        CROSSTAB_TEST_CSV,
    ]

    for file_path in input_files:
        check_file(file_path)

    feature_columns = read_feature_columns(FEATURE_COLUMNS_TXT)

    features_df = pd.read_csv(FEATURES_CSV)
    train_std_df = pd.read_csv(TRAIN_STANDARDIZED_CSV)
    test_std_df = pd.read_csv(TEST_STANDARDIZED_CSV)
    standardization_parameters_df = pd.read_csv(STANDARDIZATION_PARAMETERS_CSV)

    pca_train_df = pd.read_csv(PCA_TRAIN_RESULTS_CSV)
    pca_test_df = pd.read_csv(PCA_TEST_RESULTS_CSV)
    pca_explained_df = pd.read_csv(PCA_EXPLAINED_VARIANCE_CSV)
    pca_loadings_df = pd.read_csv(PCA_LOADINGS_CSV, index_col=0)

    kmeans_train_df = pd.read_csv(KMEANS_TRAIN_RESULTS_CSV)
    kmeans_test_df = pd.read_csv(KMEANS_TEST_RESULTS_CSV)

    validation_metrics_df = pd.read_csv(VALIDATION_METRICS_CSV)
    crosstab_train_df = pd.read_csv(CROSSTAB_TRAIN_CSV, index_col=0)
    crosstab_test_df = pd.read_csv(CROSSTAB_TEST_CSV, index_col=0)

    created_files = []
    outlier_messages = []

    print("Generiram grafove standardizacije...")
    files = generate_standardization_plots(
        features_df,
        train_std_df,
        test_std_df,
        standardization_parameters_df,
        feature_columns,
    )
    created_files.extend(files)

    print("Generiram PCA grafove...")
    files, messages = generate_pca_plots(
        pca_train_df,
        pca_test_df,
        pca_explained_df,
        pca_loadings_df,
    )
    created_files.extend(files)
    outlier_messages.extend(messages)

    print("Generiram K-means grafove...")
    files, messages = generate_kmeans_plots(
        kmeans_train_df,
        kmeans_test_df,
    )
    created_files.extend(files)
    outlier_messages.extend(messages)

    print("Generiram validacijske grafove...")
    files = generate_validation_plots(
        validation_metrics_df,
        crosstab_train_df,
        crosstab_test_df,
    )
    created_files.extend(files)

    report = f"""
04_generate_plots.py report

Generirani grafovi:
{chr(10).join(created_files)}

Outlier filtering:
Outlieri su uklonjeni samo iz 2D vizualizacija, ne iz analize.
Korišteni kvantili:
LOW_Q = {OUTLIER_LOW_Q}
HIGH_Q = {OUTLIER_HIGH_Q}

Broj uklonjenih točaka iz pojedinih prikaza:
{chr(10).join(outlier_messages)}

Napomena:
K-means je proveden na svih 10 PCA komponenti.
PC1-PC2 grafovi služe samo za vizualizaciju rezultata u dvije dimenzije.
Zato se preklapanje na 2D grafu ne mora potpuno slagati s K-means dodjelom u 10D PCA prostoru.
""".strip()

    REPORT_TXT.write_text(report, encoding="utf-8")

    print()
    print("=" * 80)
    print("Gotovo.")
    print("=" * 80)
    print(f"Broj generiranih grafova: {len(created_files)}")
    print(f"Report: {REPORT_TXT}")
    print()
    print("Grafovi su spremljeni u:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()