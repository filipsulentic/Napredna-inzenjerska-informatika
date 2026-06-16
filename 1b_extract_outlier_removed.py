"""
01b_filter_feature_outliers.py

Uklanjanje ekstremnih outliera iz već izračunatih geometrijskih značajki.

Ova skripta NE mijenja originalni features_modelnet10.csv.
Stvara novi CSV:
    features_modelnet10_iqr_filtered.csv

Workflow:
    1. Učita features_modelnet10.csv
    2. Učita feature_columns.txt
    3. IQR granice računa samo na train skupu
    4. Iste granice primjenjuje na train i test
    5. Uklanja retke koji imaju ekstremnu vrijednost u barem jednoj značajki
    6. Sprema filtrirani CSV i report

Važno:
    - Outlieri se uklanjaju prije standardizacije.
    - IQR granice se računaju na train skupu da test skup ne ulazi u analizu.
    - Originalni output ostaje sačuvan.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd


# ============================================================
# 1. POSTAVKE
# ============================================================

BASE_DIR = Path(r"C:\Users\filip\Desktop\NII")

OUTPUT_DIR = BASE_DIR / "outputs_final"
FEATURES_DIR = OUTPUT_DIR / "01_features"
LOGS_DIR = OUTPUT_DIR / "logs"

FEATURES_CSV = FEATURES_DIR / "features_modelnet10.csv"
FEATURE_COLUMNS_TXT = FEATURES_DIR / "feature_columns.txt"

FILTERED_FEATURES_CSV = FEATURES_DIR / "features_modelnet10_iqr_filtered.csv"
OUTLIER_BOUNDS_CSV = FEATURES_DIR / "feature_outlier_bounds_iqr.csv"
REMOVED_OUTLIERS_CSV = FEATURES_DIR / "feature_outlier_removed_rows.csv"
REPORT_TXT = LOGS_DIR / "01b_filter_feature_outliers_report.txt"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Blaži IQR kriterij. Standardno je 1.5, ali ovdje želimo ukloniti samo ekstremne vrijednosti.
IQR_MULTIPLIER = 3.0


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


def validate_input(df: pd.DataFrame, feature_columns: List[str]) -> None:
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
        raise ValueError(f"Nedostaju obavezni stupci: {missing_required}")

    if missing_features:
        raise ValueError(f"Nedostaju feature stupci: {missing_features}")


def compute_iqr_bounds(train_df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Računa IQR granice na train skupu.

    lower = Q1 - IQR_MULTIPLIER * IQR
    upper = Q3 + IQR_MULTIPLIER * IQR
    """

    rows = []

    for feature in feature_columns:
        values = train_df[feature].replace([np.inf, -np.inf], np.nan).dropna()

        if len(values) == 0:
            rows.append({
                "feature": feature,
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "used_for_filtering": False,
                "reason": "no_valid_values",
            })
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        if not np.isfinite(iqr) or iqr == 0:
            rows.append({
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "used_for_filtering": False,
                "reason": "iqr_zero_or_invalid",
            })
            continue

        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr

        rows.append({
            "feature": feature,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "used_for_filtering": True,
            "reason": "ok",
        })

    return pd.DataFrame(rows)


def mark_outliers(df: pd.DataFrame, bounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Označava retke koji su outlieri prema IQR granicama.

    Redak se uklanja ako je barem jedna značajka izvan dopuštenog raspona.
    """

    result = df.copy()

    result["is_outlier_iqr"] = False
    result["outlier_features"] = ""

    active_bounds = bounds_df[bounds_df["used_for_filtering"] == True].copy()

    for idx, row in result.iterrows():
        outlier_features = []

        for _, b in active_bounds.iterrows():
            feature = b["feature"]
            lower = b["lower_bound"]
            upper = b["upper_bound"]

            value = row[feature]

            if not np.isfinite(value):
                outlier_features.append(feature)
                continue

            if value < lower or value > upper:
                outlier_features.append(feature)

        if len(outlier_features) > 0:
            result.at[idx, "is_outlier_iqr"] = True
            result.at[idx, "outlier_features"] = ", ".join(outlier_features)

    return result


# ============================================================
# 3. GLAVNI PROGRAM
# ============================================================

def main() -> None:
    print("=" * 80)
    print("01b_filter_feature_outliers.py")
    print("Uklanjanje ekstremnih outliera iz geometrijskih značajki")
    print("=" * 80)

    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Ne postoji features CSV: {FEATURES_CSV}")

    feature_columns = read_feature_columns(FEATURE_COLUMNS_TXT)

    df = pd.read_csv(FEATURES_CSV)
    df["split"] = df["split"].astype(str).str.lower()

    validate_input(df, feature_columns)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    if len(train_df) == 0:
        raise RuntimeError("Train skup je prazan.")

    if len(test_df) == 0:
        raise RuntimeError("Test skup je prazan.")

    print(f"Ulaz: {FEATURES_CSV}")
    print(f"Broj modela ukupno: {len(df)}")
    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print(f"IQR multiplier: {IQR_MULTIPLIER}")
    print()

    # ------------------------------------------------------------
    # 3.1 IQR granice na train skupu
    # ------------------------------------------------------------

    bounds_df = compute_iqr_bounds(train_df, feature_columns)

    bounds_df.to_csv(
        OUTLIER_BOUNDS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3.2 Primjena granica na cijeli skup
    # ------------------------------------------------------------

    marked_df = mark_outliers(df, bounds_df)

    filtered_df = marked_df[marked_df["is_outlier_iqr"] == False].copy()
    removed_df = marked_df[marked_df["is_outlier_iqr"] == True].copy()

    # U glavnom filtriranom CSV-u ne trebaju pomoćni stupci.
    filtered_output_df = filtered_df.drop(
        columns=["is_outlier_iqr", "outlier_features"],
        errors="ignore"
    )

    filtered_output_df.to_csv(
        FILTERED_FEATURES_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    removed_df.to_csv(
        REMOVED_OUTLIERS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3.3 Statistika
    # ------------------------------------------------------------

    n_total = len(df)
    n_removed = len(removed_df)
    n_kept = len(filtered_output_df)

    train_removed = int((removed_df["split"] == "train").sum())
    test_removed = int((removed_df["split"] == "test").sum())

    train_kept = int((filtered_output_df["split"] == "train").sum())
    test_kept = int((filtered_output_df["split"] == "test").sum())

    removed_by_class = removed_df["class_label"].value_counts().sort_index()
    kept_by_class = filtered_output_df["class_label"].value_counts().sort_index()

    outlier_feature_counts: Dict[str, int] = {}

    for text in removed_df["outlier_features"].dropna():
        features = [x.strip() for x in str(text).split(",") if x.strip()]
        for feature in features:
            outlier_feature_counts[feature] = outlier_feature_counts.get(feature, 0) + 1

    outlier_feature_counts_df = pd.DataFrame(
        [
            {"feature": k, "n_outliers": v}
            for k, v in sorted(outlier_feature_counts.items(), key=lambda x: x[1], reverse=True)
        ]
    )

    OUTLIER_FEATURE_COUNTS_CSV = FEATURES_DIR / "feature_outlier_counts.csv"
    outlier_feature_counts_df.to_csv(
        OUTLIER_FEATURE_COUNTS_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    # ------------------------------------------------------------
    # 3.4 Report
    # ------------------------------------------------------------

    report = f"""
01b_filter_feature_outliers.py report

Ulaz:
{FEATURES_CSV}

Izlaz:
{FILTERED_FEATURES_CSV}

Metoda:
IQR filtering

Granice:
lower = Q1 - {IQR_MULTIPLIER} * IQR
upper = Q3 + {IQR_MULTIPLIER} * IQR

Važno:
IQR granice su izračunate samo na train skupu.
Iste granice su zatim primijenjene na train i test.
Outlieri su uklonjeni prije standardizacije.
Originalni features_modelnet10.csv nije izmijenjen.

Broj modela prije filtriranja:
{n_total}

Broj uklonjenih modela:
{n_removed}

Broj zadržanih modela:
{n_kept}

Uklonjeno train:
{train_removed}

Uklonjeno test:
{test_removed}

Zadržano train:
{train_kept}

Zadržano test:
{test_kept}

Uklonjeno po klasama:
{removed_by_class.to_string() if len(removed_by_class) > 0 else "Nema uklonjenih modela."}

Zadržano po klasama:
{kept_by_class.to_string()}

Značajke koje su najčešće uzrokovale uklanjanje:
{outlier_feature_counts_df.to_string(index=False) if len(outlier_feature_counts_df) > 0 else "Nema uklonjenih outliera."}

Datoteke:
Filtrirani features CSV:
{FILTERED_FEATURES_CSV}

IQR granice:
{OUTLIER_BOUNDS_CSV}

Uklonjeni redci:
{REMOVED_OUTLIERS_CSV}

Broj uklanjanja po značajci:
{OUTLIER_FEATURE_COUNTS_CSV}
""".strip()

    REPORT_TXT.write_text(report, encoding="utf-8")

    # ------------------------------------------------------------
    # 3.5 Ispis
    # ------------------------------------------------------------

    print("Gotovo.")
    print()
    print(f"Uklonjeno modela: {n_removed}/{n_total}")
    print(f"Zadržano modela: {n_kept}/{n_total}")
    print()
    print(f"Uklonjeno train: {train_removed}")
    print(f"Uklonjeno test: {test_removed}")
    print()
    print("Značajke koje su najčešće uzrokovale uklanjanje:")
    if len(outlier_feature_counts_df) > 0:
        print(outlier_feature_counts_df)
    else:
        print("Nema uklonjenih outliera.")
    print()
    print(f"Filtrirani CSV: {FILTERED_FEATURES_CSV}")
    print(f"Report: {REPORT_TXT}")


if __name__ == "__main__":
    main()