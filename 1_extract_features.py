"""
01_extract_features.py

Ekstrakcija geometrijskih značajki iz ModelNet10 .off modela.

Workflow:
    1. Učitavanje metadata_modelnet10.csv
    2. Učitavanje .off modela pomoću trimesh
    3. Osnovno zatvaranje/popravak mreže prije računanja značajki
    4. Izračun geometrijskih značajki
    5. Spremanje rezultata u outputs_final/01_features

Ulaz:
    C:\\Users\\filip\\Desktop\\NII\\metadata_modelnet10.csv
    C:\\Users\\filip\\Desktop\\NII\\ModelNet10\\...

Izlaz:
    outputs_final\\01_features\\features_modelnet10.csv
    outputs_final\\01_features\\feature_columns.txt
    outputs_final\\01_features\\feature_extraction_errors.csv
    outputs_final\\01_features\\mesh_repair_log.csv
    outputs_final\\logs\\01_feature_extraction_report.txt
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import trimesh
import trimesh.repair


# ============================================================
# 1. POSTAVKE
# ============================================================

BASE_DIR = Path(r"C:\Users\filip\Desktop\NII")
MODELNET_DIR = BASE_DIR / "ModelNet10"
METADATA_PATH = BASE_DIR / "metadata_modelnet10.csv"

OUTPUT_DIR = BASE_DIR / "outputs_final"
FEATURES_DIR = OUTPUT_DIR / "01_features"
LOGS_DIR = OUTPUT_DIR / "logs"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = FEATURES_DIR / "features_modelnet10.csv"
FEATURE_COLUMNS_TXT = FEATURES_DIR / "feature_columns.txt"
ERRORS_CSV = FEATURES_DIR / "feature_extraction_errors.csv"
REPAIR_LOG_CSV = FEATURES_DIR / "mesh_repair_log.csv"
REPORT_TXT = LOGS_DIR / "01_feature_extraction_report.txt"

SUPPORTED_EXTENSIONS = [".off"]


# ============================================================
# 2. POMOĆNE FUNKCIJE
# ============================================================

def safe_divide(a: float, b: float) -> float:
    """
    Sigurno dijeljenje.

    Ako nazivnik nije valjan, vraća NaN.
    """

    if b is None:
        return np.nan

    if not np.isfinite(b):
        return np.nan

    if abs(b) < 1e-12:
        return np.nan

    return a / b


def validate_metadata_columns(metadata_df: pd.DataFrame) -> None:
    """
    Provjerava postoje li potrebni stupci u metadata CSV-u.
    """

    required_columns = ["object_id", "class", "split", "object_path"]
    missing = [col for col in required_columns if col not in metadata_df.columns]

    if missing:
        raise ValueError(
            "metadata_modelnet10.csv nema potrebne stupce.\n"
            f"Nedostaju: {missing}\n"
            f"Pronađeni stupci: {list(metadata_df.columns)}"
        )


def resolve_object_path(object_path_value: str) -> Path:
    """
    Pretvara vrijednost iz stupca object_path u stvarnu putanju.

    Podržava:
        1. apsolutnu putanju
        2. putanju relativnu na BASE_DIR
        3. putanju relativnu na MODELNET_DIR
        4. slučaj kada ekstenzija nije navedena
    """

    raw = str(object_path_value).strip().strip('"').strip("'")

    if raw == "" or raw.lower() == "nan":
        raise ValueError("Prazan object_path u metadata tablici.")

    raw_slash = raw.replace("\\", "/")

    raw_variants = [
        raw,
        raw_slash,
    ]

    candidates: List[Path] = []

    for variant in raw_variants:
        p = Path(variant)

        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(BASE_DIR / p)
            candidates.append(MODELNET_DIR / p)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    extended_candidates: List[Path] = []

    for candidate in candidates:
        if candidate.suffix.lower() == "":
            for ext in SUPPORTED_EXTENSIONS:
                extended_candidates.append(candidate.with_suffix(ext))

    for candidate in extended_candidates:
        if candidate.exists():
            return candidate

    checked = candidates + extended_candidates
    checked_txt = "\n".join(str(c) for c in checked[:20])

    raise FileNotFoundError(
        "Nije moguće pronaći model za object_path:\n"
        f"{object_path_value}\n\n"
        "Provjerene putanje, prvih 20:\n"
        f"{checked_txt}"
    )


def load_mesh(path: Path) -> trimesh.Trimesh:
    """
    Učitava .off model kao trimesh.Trimesh.

    Ako trimesh vrati Scene, sve geometrije se spajaju u jedan mesh.
    """

    obj = trimesh.load(path, force="mesh", process=True)

    if isinstance(obj, trimesh.Scene):
        geometries = list(obj.geometry.values())

        if len(geometries) == 0:
            raise ValueError("Scene ne sadrži geometriju.")

        obj = trimesh.util.concatenate(geometries)

    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Objekt nije Trimesh nego: {type(obj)}")

    if obj.vertices is None or obj.faces is None:
        raise ValueError("Mesh nema vrhove ili lica.")

    if len(obj.vertices) == 0 or len(obj.faces) == 0:
        raise ValueError("Mesh je prazan.")

    return obj


def mesh_status(mesh: trimesh.Trimesh) -> Dict[str, object]:
    """
    Vraća osnovne informacije o stanju mreže.

    Ove vrijednosti se spremaju u poseban repair log,
    ali ne ulaze u glavni CSV značajki.
    """

    return {
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "is_volume": bool(mesh.is_volume),
        "euler_number": int(mesh.euler_number) if mesh.euler_number is not None else np.nan,
    }


def repair_mesh(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    """
    Osnovno zatvaranje/popravak mreže prije izračuna značajki.

    Koraci:
        - kopiranje mreže
        - process(validate=True)
        - fix_normals
        - fix_winding
        - fix_inversion
        - fill_holes

    Napomena:
        Ovo ne garantira savršeno zatvaranje svakog modela,
        ali provodi osnovnu korekciju prikladnu za projektni zadatak.
    """

    repaired = mesh.copy()

    before = mesh_status(repaired)

    repair_messages: List[str] = []

    try:
        repaired.process(validate=True)
        repair_messages.append("process_validate_ok")
    except Exception as e:
        repair_messages.append(f"process_validate_error: {e}")

    try:
        trimesh.repair.fix_normals(repaired, multibody=True)
        repair_messages.append("fix_normals_ok")
    except Exception as e:
        repair_messages.append(f"fix_normals_error: {e}")

    try:
        trimesh.repair.fix_winding(repaired)
        repair_messages.append("fix_winding_ok")
    except Exception as e:
        repair_messages.append(f"fix_winding_error: {e}")

    try:
        trimesh.repair.fix_inversion(repaired, multibody=True)
        repair_messages.append("fix_inversion_ok")
    except Exception as e:
        repair_messages.append(f"fix_inversion_error: {e}")

    try:
        fill_result = trimesh.repair.fill_holes(repaired)
        repair_messages.append(f"fill_holes_result: {fill_result}")
    except Exception as e:
        repair_messages.append(f"fill_holes_error: {e}")

    after = mesh_status(repaired)

    info = {
        "before_n_vertices": before["n_vertices"],
        "before_n_faces": before["n_faces"],
        "before_is_watertight": before["is_watertight"],
        "before_is_winding_consistent": before["is_winding_consistent"],
        "before_is_volume": before["is_volume"],
        "before_euler_number": before["euler_number"],

        "after_n_vertices": after["n_vertices"],
        "after_n_faces": after["n_faces"],
        "after_is_watertight": after["is_watertight"],
        "after_is_winding_consistent": after["is_winding_consistent"],
        "after_is_volume": after["is_volume"],
        "after_euler_number": after["euler_number"],

        "repair_messages": " | ".join(repair_messages),
    }

    return repaired, info


def compute_features(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """
    Računa tražene geometrijske značajke.

    Izlazne značajke:
        volume
        surface_area
        bbox_x
        bbox_y
        bbox_z
        ratio_xy
        ratio_xz
        ratio_yz
        compactness
        mesh_density

    Kompaktnost:
        Compactness 1 iz PyRadiomicsa

        C1 = V / (sqrt(pi) * A^(3/2))

    Gustoća mreže:
        mesh_density = broj lica / površina
    """

    n_faces = int(len(mesh.faces))

    surface_area = float(mesh.area)

    # abs() zbog mogućeg negativnog predznaka volumena nakon orijentacije normala
    volume = float(abs(mesh.volume))

    bbox_extents = np.asarray(mesh.bounding_box.extents, dtype=float)
    bbox_x, bbox_y, bbox_z = bbox_extents.tolist()

    ratio_xy = safe_divide(bbox_x, bbox_y)
    ratio_xz = safe_divide(bbox_x, bbox_z)
    ratio_yz = safe_divide(bbox_y, bbox_z)

    # Compactness 1:
    # C1 = V / (sqrt(pi) * A^(3/2))
    compactness = safe_divide(
        volume,
        math.sqrt(math.pi) * (surface_area ** 1.5)
    )

    mesh_density = safe_divide(
        float(n_faces),
        surface_area
    )

    return {
        "volume": volume,
        "surface_area": surface_area,
        "bbox_x": float(bbox_x),
        "bbox_y": float(bbox_y),
        "bbox_z": float(bbox_z),
        "ratio_xy": ratio_xy,
        "ratio_xz": ratio_xz,
        "ratio_yz": ratio_yz,
        "compactness": compactness,
        "mesh_density": mesh_density,
    }


def write_feature_columns_file() -> None:
    """
    Sprema popis značajki koje ulaze u standardizaciju, PCA i K-means.
    """

    feature_columns = [
        "volume",
        "surface_area",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "ratio_xy",
        "ratio_xz",
        "ratio_yz",
        "compactness",
        "mesh_density",
    ]

    FEATURE_COLUMNS_TXT.write_text(
        "\n".join(feature_columns),
        encoding="utf-8"
    )


# ============================================================
# 3. GLAVNI PROGRAM
# ============================================================

def main() -> None:
    print("=" * 80)
    print("01_extract_features.py")
    print("Ekstrakcija geometrijskih značajki iz ModelNet10 .off modela")
    print("=" * 80)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Ne postoji metadata CSV: {METADATA_PATH}")

    metadata_df = pd.read_csv(METADATA_PATH)
    validate_metadata_columns(metadata_df)

    print(f"Radni direktorij: {BASE_DIR}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Broj redaka u metadata CSV-u: {len(metadata_df)}")
    print(f"Izlazni direktorij: {FEATURES_DIR}")
    print()

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []
    repair_logs: List[Dict[str, object]] = []

    for idx, meta_row in metadata_df.iterrows():
        object_id = meta_row["object_id"]
        class_label = str(meta_row["class"])
        split = str(meta_row["split"]).lower()
        object_path_metadata = meta_row["object_path"]

        try:
            model_path = resolve_object_path(object_path_metadata)

            mesh = load_mesh(model_path)
            mesh_repaired, repair_info = repair_mesh(mesh)

            features = compute_features(mesh_repaired)

            out_row: Dict[str, object] = {
                "object_id": object_id,
                "class_label": class_label,
                "split": split,
                "object_path": object_path_metadata,
                "resolved_path": str(model_path),
            }

            out_row.update(features)
            rows.append(out_row)

            repair_log_row: Dict[str, object] = {
                "object_id": object_id,
                "class_label": class_label,
                "split": split,
                "object_path": object_path_metadata,
                "resolved_path": str(model_path),
            }

            repair_log_row.update(repair_info)
            repair_logs.append(repair_log_row)

        except Exception as e:
            errors.append({
                "object_id": object_id,
                "class_label": class_label,
                "split": split,
                "object_path": object_path_metadata,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

        n_done = idx + 1

        if n_done % 100 == 0 or n_done == len(metadata_df):
            print(
                f"Obrađeno: {n_done}/{len(metadata_df)} | "
                f"uspješno: {len(rows)} | "
                f"greške: {len(errors)}"
            )

    features_df = pd.DataFrame(rows)
    errors_df = pd.DataFrame(errors)
    repair_log_df = pd.DataFrame(repair_logs)

    if len(features_df) == 0:
        if len(errors_df) > 0:
            errors_df.to_csv(ERRORS_CSV, index=False, encoding="utf-8-sig")
        raise RuntimeError("Nijedan model nije uspješno obrađen.")

    ordered_columns = [
        "object_id",
        "class_label",
        "split",
        "object_path",
        "resolved_path",
        "volume",
        "surface_area",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "ratio_xy",
        "ratio_xz",
        "ratio_yz",
        "compactness",
        "mesh_density",
    ]

    features_df = features_df[ordered_columns]

    features_df.to_csv(
        FEATURES_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    if len(errors_df) > 0:
        errors_df.to_csv(
            ERRORS_CSV,
            index=False,
            encoding="utf-8-sig"
        )

    if len(repair_log_df) > 0:
        repair_log_df.to_csv(
            REPAIR_LOG_CSV,
            index=False,
            encoding="utf-8-sig"
        )

    write_feature_columns_file()

    # Osnovna statistika za report
    split_counts = features_df["split"].value_counts().to_dict()
    class_counts = features_df["class_label"].value_counts().to_dict()

    n_train = int((features_df["split"] == "train").sum())
    n_test = int((features_df["split"] == "test").sum())

    if len(repair_log_df) > 0:
        before_watertight = int(repair_log_df["before_is_watertight"].sum())
        after_watertight = int(repair_log_df["after_is_watertight"].sum())

        before_volume = int(repair_log_df["before_is_volume"].sum())
        after_volume = int(repair_log_df["after_is_volume"].sum())
    else:
        before_watertight = 0
        after_watertight = 0
        before_volume = 0
        after_volume = 0

    report = f"""
01_extract_features.py report

Ulaz:
{METADATA_PATH}

ModelNet direktorij:
{MODELNET_DIR}

Broj redaka u metadata CSV-u:
{len(metadata_df)}

Broj uspješno obrađenih modela:
{len(features_df)}

Broj grešaka:
{len(errors_df)}

Broj train modela:
{n_train}

Broj test modela:
{n_test}

Split counts:
{split_counts}

Class counts:
{class_counts}

Zatvorenost mreže:
Watertight prije popravka: {before_watertight}
Watertight nakon popravka: {after_watertight}

Is volume:
Is volume prije popravka: {before_volume}
Is volume nakon popravka: {after_volume}

Korištene značajke:
volume
surface_area
bbox_x
bbox_y
bbox_z
ratio_xy
ratio_xz
ratio_yz
compactness
mesh_density

Formula za compactness:
Compactness 1 = V / (sqrt(pi) * A^(3/2))

Formula za mesh_density:
mesh_density = n_faces / surface_area

Izlazi:
{FEATURES_CSV}
{FEATURE_COLUMNS_TXT}
{REPAIR_LOG_CSV}
{ERRORS_CSV if len(errors_df) > 0 else "Nema error CSV-a jer nije bilo grešaka."}
""".strip()

    REPORT_TXT.write_text(report, encoding="utf-8")

    print()
    print("=" * 80)
    print("Gotovo.")
    print("=" * 80)
    print(f"Features CSV: {FEATURES_CSV}")
    print(f"Feature columns: {FEATURE_COLUMNS_TXT}")
    print(f"Mesh repair log: {REPAIR_LOG_CSV}")
    print(f"Report: {REPORT_TXT}")

    if len(errors_df) > 0:
        print(f"Errors CSV: {ERRORS_CSV}")

    print()
    print("Prvih nekoliko redaka features CSV-a:")
    print(features_df.head())

    print()
    print("Broj modela po splitu:")
    print(features_df["split"].value_counts())

    print()
    print("Broj modela po klasi:")
    print(features_df["class_label"].value_counts())


if __name__ == "__main__":
    main()