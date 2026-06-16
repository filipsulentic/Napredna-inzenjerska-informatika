from pathlib import Path
import pandas as pd


BASE_DIR = Path(r"C:\Users\filip\Desktop\NII")
METADATA_PATH = BASE_DIR / "metadata_modelnet10.csv"
BACKUP_PATH = BASE_DIR / "metadata_modelnet10_backup.csv"


df = pd.read_csv(METADATA_PATH)

print("Stupci:")
print(df.columns.tolist())
print()

print("Jedinstvene klase:")
print(sorted(df["class"].astype(str).unique()))
print()

# Backup
df.to_csv(BACKUP_PATH, index=False, encoding="utf-8-sig")

before = df["object_path"].astype(str).copy()

# Ispravak:
# night/train/night_stand_0195.off  -> night_stand/train/night_stand_0195.off
# night/test/night_stand_0195.off   -> night_stand/test/night_stand_0195.off
df["object_path"] = (
    df["object_path"]
    .astype(str)
    .str.replace(
        r"(^|[/\\])night([/\\])(train|test)([/\\]night_stand_)",
        r"\1night_stand\2\3\4",
        regex=True
    )
)

after = df["object_path"].astype(str)

changed_mask = before != after
n_changed = int(changed_mask.sum())

df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

print("Gotovo.")
print(f"Backup: {BACKUP_PATH}")
print(f"Ispravljeno redaka: {n_changed}")
print()

if n_changed > 0:
    print("Primjeri ispravljenih redaka:")
    preview = pd.DataFrame({
        "before": before[changed_mask].head(10),
        "after": after[changed_mask].head(10),
    })
    print(preview)
else:
    print("Nije pronađen uzorak oblika:")
    print("night/train/night_stand_*.off")
    print("ili")
    print("night/test/night_stand_*.off")