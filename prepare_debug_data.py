import os
import shutil
import pandas as pd

# Percorsi di origine
source_img_dir = "data/exercise_2/images"
source_csv = "data/exercise_2/labels.csv"

# Percorsi di destinazione
target_dir = "data/exercise_2_debug"
target_img_dir = os.path.join(target_dir, "images")
target_csv = os.path.join(target_dir, "labels.csv")

# Crea le cartelle di destinazione
os.makedirs(target_img_dir, exist_ok=True)

# Leggi i file immagine
all_images = sorted(os.listdir(source_img_dir))[:5000]  # solo i primi 1000
selected_ids = [int(os.path.splitext(f)[0]) for f in all_images]

# Copia immagini
for img_name in all_images:
    shutil.copy(
        os.path.join(source_img_dir, img_name),
        os.path.join(target_img_dir, img_name)
    )

# Filtra e salva il CSV
df = pd.read_csv(source_csv)
df_filtered = df[df["GalaxyID"].isin(selected_ids)]
df_filtered.to_csv(target_csv, index=False)

print(f"✔️ Copiati 1000 file e CSV in {target_dir}")
