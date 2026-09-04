import os
import random
import json
from utils.paths import PROC_DIR, SIMULATED_PLANTS_DIR


# --------------------------
# FUNCTION: Create one simulated plant
# --------------------------
def simulate_one_plant(
        plant_id="plant_001",
        min_leaves=10,
        max_leaves=40,
        max_disease_types=4
    ):
    
    # Get list of all disease classes from processed_images folder
    class_names = sorted([
        d for d in os.listdir(PROC_DIR)
        if os.path.isdir(os.path.join(PROC_DIR, d))
    ])

    n_leaves = random.randint(min_leaves, max_leaves)
    chosen_classes = random.sample(
        class_names,
        k=min(max_disease_types, len(class_names))
    )

    # Where to save this simulated plant
    plant_folder = os.path.join(SIMULATED_PLANTS_DIR, plant_id)
    os.makedirs(plant_folder, exist_ok=True)

    leaves = []
    for i in range(1, n_leaves + 1):
        cls = random.choice(chosen_classes)
        class_dir = os.path.join(PROC_DIR, cls)

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        img_name = random.choice(images)
        src_path = os.path.join(class_dir, img_name)

        leaf_id = f"leaf_{i:03d}"
        dst_path = os.path.join(plant_folder, f"{leaf_id}_{cls}.jpg")

        # Copy the image to simulated plant folder
        with open(src_path, "rb") as f_in, open(dst_path, "wb") as f_out:
            f_out.write(f_in.read())

        leaves.append({
            "leaf_id": leaf_id,
            "class_name": cls,
            "image_path": dst_path
        })

    plant = {"plant_id": plant_id, "leaves": leaves}
    return plant


# --------------------------
# FUNCTION: Save metadata.json
# --------------------------
def save_plant_metadata(plant):
    plant_id = plant["plant_id"]
    folder = os.path.join(SIMULATED_PLANTS_DIR, plant_id)
    os.makedirs(folder, exist_ok=True)

    meta_path = os.path.join(folder, "metadata.json")

    with open(meta_path, "w") as f:
        json.dump(plant, f, indent=2)

    print("✅ Metadata saved:", meta_path)
