import os

# 🚩 ABSOLUTE PROJECT ROOT – make sure this path is correct
PROJECT_ROOT = "/content/drive/MyDrive/TomatoLeafAI_Project"

# Main directories used everywhere
RAW_DIR = os.path.join(PROJECT_ROOT, "raw_dataset")
PROC_DIR = os.path.join(PROJECT_ROOT, "processed_images")
SIMULATED_PLANTS_DIR = os.path.join(PROJECT_ROOT, "simulated_plants")
REALWORLD_DIR = os.path.join(PROJECT_ROOT, "realworld_processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ---- Extra paths for CNN model ----
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure all important folders exist
for d in [RAW_DIR, PROC_DIR, SIMULATED_PLANTS_DIR, REALWORLD_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

