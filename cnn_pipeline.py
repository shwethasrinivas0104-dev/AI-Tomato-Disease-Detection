import os
import json
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Import your folder paths
from utils.paths import PROC_DIR, SIMULATED_PLANTS_DIR, MODELS_DIR, RESULTS_DIR

# --------------------------
# CONFIG
# --------------------------
DISEASE_CLASSES = [
    "healthy",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "spider_mites",
    "target_spot",
    "yellow_leaf_curl_virus",
    "mosaic_virus",
    "bacterial_spot"
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(DISEASE_CLASSES)}
INDEX_TO_CLASS = {i: name for name, i in CLASS_TO_INDEX.items()}

IMG_SIZE = (224, 224)
NUM_CLASSES = len(DISEASE_CLASSES)
MODEL_PATH = os.path.join(MODELS_DIR, "leaf_cnn_model.h5")

# --------------------------
# BUILD CNN
# --------------------------
def build_cnn():
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Rescaling(1/255.0)(inputs)

    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --------------------------
# TRAIN MODEL
# --------------------------
def train_model():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.1
    )

    train_ds = datagen.flow_from_directory(
        PROC_DIR, target_size=IMG_SIZE, batch_size=32,
        class_mode="sparse", subset="training"
    )

    val_ds = datagen.flow_from_directory(
        PROC_DIR, target_size=IMG_SIZE, batch_size=32,
        class_mode="sparse", subset="validation"
    )

    model = build_cnn()
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    model.save(MODEL_PATH)
    print("Model saved at:", MODEL_PATH)
    return model

# --------------------------
# PREDICT ONE LEAF
# --------------------------
def predict_leaf(img_path, model):
    img = load_img(img_path, target_size=IMG_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return INDEX_TO_CLASS[idx], float(probs[idx])

# --------------------------
# ANALYSE ONE PLANT
# --------------------------
def analyse_plant(json_path, model):
    with open(json_path) as f:
        plant = json.load(f)

    leaves = plant.get("leaves", [])
    total = len(leaves)
    disease_counts = Counter()
    healthy = 0

    for leaf in leaves:
        # image_path in the JSON must be a valid path accessible in Colab
        disease, conf = predict_leaf(leaf["image_path"], model)
        if disease == "healthy":
            healthy += 1
        else:
            disease_counts[disease] += 1

    health_percent = round((healthy / max(1, total)) * 100, 2)

    report = {
        "plant_id": plant.get("plant_id", "unknown"),
        "total_leaves": total,
        "healthy_leaves": healthy,
        "health_percentage": health_percent,
        "diseases": dict(disease_counts)
    }

    return report

# --------------------------
# ADD-ON: Severity + Recommendations (safe, per-plant)
# --------------------------
DISEASE_TYPE = {
    "early_blight": "fungal",
    "late_blight": "fungal",
    "leaf_mold": "fungal",
    "septoria_leaf_spot": "fungal",
    "target_spot": "fungal",
    "bacterial_spot": "bacterial",
    "spider_mites": "pest",
    "yellow_leaf_curl_virus": "viral",
    "mosaic_virus": "viral",
    "healthy": "healthy"
}

def compute_severity_for_report(report):
    total = max(1, report.get("total_leaves", 1))
    disease_counts = report.get("diseases", {})
    severity = {}
    for dname, info in disease_counts.items():
        count = info.get("leaf_count") if isinstance(info, dict) else int(info)
        frac = count / total
        pct = round(frac * 100, 2)
        if pct > 50:
            level = "severe"
        elif pct >= 20:
            level = "moderate"
        else:
            level = "mild"
        severity[dname] = {
            "leaf_count": int(count),
            "percentage": pct,
            "severity_level": level
        }
    return severity

def _type_based_message(disease, severity_level):
    dtype = DISEASE_TYPE.get(disease, "unknown")
    if dtype == "fungal":
        base = ("Likely fungal infection. Remove infected leaves, improve airflow, "
                "avoid wet foliage or overhead irrigation.")
    elif dtype == "bacterial":
        base = ("Likely bacterial disease. Remove infected leaves, sanitize tools, avoid working with wet plants.")
    elif dtype == "viral":
        base = ("Likely viral infection. Cannot be cured chemically; remove severely infected parts and control vectors.")
    elif dtype == "pest":
        base = ("Pest infestation suspected. Start with biological control or mechanical washing; monitor regularly.")
    else:
        base = ("Disease detected. Follow standard control practices and consult local experts.")
    if severity_level == "severe":
        extra = " Severe level—immediate action advised. Use chemical control only as per product label."
    elif severity_level == "moderate":
        extra = " Moderate—start control measures and monitor."
    else:
        extra = " Mild—monitor regularly and apply cultural controls."
    return base + extra

def build_time_of_day_schedule(severity_dict):
    schedule = {"morning": [], "afternoon": [], "night": []}
    for disease, info in severity_dict.items():
        entry = {
            "disease": disease,
            "severity": info["severity_level"],
            "leaf_count": info["leaf_count"],
            "percentage": info["percentage"],
            "advice": _type_based_message(disease, info["severity_level"])
        }
        if info["severity_level"] == "severe":
            schedule["morning"].append(entry)
        elif info["severity_level"] == "moderate":
            schedule["afternoon"].append(entry)
        else:
            schedule["night"].append(entry)
    return schedule

def environment_recommendation(health_percentage):
    env = {}
    if health_percentage >= 85:
        env["health_status"] = "Mostly healthy"
        env["watering_ml_per_time"] = 250
        env["watering_frequency"] = "once every 2-3 days or when soil dries"
        env["sunlight_hours"] = "6–8 hours bright sunlight"
    elif health_percentage >= 60:
        env["health_status"] = "Moderately stressed"
        env["watering_ml_per_time"] = 200
        env["watering_frequency"] = "daily morning"
        env["sunlight_hours"] = "5–6 hours sunlight; avoid harsh midday heat"
    else:
        env["health_status"] = "Severely stressed"
        env["watering_ml_per_time"] = 150
        env["watering_frequency"] = "daily morning; keep soil moist but not waterlogged"
        env["sunlight_hours"] = "4–5 hours filtered sunlight"
    env["note"] = ("Values are approximate for a single tomato plant; adjust for pot size and climate.")
    return env

def nutrient_recommendation(report, severity_dict):
    health = report.get("health_percentage", 100.0)
    fungal_sum = sum(info["leaf_count"] for d, info in severity_dict.items()
                     if DISEASE_TYPE.get(d) == "fungal")
    total = report.get("total_leaves", 1)
    fungal_frac = fungal_sum / max(1, total)

    nr = {}
    if fungal_frac > 0.4:
        nr["primary"] = "Fungal diseases dominant. Reduce nitrogen; increase potassium; add compost."
    elif health < 50:
        nr["primary"] = "Plant stressed. Give mild balanced NPK + compost; apply micronutrients if needed."
    else:
        nr["primary"] = "Plant stable. Maintain balanced NPK and organic compost."
    nr["frequency"] = "Follow product label for exact gram/ml dosage."
    nr["note"] = "System gives directional nutrient advice; exact doses must follow fertilizer label."
    return nr

def enrich_report(report):
    severity = compute_severity_for_report(report)
    schedule = build_time_of_day_schedule(severity)
    env = environment_recommendation(report.get("health_percentage", 100.0))
    nutr = nutrient_recommendation(report, severity)

    report["severity"] = severity
    report["recommendations"] = {
        "time_schedule": schedule,
        "environment": env,
        "nutrients": nutr,
        "chemical_control_policy": (
            "Chemical doses NOT provided. When chemical control is needed, follow ICAR/State Agri/Product labels."
        )
    }
    return report

# --------------------------
# RUN PIPELINE
# --------------------------
def run_pipeline(train=False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if train or not os.path.exists(MODEL_PATH):
        print("📌 Training model (train=True or no model found)...")
        model = train_model()
    else:
        print("📌 Loading existing model from:", MODEL_PATH)
        # load_model with compile=False to avoid compile-time metric building errors
        model = load_model(MODEL_PATH, compile=False)

    print("🌿 Analysing simulated plants in:", SIMULATED_PLANTS_DIR)

    # Find all .json files inside simulated_plants and its subfolders (plant001, plant002, ...)
    plant_json_paths = []
    for root, dirs, files in os.walk(SIMULATED_PLANTS_DIR):
        for filename in files:
            if filename.lower().endswith(".json"):
                full_path = os.path.join(root, filename)
                plant_json_paths.append(full_path)

    plant_json_paths.sort()

    if not plant_json_paths:
        print("⚠️ No plant JSON files found anywhere under simulated_plants.")
        return

    for plant_path in plant_json_paths:
        print(" ➜ Processing plant JSON:", plant_path)
        basic_report = analyse_plant(plant_path, model)
        final_report = enrich_report(basic_report)

        out_name = f"{final_report['plant_id']}_report.json"
        out_path = os.path.join(RESULTS_DIR, out_name)

        with open(out_path, "w") as f:
            json.dump(final_report, f, indent=2)

        print(" ✅ Saved enriched report:", out_path)

