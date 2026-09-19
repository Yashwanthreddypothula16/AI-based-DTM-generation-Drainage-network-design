import os
import sys
import pickle
import laspy
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier

# Determine workspace base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_ground_classifier.pkl")


def extract_features(kdtree, coords, full_z_vals, k=15):
    """
    Extract geometric features for each point using its k-nearest neighbors.
    Features:
    1. Height above local minimum
    2. Local Z-range
    3. Local Z-variance (Roughness)
    """
    distances, indices = kdtree.query(coords, k=k, workers=-1)
    neighbor_z = full_z_vals[indices]

    local_min = np.min(neighbor_z, axis=1)
    local_max = np.max(neighbor_z, axis=1)
    local_var = np.var(neighbor_z, axis=1)

    height_above_min = coords[:, 2] - local_min
    z_range = local_max - local_min
    roughness = local_var

    return np.column_stack((height_above_min, z_range, roughness))


def pseudo_label(height_above_min, roughness):
    """
    Generate synthetic ground/non-ground training labels.
    Class 2: Ground
    Class 1: Non-ground
    """
    labels = np.ones(len(height_above_min), dtype=np.uint8)
    ground_mask = (height_above_min < 0.3) & (roughness < 0.8)
    labels[ground_mask] = 2
    return labels


def train_save_model():
    """
    Trains the Random Forest model on Rajasthan's cleaned point cloud dataset
    and saves the persistent model to the trained_models folder.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Locate Rajasthan cleaned dataset
    raj_las_path = os.path.join(
        BASE_DIR, "Rajasthan_Point_Cloud", "67169_5NKR_CHAKHIRASINGH_cleaned.las"
    )
    if not os.path.exists(raj_las_path):
        print(
            f"Error: Could not find Rajasthan cleaned point cloud at {raj_las_path} to train model."
        )
        sys.exit(1)

    print(f"Training Random Forest on Rajasthan data: {raj_las_path}...")
    las = laspy.read(raj_las_path)

    coords = np.vstack((las.x, las.y, las.z)).transpose()
    z_vals = las.z
    total_points = len(coords)

    print(f"Loaded {total_points:,} points. Building cKDTree...")
    tree = cKDTree(coords)

    # Sample 25,000 points matching 02_ml_classifier.py
    train_size = min(25000, total_points)
    np.random.seed(42)
    train_indices = np.random.choice(total_points, train_size, replace=False)

    train_coords = coords[train_indices]

    print("Extracting features using k=15 neighbors...")
    X_train = extract_features(tree, train_coords, z_vals, k=15)
    y_train = pseudo_label(X_train[:, 0], X_train[:, 2])

    print(
        f"Training distribution - Ground: {np.sum(y_train == 2)}, Non-Ground: {np.sum(y_train == 1)}"
    )

    rf_model = RandomForestClassifier(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Save using pickle
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
    print(f"Successfully trained and saved model to {MODEL_PATH}")
    return rf_model


def load_model():
    """
    Loads the persistent pre-trained model. If the model is not found,
    it automatically triggers training using Rajasthan cleaned data.
    """
    if os.path.exists(MODEL_PATH):
        print(f"Loading pre-trained Random Forest model from {MODEL_PATH}...")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        print(
            "Pre-trained model not found. Generating model from Rajasthan baseline..."
        )
        return train_save_model()


if __name__ == "__main__":
    load_model()
