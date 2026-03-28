import laspy
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial import cKDTree
import time

def extract_features(kdtree, coords, full_z_vals, k=15):
    """
    Extract geometric features for each point using its k-nearest neighbors.
    Features:
    1. Height above local minimum
    2. Local Z-range
    3. Local Z-variance (Roughness)
    """
    # Query k-nearest neighbors (including the point itself)
    distances, indices = kdtree.query(coords, k=k, workers=-1)
    
    # Get the Z values of the neighbors
    neighbor_z = full_z_vals[indices]
    
    local_min = np.min(neighbor_z, axis=1)
    local_max = np.max(neighbor_z, axis=1)
    local_var = np.var(neighbor_z, axis=1)
    
    # Feature 1: Height above local minimum (Critical for finding ground vs roofs)
    height_above_min = full_z_vals[indices[:, 0]] - local_min
    # Feature 2: Local vertical range
    z_range = local_max - local_min
    # Feature 3: Roughness / variance
    roughness = local_var
    
    return np.column_stack((height_above_min, z_range, roughness))

def pseudo_label(height_above_min, roughness):
    """
    Generate synthetic training labels.
    Assume 'Ground' (Class 2) is very smooth (low roughness) and near the local minimum.
    Assume 'Non-Ground' (Class 1) is elevated or rough (trees/buildings).
    """
    labels = np.ones(len(height_above_min), dtype=np.uint8) # Default Class 1 (Unclassified/Non-Ground)
    
    # Ground heuristic: Not more than 30cm above local minimum, and relatively low variance
    ground_mask = (height_above_min < 0.3) & (roughness < 0.8)
    labels[ground_mask] = 2 # Class 2 = Ground
    
    return labels

def main():
    print("Initializing Random Forest AI/ML Classification Workflow...")
    
    input_las = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH_cleaned.las"
    output_las = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH_ml_classified.las"
    
    if not os.path.exists(input_las):
        print(f"Error: Could not find {input_las}")
        return

    print("Loading point cloud...")
    start = time.time()
    las = laspy.read(input_las)
    
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    z_vals = las.z
    total_points = len(coords)
    
    print(f"Loaded {total_points:,} points in {time.time()-start:.1f}s")
    
    print("Building cKDTree for spatial querying (this takes memory and time)...")
    start = time.time()
    tree = cKDTree(coords)
    print(f"KDTree built in {time.time()-start:.1f}s")
    
    # --- PHASE 1: GENERATE TRAINING DATA ---
    # We sample 25,000 points to build out the Random Forest model training dataset.
    print("PHASE 1: Extracting Spatial Features for Training Dataset...")
    train_size = min(25000, total_points)
    train_indices = np.random.choice(total_points, train_size, replace=False)
    
    train_coords = coords[train_indices]
    train_z = z_vals[train_indices]
    
    print("Extracting features using k=15 nearest neighbors...")
    X_train = extract_features(tree, train_coords, z_vals, k=15)
    
    # Create the Pseudo Labels (Teacher logic)
    y_train = pseudo_label(X_train[:, 0], X_train[:, 2])
    
    print(f"Training distribution - Ground: {np.sum(y_train==2)}, Non-Ground: {np.sum(y_train==1)}")
    
    # --- PHASE 2: TRAIN RANDOM FOREST ---
    print("\nPHASE 2: Training RandomForestClassifier model...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    
    start = time.time()
    rf_model.fit(X_train, y_train)
    print(f"Model trained in {time.time()-start:.1f}s")
    
    # --- PHASE 3: AI/ML INFERENCE (BATCHED) ---
    print("\nPHASE 3: Running AI Inference on the full 9.6M point cloud (Batched)...")
    batch_size = 500000 
    all_predictions = np.zeros(total_points, dtype=np.uint8)
    
    for i in range(0, total_points, batch_size):
        batch_end = min(i + batch_size, total_points)
        print(f"  -> Predicting points {i:,} to {batch_end:,}...")
        
        batch_coords = coords[i:batch_end]
        
        batch_X = extract_features(tree, batch_coords, z_vals, k=15)
        batch_pred = rf_model.predict(batch_X)
        
        all_predictions[i:batch_end] = batch_pred
        
    # Apply classifications
    las.classification = all_predictions
    
    print(f"Final Count - ML Selected Ground Points: {np.sum(all_predictions==2):,}")
    
    print(f"\nSaving ML classified Point Cloud to {output_las}...")
    las.write(output_las)
    print("Done! Classification completed successfully.")

if __name__ == "__main__":
    main()
