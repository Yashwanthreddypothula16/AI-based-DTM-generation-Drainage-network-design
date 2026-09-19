import laspy
import numpy as np
import os
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


def extract_refined_features(kdtree, coords, full_z_vals, k=15):
    """
    Extract geometric features for refined classification.
    """
    distances, indices = kdtree.query(coords, k=k, workers=-1)
    neighbor_z = full_z_vals[indices]

    local_min = np.min(neighbor_z, axis=1)
    local_max = np.max(neighbor_z, axis=1)
    local_var = np.var(neighbor_z, axis=1)

    height_above_min = full_z_vals[indices[:, 0]] - local_min
    z_range = local_max - local_min
    roughness = local_var

    return height_above_min, z_range, roughness


def main():
    print("🚀 Initializing Refined LiDAR Classification (Ground, Trees, Houses)...")

    input_las_path = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH_ml_classified.las"
    output_las_path = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH_refined.las"

    if not os.path.exists(input_las_path):
        print(f"❌ Error: {input_las_path} not found.")
        return

    las = laspy.read(input_las_path)
    coords = np.vstack((las.x, las.y, las.z)).transpose()
    z_vals = las.z
    total_points = len(coords)

    print(f"📊 Loaded {total_points:,} points.")

    # 1. Build KDTree for spatial analysis
    print("🔍 Building spatial index...")
    cKDTree(
        coords[::10]
    )  # Use a decimated tree for speed if needed, but let's try full if memory allows
    # Actually, for 9.6M points, a full tree might be heavy. Let's use decimated tree for feature extraction of a sample if we just want stats,
    # OR process in batches.

    # Let's process in batches to refine the classification
    print("🛠️ Refining classifications in batches...")
    batch_size = 500000
    refined_classes = np.copy(las.classification)

    # Re-build tree on full coords for accurate local min
    full_tree = cKDTree(coords)

    # Optimization: Pre-calculate a global ground surface or use a larger K for local min
    for i in range(0, total_points, batch_size):
        end = min(i + batch_size, total_points)
        batch_coords = coords[i:end]

        # Using K=100 for a much more robust "local ground" estimate
        h_min, z_rng, rough = extract_refined_features(
            full_tree, batch_coords, z_vals, k=100
        )

        # Heuristic Logic (Relaxed for this terrain):
        non_ground_mask = refined_classes[i:end] == 1

        # Buildings (6): Elevated, flat-ish
        building_mask = non_ground_mask & (h_min > 1.5) & (rough < 1.5)
        refined_classes[i:end][building_mask] = 6

        # Trees (5): Elevated, rough
        tree_mask = non_ground_mask & (h_min > 1.0) & (rough >= 1.5)
        refined_classes[i:end][tree_mask] = 5

        if (i // batch_size) % 5 == 0:
            print(f"   Processed {end:,} / {total_points:,} points...")

    las.classification = refined_classes

    # 2. Estimate Number of Houses using Clustering on Class 6
    print("\n🏠 Estimating number of individual houses...")
    building_idx = np.where(refined_classes == 6)[0]
    if len(building_idx) > 0:
        # Sample points for clustering to avoid memory overflow
        # Typically houses are large groups of points
        sample_size = min(len(building_idx), 50000)
        sample_idx = np.random.choice(building_idx, sample_size, replace=False)
        cluster_coords = coords[sample_idx][:, :2]  # Only X, Y for footprint clustering

        # DBSCAN clustering: eps is distance in meters (assuming UTM), min_samples is points
        # Assuming laser data is in meters. Rajasthan data likely UTM.
        db = DBSCAN(eps=5.0, min_samples=20).fit(cluster_coords)
        n_houses = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        # Scale up the count based on sampling
        total_houses_est = int(n_houses * (len(building_idx) / sample_size))
    else:
        total_houses_est = 0

    # 3. Estimate Number of Trees using Clustering on Class 5
    print("🌳 Estimating number of individual trees...")
    tree_points_idx = np.where(refined_classes == 5)[0]
    if len(tree_points_idx) > 0:
        sample_size = min(len(tree_points_idx), 50000)
        sample_idx = np.random.choice(tree_points_idx, sample_size, replace=False)
        cluster_coords = coords[sample_idx][:, :2]

        db = DBSCAN(eps=3.0, min_samples=15).fit(cluster_coords)
        n_trees = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        total_trees_est = int(n_trees * (len(tree_points_idx) / sample_size))
    else:
        total_trees_est = 0

    print("\n✅ Refined Classification Results:")
    print(f"   - Ground Points: {np.sum(refined_classes == 2):,}")
    print(f"   - Building Points: {np.sum(refined_classes == 6):,}")
    print(f"   - Tree Points: {np.sum(refined_classes == 5):,}")
    print(f"   - Unclassified: {np.sum(refined_classes == 1):,}")
    print(f"   - Estimated Individual Houses: {total_houses_est}")
    print(f"   - Estimated Individual Trees: {total_trees_est}")

    # Save statistics for the dashboard script to pick up
    stats_path = r"Rajasthan_Point_Cloud\classification_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"ground_points:{np.sum(refined_classes == 2)}\n")
        f.write(f"building_points:{np.sum(refined_classes == 6)}\n")
        f.write(f"tree_points:{np.sum(refined_classes == 5)}\n")
        f.write(f"unclassified_points:{np.sum(refined_classes == 1)}\n")
        f.write(f"estimated_houses:{total_houses_est}\n")
        f.write(f"estimated_trees:{total_trees_est}\n")
        f.write(f"total_points:{total_points}\n")

    print(f"\n💾 Saving refined LAS to {output_las_path}...")
    las.write(output_las_path)
    print("✨ DONE!")


if __name__ == "__main__":
    main()
