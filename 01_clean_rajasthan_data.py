import laspy
import numpy as np
import os

def clean_point_cloud():
    print("Initializing Noise Removal process for Rajasthan Point Cloud...")
    
    # We will use the smaller file first to test the cleaning process
    input_las = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH.las"
    output_las = r"Rajasthan_Point_Cloud\67169_5NKR_CHAKHIRASINGH_cleaned.las"
    
    if not os.path.exists(input_las):
        print(f"Error: Could not find {input_las}")
        return

    print("Loading point cloud into memory (this may take a moment)...")
    las = laspy.read(input_las)
    
    total_points = len(las.points)
    print(f"Original point count: {total_points:,}")
    
    # --- NOISE REMOVAL FILTERING ---
    # Common noise in Drone LiDAR includes extreme high flying points or deep subsurface reflections.
    # We will remove points beyond 3 standard deviations (Z-axis).
    
    z_points = las.z
    z_mean = np.mean(z_points)
    z_std = np.std(z_points)
    
    print(f"Elevation Statistics -> Mean: {z_mean:.2f}, Std Dev: {z_std:.2f}")
    
    # Keep points within 3 std devs (removes ~0.3% of the most extreme outlier points)
    lower_bound = z_mean - (3 * z_std)
    upper_bound = z_mean + (3 * z_std)
    
    print(f"Filtering out noise points below {lower_bound:.2f} and above {upper_bound:.2f}...")
    
    valid_mask = (las.z >= lower_bound) & (las.z <= upper_bound)
    cleaned_points = las.points[valid_mask]
    
    removed_count = total_points - len(cleaned_points)
    print(f"Removed {removed_count:,} noise points ({(removed_count/total_points)*100:.2f}%)")
    
    # --- SAVE CLEANED DATA ---
    print(f"Saving cleaned point cloud to: {output_las}")
    # Create a new las file object with the same header format
    clean_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    clean_las.points = cleaned_points
    clean_las.header.scales = las.header.scales
    clean_las.header.offsets = las.header.offsets
    
    clean_las.write(output_las)
    print("Done! Data is now clean and ready for the next step of your process.")

if __name__ == "__main__":
    clean_point_cloud()
