import numpy as np
from scipy.ndimage import label, find_objects

# Load your array here. For the sake of this example, I'm assuming it's loaded into a variable named `clustered_image`.
# clustered_image = np.load('path_to_your_np_array.npy')

def extract_largest_cluster_touching_bottom(clustered_image):
    # Label the different clusters
    labeled_array, num_features = label(clustered_image)
    
    # Find the objects and their coordinates
    slices = find_objects(labeled_array)
    
    # Determine the middle of the bottom edge
    middle_x = clustered_image.shape[1] // 2
    
    # Iterate through the slices to find the largest cluster touching the bottom near the center
    largest_cluster_size = 0
    largest_cluster_label = 0
    for i, slice_obj in enumerate(slices, start=1):
        # Check if the cluster touches the bottom near the center
        if slice_obj[1].start <= middle_x <= slice_obj[1].stop:
            if slice_obj[0].stop == clustered_image.shape[0]:
                # Calculate the cluster size
                cluster_size = np.sum(labeled_array[slice_obj] == i)
                # If it's the largest so far, remember it
                if cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    largest_cluster_label = i
    
    # Now create a mask with the largest cluster as the person (0) and the rest as the background (1)
    person_mask = np.where(labeled_array == largest_cluster_label, 0, 1)
    
    return person_mask

# Example usage
# person_mask = extract_largest_cluster_touching_bottom(clustered_image)
# You can now save or process this mask further.
