import numpy as np
from scipy.ndimage import label, find_objects

def extract_largest_cluster_touching_bottom(clustered_image):
    labeled_array, num_features = label(clustered_image)
    slices = find_objects(labeled_array)
    middle_x = clustered_image.shape[1] // 2
    
    # Iterate through the slices to find the largest cluster touching the bottom near the center
    largest_cluster_size = 0
    largest_cluster_label = 0
    for i, slice_obj in enumerate(slices, start=1):
        if slice_obj[1].start <= middle_x <= slice_obj[1].stop:
            if slice_obj[0].stop == clustered_image.shape[0]:
                cluster_size = np.sum(labeled_array[slice_obj] == i)
                if cluster_size > largest_cluster_size:
                    largest_cluster_size = cluster_size
                    largest_cluster_label = i
    
    person_mask = np.where(labeled_array == largest_cluster_label, 0, 1)
    
    return person_mask
