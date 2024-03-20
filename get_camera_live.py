import cv2
from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io, transform
import random
from scipy.spatial.distance import cdist
from skimage.util import img_as_float
from scipy.ndimage import label, binary_fill_holes

def apply_mask(original_image, mask):
    # Ensure the mask is boolean
    height, width, _ = original_image.shape
    white_background = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Apply the mask to the original image.
    # mask is assumed to be a 2D array of the same height and width as the original image,
    # where 1 represents the person and 0 represents the background.
    for c in range(3):  # Iterate over each color channel.
        white_background[:, :, c] = np.where(mask == 1, original_image[:, :, c], white_background[:, :, c])

    return white_background

def extract_largest_cluster_touching_bottom(mask):
    # Label the different clusters
    
    if len(mask.shape) > 2:  # Convert to grayscale if it's a colored mask
        mask = mask[:, :, 0]
    
    # Label different clusters in the mask
    labeled, _ = label(mask)
    
    # Get the cluster numbers along the bottom center edge
    height, width = labeled.shape
    bottom_center_cluster = labeled[-1, width // 2]
    
    # Check if bottom center cluster is large enough
    if np.sum(labeled == bottom_center_cluster) < (height * width * 0.01):  # less than 1% of the image size
        print("The bottom center cluster is too small, looking for a larger cluster.")
        cluster_sizes = np.bincount(labeled.flat)
        cluster_sizes[bottom_center_cluster] = 0  # Exclude the bottom center cluster from the search
        bottom_center_cluster = cluster_sizes.argmax()  # Find the new largest cluster
    
    # Create a new mask with the person as foreground (0) and everything else as background (1)
    refined_mask = np.where(labeled == bottom_center_cluster, 0, 1)

    inverted_mask = np.invert(refined_mask.astype(bool))
    filled_mask = binary_fill_holes(inverted_mask).astype(np.float32)
    filled_mask = np.invert(filled_mask.astype(bool)).astype(np.float32)


    return filled_mask

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        prev_assignments = assignments.copy()
        distances = cdist(centers, features)
        assignments = np.argmin(distances, axis=0)
        if np.allclose(assignments, prev_assignments):
            return assignments
        for i in range(k):
            idxs = np.where(assignments == i)
            centers[i,:] = np.mean(features[idxs],axis=0)
        ### END YOUR CODE

    return assignments

def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)

    ### YOUR CODE HERE
    features = img.reshape((H * W, C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE
    coords = np.dstack(np.mgrid[0:H, 0:W])
    features = np.append(color, coords, axis=2)
    features = features.reshape((H*W, C+2))
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    ### END YOUR CODE

    return features

def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=color_position_features,
        scale=0):
    """ Compute a segmentation for an image.

    First a feature vector is extracted from each pixel of an image. Next a
    clustering algorithm is applied to the set of all feature vectors. Two
    pixels are assigned to the same segment if and only if their feature
    vectors are assigned to the same cluster.

    Args:
        img - An array of shape (H, W, C) to segment.
        k - The number of segments into which the image should be split.
        clustering_fn - The method to use for clustering. The function should
            take an array of N points and an integer value k as input and
            output an array of N assignments.
        feature_fn - A function used to extract features from the image.
        scale - (OPTIONAL) parameter giving the scale to which the image
            should be in the range 0 < scale <= 1. Setting this argument to a
            smaller value will increase the speed of the clustering algorithm
            but will cause computed segments to be blockier. This setting is
            usually not necessary for kmeans clustering, but when using HAC
            clustering this parameter will probably need to be set to a value
            less than 1.
    """

    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        # Scale down the image for faster computation.
        img = transform.rescale(img, scale)

    features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        # Resize segmentation back to the image's original size
        segments = transform.resize(segments, (H, W), preserve_range=True)

        # Resizing results in non-interger values of pixels.
        # Round pixel values to the closest interger
        segments = np.rint(segments).astype(int)

    return segments


def capture_and_display():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    frames_processed = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if frames_processed < 10:
            frames_processed += 1
            continue
        processed_frame = compute_segmentation(frame, 2, kmeans_fast, color_features, 0.3)
        mask = extract_largest_cluster_touching_bottom(processed_frame)
        mask = np.invert(mask.astype(bool))
        frame_with_filter = apply_mask(frame, mask)
        print(frame_with_filter.shape)
        # new_processed_frame = frame_with_filter.astype(np.float32)
        # processed_frame = processed_frame.astype(np.float32)
        # cv2.imshow('frame', processed_frame)
        cv2.imshow("frame", frame_with_filter)
        cv2.imwrite('current_frame.jpg', frame)
        cv2.imwrite("processed_frame.jpg", processed_frame)    
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_display()
