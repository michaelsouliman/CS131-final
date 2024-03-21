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
    height, width, _ = original_image.shape
    white_background = np.ones((height, width, 3), dtype=np.uint8) * 255
    for c in range(3):  # Iterate over each color channel.
        white_background[:, :, c] = np.where(mask == 1, original_image[:, :, c], white_background[:, :, c])

    return white_background

def extract_largest_cluster_touching_bottom(mask):
    if len(mask.shape) > 2:  # Convert to grayscale if it's a colored mask
        mask = mask[:, :, 0]
    
    labeled, _ = label(mask)

    height, width = labeled.shape
    bottom_center_cluster = labeled[-1, width // 2]
    
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
    """
    NOTE: This algorithm was taken from the Second in-class project
    """
    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        prev_assignments = assignments.copy()
        distances = cdist(centers, features)
        assignments = np.argmin(distances, axis=0)
        if np.allclose(assignments, prev_assignments):
            return assignments
        for i in range(k):
            idxs = np.where(assignments == i)
            centers[i,:] = np.mean(features[idxs],axis=0)

    return assignments

def color_features(img):
    H, W, C = img.shape
    img = img_as_float(img)

    features = img.reshape((H * W, C))

    return features

def color_position_features(img):
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    coords = np.dstack(np.mgrid[0:H, 0:W])
    features = np.append(color, coords, axis=2)
    features = features.reshape((H*W, C+2))
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features


def edge_features(img, scale=1):
    """ Retrieves edge features from a given image

    Utilize Canny edge detection on the gray scaled image
    to extract edge features."""
    H, W, C = img.shape
    img = (img*255).astype(np.uint8)

    if scale == 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    edges = cv2.Canny(gray, 100, 200)  # You can adjust the threshold values as needed
    edge_features = edges.reshape(-1, 1)
    color_features = img.reshape((H*W,C))
    features = np.hstack((color_features, edge_features))
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features

def compute_segmentation(img, k,
        clustering_fn=kmeans_fast,
        feature_fn=color_position_features,
        scale=0):
    """
    NOTE: This function is built from project 2, although modifications have been made
    so that it is compatible with our edge features implementation
    """
    assert scale <= 1 and scale >= 0, \
        'Scale should be in the range between 0 and 1'

    H, W, C = img.shape

    if scale > 0:
        img = transform.rescale(img, scale)

    if feature_fn == edge_features:
        features = feature_fn(img, scale=scale)
    else:
        features = feature_fn(img)
    assignments = clustering_fn(features, k)
    segments = assignments.reshape((img.shape[:2]))

    if scale > 0:
        segments = transform.resize(segments, (H, W), preserve_range=True)
        segments = np.rint(segments).astype(int)

    return segments


def capture_and_display(feature_fn=color_features):
    """
    The function that will be used to start the video feed and perform background filtering
    by calling the segmentation algorithms.
    """
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
        processed_frame = compute_segmentation(frame, 2, kmeans_fast, feature_fn, 0.3)
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
    capture_and_display(feature_fn=edge_features)