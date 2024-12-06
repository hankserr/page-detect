# Trying to match a query to a database image using HOG global feature matching

import sys
import numpy as np
import cv2
import os
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity


def extract_hog_features(image, resize_shape=(128, 128)):
    """
    Extract HOG features from an image.
    """
    # Resize image to a consistent size
    resized_image = cv2.resize(image, resize_shape)
    # Compute HOG features
    features, _ = hog(resized_image,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=True,
                      multichannel=False)
    return features


def map(query, tester):
    """
    Compute similarity between query and tester images using HOG features.
    """
    query_img = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    tester_img = cv2.imread(tester, cv2.IMREAD_GRAYSCALE)

    # Extract HOG features for both images
    query_features = extract_hog_features(query_img)
    tester_features = extract_hog_features(tester_img)

    # Compute cosine similarity
    similarity = cosine_similarity([query_features], [tester_features])[0][0]

    # Convert similarity to a score (higher is better, 1.0 is max similarity)
    match_score = similarity
    # Optionally calculate an inverted "distance" for ranking
    distance = 1 - similarity

    return match_score, distance


def call_map(query, database):
    """
    Compare the query image against all images in the database using HOG features.
    """
    diffs = []
    for filename in os.listdir(database):
        name = f"{database}/{filename}"
        this_score, this_distance = map(query, name)
        diffs.append((this_score, this_distance, filename))
    # Sort by match score (higher is better)
    diffs.sort(reverse=True, key=lambda x: x[0])
    return diffs[:25]


if __name__ == "__main__":
    args = sys.argv[1:]

    best_score = -1
    best_dist = -1
    tester = "test_suite/page4-widescreen-phone.png"

    if args:
        match args[0]:
            case '-c':
                out = map(args[1], args[2])
                print(f"Match score: {out[0]:.4f}, Total distance: {out[1]:.4f}")
                exit(0)
    out = call_map(tester, "test_suite/slides")
    print(f"Testing on {tester}")
    print(f"Best image: {out[0][2]}, Score: {out[0][0]:.4f}, Distance: {out[0][1]:.4f}")
    best_score = out[0][0]
    best_dist = out[0][1]
    for score, dist, name in out:
        print(f"{score:.4f}, {dist:.4f}, {name}")

    ans = input(f"{best_score:.4f}, {best_dist:.4f} Add results? (y/n)")
    if ans == "y":
        with open("test_suite/f_map_results.txt", "a") as file:
            file.write(f"{best_score:.4f} {best_dist:.4f} {tester}\n")
        print("Results added to test_suite/f_map_results.txt")
