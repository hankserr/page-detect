# Trying to match a query to a database image using ORB feature matching

import sys
import numpy as np
import cv2
import os


def map(query, tester):
    query_img = cv2.imread(query, cv2.IMREAD_GRAYSCALE)
    tester_img = cv2.imread(tester, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    query_kp, query_des = orb.detectAndCompute(query_img, None)
    tester_kp, tester_des = orb.detectAndCompute(tester_img, None)

    # Initialize matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    try:
        matches = matcher.match(query_des, tester_des)
    except cv2.error:
        return 0, float('inf')

    # Filter good matches based on distance
    good_matches = [m for m in matches if m.distance < 50]  # Adjust threshold as needed

    # Calculate a match score
    match_score = len(good_matches)  # Number of good matches
    # Optional: Sum of distances of good matches
    total_distance = sum(m.distance for m in good_matches) if good_matches else float('inf')

    # Return match score and optional total distance
    return match_score, total_distance


def call_map(query, database):
    diffs = []
    for filename in os.listdir(database):
        name = f"{database}/{filename}"
        this_score, this_distance = map(query, name)
        diffs.append((this_score, this_distance, filename))
    diffs.sort(reverse=True)
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
                print(f"Match score: {out[0]}, Total distance: {out[1]}")
                exit(0)
    out = call_map(tester, "test_suite/slides")
    print(f"Testing on {tester}")
    print(f"Best image: {out[0][2]}, Difference: {out[0][1]}")
    best_score = out[0][0]
    best_dist = out[0][1]
    for score, dist, name in out:
        print(score, dist, name)

    ans = input(f"{best_score} {best_dist} Add results? (y/n)")
    if ans == "y":
        with open("test_suite/f_map_results.txt", "a") as file:
            file.write(str(best_score))
            file.write(" ")
            file.write(str(best_dist))
            file.write(" ")
            file.write(tester)
            file.write("\n")
        print("Results added to test_suite/f_map_results.txt")