import sys
import logging
import os
import scipy
import numpy as np
import cv2
from apryse_sdk import PDFDoc, PDFDraw, PDFNet
from scipy.spatial.distance import hamming, cosine, euclidean, jaccard
import time

logger = logging.getLogger(__name__)
logging.basicConfig(filename='p_hash.log', encoding='utf-8', level=logging.DEBUG)


LICENSE_KEY = (
    "Echomark Inc.:OEM:EchoMark::LM+:AMS(20250218):"
    "F8572B231FA7F4D0530352185F616F2F2292DC25164CEEC2B76C28EE7AC2B6F5C7"
)
PDFNet.Initialize(LICENSE_KEY)

def start_hash(file1, file2):
    hash1 = hash_hex_to_hash_array(get_hash(file1))
    hash2 = hash_hex_to_hash_array(get_hash(file2))
    minDist = min(len(hash1), len(hash2))
    hash1 = hash1[:minDist]
    hash2 = hash2[:minDist]
    diff = hamming(
        hash1,
        hash2
    )
    diff_2 = 1 - cosine(
        hash1,
        hash2
    )
    diff_3 = euclidean(
        hash1,
        hash2
    )
    print(f"Hamming distance: {diff}")
    print(f"Cosine distance: {diff_2}")
    print(f"Euclidean distance: {diff_3}")

    return diff

def rv_hash(file1, file2):
    hash1 = get_hash_rv(file1)
    hash1_start_length = len(hash1)
    hash2 = get_hash_rv(file2)
    hash2_start_length = len(hash2)
    size = min(len(hash1), len(hash2))
    hash1 = hash1[:size]
    hash2 = hash2[:size]
    if hash1_start_length != len(hash1): print("Hash 1 truncated")
    if hash2_start_length != len(hash2): print("Hash 2 truncated")
    diff = hamming(hash1,hash2)
    diff2 = cosine(hash1, hash2)
    diff3 = jaccard(hash1, hash2)
    return [diff, diff2, diff3]

def hash_array_to_hash_hex(hash_array):
  # convert hash array of 0 or 1 to hash string in hex
  hash_array = np.array(hash_array, dtype = np.uint8)
  hash_str = ''.join(str(i) for i in 1 * hash_array.flatten())
  return (hex(int(hash_str, 2)))

def hash_hex_to_hash_array(hash_hex):
  # convert hash string in hex to hash values of 0 or 1
  hash_str = int(hash_hex, 16)
  array_str = bin(hash_str)[2:]
  return np.array([i for i in array_str], dtype = np.float32)

def rank_images(diff):
    scores, _ = diff
    total_score = sum(scores)  # Total similarity score for normalization
    weights = [score / total_score for score in scores]  # Normalize scores
    final_score = sum(weight * score for weight, score in zip(weights, scores))
    # final_score = sum(scores) / 6
    return final_score

def split_up_hash(name, given_hash):
    img = cv2.imread(name)
    if img is None:
        print(f"Error reading {name}")
        import pdb; pdb.set_trace()
    height, width, _ = img.shape
    segment_height = height // 2
    segment_width = width // 2

    top_left = img[:segment_height, :segment_width]  # Top-left
    top_right = img[:segment_height, segment_width:]  # Top-right
    bottom_left = img[segment_height:, :segment_width]  # Bottom-left
    bottom_right = img[segment_height:, segment_width:]  # Bottom-right

    # Extract the middle section of the same size as one segment
    middle = img[segment_height//2 : segment_height//2 + segment_height,
                segment_width//2 : segment_width//2 + segment_width]

    segments = [get_hash(top_left, True), get_hash(top_right, True), get_hash(bottom_left, True),
                get_hash(bottom_right, True), get_hash(middle, True), get_hash(img, True)]

    diffs = []
    for segment in segments:
        segment_array = hash_hex_to_hash_array(segment)
        if segment_array.shape[0] != given_hash.shape[0]:
            min_diff = 1
            print(f"Bad shape: {segment_array.shape[0]} vs {given_hash.shape[0]}, {name}")
            continue
        diff = hamming(
            segment_array,
            given_hash
        )
        diffs.append(diff)
    return diffs

def get_hash_rv(name, is_split=False):
    if not is_split:
        img = cv2.imread(name)
    else:
        img = name
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute the radial variance hash (using OpenCV's ximgproc module)
    hash_value = cv2.img_hash.radialVarianceHash(img).flatten()
    # Convert the binary hash array to a hex string
    # hash_hex = hash_array_to_hash_hex(hash_value)

    return hash_value

def get_hash(name, is_split=False):
    if not is_split:
        img = cv2.imread(name)
    else:
        img = name
    # resize image and convert to gray scale
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype = np.float32)
    # calculate dct of image
    dct = cv2.dct(img)
    # to reduce hash length take only 8*8 top-left block
    # as this block has more information than the rest
    dct_block = dct[: 8, : 8]
    # dct_block = dct
    # caclulate mean of dct block excluding first term i.e, dct(0, 0)
    dct_average = (dct_block.mean() * dct_block.size - dct_block[0, 0]) / (dct_block.size - 1)
    # convert dct block to binary values based on dct_average
    dct_block[dct_block < dct_average] = 0.0
    dct_block[dct_block != 0] = 1.0
    return hash_array_to_hash_hex(dct_block.flatten())

def get_image(directory, original_file):
    original_hash = get_hash(original_file)
    best_diff = 2147483647
    best_image = ""
    file_num = 0
    diffs = []
    for filename in os.listdir(directory):
        name = f"{directory}/{filename}"
        this_og_hash = hash_hex_to_hash_array(original_hash)
        segment_diff = split_up_hash(name, this_og_hash)
        diffs.append((segment_diff, filename))
    diffs.sort(key=rank_images)

    return best_diff, best_image, diffs

def get_image_rv(directory, original_file):
    original_hash = get_hash_rv(original_file)
    best_diff = 2147483647
    best_image = ""
    file_num = 0
    diffs = []
    for filename in os.listdir(directory):
        name = f"{directory}/{filename}"
        this_hash = get_hash_rv(name)
        diff = euclidean(original_hash, this_hash)
        diffs.append((diff, filename))
    diffs.sort()

    return diffs[0][0], diffs[0][1], diffs

def test_hash(directory, tresh=0.90, is_rv = True):
    correct_guesses = 0
    bad_guess = []
    ave_confidence = 0
    good = False
    ave_gap = 0
    for filename in os.listdir(directory):
        original_file = f"{directory}/{filename}"
        if is_rv: this_diff, this_image, diffs = get_image_rv(directory, original_file)
        else: this_diff, this_image, diffs = get_image(directory, original_file)
        if this_image == filename:
            correct_guesses += 1
            good = True
        else:
            bad_guess.append((filename, this_image, this_diff))
        ave_confidence += this_diff
        ave_gap += diffs[1][0] - diffs[0][0]
        print(f"\rTesting on {filename}, good result = {good}".ljust(80), end='', flush=False)
        good = False

    print(f"\nTotal correct guesses: {correct_guesses}/{len(os.listdir(directory))}")
    print(f"Incorrect guesses: {len(bad_guess)}")
    for guess in bad_guess:
        print(f"Original file: {guess[0]}, Guessed file: {guess[1]}, Difference: {guess[2]}")
    print(f"Average confidence: {ave_confidence / len(os.listdir(directory))}")
    print(f"Average gap: {ave_gap / len(os.listdir(directory))}")

def pdf_to_png(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            doc = PDFDoc(f"{directory}/{filename}")
            image = PDFDraw()
            image.SetDPI(92)
            image.SetImageSize(1000, 1000)
            itr = doc.GetPageIterator()
            filename = filename[:-4]
            image.Export(itr.Current(), f"{directory}/{filename}.png")
            os.remove(f"{directory}/{filename}.pdf")



if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) != 0:
        match args[0]:
            case '-b':
                print("No base files chosen")
                exit(1)
            case '-t':
                if len(args) == 2:
                    test_hash(args[1])
                else:
                    test_hash(args[1], int(args[2]))
                exit(0)
            case '-h':
                print(
                        "-b\tp_hash.py <-b>\n"
                        "-t\tp_hash.py <-t> <test_directory> <(opt) threshold>\n"
                        "-s\tp_hash.py <-s> <directory>\n"
                        "\tp_hash.py <-g> <filename> <directory>\n"
                        "\tp_hash.py <-r> <filename> <filename>\n"
                        "\tp_hash.py <filename> <original_file>\n"
                )
                exit(0)
            case '-s':
                if len(args) == 2:
                    pdf_to_png(args[1])
                exit(0)

            case '-g':
                if len(args) == 3:
                    diff, image, diffs = get_image_rv(args[2], args[1])
                    print(f"Best image: {image}, Difference: {diff}")
                    print(f"Top 3 differences: {diffs[:3]}")
                exit(0)

            case '-r':
                diffs = rv_hash(args[1], args[2])
                print("Hamming, Cosine, Jaccard", diffs)
                exit(0)


            case _ :
                if len(args) == 2:
                    start = time.time()
                    result = start_hash(args[0], args[1])
                    end = time.time()
                    print(result)
                    print(f"Time taken: {end - start}")
                    exit(0)

    print("Incorrect usage. [-h] for help")
