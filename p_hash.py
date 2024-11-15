import sys
import logging
import os
import scipy
import numpy as np
import cv2
from apryse_sdk import PDFDoc, PDFDraw, PDFNet
from scipy.spatial.distance import hamming, cosine, euclidean
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

def get_hash(name):
    img = cv2.imread(name)
    # resize image and convert to gray scale
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(img, dtype = np.float32)
    # calculate dct of image
    dct = cv2.dct(img)
    # to reduce hash length take only 8*8 top-left block
    # as this block has more information than the rest
    # dct_block = dct[: 8, : 8]
    dct_block = dct
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
        hash = get_hash(f"{directory}/{filename}")
        this_hash = hash_hex_to_hash_array(hash)
        this_og_hash = hash_hex_to_hash_array(original_hash)
        minDist = min(len(this_hash), len(this_og_hash))
        this_hash = this_hash[:minDist]
        this_og_hash = this_og_hash[:minDist]
        this_diff = hamming(
            this_hash,
            this_og_hash
        )
        other_diff = 1-cosine(
            this_hash,
            this_og_hash
        )
        other_diff_2 = euclidean(
            this_hash,
            this_og_hash
        )
        ave_diff = (this_diff + other_diff + other_diff_2*.01) / 3
        if (ave_diff) < best_diff:
            best_diff = ave_diff
            best_image = filename
        file_num += 1
        diffs.append((this_diff, other_diff, other_diff_2, ave_diff, filename))
    diffs.sort()
    return best_diff, best_image, diffs

def test_hash(directory, tresh=0.90):
    correct_guesses = 0
    bad_guess = []
    ave_confidence = 0
    good = False
    ave_gap = 0
    for filename in os.listdir(directory):
        original_file = f"{directory}/{filename}"
        this_diff, this_image, diffs = get_image(directory, original_file)
        if this_image == filename:
            correct_guesses += 1
            good = True
        else:
            bad_guess.append((filename, this_image, this_diff))
        ave_confidence += this_diff
        ave_gap += diffs[1] - diffs[0]
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
                        "Usage: [-b][-t][-s]\n"
                        "-b\tp_hash.py <-b>\n"
                        "-t\tp_hash.py <-t> <test_directory> <(opt) threshold>\n"
                        "-s\tp_hash.py <-s> <directory>\n"
                        "\tp_hash.py <-g> <filename> <directory>\n"
                        "\tp_hash.py <filename> <original_file>\n"
                )
                exit(0)
            case '-s':
                if len(args) == 2:
                    pdf_to_png(args[1])
                exit(0)

            case '-g':
                if len(args) == 3:
                    diff, image, diffs = get_image(args[2], args[1])
                    print(f"Best image: {image}, Difference: {diff}")
                    print(f"Top 3 differences: {diffs[:3]}")
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
