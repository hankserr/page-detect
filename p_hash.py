import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='p_hash.log', encoding='utf-8', level=logging.DEBUG)

# TODO: implement
# This returns the top k documents, within threshold thresh.
def start_hash(filename, original_file, k=1, thresh=0.50):
    return

# TODO: implement
# This calls start_hash, it takes in two files, selects page number
# from original file, and calls start_hash
def call_hash(filename, original_file, page_number):
    return

# TODO:
# Input: a directory of files: test-n.pdf, test-n-marked.pdf
# Output: the percentage correct, the incorrect guesses, the average confidence, the ave gap
def test_hash(directory, tresh=0.90):
    correct_guesses = 0
    bad_guess = []
    ave_confidence = 0
    ave_gap = 0

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
            case '-h':
                print(
                        "Usage: [-b][-t]\n"
                        "-b\tp_hash.py <-b>\n"
                        "-t\tp_hash.py <-t> <test_directory> <(opt) threshold>\n"
                        "\tp_hash.py <filename> <original_file> <page_number>\n"
                )
            case _ :
                if len(args) == 3:
                    call_hash(args[0], args[1], args[2])
    print("Incorrect usage. [-h] for help")
