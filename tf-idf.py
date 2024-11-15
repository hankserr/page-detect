from apryse_sdk import PDFDoc, PDFDraw, PDFNet, TextExtractor  # type: ignore # pylint: disable=no-name-in-module
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import logging
import random
import sys

LICENSE_KEY = (
    "Echomark Inc.:OEM:EchoMark::LM+:AMS(20250218):"
    "F8572B231FA7F4D0530352185F616F2F2292DC25164CEEC2B76C28EE7AC2B6F5C7"
)

def auto_pg_detect(file_name, original_file_name, original_page_number):
    train_set = []

    PDFNet.Initialize(LICENSE_KEY)
    doc = PDFDoc(file_name)
    for i in range(doc.GetPageCount()):
        txt = TextExtractor()
        page = doc.GetPage(i + 1)
        txt.Begin(page)
        train_set.append(txt.GetAsText())

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    query_doc = PDFDoc(original_file_name)
    if query_doc.GetPageCount() != 1:
        if original_page_number >= query_doc.GetPageCount():
            logging.error("Original page number is out of range")
            exit(1)

    query = query_doc.GetPage(original_page_number)
    txt = TextExtractor()
    txt.Begin(query)

    logging.info("Training on %s pages", len(train_set))

    trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
    testVectorizerArray = vectorizer.transform([txt.GetAsText()]).toarray()

    transformer.fit(trainVectorizerArray)
    X = transformer.transform(trainVectorizerArray).toarray()

    transformer.fit(testVectorizerArray)
    Y = transformer.transform(testVectorizerArray).toarray()

    sims = cosine_similarity(X, Y).tolist()
    for i in range(len(sims)):
        sims[i] = (sims[i], i+1)
    sims.sort(reverse=True)
    top_pgs= sims[:3]
    return top_pgs

def single_page(file_name, test_name):
    PDFNet.Initialize(LICENSE_KEY)
    file = PDFDoc(file_name)
    txt = TextExtractor()
    txt.Begin(file.GetPage(1))

    test_file = PDFDoc(test_name)
    test_txt = TextExtractor()
    test_txt.Begin(test_file.GetPage(1))

    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()

    trainVectorizerArray = vectorizer.fit_transform([txt.GetAsText()]).toarray()
    testVectorizerArray = vectorizer.transform([test_txt.GetAsText()]).toarray()

    transformer.fit(trainVectorizerArray)
    X = transformer.transform(trainVectorizerArray).toarray()

    transformer.fit(testVectorizerArray)
    Y = transformer.transform(testVectorizerArray).toarray()

    sims = cosine_similarity(X, Y).tolist()

    import editdistance
    eval = editdistance.eval(txt.GetAsText(), test_txt.GetAsText())
    return sims, eval

def start_auto_pg_detect(pdf_file, original_pdf_file, original_page_number):
    logging.info(str(f"Original file: {original_pdf_file}, Original page number: {original_page_number}, testing on {pdf_file}"))
    top_pgs = auto_pg_detect(pdf_file, original_pdf_file, original_page_number)
    logging.info(f"Top 3 pages: {top_pgs}\n")
    return top_pgs

# Each original file is named test-<test_number>.pdf
# Each marked file is named test-<test_number>-marked.pdf
# Compare the marked file to the original file, selecting a random page from the original file
def start_mass_testing(test_directory):
    import glob
    logging.info(f"Testing on all pdf files in {test_directory}")
    directory_size = len(glob.glob(f"{test_directory}/*.pdf"))
    if directory_size == 0:
        test_directory = "./test_suite/"
        directory_size = len(glob.glob(f"{test_directory}/*.pdf"))
    PDFNet.Initialize(LICENSE_KEY)
    tot_right = 0
    incorrect_guesses = []
    lowest_guess = 1
    lowest_gap_diff = 1
    lowest_gap = 1
    lowest_gap_confidence = 0
    average_guess = 0
    average_gap = 0
    iterations = 5
    for j in range(iterations):
        for i in range(1, int(directory_size / 2) + 1):
            original_pdf_file = f"{test_directory}/test-{i}.pdf"
            marked_pdf_file = f"{test_directory}/test-{i}-marked.pdf"
            original_page_number = random.randint(1, PDFDoc(original_pdf_file).GetPageCount()-1)
            top_3 = start_auto_pg_detect(marked_pdf_file, original_pdf_file, original_page_number)
            average_gap += top_3[0][0][0] - top_3[1][0][0]
            if top_3[0][0][0] - top_3[1][0][0] < lowest_gap:
                lowest_gap = top_3[0][0][0] - top_3[1][0][0]
                lowest_gap_confidence = top_3[0][0][0]
            top = top_3[0]
            this_guess = top[0][0]
            average_guess += this_guess
            if(this_guess < lowest_guess):
                lowest_guess = this_guess
                lowest_gap_diff = top_3[0][0][0] - top_3[1][0][0]
            if top[1] == original_page_number:
                tot_right += 1
            else:
                incorrect_guesses.append((original_pdf_file, marked_pdf_file, original_page_number, top))
    print(f"Total correct guesses: {tot_right}/{int(directory_size / 2) * iterations}")
    print(f"Incorrect guesses: {len(incorrect_guesses)}")
    for guess in incorrect_guesses:
        print(f"Original file: {guess[0]}, Marked file: {guess[1]}, Original page number: {guess[2]}, Top 3 pages: {guess[3]}")
    print(f"Lowest guess: {lowest_guess}, lowest_gap_diff: {lowest_gap_diff}")
    print(f"Lowest gap: {lowest_gap}, lowest gap confidence: {lowest_gap_confidence}")
    print(f"Average guess: {average_guess / (directory_size / 2 * iterations)}")
    print(f"Average gap: {average_gap / (directory_size / 2 * iterations)}")

# Iterate test directory, testing every file with files1
# Return the 3 documents that are below threshold, but closest to threshold
def find_test_page(file1, directory, threshold=0.85, gap=0.15):
    import glob
    logging.info(f"Testing on all pdf files in {directory}")
    directory_size = len(glob.glob(f"{directory}/*.pdf"))
    if directory_size == 0:
        directory = "./test_suite/pdf"
        directory_size = len(glob.glob(f"{directory}/*.pdf"))
    PDFNet.Initialize(LICENSE_KEY)
    top_3 = []
    for i in range(1, int(directory_size / 2) + 1):
        doc = f"{directory}/test-{i}.pdf"
        doc_len = PDFDoc(doc).GetPageCount()
        for j in range(1, doc_len):
            top = auto_pg_detect(file1, doc, j)
            if top[0][0][0] >= threshold and top[0][1] == j and top[0][0][0] - top[1][0][0] > gap:
                top_3.append((doc[len(directory):], top[0][0][0], j, i))
    # Sort top_3 by the second element in the tuple
    top_3.sort(key=lambda x: x[1], reverse=True)
    top_3 = top_3[:3]
    return top_3





if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Set logging level to DEBUG
        format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages
        filename='app.log',  # File to write logs to
        filemode='a'  # 'w' overwrites the file; 'a' appends to the file
    )
    args = sys.argv[1:]
    # import pdb; pdb.set_trace()

    # test = False
    # if len(args) == 2 and args[0] == '-t':
    #     test = True
    # elif len(args) != 3:
    #     print("Usage: python3 tf-idf.py <pdf_file> <original_pdf_file> <original_page_number>")
    #     print("Test usage: python3 tf-idf.py <-t> <path-to-test_directory>")
    #     exit(1)
    # if test :
    #     start_mass_testing(args[1])
    # else:
    if args[0] == '-t':
        bar = find_test_page(args[1], args[2])
        print(bar)
    elif args[0] == '-s':
        sims, eval = single_page(args[1], args[2])
        print(sims, eval)
    else:
        bar = start_auto_pg_detect(args[0], args[1], int(args[2]))
        print(bar)