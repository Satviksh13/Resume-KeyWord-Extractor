import nltk
import PyPDF2
from collections import Counter

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')   
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def extract_text_from_pdf(pdf_path):
    text = ""

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    return text


def extract_keywords(text):

    stop_words = set(stopwords.words('english'))

    words = word_tokenize(text)

    words = [
        word.lower()
        for word in words
        if word.isalpha() and word.lower() not in stop_words
    ]

    tagged_words = nltk.pos_tag(words)

    keywords = []

    for word, tag in tagged_words:
        if tag in ['NN', 'NNP', 'NNS', 'NNPS']:
            keywords.append(word)

    return Counter(keywords).most_common(20)


if __name__ == "__main__":

    resume_path = "resumes/Satvik Sharma-Resume.pdf"


    text = extract_text_from_pdf(resume_path)

    if not text.strip():
        print("No text found in PDF.")
        exit()

    keywords = extract_keywords(text)

    print("\nTop Resume Keywords:\n")

    for word, count in keywords:
        print(f"{word}: {count}")