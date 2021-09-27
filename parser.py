import os
import random
import bs4 as bs
import string
import re
import time

# Function to pre-process the text when writing to the lexicon file
def cleanText(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = " ".join(text.split())
    return text


# Function to go through each corpus and write the pre-processed sentences from the xml files of the corpus to the
# lexicon.
# The pre-processed sentences are appended the start and end tokens for use when generating the language models.
def processEnglish():
    lexicon = open("lexicon/lexicon.txt", "w", encoding="utf8")

    for directory in os.listdir(r'corpus'):
        print("\nCurrent directory: " + directory + "\n")
        directory_path = 'corpus/' + directory

        for file in os.listdir(directory_path):
            if file.endswith(".xml"):
                print(file)
                file_path = directory_path + "/" + file

                raw_xml = open(file_path, 'r', encoding="utf8")
                soup_xml = bs.BeautifulSoup(raw_xml, 'lxml')
                lex_sentence = ""

                for paragraph in soup_xml.find_all('p'):
                    for sentence in paragraph.find_all('s'):
                        for word in sentence.find_all('w'):
                            lex_sentence = lex_sentence + word.text + " "

                        lex_sentence = "<s> " + cleanText(lex_sentence) + " </s>\n"
                        lexicon.write(lex_sentence)
                        lex_sentence = ""
            else:
                continue

    lexicon.close()


# Function to split the newly generated lexicon to a training set and a test set.
# The split occurs randomly using an 80:20 split. This is done by generating a random float for each sentence in the lexicon.
# If the the randomly generated number falls within the 1 to 0.2 range, the sentence is written to the training set
# If the the randomly generated number falls within the 0 to 0.2 range, the sentence is written to the test set
def splitCorpus():
    training_set = open("lexicon/training.lex.txt", "w", encoding="utf8")
    test_set = open("lexicon/test.lex.txt", "w", encoding="utf8")
    lexicon = open("lexicon/lexicon.txt", "r", encoding="utf8")

    for line in lexicon:
        rnum = random.random()

        if rnum >= 0.2:
            training_set.write(line)
        else:
            test_set.write(line)

    training_set.close()
    test_set.close()
    lexicon.close()

start_time = time.time()
processEnglish()
splitCorpus()
print("\n--- %s seconds to build lexicon ---" % (time.time() - start_time))