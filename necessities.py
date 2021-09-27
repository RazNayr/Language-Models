from collections import defaultdict
from enum import Enum


# Enumerator class to store the type of models able to be used
class Model(Enum):
    Unigram = 1
    Bigram = 2
    Trigram = 3
    Interpolation = 4


# Enumerator class to store the version, models can have
class Version(Enum):
    Vanilla = 1
    Laplace = 2
    Unk = 3


# Function to generate and return unigram, bigram and trigram counts to be copied when generating the version specific
# models. The number of distinct words is also calculated and returned to be used within the laplace models.
def getNecessities(lexicon):
    unigram_counts = defaultdict(lambda: 0)
    bigram_counts = defaultdict(lambda: defaultdict(lambda: 0))
    trigram_counts = defaultdict(lambda: defaultdict(lambda: 0))

    vocab_size = 0
    words_found = {}

    # -------------- UNIGRAM COUNTS --------------
    lexicon.seek(0)

    for line in lexicon:
        for term in line.split():
            unigram_counts[term] += 1

            if term not in words_found:
                words_found[term] = None
                vocab_size += 1

    # -------------- BIGRAM COUNTS --------------
    lexicon.seek(0)

    for line in lexicon:
        sentence = line.split()

        for i in range(len(sentence) - 1):
            prev_term = sentence[i]
            term = sentence[i + 1]
            bigram_counts[prev_term][term] += 1

    # -------------- TRIGRAM COUNTS --------------
    lexicon.seek(0)

    for line in lexicon:
        sentence = line.split()

        for i in range(len(sentence) - 2):
            prev_term1 = sentence[i]
            prev_term2 = sentence[i + 1]
            term = sentence[i + 2]
            trigram_counts[(prev_term1, prev_term2)][term] += 1

    return vocab_size, unigram_counts, bigram_counts, trigram_counts
