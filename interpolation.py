import random
import copy
from necessities import Version
from necessities import Model

# Class to implement Linear Interpolation using the models passed to it.
# Contains a function to generate text when called, using the given sequence. The function checks for the version of the
# passed models and applies version-specific measures accordingly.
# Contains a function to calculate the probability of a given sequence. The function checks for the version of the
# passed models and applies version-specific measures accordingly.
# Contains a function to return the max range determining the max number that can be randomly generating when generating
# text.
# Contains a function to return the laplace value of a model whose term is zero.
class LinearInterpolation:
    def __init__(self, version, vocab_size, unigram, bigram, trigram, unigram_counts, bigram_counts):
        self.version = version
        self.vocab_size = vocab_size
        self.unigram = unigram
        self.bigram = bigram
        self.trigram = trigram
        self.unigram_counts = copy.deepcopy(unigram_counts)
        self.bigram_counts = copy.deepcopy(bigram_counts)
        self.tri_w = 0.6
        self.bi_w = 0.3
        self.uni_w = 0.1

    def generateText(self, sequence):

        generated_text = ['<s>']
        generated_word = False
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

        if self.version == Version.Vanilla:

            while not end_of_text:
                generated_word = False
                accumulator = 0
                last_gen_w1w2 = list(generated_text[-2:])

                w1 = last_gen_w1w2[0]
                w2 = last_gen_w1w2[1]

                rnum = random.uniform(0, self.getMaxRange(w1, w2))

                if rnum == 0:
                    generated_word = False
                    break
                else:
                    for w3 in self.unigram:
                        interpolation = self.tri_w * self.trigram[(w1, w2)][w3]
                        interpolation += self.bi_w * self.bigram[w2][w3]
                        interpolation += self.uni_w * self.unigram[w3]

                        accumulator += interpolation

                        if accumulator >= rnum:
                            if w3 == "</s>":
                                end_of_text = True

                            generated_word = True
                            generated_text.append(w3)
                            break

                if not generated_word:
                    break

        elif self.version == Version.Laplace:

            while not end_of_text:
                generated_word = False
                accumulator = 0

                last_gen_w1w2 = list(generated_text[-2:])
                w1 = last_gen_w1w2[0]
                w2 = last_gen_w1w2[1]

                rnum = random.uniform(0, self.getMaxRange(w1, w2))

                if rnum == 0:
                    break
                else:

                    bigram_laplace = self.laplaceValue(Model.Bigram, w2)
                    trigram_laplace = self.laplaceValue(Model.Trigram, (w1, w2))

                    for w3 in self.unigram:
                        if w3 not in self.trigram[(w1, w2)]:
                            interpolation = self.tri_w * trigram_laplace
                        else:
                            interpolation = self.tri_w * self.trigram[(w1, w2)][w3]

                        if w3 not in self.bigram[w2]:
                            interpolation += self.bi_w * bigram_laplace
                        else:
                            interpolation += self.bi_w * self.bigram[w2][w3]

                        interpolation += self.uni_w * self.unigram[w3]
                        accumulator += interpolation

                        if accumulator >= rnum:
                            if w3 == "</s>":
                                end_of_text = True

                            generated_word = True
                            generated_text.append(w3)
                            break

        elif self.version == Version.Unk:
            unk_token = "<UNK>"

            for i in range(len(generated_text)):
                if self.unigram_counts[generated_text[i]] <= 1:
                    generated_text[i] = unk_token

            while not end_of_text:
                accumulator = 0
                last_gen_w1w2 = list(generated_text[-2:])
                w1 = last_gen_w1w2[0]
                w2 = last_gen_w1w2[1]

                rnum = random.uniform(0, self.getMaxRange(w1, w2))

                if rnum == 0:
                    generated_word = False
                    break
                else:
                    for w3 in self.unigram:
                        interpolation = self.tri_w * self.trigram[(w1, w2)][w3]
                        interpolation += self.bi_w * self.bigram[w2][w3]
                        interpolation += self.uni_w * self.unigram[w3]

                        accumulator += interpolation

                        if accumulator >= rnum:
                            if w3 == "</s>":
                                end_of_text = True

                            generated_word = True
                            generated_text.append(w3)
                            break

        if generated_word:
            generated_text = " ".join(generated_text)
            return generated_text
        else:
            return None

    def getMaxRange(self, w1, w2):

        max_range = 0

        if self.version == Version.Laplace:

            bigram_laplace = self.laplaceValue(Model.Bigram, w2)
            trigram_laplace = self.laplaceValue(Model.Trigram, (w1, w2))

            for w3 in self.unigram:

                if w3 not in self.trigram[(w1, w2)]:
                    interpolation = self.tri_w * trigram_laplace
                else:
                    interpolation = self.tri_w * self.trigram[(w1, w2)][w3]

                if w3 not in self.bigram[w2]:
                    interpolation += self.bi_w * bigram_laplace
                else:
                    interpolation += self.bi_w * self.bigram[w2][w3]

                interpolation += self.uni_w * self.unigram[w3]
                max_range += interpolation

        else:

            for w3 in self.unigram:
                interpolation = self.tri_w * self.trigram[(w1, w2)][w3]
                interpolation += self.bi_w * self.bigram[w2][w3]
                interpolation += self.uni_w * self.unigram[w3]

                max_range += interpolation

        return max_range

    def findProbability(self, sequence, test_set):

        if not test_set:
            words_list = ["<s>"]
            words_list += sequence.split()
            words_list.append("</s>")
        else:
            words_list = sequence.split()

        probability = 1

        if self.version == Version.Vanilla:

            for i in range(len(words_list) - 2):
                prev_term1 = words_list[i]
                prev_term2 = words_list[i + 1]
                term = words_list[i + 2]

                interpolation = self.tri_w * self.trigram[(prev_term1, prev_term2)][term]
                interpolation += self.bi_w * self.bigram[prev_term2][term]
                interpolation += self.uni_w * self.unigram[term]

                probability *= interpolation

        elif self.version == Version.Laplace:

            for i in range(len(words_list) - 2):
                prev_term1 = words_list[i]
                prev_term2 = words_list[i + 1]
                term = words_list[i + 2]

                if self.trigram[(prev_term1, prev_term2)][term] == 0 and term in self.unigram:
                    interpolation = self.tri_w * self.laplaceValue(Model.Trigram, (prev_term1, prev_term2))
                else:
                    interpolation = self.tri_w * self.trigram[(prev_term1, prev_term2)][term]

                if self.bigram[prev_term2][term] == 0 and term in self.unigram:
                    interpolation += self.bi_w * self.laplaceValue(Model.Bigram, prev_term2)
                else:
                    interpolation += self.bi_w * self.bigram[prev_term2][term]

                interpolation += self.uni_w * self.unigram[term]
                probability *= interpolation

        elif self.version == Version.Unk:

            unk_token = "<UNK>"

            for i in range(len(words_list)):
                if self.unigram_counts[words_list[i]] <= 1:
                    words_list[i] = unk_token

            for i in range(len(words_list) - 2):
                prev_term1 = words_list[i]
                prev_term2 = words_list[i + 1]
                term = words_list[i + 2]

                interpolation = self.tri_w * self.trigram[(prev_term1, prev_term2)][term]
                interpolation += self.bi_w * self.bigram[prev_term2][term]
                interpolation += self.uni_w * self.unigram[term]

                probability *= interpolation

        return probability

    def laplaceValue(self, model, term):

        if model == Model.Bigram:
            word_count = self.unigram_counts[term]
            return 1 / (word_count + self.vocab_size)

        elif model == Model.Trigram:
            w1 = term[0]
            w2 = term[1]

            tuple_count = self.bigram_counts[w1][w2]
            return 1 / (tuple_count + self.vocab_size)

