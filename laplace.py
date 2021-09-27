import random
import copy


# Class to generate the Laplace-Unigram Model.
# Contains a function to generate the model using a copy of the pre-generated unigram counts and applying the laplace
# formula on it.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class UnigramModel:
    def __init__(self, unigram, vocab_size):
        self.vocab_size = vocab_size
        self.model = copy.deepcopy(unigram)

    def generateModel(self):
        total_count = sum(self.model.values())

        for word in self.model:
            self.model[word] = (self.model[word] + 1) / (total_count + self.vocab_size)

        return self

    def generateText(self, sequence):

        generated_text = ['<s>']
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

        while not end_of_text:
            rnum = random.random()
            accumulator = 0

            for word in self.model:
                if word == '<s>':
                    continue
                else:
                    accumulator += self.model[word]
                    if accumulator >= rnum:
                        if word == "</s>":
                            end_of_text = True

                        generated_text.append(word)
                        break

        generated_text = " ".join(generated_text)
        return generated_text

    def findProbability(self, sequence, test_set):

        if not test_set:
            words_list = ["<s>"]
            words_list += sequence.split()
            words_list.append("</s>")
        else:
            words_list = sequence.split()

        probability = 1

        for word in words_list:
            probability *= self.model[word]

        return probability

    def getModel(self):
        return self.model


# Class to generate the Laplace-Bigram Model.
# Contains a function to generate the model using a copy of the pre-generated bigram and unigram counts and applying the
# laplace formula on it.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the laplace value of a specific term whose count is 0
# Contains a function to return the model's dictionary of values to be used for interpolation
class BigramModel:
    def __init__(self, unigram, bigram, vocab_size):
        self.vocab_size = vocab_size
        self.unigram = copy.deepcopy(unigram)
        self.model = copy.deepcopy(bigram)

    def generateModel(self):
        for word in self.model:
            for following_word in self.model[word]:
                count_bigram = self.model[word][following_word]
                count_unigram = self.unigram[word]
                self.model[word][following_word] = (1 + count_bigram) / (count_unigram + self.vocab_size)

        return self

    def generateText(self, sequence):
        generated_text = ['<s>']
        generated_word = False
        end_of_text = False
        words_left = 10

        if len(sequence) != 0:
            generated_text += sequence.split()

        while not end_of_text and words_left != 0:
            generated_word = False
            rnum = random.random()
            accumulator = 0
            last_gen_word = generated_text[len(generated_text)-1]
            laplace_value = self.laplaceValue(last_gen_word)

            for following_word in self.unigram:
                if following_word in self.model[last_gen_word]:
                    accumulator += self.model[last_gen_word][following_word]
                else:
                    accumulator += laplace_value

                if accumulator >= rnum:
                    if following_word == "</s>":
                        end_of_text = True

                    words_left -= 1
                    generated_word = True
                    generated_text.append(following_word)
                    break

            if not generated_word:
                break

        if generated_word:
            end_token = "</s>"

            if end_token not in generated_text:
                generated_text.append(end_token)

            generated_text = " ".join(generated_text)
            return generated_text
        else:
            return None

    def findProbability(self, sequence, test_set):

        if not test_set:
            words_list = ["<s>"]
            words_list += sequence.split()
            words_list.append("</s>")
        else:
            words_list = sequence.split()

        probability = 1

        for i in range(len(words_list) - 1):
            prev_term = words_list[i]
            term = words_list[i + 1]

            if self.model[prev_term][term] == 0 and term in self.unigram:
                probability *= self.laplaceValue(prev_term)
            else:
                probability *= self.model[prev_term][term]

        return probability

    def laplaceValue(self, word):
        word_count = self.unigram[word]
        return 1 / (word_count + self.vocab_size)

    def getModel(self):
        return self.model


# Class to generate the Laplace-Trigram Model.
# Contains a function to generate the model using a copy of the pre-generated trigram, bigram and unigram counts and
# applying the laplace formula on it.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the laplace value of a specific term whose count is 0
# Contains a function to return the model's dictionary of values to be used for interpolation
class TrigramModel:
    def __init__(self, unigram, bigram, trigram, vocab_size):
        self.vocab_size = vocab_size
        self.unigram = copy.deepcopy(unigram)
        self.bigram = copy.deepcopy(bigram)
        self.model = copy.deepcopy(trigram)

    def generateModel(self):
        for w1w2 in self.model:
            w1 = w1w2[0]
            w2 = w1w2[1]

            for w3 in self.model[w1w2]:
                self.model[w1w2][w3] = (self.model[w1w2][w3] + 1) / (self.bigram[w1][w2] + self.vocab_size)

        return self

    def generateText(self, sequence):
        generated_text = ['<s>']
        generated_word = False
        end_of_text = False
        words_left = 10

        if len(sequence) != 0:
            generated_text += sequence.split()

        while not end_of_text and words_left != 0:
            generated_word = False
            rnum = random.random()
            accumulator = 0
            last_gen_w1w2 = tuple(generated_text[-2:])
            laplace_value = self.laplaceValue(last_gen_w1w2)

            for w3 in self.unigram:
                if w3 in self.model[last_gen_w1w2]:
                    accumulator += self.model[last_gen_w1w2][w3]
                else:
                    accumulator += laplace_value

                if accumulator >= rnum:
                    if w3 == "</s>":
                        end_of_text = True

                    words_left -= 1
                    generated_word = True
                    generated_text.append(w3)
                    break

            if not generated_word:
                break

        if generated_word:
            end_token = "</s>"

            if end_token not in generated_text:
                generated_text.append(end_token)

            generated_text = " ".join(generated_text)
            return generated_text
        else:
            return None

    def findProbability(self, sequence, test_set):

        if not test_set:
            words_list = ["<s>"]
            words_list += sequence.split()
            words_list.append("</s>")
        else:
            words_list = sequence.split()

        probability = 1

        for i in range(len(words_list) - 2):
            prev_term1 = words_list[i]
            prev_term2 = words_list[i + 1]
            term = words_list[i + 2]

            if self.model[(prev_term1, prev_term2)][term] == 0 and term in self.unigram:
                probability *= self.laplaceValue((prev_term1, prev_term2))
            else:
                probability *= self.model[(prev_term1, prev_term2)][term]

        return probability

    def laplaceValue(self, w1w2):
        tuple_count = self.bigram[w1w2[0]][w1w2[1]]
        return 1 / (tuple_count + self.vocab_size)

    def getModel(self):
        return self.model
