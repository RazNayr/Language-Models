from collections import defaultdict
import copy
import random

# Class to generate the UNK-Unigram Model.
# Contains a function to generate the model using a copy of the unigram counts and using the lexicon to build a model
# containing UNK tokens. After this is done, the probabilities for each term to occur is found.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class UnigramModel:
    def __init__(self, unigram, lexicon):
        self.lexicon = lexicon
        self.unigram = copy.deepcopy(unigram)
        self.model = defaultdict(lambda: 0)

    def generateModel(self):

        self.lexicon.seek(0)
        unk_token = "<UNK>"

        for line in self.lexicon:
            for term in line.split():

                if self.unigram[term] <= 1:
                    term = unk_token

                self.model[term] += 1

        total_count = sum(self.model.values())

        for word in self.model:
            self.model[word] /= total_count

        return self

    def generateText(self, sequence):

        generated_text = ['<s>']
        unk_token = "<UNK>"
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

        for i in range(len(generated_text)):
            if self.unigram[generated_text[i]] <= 1:
                generated_text[i] = unk_token

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
        unk_token = "<UNK>"

        for i in range(len(words_list)):
            if self.model[words_list[i]] <= 1:
                words_list[i] = unk_token

        for word in words_list:
            probability *= self.model[word]

        return probability

    def getModel(self):
        return self.model


# Class to generate the UNK-Bigram Model.
# Contains a function to generate the model using a copy of the unigram counts and using the lexicon to build a model
# containing UNK tokens. After this is done, the probabilities for each term to occur is found.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class BigramModel:
    def __init__(self, unigram, lexicon):
        self.lexicon = lexicon
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self.unigram = copy.deepcopy(unigram)

    def generateModel(self):
        self.lexicon.seek(0)
        unk_token = "<UNK>"

        for line in self.lexicon:
            sentence = line.split()

            for i in range(len(sentence) - 1):
                prev_term = sentence[i]
                term = sentence[i+1]

                if self.unigram[prev_term] <= 1:
                    prev_term = unk_token

                if self.unigram[term] <= 1:
                    term = unk_token

                self.model[prev_term][term] += 1

        for word in self.model:
            word_count = sum(self.model[word].values())
            for following_word in self.model[word]:
                self.model[word][following_word] /= word_count

        return self

    def generateText(self, sequence):
        generated_text = ['<s>']
        unk_token = "<UNK>"
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

        for i in range(len(generated_text)):
            if self.unigram[generated_text[i]] <= 1:
                generated_text[i] = unk_token

        while not end_of_text:
            rnum = random.random()
            accumulator = 0
            last_gen_word = generated_text[len(generated_text)-1]

            for following_word in self.model[last_gen_word].keys():
                accumulator += self.model[last_gen_word][following_word]

                if accumulator >= rnum:
                    if following_word == "</s>":
                        end_of_text = True

                    generated_text.append(following_word)
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

        unk_token = "<UNK>"
        probability = 1

        for i in range(len(words_list)):
            if self.unigram[words_list[i]] <= 1:
                words_list[i] = unk_token

        for i in range(len(words_list) - 1):
            prev_term = words_list[i]
            term = words_list[i + 1]

            probability *= self.model[prev_term][term]

        return probability

    def getModel(self):
        return self.model


# Class to generate the UNK-Trigram Model.
# Contains a function to generate the model using a copy of the unigram counts and using the lexicon to build a model
# containing UNK tokens. After this is done, the probabilities for each term to occur is found.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class TrigramModel:
    def __init__(self, unigram, lexicon):
        self.lexicon = lexicon
        self.unigram = copy.deepcopy(unigram)
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def generateModel(self):
        self.lexicon.seek(0)
        unk_token = "<UNK>"

        for line in self.lexicon:
            sentence = line.split()

            for i in range(len(sentence) - 2):
                prev_term1 = sentence[i]
                prev_term2 = sentence[i+1]
                term = sentence[i+2]

                if self.unigram[prev_term1] <= 1:
                    prev_term1 = unk_token

                if self.unigram[prev_term2] <= 1:
                    prev_term2 = unk_token

                if self.unigram[term] <= 1:
                    term = unk_token

                self.model[(prev_term1, prev_term2)][term] += 1

        for w1w2 in self.model:
            word_count = sum(self.model[w1w2].values())
            for w3 in self.model[w1w2]:
                self.model[w1w2][w3] /= word_count

        return self

    def generateText(self, sequence):

        generated_text = ['<s>']
        unk_token = "<UNK>"
        generated_word = False
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

        for i in range(len(generated_text)):
            if self.unigram[generated_text[i]] <= 1:
                generated_text[i] = unk_token

        while not end_of_text:
            generated_word = False
            rnum = random.random()
            accumulator = 0
            last_gen_w1w2 = tuple(generated_text[-2:])

            for following_word in self.model[last_gen_w1w2]:
                accumulator += self.model[last_gen_w1w2][following_word]

                if accumulator >= rnum:
                    if following_word == "</s>":
                        end_of_text = True

                    generated_word = True
                    generated_text.append(following_word)
                    break

            if not generated_word:
                break

        if generated_word:
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

        unk_token = "<UNK>"
        probability = 1

        for i in range(len(words_list)):
            if self.unigram[words_list[i]] <= 1:
                words_list[i] = unk_token

        for i in range(len(words_list) - 2):
            prev_term1 = words_list[i]
            prev_term2 = words_list[i + 1]
            term = words_list[i + 2]

            probability *= self.model[(prev_term1, prev_term2)][term]

        return probability

    def getModel(self):
        return self.model
