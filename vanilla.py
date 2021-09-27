import random
import copy

# Class to generate the Vanilla-Unigram Model.
# Contains a function to generate the model using a copy of the pre-generated unigram counts and calculating
# probabilities for each term.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class UnigramModel:
    def __init__(self, unigram):
        self.model = copy.deepcopy(unigram)

    def generateModel(self):
        total_count = sum(self.model.values())

        for word in self.model:
            self.model[word] /= total_count

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


# Class to generate the Vanilla-Bigram Model.
# Contains a function to generate the model using a copy of the pre-generated bigram counts and calculating
# probabilities for each term.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class BigramModel:
    def __init__(self, bigram):
        self.model = copy.deepcopy(bigram)

    def generateModel(self):
        for word in self.model:
            word_count = sum(self.model[word].values())
            for following_word in self.model[word]:
                self.model[word][following_word] /= word_count

        return self

    def generateText(self, sequence):
        generated_text = ['<s>']
        generated_word = False
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

        while not end_of_text:
            generated_word = False
            rnum = random.random()
            accumulator = 0
            last_gen_word = generated_text[len(generated_text)-1]

            for following_word in self.model[last_gen_word]:
                accumulator += self.model[last_gen_word][following_word]

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

        probability = 1

        for i in range(len(words_list) - 1):
            prev_term = words_list[i]
            term = words_list[i + 1]
            probability *= self.model[prev_term][term]

        return probability

    def getModel(self):
        return self.model


# Class to generate the Vanilla-Trigram Model.
# Contains a function to generate the model using a copy of the pre-generated trigram counts and calculating
# probabilities for each term.
# Contains a function to generate text when called, using the given sequence.
# Contains a function to calculate the probability of a given sequence when called.
# Contains a function to return the model's dictionary of values to be used for interpolation
class TrigramModel:
    def __init__(self, trigram):
        self.model = copy.deepcopy(trigram)

    def generateModel(self):
        for w1w2 in self.model:
            word_count = sum(self.model[w1w2].values())
            for w3 in self.model[w1w2]:
                self.model[w1w2][w3] /= word_count

        return self

    def generateText(self, sequence):

        generated_text = ['<s>']
        generated_word = False
        end_of_text = False

        if len(sequence) != 0:
            generated_text += sequence.split()

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

        probability = 1

        if not test_set:
            words_list = ["<s>"]
            words_list += sequence.split()
            words_list.append("</s>")
        else:
            words_list = sequence.split()

        for i in range(len(words_list) - 2):
            prev_term1 = words_list[i]
            prev_term2 = words_list[i + 1]
            term = words_list[i + 2]
            probability *= self.model[(prev_term1, prev_term2)][term]

        return probability

    def getModel(self):
        return self.model
