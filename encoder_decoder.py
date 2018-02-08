from collections import Counter
import numpy as np

import tokenize as tk
from io import BytesIO

PADDING = "PADDING"

class EncoderDecoder():

    def __init__(self, min_count, unknown):
        self.min_count = min_count
        self.unknown = unknown
        self.questions = []
        self.answers = []
        self.ex, self.dx = None, None
        self.ey, self.dy = None, None
        self.X, self.y = self.build_data()

    def build_data(self):
        raise NotImplementedError

    def encode_x(self, x):
        return self.ex.get(x, 0)

    def encode_y(self, y):
        return self.ey.get(y, 0)

    def decode_x(self, x):
        return self.dx.get(x, self.unknown)

    def decode_y(self, y):
        return self.dy.get(y, self.unknown)

    def pad(self, tokens):
        seqlen = len(tokens)
        return [PADDING] * (self.maxlen - seqlen + 1) + tokens

    def build_coders(self, tokens):
        tokens = [item for sublist in tokens for item in sublist]
        word_to_index = {k: v for k, v in Counter(tokens).items() if v >= self.min_count}
        word_to_index = {k: i for i, (k, v) in enumerate(word_to_index.items(), 1)}
        word_to_index[self.unknown] = 0
        index_to_word = {v: k for k, v in word_to_index.items()}
        index_to_word[0] = self.unknown
        print("word_to_index", word_to_index)
        print("index_to_word", index_to_word)
        return word_to_index, index_to_word

    def build_qa_coders(self):
        self.ex, self.dx = self.build_coders(self.sequences)
        print("unique question tokens:", len(self.ex))
        self.ey, self.dy = self.build_coders(self.sequences)
        print("unique answer tokens:", len(self.ey))

    def get_xy(self):
        n = len(self.questions)
        X = np.zeros((n, self.maxlen + 1, len(self.ex)), dtype=np.bool)
        y = np.zeros((n, len(self.ey)), dtype=np.bool)
        for num_pair, (question, answer) in enumerate(zip(self.questions, self.answers)):
            for num_token, q_token in enumerate(question):
                X[num_pair, num_token, self.encode_x(q_token)] = 1
            y[num_pair, self.encode_y(answer)] = 1
        return X, y

    def encode_question(self, sequence):
        X = np.zeros((1, self.maxlen, len(self.ex)), dtype=np.bool)
        for num, x in enumerate(sequence[1:]):
            X[0, num, self.encode_x(x)] = 1
        return X


class TextEncoderDecoder(EncoderDecoder):

    def __init__(self, sequences, min_count=1,
                 unknown="UNKNOWN"):
        self.sequences = sequences
        c = super(TextEncoderDecoder, self)
        c.__init__(min_count, unknown)

    def build_data(self):
        self.questions = []
        self.answers = []
        self.maxlen = max(len(sequence) for sequence in self.sequences)
        for sequence in self.sequences:
            seqlen = len(sequence)
            for i in range(1, seqlen):
                self.questions.append(self.pad(sequence[ : i]))
                self.answers.append(sequence[i])
        self.build_qa_coders()
        print("number of QA pairs:", len(self.questions))
        return self.get_xy()

if __name__ == "__main__":
    encoder = TextEncoderDecoder([['import', 'React', 'from', 'react', ';'], ['import', 'react']])
    encoder.build_data()
