import numpy as np

SEQ_END = "SEQ_END"
UNKNOWN = "UNKNOWN"
PAD = "PAD"

class Encoder(object):
    def __init__(self, sequences):
        item_to_index_map, index_to_item_map, vocabulary_size = self.get_vocabulary(sequences)

        self.sequences = sequences
        self.item_to_index_map = item_to_index_map
        self.index_to_item_map = index_to_item_map
        self.vocabulary_size = vocabulary_size
        self.max_sequence_length = 0

    def get_vocabulary(self, sequences):
        unique_items = set([UNKNOWN, SEQ_END, PAD])

        for sequence in sequences:
            for item in sequence:
                unique_items.add(item)

        item_to_index_map = {}
        index_to_item_map = {}
        for index, item in enumerate(unique_items):
            item_to_index_map[item] = index
            index_to_item_map[index] = item

        return item_to_index_map, index_to_item_map, len(unique_items)

    def get_questions_answers_pairs(self):
        questions = []
        answers = []

        for sequence in self.sequences:
            extended_sequence = sequence + [SEQ_END]
            for index in range(len(extended_sequence) - 1):
                questions.append(extended_sequence[:index + 1])
                answers.append(extended_sequence[index + 1])

        return questions, answers, max([len(question) for question in questions])

    def get_encoded_questions_answers(self):
        questions, answers, max_sequence_length = self.get_questions_answers_pairs()

        self.max_sequence_length = max_sequence_length

        questions_count = len(questions)
        encoded_questions = np.zeros((questions_count, self.max_sequence_length + 1, self.vocabulary_size))
        encoded_answers = np.zeros((questions_count, self.vocabulary_size))

        for index, (question, answer) in enumerate(zip(questions, answers)):
            normalized_question = self.normalize_sequence(question)

            for inner_index, item in enumerate(normalized_question):
                encoded_questions[index, inner_index, self.get_item_index(item)] = 1
            encoded_answers[index, self.get_item_index(answer)] = 1

        return encoded_questions, encoded_answers

    def encode_question(self, question):
        encoded_question = np.zeros((1, self.max_sequence_length + 1, self.vocabulary_size))

        for index, item in enumerate(self.normalize_sequence(question)):
            encoded_question[0, index, self.get_item_index(item)] = 1

        return encoded_question

    def normalize_sequence(self, sequence):
        return [PAD] * (self.max_sequence_length - len(sequence)) + sequence

    def get_item_index(self, item):
        if item in self.item_to_index_map:
            return self.item_to_index_map[item]
        else:
            return self.item_to_index_map[UNKNOWN]

    def get_item_from_index(self, index):
        return self.index_to_item_map[index]

    def encode_sequence(self, sequence):
        return [self.get_item_index(item) for item in sequence]

    def decode_sequence(self, num_sequence):
        return [self.get_item_from_index(index) for index in num_sequence]

if __name__ == "__main__":
    sequences = [
        ["a", "b", "c"],
        ["b", "d", "a", "d", "e"],
        ["a"],
        ["d", "c"]
    ]

    encoder_decoder = EncoderDecoder(sequences)
    questions, answers = encoder_decoder.get_encoded_questions_answers()
    print(questions)
    print(answers)
