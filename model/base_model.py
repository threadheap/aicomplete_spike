from keras.models import load_model
from encoder import SEQ_END
import numpy as np
import just


def sample(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probas = np.random.multinomial(1, predictions, 1)

    return np.argmax(probas)


class Model(object):

    def __init__(self, model_name, encoder_decoder=None, base_path="./models/"):
        self.model_name = model_name
        self.h5_path = base_path + model_name + ".h5"
        self.pkl_path = base_path + model_name + ".pkl"
        self.encoder_decoder = encoder_decoder

    def build_model(self):
        raise Exception('Not implemented')

    def train(self, test_cases=None, batch_size=256, iterations=20, epochs=10):
        self.model = self.build_model()
        X, y = self.encoder_decoder.get_encoded_questions_answers()
        self.X = X
        self.y = y

        for iteraction in range(iterations):
            print("Iteration", iteraction)
            self.model.fit(X, y, batch_size=batch_size, epochs=epochs)
            self.test_output(test_cases)

    def test_output(self, test_cases=None):
        if test_cases is not None:
            for test_case in test_cases:
                for diversity in [0.2, 0.5, 1]:
                    print("".join(self.predict(test_case, diversity)))

    def predict(self, sequence, diversity=1.0):
        max_steps = self.encoder_decoder.max_sequence_length - len(sequence)

        result = [] + sequence

        for _ in range(max_steps):
            vect = self.encoder_decoder.encode_question(result)
            predictions = self.model.predict(vect, verbose=0)[0]
            new_index = sample(predictions, diversity)
            new_token = self.encoder_decoder.get_item_from_index(new_index)

            if new_token == SEQ_END:
                break;
            result.append(new_token)

        return result

    def save(self):
        just.write(self.encoder_decoder, self.pkl_path)
        self.model.save(self.h5_path)

    def load(self):
        self.model = load_model(self.h5_path)
        self.encoder_decoder = just.read(self.pkl_path)
