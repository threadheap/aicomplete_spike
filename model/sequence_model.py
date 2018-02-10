from model.base_model import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

class SequenceModel(Model):
    def build_model(self):
        model = Sequential()

        model.add(LSTM(128, return_sequences=True, input_shape=(None, self.encoder_decoder.vocabulary_size), dropout=0.6))
        model.add(Dense(self.encoder_decoder.vocabulary_size, activation="relu"))
        model.add(LSTM(32, dropout=0.4))
        model.add(Dense(self.encoder_decoder.vocabulary_size))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='nadam')

        return model
