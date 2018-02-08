import glob

from encoder_decoder import TextEncoderDecoder
from model import LSTMBase
from scrapper import get_data

TRAINING_TEST_CASES = [["import", "fs", "from"]]


def train(model_name):
    data = get_data()
    lb = LSTMBase(model_name, TextEncoderDecoder(data))
    try:
        lb.train(test_cases=TRAINING_TEST_CASES)
    except KeyboardInterrupt:
        pass
    print("saving")
    lb.save()


if __name__ == "__main__":
    import sys
    print(sys.argv)
    model_name = "char"
    train(model_name)
