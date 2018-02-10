import glob

from encoder import Encoder
from model import SequenceModel, IdentifiersModel
from scrapper import get_data


IDENTIFIERS_TEST_CASES = [
    ["i", "m", "p"],
    ["R", "a", "c", "t"],
    ["C", "h", "i"]
]


SEQUENCES_TEST_CASES = [
    ["import", "React", "from"],
    ["import", "type"],
    ["import", "{"]
]


def train(model_type):
    data = get_data(model_type)
    encoder = Encoder(data)
    model = None
    test_cases = None
    iterations = 20

    if model_type == "sequences":
        iterations = 100
        test_cases = SEQUENCES_TEST_CASES
        model = SequenceModel(model_type, encoder)
    else:
        test_cases = IDENTIFIERS_TEST_CASES
        model = IdentifiersModel(model_type, encoder)

    try:
        model.train(test_cases=test_cases, iterations=iterations)
    except KeyboardInterrupt:
        pass
    print("saving")
    model.save()


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1]
    train(model_type)
