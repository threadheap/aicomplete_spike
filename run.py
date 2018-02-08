import os

from train import neural_complete
from train import get_model


def read_models(base_path="models/"):
    return set([x.split(".")[0] for x in os.listdir(base_path)])

models = {x: get_model(x) for x in read_models()}

def predict(model_type, sentence):
    model_name = model_type
    if model_name not in models:
        models[model_name] = get_model(model_name)
    suggestions = neural_complete(models[model_name], sentence, [0.2, 0.5, 1])
    print("result: {}".format({"data": {"results": [x.strip() for x in suggestions]}}))

if __name__ == "__main__":
    import sys
    print(sys.argv)
    predict(sys.argv[1], sys.argv[2])
