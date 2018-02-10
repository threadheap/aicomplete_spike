from model import SequenceModel, IdentifiersModel
from flask import Flask, request

identifiers_model = IdentifiersModel("identifiers")
sequence_model = SequenceModel("sequences")

identifiers_model.load()
sequence_model.load()

app = Flask(__name__)

@app.route("/")
def index():
    sequence = request.args.get("sequence")

    words = sequence.split(" ")
    last_word = words[-1]

    if len(last_word) > 0:
        last_predicted_word = identifiers_model.predict(list(last_word))
        words = words[:-1] + ["".join(last_predicted_word)]

    return " ".join(sequence_model.predict(words))

if __name__ == "__main__":
    app.run()
