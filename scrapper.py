import re

word_regex_template = "[a-z0-9_-]+"
word_regex = re.compile(word_regex_template, re.IGNORECASE)

def get_data(model_type):
    file = open("./dataset.txt")
    content = file.read()

    if model_type == "sequences":
        return [line.split(" ") for line in content.split("\n")]
    else:
        res = []

        for line in content.split("\n"):
            for word in line.split(" "):
                if word_regex.match(word):
                    res.append(list(word))

        return res
