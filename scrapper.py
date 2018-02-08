import json

def get_data():
    file = open("./dataset.json", "r")
    return json.loads(file.read())
