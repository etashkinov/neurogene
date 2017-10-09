#!flask/bin/python
from flask import Flask
from flask import request

import base64
import io
from matplotlib import pyplot as plt
from PIL import Image

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello! I'm a simple neural network."


@app.route('/evaluate', methods=('POST',))
def evaluate():
    data = request.get_data()

    decode = base64.b64decode(data)
    bio = io.BytesIO(decode)
    pim = Image.open(bio)

    image = pim.convert('LA').resize((28, 28), 1)

    return str(data)


@app.route('/train', methods=('POST',))
def train():
    data = request.get_data()
    category = int(request.args.get('category'))
    return str(data)


if __name__ == '__main__':
    app.run(debug=True)
