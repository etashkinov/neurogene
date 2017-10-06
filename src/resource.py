#!flask/bin/python
from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello! I'm a simple neural network."


@app.route('/evaluate', methods=('POST',))
def evaluate():
    data = request.get_data()
    return str(data)


if __name__ == '__main__':
    app.run(debug=True)
