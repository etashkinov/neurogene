from PIL import Image
from numpy import asarray, argmax, array

LABELS = ('circle', 'line', 'arch')


def get_one_hot(label):
    result = [0] * len(LABELS)
    result[LABELS.index(label)] = 1
    return result


def get_label(one_hot):
    return LABELS[argmax(one_hot)[0]]


def __to2d__(img):
    result = img.convert('L')
    result = asarray(result)
    x = []
    for row in result:
        we = []
        x.append(we)
        for value in row:
            we.append(1 - value / 255)
    return array(x).reshape(28, 28, 1)


class Figure:
    def __init__(self, image, label) -> None:
        super().__init__()
        self.image = image.resize((28, 28), Image.ANTIALIAS)
        self.label = label
        self.encoding = __to2d__(image)
        self.one_hot = get_one_hot(label)
