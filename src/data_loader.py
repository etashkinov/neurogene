from os import walk
from PIL import Image
import figures


def get_data(path):
    data = []
    for (dir_path, dir_names, file_names) in walk(path):
        for dir_name in dir_names:
            print(dir_name)

            for (fig_path, fig_dirs, figs) in walk(dir_path + '/' + dir_name):
                for fig in figs:
                    image = Image.open(fig_path + '/' + fig).resize((28, 28), Image.ANTIALIAS)
                    data.append(figures.Figure(image, dir_name))

    return data

