import visdom,sys
import numpy as np


def show_loss(path, name, step=1):
    with open(path, "r") as f:
        data = f.read()
    data = data.split("\n")[:-1]
    x = np.linspace(1, len(data) + 1, len(data)) * step
    y = []
    for i in range(len(data)):
        y.append(float(data[i]))

    vis = visdom.Visdom(env='loss')
    vis.line(X=x, Y=y, win=name, opts={'title': name, "xlabel": "epoch", "ylabel": name})


def compare2(path_1, path_2, title="xxx", legends=["a", "b"], x="epoch", step=20):
    with open(path_1, "r") as f:
        data_1 = f.read()
    data_1 = data_1.split("\n")[:-1]

    with open(path_2, "r") as f:
        data_2 = f.read()
    data_2 = data_2.split("\n")[:-1]

    x = np.linspace(1, len(data_1) + 1, len(data_1)) * step
    y = []
    for i in range(len(data_1)):
        y.append([float(data_1[i]), float(data_2[i])])

    vis = visdom.Visdom(env='loss')
    vis.line(X=x, Y=y, win="compare",
             opts={"title": "compare " + title, "legend": legends, "xlabel": "epoch", "ylabel": title})


if __name__ == "__main__":
    show_loss(sys.argv[1], "loss")
    # compare2("precision1.txt", "precision2.txt", title="precision", step=20)

