import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# plt.switch_backend('agg')

count = 0


def get_bounds(data, factor):
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


class StrokesPath(Path):

    def __init__(self, data, factor=.2, *args, **kwargs):

        vertices = np.cumsum(data[::, :-1], axis=0) / factor
        codes = np.roll(self.to_code(data[::, -1].astype(int)),
                        shift=1)

        super(StrokesPath, self).__init__(vertices, codes, *args, **kwargs)

    @staticmethod
    def to_code(cmd):
        # if cmd == 0, the code is LINETO
        # if cmd == 1, the code is MOVETO (which is LINETO - 1)
        return Path.LINETO - cmd


fig, ax = plt.subplots(figsize=(3, 3))


def trans_lines2png(stroke, out_dir):
    global ax, plt
    strokes = StrokesPath(stroke)

    patch = patches.PathPatch(strokes, facecolor='none')
    ax.add_patch(patch)

    x_min, x_max, y_min, y_max = get_bounds(data=stroke, factor=.2)

    ax.set_xlim(x_min - 5, x_max + 5)
    ax.set_ylim(y_max + 5, y_min - 5)

    ax.axis('off')

    global count
    plt.savefig('{0}/{1}.png'.format(out_dir, count), format='png')
    count += 1
    plt.cla()


def trans_dataset2png(infile, class_name):
    dataset = np.load(infile, encoding='latin1')
    sets = ['train', 'valid', 'test']
    for phase in sets:
        if phase == 'train':
            # set = np.random.choice(dataset[phase], int(len(dataset[phase]) / 70))
            set = np.random.choice(dataset[phase], 1000)
        else:
            # set = np.random.choice(dataset[phase], int(len(dataset[phase]) / 10))
            set = np.random.choice(dataset[phase], 250)
        for stroke in set:
            out_dir = 'png_data/{0}/{1}'.format(phase, class_name)
            if not os.path.exists(out_dir):
                os.system('mkdir -p {0}'.format(out_dir))
            trans_lines2png(stroke, out_dir)


if __name__ == '__main__':
    class_names = ['bee', 'bird', 'butterfly', 'rabbit', 'flower', 'grass']
    for class_name in class_names:
        filename = 'npz_data/{0}.npz'.format(class_name)
        trans_dataset2png(filename, class_name)
