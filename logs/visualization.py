import math
import numpy as np
import matplotlib.pyplot as plt


def plot_offset(offset):
    """
    :param offset: Numpy.array. (2*k^2, )
    :return:
    """
    k = int(math.sqrt(offset.shape[0] / 2))
    # 1. Reshape offset
    offset = np.reshape(offset, newshape=(k, k, 2))
    # 2. Original & offset kernel locations
    xs, ys = np.meshgrid(np.arange(0, k), np.arange(0, k))
    orig_loc = np.concatenate((ys[:, :, np.newaxis], xs[:, :, np.newaxis]), axis=2)
    offset_loc = orig_loc + offset
    # 3. Location offset
    tmp = np.concatenate((orig_loc[:, :, :, np.newaxis], offset_loc[:, :, :, np.newaxis]), axis=3)
    min_y = np.min(tmp[:, :, 0])
    min_x = np.min(tmp[:, :, 1])
    max_y = np.max(tmp[:, :, 0])
    max_x = np.max(tmp[:, :, 1])
    center = np.array([[[(max_y + min_y) / 2, (max_x + min_x) / 2]]])
    scale = np.array([[[max_y - min_y, max_x - min_x]]])
    orig_loc = orig_loc + scale - center
    offset_loc = offset_loc + scale - center
    orig_loc[:, :, 0] = 2 * scale[0, 0, 0] - orig_loc[:, :, 0]
    offset_loc[:, :, 0] = 2 * scale[0, 0, 0] - offset_loc[:, :, 0]
    # 4. Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 2 * scale[0, 0, 1])
    ax.set_ylim(0, 2 * scale[0, 0, 0])
    ax.grid()
    ax.set_aspect('equal')
    # (1) points
    orig_points = np.reshape(orig_loc, newshape=(orig_loc.shape[0] * orig_loc.shape[1], 2))
    offset_points = np.reshape(offset_loc, newshape=(offset_loc.shape[0] * offset_loc.shape[1], 2))
    for y, x in orig_points: plt.scatter(x, y, s=100, c='b')
    for y, x in offset_points: plt.scatter(x, y, s=100, c='r')
    # (2) Arrows
    for orig, offset in zip(orig_points, offset_points):
        ax.arrow(orig[1], orig[0], offset[1] - orig[1], offset[0] - orig[0],
                 length_includes_head=True, head_width=0.25, head_length=0.5, fc='r', ec='b')
    plt.show()
    plt.tight_layout()


def conver2dict(items):
    d = {}
    for item in items:
        t = item.split('=')
        if t.__len__() != 2:
            continue
        d[t[0]] = t[1]
    return d


def compare_logs(log_paths, kind_of_y='RCNNAcc', save_path=None, colors=None):
    def plot_log(log_path, color, kind_of_y):
        fontsize = 60
        f = open(log_path)
        y = []
        for line in f.readlines():
            if line.__len__() < 200:
                continue
            line = line.replace(',', '')
            items = conver2dict(line.split('\t'))
            y.append(float(items[kind_of_y]))

        f.close()
        if log_path.startswith('1080/'):
            # the batch num on 1080 is double of others, so halve the batch num
            y = y[1::2]
        plt.plot(y, label=log_path, color=color, linewidth=6)
        plt.legend(loc='upper left', fontsize=fontsize)
        plt.xlabel('batch num in 7 epochs', fontsize=fontsize)
        plt.ylabel(kind_of_y, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title('Comparation of ' + kind_of_y, fontsize=fontsize)
        return

    plt.figure(figsize=[25, 20])
    if not colors: colors = [None for i in log_paths]
    for log_path, color in zip(log_paths, colors):
        plot_log(log_path, color, kind_of_y)

    if not save_path:
        fname = kind_of_y
        for lname in log_paths:
            fname += '+' + lname
        fname = fname.replace('/', '@')
        plt.savefig('figures/' + fname + '.jpg')
    else:
        plt.savefig(save_path)
    return


def vis_offset_from_model(model_path, epoch):
    from lib.utils.load_model import load_param
    arg, aux = load_param(model_path, epoch)

    pass


vis_offset_from_model('params/68/0e-4', 7)
# compare_logs(['1080/0.log', '1080/1.0.log'], kind_of_y='RCNNAcc', colors=['r', 'b'])
vis_offset(np.array([0.5, 1, 0.5, 1, 1, 0.5,
                     1, 1, 1, 0.5, 1, 0.5,
                     1, 1, 0.5, 1, 1, 1]))
