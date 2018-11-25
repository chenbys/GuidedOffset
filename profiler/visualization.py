import math
import numpy as np
import matplotlib.pyplot as plt


def vis_offset(offset):
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


def vis_log2(log_file_path1, log_file_path2=None, kind_of_y='RCNNAcc'):
    """
    Plot a curve for a log_file.
    x-axis is 'Batch' count, y-axis is 'RCNNAcc' or 'RCNNLogLoss' or 'RCNNL1Loss'
    Batch count should not return to zero when a new epoch begin.
    :param log_file_path:
    :param kind_of_y:
    :return:
    """

    def illu_line(line, mode=1):
        """
        :param line:
        :return:
        """
        try:
            r = {}
            if mode == 1:
                date, time, epoch, _, batch, _, speed, _, train_rpnacc, rpnlogloss, rpnl1loss, rcnnacc, rcnnlogloss, rcnnl1loss = line.split()
                r.update({'Train-RPNAcc': float(str.split(train_rpnacc, '=')[-1][:-1])})
            else:
                date, time, epoch, _, batch, _, speed, _, trainapenalty, bpenalty, cpenalty, rcnnacc, rpnacc, rpnlogloss, rpnl1loss, rcnnlogloss, rcnnl1loss = line.split()
                r.update({
                    'BPenalty'      : float(bpenalty[9:-1]),
                    'CPenalty'      : float(cpenalty[9:-1]),
                    'RPNAcc'        : float(rpnacc[7:-1]),
                    'Train-APenalty': float(trainapenalty[15:-1])
                })
            r.update({
                'Epoch'      : int(epoch[6:-1]),
                'Batch'      : int(batch[1:-1]),
                'Speed'      : float(speed),
                'RPNLogLoss' : float(str.split(rpnlogloss, '=')[-1][:-1]),
                'RPNL1Loss'  : float(str.split(rpnl1loss, '=')[-1][:-1]),
                'RCNNAcc'    : float(str.split(rcnnacc, '=')[-1][:-1]),
                'RCNNLogLoss': float(str.split(rcnnlogloss, '=')[-1][:-1]),
                'RCNNL1Loss' : float(str.split(rcnnl1loss, '=')[-1][:-1])
            })
            return r
        except:
            return None

    # 1. File1
    # Data
    data1 = []
    # Read file
    f = open(log_file_path1)
    line = f.readline()
    while line:
        r = illu_line(line, mode=1)
        if r is not None:
            data1.append(r[kind_of_y])
        line = f.readline()
    f.close()
    # 2. File2
    if log_file_path2 is not None:
        data2 = []
        f = open(log_file_path2)
        line = f.readline()
        while line:
            r = illu_line(line, mode=2)
            if r is not None:
                data2.append(r[kind_of_y])
            line = f.readline()
    f.close()
    # Plot
    plt.plot(data1, color='r')
    if data2: plt.plot(data2, color='b')
    plt.title(kind_of_y + ' red:1, blue:2.')
    plt.xlabel('Batch')
    plt.ylabel(kind_of_y)
    plt.show()


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

compare_logs(['1080/0.log', '1080/1.0.log'], kind_of_y='RCNNAcc', colors=['r', 'b'])

# compare_logs(['1080/0.log', '1080/1.0.log'], kind_of_y='Train-APenalty', colors=['r', 'b'])
# compare_logs(['1080/0.log', '1080/1.0.log'], kind_of_y='BPenalty', colors=['r', 'b'])
# compare_logs(['1080/0.log', '1080/1.0.log'], kind_of_y='CPenalty', colors=['r', 'b'])
