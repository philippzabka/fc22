import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

def plot_line(df, timestamps, name, reverse=False):
    ranks = list()
    for i in range(len(df)):
        node_ranks = list(df.iloc[i])
        if reverse == True:
            n = node_ranks[2:]
            n.reverse()
            ranks.append(n)
        else:
            ranks.append(node_ranks[2:])

    # fig = plt.figure()
    # ax = fig.add_axes([0, 0, 1, 1])
    plt.ylim(0, 50)
    width = 0.08  # the width of the bars
    x = np.arange(len(timestamps))
    bar_padding = 0
    for i in range(len(ranks)):
        plt.bar(x + bar_padding, ranks[i], label="N" + str(i+1), width=width)
        bar_padding += 0.08
    plt.legend(loc=0)
    plt.xticks(ticks=x, labels=timestamps, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Timestamps', fontsize=20)
    plt.ylabel('Rank', fontsize=20)
    Path("plots/top").mkdir(parents=True, exist_ok=True)
    filePath = cwd + '/plots/top/top_10_' + name + '.png'
    plt.savefig(filePath, bbox_inches='tight', dpi=400)
    plt.show()


timestamps = [
    1554112800,
    1564653600,
    1572606000,
    1585735200,
    1596276000,
    1606820400,
    1609498800
]

timestamps_short = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
cwd = str(Path().resolve())
filepath_newest = cwd + '/top_10_newest.csv'
filepath_oldest = cwd + '/top_10_oldest.csv'

df_newest = pd.read_csv(filepath_newest)
df_oldest = pd.read_csv(filepath_oldest)

timestamps_short.reverse()
plot_line(df_newest, timestamps_short, 'newest')
plot_line(df_oldest, timestamps_short, 'oldest', reverse=True)
