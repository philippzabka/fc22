import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os


def gini(betweenness_list):
    betweenness_list = (list(filter(lambda a: a != 0.0, betweenness_list)))
    sorted_list = sorted(betweenness_list)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(betweenness_list) / 2.
    return (fair_area - area) / fair_area


def calc_betweenness_percent(betweenness_list):
    betweenness_list.reverse()
    decile_percents = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cumulative_betweenness_percents = list()
    total_betweenness_sum = sum(betweenness_list)
    for decile in decile_percents:
        height = int(len(betweenness_list) * decile)
        betweenness_sum = 0
        for i in range(height):
            betweenness_sum += betweenness_list[i]
        cumulative_betweenness_percents.append(betweenness_sum / total_betweenness_sum)
    return decile_percents, cumulative_betweenness_percents


def plot_gini_curve(decentiles, betweennes_percentiles):
    plt.plot(decentiles, decentiles, linewidth=3, label='Line of Equality')
    plt.plot(decentiles, betweennes_percentiles, linewidth=3, label='Lorenz Curve')
    plt.legend(loc=0)
    plt.xlabel('Share of Nodes', fontsize=20, labelpad=20)
    plt.ylabel('Share of Centrality', fontsize=20)
    # plt.title("Timestamp: " + str(timestamp), fontsize=12)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)

    # plt.legend().remove()
    plt.legend(fontsize=20)
    Path(str(timestamp) + "/plots/gini").mkdir(parents=True, exist_ok=True)
    filePath = cwd + "/" + str(timestamp) + '/plots/gini/gini_curve.png'
    plt.savefig(filePath, bbox_inches='tight', dpi=400)
    plt.show()


def plot_ginis_bar(dates, ginis):
    df = pd.DataFrame({'Dates': dates, 'Gini': [gini*100 for gini in ginis]})
    ax = df.plot.bar(x='Dates', y='Gini', rot=0, ylim=(70, 100), width=0.9,
                     color=['#ffffb2', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#b10026'])
    # Labels above bars
    x_offset = 0
    y_offset = 0
    for p in ax.patches:
        b = p.get_bbox()
        val = " {:.1f}".format((b.y1 + b.y0)) + '%'
        ax.annotate(val, ((b.x0 + b.x1) / 2 + x_offset, b.y1 + y_offset), horizontalalignment='center',
                    verticalalignment='bottom', rotation=90, fontsize=20)
    ax.set_xlabel('Timestamps', fontsize=20, labelpad=20)
    ax.set_ylabel('Gini Coefficient in %', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.get_legend().remove()
    plt.title('')
    Path("plots/gini").mkdir(parents=True, exist_ok=True)
    filePath = cwd + '/plots/gini/ginis_2.png'
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

timestamp = timestamps[0]
baseAmount = [10000000, 1000000000, 10000000000]

cwd = str(Path().resolve())
filepath = cwd + '/' + str(timestamp) + '/' + str(baseAmount[0])
filenames = next(os.walk(filepath), (None, None, []))[2]  # [] if no file
df = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[0]) + '/' + filenames[3])
df_2 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[1]) + '/' + filenames[3])
df_3 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[2]) + '/' + filenames[3])

betweenness = (list(filter(lambda a: a != 0.0, df['betweenness'])))
betweenness_2 = (list(filter(lambda a: a != 0.0, df_2['betweenness'])))
betweenness_3 = (list(filter(lambda a: a != 0.0, df_3['betweenness'])))

avg_betwenness = list()
for b1, b2, b3 in zip(betweenness, betweenness_2, betweenness_3):
    avg_betwenness.append(np.average([b1, b2, b3]))

# Calc Gini Coefficient from Lorenz Curve per timestamp
decentiles, betweenness_percentiles = calc_betweenness_percent(avg_betwenness)
print(decentiles, betweenness_percentiles)
# plot_gini_curve(decentiles, betweenness_percentiles)

# Calc all Gini Coefficient from all timestamps and plot as bar chart
ginis = list()
for timestamp in timestamps:
    filepath = cwd + '/' + str(timestamp) + '/' + str(baseAmount[0])
    df = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[0]) + '/' + filenames[3])
    ginis.append(gini(df['betweenness']))

timestamps_short = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
dates = ['01 Apr. 2019', '01 Aug. 2019', '01 Nov. 2019', '01 Apr. 2020', '01 Aug. 2020', '01 Dec. 2020', '01 Jan. 2021']
plot_ginis_bar(timestamps_short, ginis)
