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
    plt.plot(decentiles, decentiles)
    plt.plot(decentiles, betweennes_percentiles)
    plt.legend(loc=0)
    plt.xlabel('% of Nodes')
    plt.ylabel('% of Betweenness')
    plt.title("Timestamp: " + str(timestamp), fontsize=12)
    Path(str(timestamp) + "/plots/gini").mkdir(parents=True, exist_ok=True)
    filePath = cwd + "/" + str(timestamp) + '/plots/gini/gini_curve.png'
    plt.savefig(filePath, bbox_inches='tight', dpi=400)
    plt.show()


def plot_ginis_bar(timestamps, ginis):
    df = pd.DataFrame({'Timestamps': timestamps, 'Gini': ginis})
    ax = df.plot.bar(x='Timestamps', y='Gini', rot=20, ylim=(0, 1), width=0.9)
    ax.set_xlabel('Timestamps')
    ax.set_ylabel('Gini Coefficient')
    ax.get_legend().remove()
    plt.title('')
    Path("plots/gini").mkdir(parents=True, exist_ok=True)
    filePath = cwd + '/plots/gini/ginis.png'
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
plot_gini_curve(decentiles, betweenness_percentiles)

# Calc all Gini Coefficient from all timestamps and plot as bar chart
ginis = list()
for timestamp in timestamps:
    filepath = cwd + '/' + str(timestamp) + '/' + str(baseAmount[0])
    df = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount[0]) + '/' + filenames[3])
    ginis.append(gini(df['betweenness']))

plot_ginis_bar(timestamps, ginis)

