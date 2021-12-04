import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def calc_betweenness_centrality(G):
    return nx.betweenness_centrality(G, weight="weight", normalized=False)


def create_graph_from_graphml(filePath, baseAmount):
    g = nx.read_graphml(filePath)
    G = nx.DiGraph()

    weights = list()
    for edge in g.edges(data=True):
        fee = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
        if fee != 0:
            weights.append(fee)

    min_weight = min(weights)
    for edge in g.edges(data=True):

        weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
        if weight == 0:
            weight = min_weight / len(G.nodes)
        G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

    return G


def init_process(timestamp, baseAmount, filePath):
    G = create_graph_from_graphml(filePath, baseAmount)

    Path(str(timestamp) + "/" + str(baseAmount)).mkdir(parents=True, exist_ok=True)
    cwd = str(Path().resolve().parent)

    try:
        betweenness = calc_betweenness_centrality(G)
        betweenness_sorted = dict(sorted(betweenness.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(betweenness_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/betweenness_centrality.csv',
                  index=True, index_label=['node_id', 'betweenness'])
    except Exception as e:
        print(e)
        pass


def split_betweenness_into_ranges(df):
    data = df['betweenness'].values
    data.sort()

    removed, toHundred, toTenThousand, toMillion, toMax, toLog = list(), list(), list(), list(), list(), list()
    [removed.append(x) for x in data if x < 1]
    [toHundred.append(x) for x in data if 1 <= x <= 100]
    [toTenThousand.append(x) for x in data if 101 <= x <= 10000]
    [toMillion.append(x) for x in data if 10001 <= x <= 1000000]
    [toMax.append(x) for x in data if 1000001 <= x]
    [toLog.append(x) for x in data if 1 <= x]

    ranges = [toHundred, toTenThousand, toMillion, toMax, toLog]
    titles = ['1 to 100', '101 to 10000', '10001 to 1000000', '1000001 to 8500000', '1 to 8500000']

    return ranges, titles


def plot_bt_cdf(df, df2, df3, timestamp):
    ranges, titles = split_betweenness_into_ranges(df)
    ranges2, titles = split_betweenness_into_ranges(df2)
    ranges3, titles = split_betweenness_into_ranges(df3)

    Path(str(timestamp) + "/plots/cdf").mkdir(parents=True, exist_ok=True)
    cwd = str(Path().resolve().parent)

    for r, r2, r3, title, index in zip(ranges, ranges2, ranges3, titles, range(len(ranges))):
        y, y2, y3 = np.zeros(len(r)), np.zeros(len(r2)), np.zeros(len(r3))

        print(index)
        for i in range(len(r)):
            y[i] = (i + 1) / len(y)
        for i in range(len(r2)):
            y2[i] = (i + 1) / len(y2)
        for i in range(len(r3)):
            y3[i] = (i + 1) / len(y3)
        plt.ylim(0, 1)

        filePath = cwd + "/" + str(timestamp) + '/plots/cdf/cdf_' + str(index) + '_grp.png'
        # If last element in list make log scale x-axis
        if index == len(ranges) - 1:
            plt.xscale('log')
        plt.plot(r, y, label="0,0001 BTC", linewidth=3)
        plt.plot(r2, y2, label="0,01 BTC", linewidth=3)
        plt.plot(r3, y3, label="0,1 BTC", linewidth=3)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc=0, fontsize=19)
        plt.xlabel('Centrality', fontsize=20)
        plt.ylabel('Share of Nodes in %', fontsize=20)
        plt.savefig(filePath, bbox_inches='tight', dpi=400)
        plt.show()


timestamps = [
    # 1554112800,
    # 1564653600,
    # 1572606000,
    # 1585735200,
    # 1596276000,
    # 1606820400,
    1609498800
]

baseAmounts = [10000000, 1000000000, 10000000000]

for timestamp in timestamps:
    for baseAmount in baseAmounts:
        pass
        filePath = '../../graphs/' + str(timestamp) + '_lngraph.graphml'
        init_process(timestamp, baseAmount, filePath)

    cwd = str(Path().resolve().parent)
    filepath = cwd + '/' + str(timestamp) + '/' + str(baseAmounts[0])
    filenames = next(os.walk(filepath), (None, None, []))[2]
    df_plot = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmounts[0]) + '/' + filenames[0])
    df_plot_2 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmounts[1]) + '/' + filenames[0])
    df_plot_3 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmounts[2]) + '/' + filenames[0])
    plot_bt_cdf(df_plot, df_plot_2, df_plot_3, timestamp)


