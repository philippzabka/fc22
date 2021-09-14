import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# TODO Alle timestamps + cdf & histogramme von betweenness
# TODO Kleiner Report der Methodik für P1

"""
Fee = fee_base_msat +(amount * fee_proportional_millionths / 10**6)
"""

def calc_closeness_centrality(G):
    return nx.closeness_centrality(G, distance="weight")


def calc_in_degree_centrality(G):
    return nx.in_degree_centrality(G)


def calc_out_degree_centrality(G):
    return nx.out_degree_centrality(G)


def calc_degree_centrality(G):
    return nx.degree_centrality(G)


def calc_edge_betweenness(G):
    return nx.edge_betweenness_centrality(G, normalized=False)


def calc_page_rank(G):
    return nx.pagerank(G, weight="weight")


def calc_betweenness_centrality(G):
    return nx.betweenness_centrality(G, weight="weight", normalized=False)


def calc_triangles(G):
    return nx.triangles(G)


def calc_clustering_coefficient(G):
    return nx.clustering(G)


def calc_avg_clustering_coefficient(G):
    return nx.average_clustering(G)


def createGraphFromGraphML(filePath, baseAmount):
    g = nx.read_graphml(filePath)
    G = nx.DiGraph()

    seen_source = set()
    bar_dict = dict()
    bar_dict['both'] = 0
    bar_dict['fee_base'] = 0
    bar_dict['fee_prop'] = 0

    weights = list()
    for edge in g.edges(data=True):
        fee = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
        if fee != 0:
            weights.append(fee)

    min_weight = min(weights)
    for edge in g.edges(data=True):
        # Weight = Fee
        if edge[2]['source'] not in seen_source:
            seen_source.add(edge[2]['source'])
            if edge[2]['fee_base_msat'] == 0 and edge[2]['fee_proportional_millionths'] == 0:
                bar_dict['both'] += 1
            elif edge[2]['fee_base_msat'] == 0:
                bar_dict['fee_base'] += 1
            elif edge[2]['fee_proportional_millionths'] == 0:
                bar_dict['fee_prop'] += 1

        weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
        if weight == 0:
            weight = min_weight / len(G.nodes)
        G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

    return G


def calc_degrees(G, in_deg=False, out_deg=False):
    degrees = dict()
    for node in G.nodes:
        degrees[node] = dict()

        adj_nodes = G.adj[node]
        deg_one_sum, deg_two_sum, deg_three_sum, deg_greater = 0, 0, 0, 0
        for key in adj_nodes:

            if in_deg:
                deg = G.in_degree[key]
            elif out_deg:
                deg = G.out_degree[key]
            else:
                deg = G.degree[key]

            if deg == 1:
                deg_one_sum += 1
            elif deg == 2:
                deg_two_sum += 1
            elif deg == 3:
                deg_three_sum += 1
            else:
                deg_greater += 1

        if in_deg:
            degrees[node]['deg'] = G.in_degree[node]
        elif out_deg:
            degrees[node]['deg'] = G.out_degree[node]
        else:
            degrees[node]['deg'] = G.degree[node]

        degrees[node]['deg_one'] = (deg_one_sum / G.degree[node])
        degrees[node]['deg_two'] = (deg_two_sum / G.degree[node])
        degrees[node]['deg_three'] = (deg_three_sum / G.degree[node])

    return degrees


# Init all necessary calculations
def initProcess(timestamp, baseAmount, filePath, betweenness=False, clustering=False, page=False, deg_cen=False,
                in_deg_cen=False, out_deg_cen=False, closeness=False, edge_betweenness=False, deg=False, in_deg=False,
                out_deg=False):
    G = createGraphFromGraphML(filePath, baseAmount)

    Path(str(timestamp) + "/" + str(baseAmount)).mkdir(parents=True, exist_ok=True)
    cwd = str(Path().resolve())

    try:
        if deg:
            degrees = calc_degrees(G)
            degrees_sorted = dict(sorted(degrees.items(), key=lambda item: item[1]['deg'], reverse=True))
            df = pd.DataFrame.from_dict(degrees_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/degrees.csv',
                      index=True, index_label='node_id')
    except Exception as e:
        print(e)
        pass

    try:
        if in_deg:
            in_degrees = calc_degrees(G, in_deg=in_deg)
            in_degrees_sorted = dict(sorted(in_degrees.items(), key=lambda item: item[1]['deg'], reverse=True))
            df = pd.DataFrame.from_dict(in_degrees_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/in_degrees.csv',
                      index=True, index_label='node_id')
    except Exception as e:
        print(e)
        pass

    try:
        if out_deg:
            out_degrees = calc_degrees(G, out_deg=out_deg)
            out_degrees_sorted = dict(sorted(out_degrees.items(), key=lambda item: item[1]['deg'], reverse=True))
            df = pd.DataFrame.from_dict(out_degrees_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/out_degrees.csv',
                      index=True, index_label='node_id')
    except Exception as e:
        print(e)
        pass

    # Calc betweenness centrality
    try:
        if betweenness:
            betweenness = calc_betweenness_centrality(G)
            betweenness_sorted = dict(sorted(betweenness.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(betweenness_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/betweenness_centrality.csv',
                      index=True, index_label=['node_id', 'betweenness'])
    except Exception as e:
        print(e)
        pass

    # Calc clustering
    try:
        if clustering:
            clustering_coeff = calc_clustering_coefficient(G)
            clustering_coeff_sorted = dict(sorted(clustering_coeff.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(clustering_coeff_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/clustering.csv',
                      index=True, index_label=['node_id', 'clustering_coeff'])
    except Exception as e:
        print(e)
        pass

    # Calc pagerank
    try:
        if page:
            page_rank = calc_page_rank(G)
            page_sorted = dict(sorted(page_rank.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(page_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/pagerank.csv',
                      index=True, index_label=['node_id', 'pagerank'])
    except Exception as e:
        print(e)
        pass

    # Calc node degree centrality
    try:
        if deg_cen:
            degree_centrality = calc_degree_centrality(G)
            degree_centrality_sorted = dict(sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(degree_centrality_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/degree_centrality.csv',
                      index=True, index_label=['node_id', 'degree_centrality'])
    except Exception as e:
        print(e)
        pass

    # Calc node in-degree centrality
    try:
        if in_deg_cen:
            degree_in_centrality = calc_in_degree_centrality(G)
            degree_in_sorted = dict(sorted(degree_in_centrality.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(degree_in_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/in_degree_centrality.csv',
                      index=True, index_label=['node_id', 'in_degree_centrality'])
    except Exception as e:
        print(e)
        pass

    # Calc node out-degree centrality
    try:
        if out_deg_cen:
            degree_out_centrality = calc_out_degree_centrality(G)
            degree_out_sorted = dict(sorted(degree_out_centrality.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(degree_out_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/out_degree_centrality.csv',
                      index=True, index_label=['node_id', 'out_degree_centrality'])
    except Exception as e:
        print(e)
        pass

    # Calc closeness centrality
    try:
        if closeness:
            closeness = calc_closeness_centrality(G)
            closeness_sorted = dict(sorted(closeness.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(closeness_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/closeness_centrality.csv',
                      index=True, index_label=['node_id', 'closeness_centrality'])
    except Exception as e:
        print(e)
        pass

    # Calc edge betweenness centrality
    try:
        if edge_betweenness:
            edge_betweenness = calc_edge_betweenness(G)
            edge_betweenness_sorted = dict(sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True))
            df = pd.DataFrame.from_dict(edge_betweenness_sorted, orient='index')
            df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/edge_betweenness_centrality.csv',
                      index=True, index_label=['edge', 'edge_betweenness'])
    except Exception as e:
        print(e)
        pass


def splitBetweennessIntoRanges(df):
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
    ranges, titles = splitBetweennessIntoRanges(df)
    ranges2, titles = splitBetweennessIntoRanges(df2)
    ranges3, titles = splitBetweennessIntoRanges(df3)

    # Path(str(timestamp) + "/" + str(baseAmount) + "/plots/cdf").mkdir(parents=True, exist_ok=True)
    Path(str(timestamp) + "/plots/cdf").mkdir(parents=True, exist_ok=True)
    cwd = str(Path().resolve())

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

        # filePath = cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/plots/cdf/cdf_' + str(index) + 'grp.png'
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
        # plt.title("Timestamp: " + str(timestamp) + ", Range: " + title + ", Base: " + str(base_amount), fontsize=10)
        # plt.title("Timestamp: " + str(timestamp), fontsize=12)
        plt.savefig(filePath, bbox_inches='tight', dpi=400)
        plt.show()


def plot_bt_histogram(df, df2, df3,  timestamp, bins=10):
    ranges, titles = splitBetweennessIntoRanges(df)
    ranges2, titles = splitBetweennessIntoRanges(df2)
    ranges3, titles = splitBetweennessIntoRanges(df3)

    # Path(str(timestamp) + "/" + str(baseAmount) + "/plots/histogram").mkdir(parents=True, exist_ok=True)
    Path(str(timestamp) + "/plots/histogram").mkdir(parents=True, exist_ok=True)

    cwd = str(Path().resolve())
    for r, r2, r3, title, index in zip(ranges, ranges2, ranges3, titles, range(len(ranges))):
        range_list = [r, r2, r3]
        longest_list = max(range_list, key=len)

        for i in range(len(longest_list)):
            if len(longest_list) > len(r):
                r.append(None)
            if len(longest_list) > len(r2):
                r2.append(None)
            if len(longest_list) > len(r3):
                r3.append(None)

        df = pd.DataFrame({
            "0,0001 BTC": r,
            "0,01 BTC": r2,
            "0,1 BTC": r3
        }, columns=["0,0001 BTC", "0,01 BTC", "0,1 BTC"],)
        plt.figure()
        ax = df.plot.hist(bins=bins, edgecolor='black', linewidth=1.2, grid=False, stacked=True, alpha=1)
        ax.set_ylabel('Share of Nodes')
        ax.set_xlabel('Node betweennees')
        filePath = cwd + "/" + str(timestamp) + '/plots/histogram/hist_' + str(index) + '_grp.png'
        plt.title("Timestamp: " + str(timestamp), fontsize=12)
        plt.savefig(filePath, bbox_inches='tight', dpi=400)
        plt.show()


def plot_cluster_histogram(df_plot, timestamp, bins=10):
    df_plot = pd.DataFrame(df_plot['clustering'])
    ax_arr = df_plot.hist(bins=bins, edgecolor='black', linewidth=1.2, grid=False)
    for ax in ax_arr.flatten():
        ax.set_xlim(left=0.)
        ax.set_ylabel('#Nodes')
        ax.set_xlabel('Clustering Coefficient')
    filePath = '../analysis/plots/histogram/clustering_' + str(timestamp) + '.png'
    plt.title("Timestamp: " + str(timestamp))
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

timestamp = timestamps[6]
baseAmount = 10000000
# 10000000 -> 0,0001 BTC -> 3€ *
# 100000000 -> 0,001 BTC -> 32€
# 1000000000 -> 0,01 BTC -> 320€ *
# 10000000000 -> 0,1 BTC -> 3200€ *
# 100000000000 ->  1 BTC ->  32000€

# filePath = '../graphs/' + str(timestamp) + '_lngraph.graphml'
# initProcess(timestamp, baseAmount, filePath, betweenness=True, clustering=True, page=True, deg_cen=True,
#             in_deg_cen=True, out_deg_cen=True, closeness=True, edge_betweenness=True, deg=True, in_deg=True,
#             out_deg=True)

print(timestamp)
cwd = str(Path().resolve())
filepath = cwd + '/' + str(timestamp) + '/' + str(baseAmount)
filenames = next(os.walk(filepath), (None, None, []))[2]  # [] if no file
print(filenames)
baseAmount, baseAmount2, baseAmount3 = 10000000, 1000000000, 10000000000
df_plot = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount) + '/' + filenames[3])
df_plot_2 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount2) + '/' + filenames[3])
df_plot_3 = pd.read_csv(cwd + '/' + str(timestamp) + '/' + str(baseAmount3) + '/' + filenames[3])


plot_bt_cdf(df_plot, df_plot_2, df_plot_3, timestamp)
# plot_bt_histogram(df_plot, df_plot_2, df_plot_3, timestamp, 20)
# plot_cluster_histgram(df_plot, graphTimestamp, 20)

# Test
# G = createGraphFromGraphML(filePath, baseAmount)
# TODO maybe try to somehow plot the graph components
# TODO Update paths & plotting functions
# print("Closeness: ", calc_closeness_centrality(G))
# print("Degree: ", calc_degree_centrality(G))
# print("Edge BT: ", calc_edge_betweenness(G))
# print("In Deg: ", calc_in_degree_centrality(G))
# print("Out Deg: ", calc_out_degree_centrality(G))

# coeff_dict = calc_clustering_coefficient(G)
# df_hist = pd.DataFrame.from_dict(coeff_dict, orient='index')
# plot_cluster_histgram(df_plot, graphTimestamp, 20)

# TODO Differenz anschauen von Node Betweenness in verschiedenen Timestamps
# TODO Min/Max von Bt über alle Timestamps und Differenz plotten --> Wichtig
# TODO Report + offene Fragen --> Wichtig