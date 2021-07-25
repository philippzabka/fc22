import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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


def createGraphFromGraphML(filePath, baseAmount, timestamp, plot=False):
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
        # print("Base: ", edge[2]['fee_base_msat'], "Prop: ", edge[2]['fee_proportional_millionths'], "Weight: ", weight)
        if weight == 0:
            weight = min_weight / len(G.nodes)
        G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

    if plot:
        df = pd.DataFrame.from_dict(bar_dict, orient='index')
        ax = df.plot.bar(rot=0)
        ax.set_ylabel('#nodes')
        ax.set_xlabel('fee parameter')
        plt.show()
        plt.savefig('../analysis/plots/fees/zero_fees_' + str(timestamp) + '.png')
    return G


def calc_node_degrees(G, sorted_nodes_dict):
    # TODO rework function
    degrees = dict()
    for node in sorted_nodes_dict:
        degrees[node] = dict()
        degrees[node]['degree'] = G.degree[node]
        # node_degree[node]['betweenness'] = G.nodes[node]['betweenness']

        adj_nodes = G.adj[node]
        deg_one_sum, deg_two_sum, deg_three_sum, deg_greater_three = 0, 0, 0, 0
        for key in adj_nodes:
            deg = G.degree[key]
            if deg == 1:
                deg_one_sum += 1
            elif deg == 2:
                deg_two_sum += 1
            elif deg == 3:
                deg_three_sum += 1
            else:
                deg_greater_three += 1

        degrees[node]['deg_one'] = (deg_one_sum / G.degree[node])
        degrees[node]['deg_two'] = (deg_two_sum / G.degree[node])
        degrees[node]['deg_three'] = (deg_three_sum / G.degree[node])
        degrees[node]['deg_greater_three'] = (deg_greater_three / G.degree[node])

    return degrees


# Init all necessary calculations
def initProcess(timestamp, baseAmount, filePath, betweenness=False, clustering=False, page=False, deg_cen=False,
                in_deg_cen=False, out_deg_cen=False, closeness=False, edge_betweenness=False):

    G = createGraphFromGraphML(filePath, baseAmount, graphTimestamp, plot=False)

    Path(str(graphTimestamp) + "/" + str(baseAmount)).mkdir(parents=True, exist_ok=True)
    cwd = str(Path().resolve())

    # Calc betweenness centrality
    if betweenness:
        betweenness = calc_betweenness_centrality(G)
        betweenness_sorted = dict(sorted(betweenness.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(betweenness_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/betweenness_centrality.csv',
                              index=True, index_label=['node_id', 'betweenness'])

    # Calc clustering
    if clustering:
        clustering_coeff = calc_clustering_coefficient(G)
        clustering_coeff_sorted = dict(sorted(clustering_coeff.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(clustering_coeff_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/clustering.csv',
                  index=True, index_label=['node_id', 'clustering_coeff'])

    # Calc pagerank
    if page:
        page_rank = calc_page_rank(G)
        page_sorted = dict(sorted(page_rank.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(page_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/pagerank.csv',
                  index=True, index_label=['node_id', 'pagerank'])

    # Calc node degree centrality
    if deg_cen:
        degree_centrality = calc_degree_centrality(G)
        degree_centrality_sorted = dict(sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(degree_centrality_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/degree_centrality.csv',
                  index=True, index_label=['node_id', 'degree_centrality'])

    # Calc node in-degree centrality
    if in_deg_cen:
        degree_in_centrality = calc_in_degree_centrality(G)
        degree_in_sorted = dict(sorted(degree_in_centrality.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(degree_in_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/in_degree_centrality.csv',
                  index=True, index_label=['node_id', 'in_degree_centrality'])

    # Calc node out-degree centrality
    if out_deg_cen:
        degree_out_centrality = calc_out_degree_centrality(G)
        degree_out_sorted = dict(sorted(degree_out_centrality.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(degree_out_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/out_degree_centrality.csv',
                  index=True, index_label=['node_id', 'out_degree_centrality'])

    # Calc closeness centrality
    if closeness:
        closeness = calc_closeness_centrality(G)
        closeness_sorted = dict(sorted(closeness.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(closeness_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/closeness_centrality.csv',
                  index=True, index_label=['node_id', 'closeness_centrality'])

    # Calc edge betweenness centrality
    if edge_betweenness:
        edge_betweenness = calc_edge_betweenness(G)
        print(edge_betweenness)
        edge_betweenness_sorted = dict(sorted(edge_betweenness.items(), key=lambda item: item[1], reverse=True))
        df = pd.DataFrame.from_dict(edge_betweenness_sorted, orient='index')
        df.to_csv(cwd + "/" + str(timestamp) + '/' + str(baseAmount) + '/edge_betweenness_centrality.csv',
                  index=True, index_label=['edge', 'edge_betweenness'])

    # # Calc node degrees
    # # degrees = calc_node_degrees(G)


def splitBetweennessIntoRanges(df):
    # TODO update values
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
    titles = ['1 to 100', '101 to 10000', '10001 to 1000000', '1000001 to 7500000', '1 to 7500000']

    return ranges, titles


def plot_bt_cdf(df_plot, timestamp, base_amount):
    ranges, titles = splitBetweennessIntoRanges(df_plot)
    for r, title, index in zip(ranges, titles, range(len(ranges))):
        y = np.zeros(len(r))
        for i in range(len(r)):
            y[i] = (i + 1) / len(y)
        plt.ylim(0, 1)

        # If last element in list
        filePath = '../analysis/plots/betweenness/cdf/cdf_' + str(timestamp) + '_' + str(index) + '.png'
        if index == len(ranges) - 1:
            plt.xscale('log')
        plt.plot(r, y)
        plt.xlabel('Node Betweenness')
        plt.ylabel('Percentage')
        # plt.title("CDF: " + str(np.ceil(int(min(r)))) + ' to ' + str(np.ceil(int(max(r)))))
        plt.title("Timestamp: " + str(timestamp) + ", Range: " + title + ", Base: " + str(base_amount), fontsize=10)
        plt.savefig(filePath, bbox_inches='tight', dpi=400)
        plt.show()


def plot_bt_histogram(df_plot, timestamp, base_amount, bins=10):
    ranges, titles = splitBetweennessIntoRanges(df_plot)
    for r, title, index in zip(ranges, titles, range(len(ranges))):
        df_plot = pd.DataFrame(r)
        ax_arr = df_plot.hist(bins=bins, edgecolor='black', linewidth=1.2, grid=False)
        for ax in ax_arr.flatten():
            ax.set_xlim(left=0.)
            ax.set_ylabel('#Nodes')
            ax.set_xlabel('Node betweennees')
        filePath = '../analysis/plots/histogram/betweenness_' + str(timestamp) + '_' + str(index) + '.png'
        plt.title("Timestamp: " + str(timestamp) + ", Range: " + title + ", Base: " + str(base_amount), fontsize=10)
        plt.savefig(filePath, bbox_inches='tight', dpi=400)
        plt.show()


def plot_cluster_histgram(df_plot, timestamp, bins=10):
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
    1522576800,
    1533117600,
    1543662000,
    1554112800,
    1564653600,
    1572606000,
    1585735200,
    1596276000,
    1606820400,
    1609498800
]

graphTimestamp = timestamps[8]
baseAmount = 1000000000
# 100000000 -> 0,001 BTC
# 1000000000 -> 0,01 BTC
# 10000000000 -> 0,1 BTC
# 100000000000 -> 1 BTC

filePath = '../graphs/' + str(graphTimestamp) + '_lngraph.graphml'
initProcess(graphTimestamp, baseAmount, filePath, betweenness=False, clustering=False, page=True, deg_cen=True,
            in_deg_cen=True, out_deg_cen=True, closeness=True, edge_betweenness=False)

# df_plot = pd.read_csv('../analysis/results/' + str(baseAmount) + '_' + str(graphTimestamp) + '.csv')
# plot_bt_cdf(df_plot, graphTimestamp, baseAmount)
# plot_bt_histogram(df_plot, graphTimestamp, baseAmount, 20)
# plot_cluster_histgram(df_plot, graphTimestamp, 20)

# Test
G = createGraphFromGraphML(filePath, baseAmount, graphTimestamp, plot=False)
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
