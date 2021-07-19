import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""
Fee = fee_base_msat +(amount * fee_proportional_millionths / 10**6)
"""


def calc_betweenness_centrality(G):
    return nx.betweenness_centrality(G, weight='weight', normalized=False)


def calc_triangles(G):
    return nx.triangles(G)


def calc_clustering_coefficient(G):
    return nx.clustering(G)


def calc_avg_clustering_coefficient(G):
    return nx.average_clustering(G)


def draw_histogram(df, title, filePath):

    # bt_vals = list()
    # for value in df.values:
    #     if value != 0:
    #         bt_vals.append(value)
    # df = pd.DataFrame(bt_vals)

    # binWidth = int(np.sqrt(2200))
    # bins = np.arange(min(df.values), max(df.values) + binWidth, binWidth)
    # print(bins)
    # print('Bins: ', bins)
    axarr = df.hist(bins=10, edgecolor='black', linewidth=1.2)
    for ax in axarr.flatten():
        ax.set_xlim(left=0.)
        ax.set_ylabel('#Nodes')
        ax.set_xlabel('Clustering coefficient')

    plt.title(title)
    plt.savefig(filePath)
    plt.show()

    # df.boxplot(meanline=True, showfliers=False, showmeans=True)
    # plt.show()


def createGraphFromGraphML(filePath, baseAmount, timestamp, plot=False):
    print("Creating graph from file")
    g = nx.read_graphml(filePath)
    G = nx.Graph()

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


def setAttributes(G, betweenness, clustering):
    nx.set_node_attributes(G, betweenness, "betweenness")
    nx.set_node_attributes(G, clustering, "clustering")


def calc_node_degrees(G, sorted_nodes_dict):
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
def initProcess(graphTimestamp, baseAmount, filePath):
    G = createGraphFromGraphML(filePath, baseAmount, graphTimestamp, plot=False)
    betweenness = calc_betweenness_centrality(G)
    clustering = calc_clustering_coefficient(G)

    # Set attributes to graph
    setAttributes(G, betweenness, clustering)

    nx.write_graphml(G, '../graphs/betweenness/' + str(baseAmount) + '_' + str(graphTimestamp) + 'norm.graphml')

    # Sort nodes according to highest betweenness
    sorted_nodes = sorted(G.nodes, key=lambda x: G.nodes[x]['betweenness'], reverse=True)

    # Calc node degrees
    degrees = calc_node_degrees(G, sorted_nodes)

    #Create dict from all attributes
    to_csv = dict()
    for node in sorted_nodes:
        to_csv[node] = dict()
        to_csv[node]['betweenness'] = G.nodes[node]['betweenness']
        to_csv[node]['clustering'] = G.nodes[node]['clustering']
        to_csv[node]['degree'] = degrees[node]['degree']
        to_csv[node]['deg_one'] = degrees[node]['deg_one']
        to_csv[node]['deg_two'] = degrees[node]['deg_two']
        to_csv[node]['deg_three'] = degrees[node]['deg_three']
        to_csv[node]['deg_greater'] = degrees[node]['deg_greater_three']

    df = pd.DataFrame.from_dict(to_csv, orient='index')
    df.to_csv('../analysis/results/' + str(baseAmount) + '_' + str(graphTimestamp) + '.csv',
              index=True, index_label='node')


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


def plot_cdf(df_plot):
    data = df_plot['betweenness'].values
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

    for r, title, index in zip(ranges, titles, range(len(ranges))):
        y = np.zeros(len(r))
        for i in range(len(r)):
            y[i] = (i+1)/len(y)
        plt.ylim(0, 1)

        # If last element in list
        filePath = '../analysis/plots/betweenness/cdf/cdf_to_' + str(int(max(r))) + '.png'
        if index == len(ranges)-1:
            plt.xscale('log')
            filePath = '../analysis/plots/betweenness/cdf/cdf_to_' + str(int(max(r))) + '_log.png'
        plt.plot(r, y)
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title("CDF: " + str(np.ceil(int(min(r)))) + ' to ' + str(np.ceil(int(max(r)))))
        plt.title("CDF: " + title)
        plt.savefig(filePath)
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

graphTimestamp = timestamps[9]
baseAmount = 1000000000
           # 100000000 -> 0,001 BTC
           # 1000000000 -> 0,01 BTC
           # 10000000000 -> 0,1 BTC
           # 100000000000 -> 1 BTC
filePath = '../graphs/' + str(graphTimestamp) + '_lngraph.graphml'
# initProcess(graphTimestamp, baseAmount, filePath)

df_plot= pd.read_csv('../analysis/results/' + str(baseAmount) + '_' + str(graphTimestamp) + '.csv')
plot_cdf(df_plot)

filePathClustering = '../analysis/plots/clustering/clustering_' + str(graphTimestamp) + '.png'
filePathBetweenness = '../analysis/plots/betweenness/betweenness' + str(graphTimestamp) + '.png'

# Only Clustering
# G = createGraphFromGraphML(filePath, baseAmount, graphTimestamp, plot=False)
# coeff_dict = calc_clustering_coefficient(G)
# df_hist = pd.DataFrame.from_dict(coeff_dict, orient='index')
# draw_histogram(df_hist, "Timestamp: " + str(graphTimestamp), filePathClustering)


