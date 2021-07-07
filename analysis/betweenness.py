import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkit as nk
from scipy.stats import stats

"""
Fee = fee_base_msat +(amount * fee_proportional_millionths / 10**6)
"""


def calc_betweenness_centrality(G):
    return nx.betweenness_centrality(G, weight='weight', normalized=False)


def calc_triangles(g):
    return nx.triangles(g)


def calc_clustering_coefficient(g):
    return nx.clustering(g)

def calc_avg_clustering_coefficient(g):
    return nx.average_clustering(g)

def drawHistogram(df_plot):
    # Z-Score Method
    # print(min(df_plot.values), max(df_plot.values))
    # print(df_plot)
    # z_scores = stats.zscore(df)
    # abs_z_scores = np.abs(z_scores)
    # filtered_entries = (abs_z_scores < 3).all(axis=1)
    # new_df_plot = df[filtered_entries]
    # print(min(new_df_plot.values), max(new_df_plot.values))

    #Quantile Method
    # max_tresh = df_plot['0'].quantile(0.95)
    # print(max_tresh)
    # print(df[df_plot['0'] > max_tresh])
    #
    # min_tresh = df_plot['0'].quantile(0.05)
    # print(min_tresh)
    # print(df[df_plot['0'] < min_tresh])
    #
    # new_plot = df[(df_plot['0'] < max_tresh) & (df_plot['0'] > min_tresh)]
    # print(new_plot)

    print(min(df_plot.values), max(df_plot.values))

    binWidth = int(np.sqrt(2200))
    print(binWidth)
    bins = np.arange(min(df_plot.values), max(df_plot.values) + binWidth, binWidth)
    print(bins)
    # df_plot.hist(bins=bins, edgecolor='black', linewidth=1.2)
    # plt.show()
    df_plot.boxplot(meanline=True, showfliers=True, showmeans=True)
    plt.show()

print("Creating graph from file")
g = nx.read_graphml("../graphs/1609498800_lngraph.graphml")
baseAmount = 1000000000
G = nx.Graph()
for edge in g.edges(data=True):
    # Weight = Fee
    weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
    G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

# print("Graph created, proceeding with SP")
# short_path_set = set()
# for node in G.nodes:
#     paths = dict(nx.single_source_shortest_path(G, source=node))
#     for key in paths:
#         # print(paths[key])
#         short_path_set.add(tuple(paths[key]))
#
# print("SP finished")
# print(len(short_path_set))

betweenness = nk.centrality.Betweenness(G)
# betweenness = calc_betweenness_centrality(G)
betweenness_filtered = dict()
print("#Nodes:", len(betweenness))
for key in betweenness:
    if betweenness[key] != 0:
        betweenness_filtered[key] = betweenness[key]
print("#Nodes!=0:", len(betweenness_filtered))
# [7.55031845e-65] [2.24284729e+100]


df = pd.DataFrame.from_dict(betweenness_filtered, orient='index')
df.to_csv('../analysis/betweenness_results/' + str(baseAmount) + '_1609498800_filtered_nw',
          index=False)

df = pd.read_csv('../analysis/betweenness_results/1000000000_1609498800_filtered_nw')
drawHistogram(df)



