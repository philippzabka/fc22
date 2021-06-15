import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import stats

"""
Fee = fee_base_msat +(amount * fee_proportional_millionths / 10**6)
"""

def calc_betweenness_centrality(g):
    return nx.betweenness_centrality(g, weight='weight', normalized=False)


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

    binWidth = 0.02
    print(min(df_plot.values), max(df_plot.values))
    bins = np.arange(min(df_plot.values), max(df_plot.values) + binWidth, binWidth)
    df_plot.hist(bins=bins, edgecolor='black', linewidth=1.2)
    # plt.show()


g = nx.read_graphml("../graphs/1609498800_lngraph.graphml")
baseAmount = 1000000000
for edge in g.edges(data=True):
    # Weight = Fee
    edge[2]['weight'] = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)

betweenness = calc_betweenness_centrality(g)
betweenness_filtered = dict()
for key in betweenness:
    if betweenness[key] != 0:
        betweenness_filtered[key] = betweenness[key]

df = pd.DataFrame.from_dict(betweenness_filtered, orient='index')
df.to_csv('./analysis/betweenness_results' + str(baseAmount) + '_1609498800_filtered',
          index=False)

# df = pd.read_csv('../analysis/betweenness_results/1000000000_1609498800_filtered')
# drawHistogram(df)

