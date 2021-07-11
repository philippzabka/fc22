import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    pass
    # print(min(df_plot.values), max(df_plot.values))
    #
    # binWidth = int(np.sqrt(2200))
    # print(binWidth)
    # bins = np.arange(min(df_plot.values), max(df_plot.values) + binWidth, binWidth)
    # print(bins)
    # # df_plot.hist(bins=bins, edgecolor='black', linewidth=1.2)
    # # plt.show()
    # df_plot.boxplot(meanline=True, showfliers=True, showmeans=True)
    # plt.show()

print("Creating graph from file")
g = nx.read_graphml("../graphs/1609498800_lngraph.graphml")
baseAmount = 1000000000
G = nx.Graph()
sum = 0
for edge in g.edges(data=True):
    # Weight = Fee
    weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
    if weight == 0:
        weight = 1
    G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

print(sum)
print(nx.number_connected_components(G))
# print("Calculating betweenness")
# betweenness = calc_betweenness_centrality(G)
# nx.set_node_attributes(G, betweenness, "betweenness")
# nx.write_graphml(G, '../graphs/betweenness/' + str(baseAmount) + _1609498800.graphml')

# G = nx.read_graphml('../graphs/betweenness/' + str(baseAmount) + '_1609498800.graphml')
# node_bt_sorted = sorted(G.nodes, key=lambda x: G.nodes[x]['betweenness'], reverse=True)
#
# node_dict = dict()
# for node in node_bt_sorted:
#     # print('Node: ', node, 'BT: ', G.nodes[node]['betweenness'], 'D: ', G.degree[node])
#     node_dict[node] = dict()
#     node_dict[node]['betweenness'] = G.nodes[node]['betweenness']
#     node_dict[node]['degree'] = G.degree[node]
#
#     adj_nodes = G.adj[node]
#     deg_one_sum, deg_two_sum = 0, 0
#     for key in adj_nodes:
#         deg = G.degree[key]
#         if deg == 1:
#             deg_one_sum += 1
#         elif deg == 2:
#             deg_two_sum += 1
#
#     node_dict[node]['deg_one_percentage'] = (deg_one_sum / G.degree[node])
#     node_dict[node]['deg_two_percentage'] = (deg_two_sum / G.degree[node])
#
# df = pd.DataFrame.from_dict(node_dict, orient='index')
# df.to_csv('../analysis/results/' + str(baseAmount) + '_1609498800.csv',
#           index=True, index_label='node')
#
# df = pd.read_csv('../analysis/betweenness_results/1000000000_1609498800_filtered_nw')
# drawHistogram(df)
#


