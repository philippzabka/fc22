import networkx as nx
import pandas as pd
from betweenness import calc_betweenness_centrality
import igraph as ig

print("Creating graph from file")
# f = open("../graphs/1609498800_lngraph.graphml")
# g = ig.Graph.Read_GraphML(f)
g = nx.read_graphml("../graphs/1609498800_lngraph.graphml")
baseAmount = 1000000000
G = ig.Graph()
for edge in g.edges(data=True):
    # Weight = Fee
    weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
    # weight = random.randint(1, baseAmount)
    if weight == 0:
        weight = 1
    G.add_vertices([edge[2]['source'], edge[2]['destination']])
    G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

print(G.is_weighted(), G.get_edge_dataframe())

print("Calculating betweenness")
betweenness = G.betweenness(weights="weight", directed=False)
print(betweenness)
sorted(betweenness, reverse=True)

# nx.set_node_attributes(G, betweenness, "betweenness")
# nx.write_graphml(G, '../graphs/betweenness/' + str(baseAmount) + '_1609498800_test.graphml')

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
# df.to_csv('../analysis/results/' + str(baseAmount) + '_1609498800_test.csv',
#           index=True, index_label='node')

