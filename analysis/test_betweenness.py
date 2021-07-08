import random

import networkx as nx
import pandas as pd
from betweenness import calc_betweenness_centrality

G = nx.Graph()
# Weight
G.add_edge("a", "b", weight=2)
G.add_edge("a", "c", weight=3)
G.add_edge("a", "g", weight=2)
G.add_edge("b", "c", weight=1)
G.add_edge("c", "d", weight=1)
G.add_edge("d", "e", weight=1)
G.add_edge("d", "f", weight=3)
G.add_edge("e", "f", weight=3)
G.add_edge("e", "h", weight=1)
G.add_edge("g", "h", weight=2)
#
# G.add_edge("a", "l", weight=1)
# G.add_edge("l", "k", weight=3)
# G.add_edge("k", "e", weight=1)

# G.add_edge("c", "i", weight=1)
# G.add_edge("e", "j", weight=3)
# G.add_edge("f", "j", weight=4)

#No weights
# G.add_edge("a", "b")
# G.add_edge("a", "c")
# G.add_edge("a", "g")
# G.add_edge("b", "c")
# G.add_edge("c", "d")
# G.add_edge("d", "e")
# G.add_edge("d", "f")
# G.add_edge("e", "f")
# G.add_edge("e", "h")
# G.add_edge("g", "h")
#
# G.add_edge("c", "i")
# G.add_edge("e", "j")
# G.add_edge("f", "j")


print("Creating graph from file")
g = nx.read_graphml("../graphs/1609498800_lngraph.graphml")
baseAmount = 10000000000
G = nx.Graph()
for edge in g.edges(data=True):
    # Weight = Fee
    # weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
    weight = random.randint(0, baseAmount)
    G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

print("Calculating betweenness")
betweenness = calc_betweenness_centrality(G)
nx.set_node_attributes(G, betweenness, "betweenness")
nx.write_graphml(G, '../graphs/betweenness/' + str(baseAmount) + '_1609498800.graphml')

# G = nx.read_graphml('../graphs/betweenness/' + str(baseAmount) + '_1609498800.graphml')
node_bt_sorted = sorted(G.nodes, key=lambda x: G.nodes[x]['betweenness'], reverse=True)

node_dict = dict()
for node in node_bt_sorted:
    # print('Node: ', node, 'BT: ', G.nodes[node]['betweenness'], 'D: ', G.degree[node])
    node_dict[node] = dict()
    node_dict[node]['betweenness'] = G.nodes[node]['betweenness']
    node_dict[node]['degree'] = G.degree[node]

    adj_nodes = G.adj[node]
    deg_one_sum, deg_two_sum = 0, 0
    for key in adj_nodes:
        deg = G.degree[key]
        if deg == 1:
            deg_one_sum += 1
        elif deg == 2:
            deg_two_sum += 1

    node_dict[node]['deg_one_percentage'] = (deg_one_sum / G.degree[node])
    node_dict[node]['deg_two_percentage'] = (deg_two_sum / G.degree[node])

df = pd.DataFrame.from_dict(node_dict, orient='index')
df.to_csv('../analysis/results/' + str(baseAmount) + '_1609498800.csv',
          index=True, index_label='node')
