import networkx as nx

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

# 100
# {'a': 5.5, 'b': 0.0, 'c': 15.666666666666666, 'g': 3.833333333333333, 'd': 12.166666666666666, 'e': 9.166666666666666, 'f': 2.333333333333333, 'h': 4.333333333333333, 'i': 0.0, 'j': 0.0}

# short_path_set = set()
# for node in G.nodes:
#     paths = dict(nx.single_source_shortest_path(G, source=node))
#     for key in paths:
#         print(paths[key])
#         short_path_set.add(tuple(paths[key]))

# print(short_path_set)
# print(len(short_path_set))

# print(nx.single_source_shortest_path(G, source="a"))
# sh_paths = dict(nx.all_pairs_shortest_path(G))
# for key in sh_paths:
#     print(sh_paths[key])

# all_pairs = dict(nx.all_pairs_dijkstra_path(G, weight="weight"))
#
# count = 0
# for key in all_pairs:
#     for key2 in all_pairs[key]:
#         count += 1
#
# # print(count)

bt = nx.betweenness_centrality(G, weight='weight', normalized=False)
nx.set_node_attributes(G, bt, "betweenness")
for node in G:
    print(node[0])


