import random

import networkx as nx
import pandas as pd
from betweenness import calc_betweenness_centrality

G = nx.Graph()
# Weight
G.add_edge("a", "b", weight=0)
G.add_edge("a", "c", weight=1)
G.add_edge("a", "g", weight=2)
G.add_edge("b", "c", weight=0)
G.add_edge("c", "d", weight=4)
G.add_edge("d", "e", weight=1)
G.add_edge("d", "f", weight=3)
G.add_edge("e", "f", weight=0)
G.add_edge("e", "h", weight=0)
G.add_edge("g", "h", weight=10)
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

print("Calculating betweenness")
betweenness = calc_betweenness_centrality(G)
print(betweenness)