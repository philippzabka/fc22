import networkx as nx
import igraph as ig

print("Creating graph from file")
# f = open("../graphs/1609498800_lngraph.graphml")
# g = ig.Graph.Read_GraphML(f)
g = nx.read_graphml("../graphs/1609498800_lngraph.graphml")
baseAmount = 1000000000
G = ig.Graph()
used_vertices = set()
for edge in g.edges(data=True):
    # Weight = Fee
    weight = edge[2]['fee_base_msat'] + (baseAmount * edge[2]['fee_proportional_millionths'] / 1000000)
    # weight = random.randint(1, baseAmount)
    if weight == 0:
        weight = 1
    if edge[2]['source'] not in used_vertices:
        G.add_vertex(edge[2]['source'])
        used_vertices.add(edge[2]['source'])
    if edge[2]['destination'] not in used_vertices:
        G.add_vertex(edge[2]['destination'])
        used_vertices.add(edge[2]['destination'])

    G.add_edge(edge[2]['source'], edge[2]['destination'], weight=weight)

print(G.is_weighted(), G.get_edge_dataframe())

print("Calculating betweenness")
betweenness = G.betweenness(weights="weight", directed=False)

df_vertex = G.get_vertex_dataframe()
df_vertex['betweenness'] = betweenness

df_vertex.to_csv('../analysis/results/' + str(baseAmount) + '_1609498800_igraph.csv',
          index=True, index_label='node')

