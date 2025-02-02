from scipy.io import mmread
import networkx as nx
a = mmread('../../graphs/test.mtx')
graph = nx.Graph(a)
nx.draw(graph)