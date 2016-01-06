from direct_graph import DirectGraph
from direct_graph import topk
import sys

seed = 100
graph = DirectGraph.from_filename('wiki-Vote.txt')

print(graph)

print('# Edges = %d\tAverage Clustering = %f' % (graph.countEdges(), graph.average_clustering()))

sys.stdout.flush()

print('# Eigenvector Centrality...')
diffsum, cscores = graph.eigenvector_centrality()
print(diffsum)
print(cscores)
top_eigenc = [a for a, b in topk(cscores, seed)]
print('# Done')
sys.stdout.flush()
print('# Betweennes centrality...')
bet = graph.betweenness()
print(bet)
top_bet = [a for a, b in topk(bet, seed)]
print('# Done')
sys.stdout.flush()
print("# Lin's index...")
lin = graph.lin_index()
print(lin)
top_lin = [a for a, b in topk(lin, seed)]
print('# Done')

sys.stdout.flush()