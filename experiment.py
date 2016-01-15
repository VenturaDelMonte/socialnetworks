from directed_graph import DirectedGraph
from directed_graph import topk
import sys

def Run_LTM(graph, seeds, rounds, centrality):
	v, e = graph.size()
	#print("* running LTM({}) TOP {} {} - graph size: {} {}".format(rounds, len(seeds), centrality, v, e))
	influenced, kept, steps = graph.ltm(seeds, rounds)
	#print("** influenced(%d) kept(%d) steps(%d)" % (len(influenced), len(kept), steps))
	return len(influenced), len(kept), steps


def experiment(graph, seed, rounds):
	#print('# Edges = %d\tAverage Clustering = %f' % (graph.countEdges(), graph.toUndirect().average_clustering()))
	sys.stdout.flush()

	#print('# Eigenvector Centrality...')
	diffsum, cscores = graph.eigenvector_centrality()
	# #print(diffsum)
	# #print(cscores)
	top_eigenc = [a for a, b in topk(cscores, seed)]
	#print(top_eigenc)
	#print('# Done')
	sys.stdout.flush()

	#print('# Betweennes centrality...')
	bet, D = graph.betweennessEx()
	# #print(bet)
	top_bet = [a for a, b in topk(bet, seed)]
	#print(top_bet)
	#print('# Done')
	sys.stdout.flush()

	#print("# Lin's index...")
	lin = graph.lin_index(D)
	##print(lin)
	top_lin = [a for a, b in topk(lin, seed)]
	#print(top_lin)
	#print('# Done')
	sys.stdout.flush()
	del D

	max_lin_influenced, _, lin_rounds = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')
	max_eigenc_influenced, _, eigenc_rounds = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')
	max_bet_influenced, _, bet_rounds = Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')
	lin_max_seed = seed
	eigenc_max_seed = seed
	bet_max_seed = seed

	
	while seed > 0:
		seed -= 5
		influenced_lin, _, _rounds = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')
		if max_lin_influenced <= influenced_lin:
			max_lin_influenced = influenced_lin
			lin_max_seed = seed
			lin_rounds = _rounds
		else:
			break
			
	seed = 100
	while seed > 0:
		seed -= 5
		influenced_eigenc, _, _rounds = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')
		if max_eigenc_influenced <= influenced_eigenc:
			max_eigenc_influenced = influenced_eigenc
			eigenc_max_seed = seed
			eigenc_rounds = _rounds
		else:
			break
			 
	seed = 100
	while seed > 0:
		seed -= 5
		influenced_bet, _, _rounds= Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')
		if max_bet_influenced <= influenced_bet:
			max_bet_influenced = influenced_bet
			bet_max_seed = seed
			bet_rounds = _rounds
		else:
			break
	
	#lin_max_values.append((lin_max_seed, max_lin_influenced))
	#eigenc_max_values.append((eigenc_max_seed, max_eigenc_influenced))
	#bet_max_values.append((bet_max_seed, max_bet_influenced))
	sys.stdout.flush()

	return [(max_lin_influenced,lin_max_seed,lin_rounds),(max_eigenc_influenced,eigenc_max_seed,lin_rounds),(max_bet_influenced,bet_max_seed,bet_rounds)]