from directed_graph import DirectGraph
from directed_graph import topk


def Run_LTM(graph, seeds, rounds, centrality):
	v, e = graph.size()
	print("* running LTM({}) TOP {} {} - graph size: {} {}".format(rounds, len(seeds), centrality, v, e))
	influenced, kept, steps = graph.ltm(seeds, rounds)
	print("** influenced(%d) kept(%d) steps(%d)" % (len(influenced), len(kept), steps))
	return len(influenced), len(kept), steps


def experiment(graph):
	print('# Edges = %d\tAverage Clustering = %f' % (graph.countEdges(), graph.average_clustering()))
	sys.stdout.flush()

	print('# Eigenvector Centrality...')
	diffsum, cscores = graph.eigenvector_centrality()
	# print(diffsum)
	# print(cscores)
	top_eigenc = [a for a, b in topk(cscores, seed)]
	print(top_eigenc)
	print('# Done')
	sys.stdout.flush()

	print('# Betweennes centrality...')
	bet = graph.betweenness()
	# print(bet)
	top_bet = [a for a, b in topk(bet, seed)]
	print(top_bet)
	print('# Done')
	sys.stdout.flush()

	print("# Lin's index...")
	lin = graph.lin_index()
	#print(lin)
	top_lin = [a for a, b in topk(lin, seed)]
	print(top_lin)
	print('# Done')
	sys.stdout.flush()

	max_lin_influenced, _, max = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')
	max_eigenc_influenced = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')
	max_bet_influenced = Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')
	lin_max_seed = seed
	eigenc_max_seed = seed
	bet_max_seed = seed

	
	while seed > 0:
		seed -= 5
		influenced_lin = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')[0]
		if max_lin_influenced <= influenced_lin:
			max_lin_influenced = influenced_lin
			lin_max_seed = seed
		else:
			break
			
	seed = 100
	while seed > 0:
		seed -= 5
		influenced_eigenc = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')[0]
		if max_eigenc_influenced <= influenced_eigenc:
			max_eigenc_influenced = influenced_eigenc
			eigenc_max_seed = seed
		else:
			break
			 
	seed = 100
	while seed > 0:
		seed -= 5
		influenced_bet = Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')[0]
		if max_bet_influenced <= influenced_bet:
			max_bet_influenced = influenced_bet
			bet_max_seed = seed
		else:
			break
	
	#lin_max_values.append((lin_max_seed, max_lin_influenced))
	#eigenc_max_values.append((eigenc_max_seed, max_eigenc_influenced))
	#bet_max_values.append((bet_max_seed, max_bet_influenced))
	sys.stdout.flush()