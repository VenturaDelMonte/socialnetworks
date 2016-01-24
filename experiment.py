from directed_graph import DirectedGraph
from directed_graph import topk
import sys
import time
import logging

logger = logging.getLogger()

def Run_LTM(graph, seeds, rounds, centrality):
	global logger
	s = time.time()
	v, e = graph.size()
	logger.info("* running LTM({}) TOP {} {} - graph size: {} {}".format(rounds, len(seeds), centrality, v, e))
	influenced, kept, steps = graph.ltm(seeds, rounds)
	logger.info("** influenced({}) kept({}) steps({}) in {}".format(len(influenced), len(kept), steps, time.time() - s))
	return len(influenced), len(kept), steps

def take_measures(graph, seed):
	nodes, edges = graph.size()
	s = time.time()
	avgc = graph.toUndirect().average_clustering()
	logger.info('# Edges = %d\tAverage Clustering = %f [%f]' % (edges, avgc, time.time() - s))
	logger.info('# Eigenvector Centrality...')
	s = time.time()
	diffsum, cscores = graph.eigenvector_centrality()
	# #logger.info(diffsum)
	# #logger.info(cscores)
	top_eigenc = [a for a, b in topk(cscores, seed)]
	#logger.info(top_eigenc)
	logger.info('# Eigenvector Centrality Done in {}'.format(time.time() - s))
	sys.stdout.flush()
	
	logger.info('# Betweennes centrality...')
	s = time.time()
	bet, D = graph.betweennessEx()
	# #logger.info(bet)
	top_bet = [a for a, b in topk(bet, seed)]
	#logger.info(top_bet)
	logger.info('# Betweennes Done in {}'.format(time.time() - s))
	sys.stdout.flush()

	logger.info("# Lin's index...")
	s = time.time()
	lin = graph.lin_index(D)
	##logger.info(lin)
	top_lin = [a for a, b in topk(lin, seed)]
	#logger.info(top_lin)
	logger.info('# Lin Done in {}'.format(time.time() - s))
	sys.stdout.flush()
	del D
	return top_eigenc, top_bet, top_lin

def simulate(graph, top_eigenc, top_bet, top_lin, seed, rounds):
	nodes = graph.size()[0]
	max_lin_influenced, _, lin_rounds = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')
	max_eigenc_influenced, _, eigenc_rounds = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')
	max_bet_influenced, _, bet_rounds = Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')
	lin_max_seed = seed
	eigenc_max_seed = seed
	bet_max_seed = seed
	
	while seed > 5:
		seed -= 5
		influenced_lin, _, _rounds = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')
		if max_lin_influenced <= influenced_lin:
			max_lin_influenced = influenced_lin
			lin_max_seed = seed
			lin_rounds = _rounds
		else:
			break
	
	seed = 100
	while seed > 5:
		seed -= 5
		influenced_eigenc, _, _rounds = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')
		if max_eigenc_influenced <= influenced_eigenc:
			max_eigenc_influenced = influenced_eigenc
			eigenc_max_seed = seed
			eigenc_rounds = _rounds
		else:
			break
	
	seed = 100
	while seed > 5:
		seed -= 5
		influenced_bet, _, _rounds= Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')
		if max_bet_influenced <= influenced_bet:
			max_bet_influenced = influenced_bet
			bet_max_seed = seed
			bet_rounds = _rounds
		else:
			break

	return (max_lin_influenced,lin_max_seed,lin_rounds),(max_eigenc_influenced,eigenc_max_seed,eigenc_rounds),(max_bet_influenced,bet_max_seed,bet_rounds)
	

def experiment(graph, seed, rounds):
	global logger
	if logger is None:
		logger = logging.getLogger()
	
	sys.stdout.flush()

	start1 = time.time()
	top_eigenc, top_bet, top_lin = take_measures(graph, seed)
	elapsed1 = time.time() - start1
	start2 = time.time()
	l, e, b = simulate(graph, top_eigenc, top_bet, top_lin, seed, rounds)
	elapsed2 = time.time() - start2
	logger.info("e1={}:::e2={}".format(elapsed1, elapsed2))
	sys.stdout.flush()
	return [l, e, b]


