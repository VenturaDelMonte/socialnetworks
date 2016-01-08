from direct_graph import DirectGraph
from direct_graph import topk
import sys
import matplotlib.pyplot as plt
import random
from datetime import datetime


def Run_LTM(graph, seeds, rounds, centrality):
	v, e = graph.size()
	print("* running LTM({}) TOP {} {} - graph size: {} {}".format(rounds, len(seeds), centrality, v, e))
	influenced, kept, steps = graph.ltm(seeds, rounds)
	print("** influenced(%d) kept(%d) steps(%d)" % (len(influenced), len(kept), steps))
	return len(influenced), len(kept), steps


if __name__ == '__main__':
	random.seed(datetime.now())	
	rounds = 10
	seed = 100

	lin_max_values = []
	eigenc_max_values = []
	bet_max_values = []

	
	for i in range(100):

		NODES = 7115
		min_edges = 75000
		max_edges = 125000
		incr = 0.001
		p = 0.3 # probability
		seed = 100
		d = int(random.randint(min_edges, max_edges) / NODES)

		graph = DirectGraph.preferentialAttachment(NODES, d, p)

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

		max_lin_influenced = Run_LTM(graph, top_lin[:seed], rounds, 'Lin')[0]
		max_eigenc_influenced = Run_LTM(graph, top_eigenc[:seed], rounds, 'Eigenvector')[0]
		max_bet_influenced = Run_LTM(graph, top_bet[:seed], rounds, 'Betweenness')[0]
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
		
		lin_max_values.append((lin_max_seed, max_lin_influenced))
		eigenc_max_values.append((eigenc_max_seed, max_eigenc_influenced))
		bet_max_values.append((bet_max_seed, max_bet_influenced))
		
		sys.stdout.flush()

	print('# Lin\tEigenvector\tBetweenness')
	for x, y, z in zip(lin_max_values, eigenc_max_values, bet_max_values):
		print("{}-{} {}-{} {}-{}".format(x[0], x[1], y[0], y[1], z[0], z[1]))

	sys.stdout.flush()



	fig, ax = plt.subplots()

	bar_width = 0.35

	opacity = 0.4

	rects1 = plt.bar(1, max_lin_influenced, width=bar_width, alpha=opacity, color='b', label='Lin')

	rects2 = plt.bar(2, max_eigenc_influenced, width=bar_width, alpha=opacity, color='r', label='Eigenvector')

	rects3 = plt.bar(3, max_bet_influenced, width=bar_width, alpha=opacity, color='y', label='Betweenness')

	plt.xlabel('Centrality Measures')
	plt.ylabel('Influenced')
	plt.title('Influenced Comparison')
	plt.xticks([1.2,2.2,3.2], ('L', 'E', 'B'))

	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	plt.tight_layout()

	plt.savefig('pa.png')
	plt.show()