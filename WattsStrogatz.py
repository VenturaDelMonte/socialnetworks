from direct_graph import DirectGraph
from direct_graph import topk
import sys
import matplotlib.pyplot as plt
import random
from datetime import datetime
import time
import concurrent.futures
from experiment import experiment
import dill

def worker_task(i):
	start_time = time.time()
	rounds = 10

	print("# iteration %d" % (i+1))
	NODES = 7115
	min_edges = 75000
	max_edges = 125000
	incr = 0.001
	p = 0.001 # probability
	seed = 100
	radius = 2
	weak_ties = [i*5 for i in range(0, 3)]
	ret = None
	avgc = 0
	edges = 0
	with DirectGraph.WS2DGraph(NODES, random.randint(min_edges, max_edges), radius, weak_ties) as graph:
		edges = graph.size()[1]
		avgc = graph.toUndirect().average_clustering()
		ret = experiment(graph, seed, rounds)
	#	print("# iteration %d done in %f" % (i+1, time.time() - start_time))
	elapsed = time.time() - start_time
	return ret.append((edges, avgc, elapsed, p, radius))

if __name__ == '__main__':
	start = time.time()
	random.seed(datetime.now())	
	
	seed = 100

	lin_max_values = []
	eigenc_max_values = []
	bet_max_values = []

	
	n = 100
	pbar = pyprind.ProgBar(n,stream=1)
	with concurrent.futures.ProcessPoolExecutor(5) as executor:
		futures = { executor.submit(worker_task, i) for i in range(n) }
		for future in concurrent.futures.as_completed(futures):
			ret = future.result()
			# lin_max_values.append(ret[0])
			# eigenc_max_values.append(ret[1])
			# bet_max_values.append(ret[2])
			result.append(ret)
			pbar.update()

	dill.dump(result, open('ws', 'wb'))


	
	sys.stdout.flush()
	'''

	max_lin = sorted(lin_max_values, key=lambda t: t[1], reverse=True)[0][1]
	max_eigenc = sorted(eigenc_max_values, key=lambda t: t[1], reverse=True)[0][1]
	max_bet = sorted(bet_max_values, key=lambda t: t[1], reverse=True)[0][1]

	fig, ax = plt.subplots()

	bar_width = 0.35

	opacity = 0.4

	rects1 = plt.bar(1, max_lin, width=bar_width, alpha=opacity, color='b', label='Lin')

	rects2 = plt.bar(2, max_eigenc, width=bar_width, alpha=opacity, color='r', label='Eigenvector')

	rects3 = plt.bar(3, max_bet, width=bar_width, alpha=opacity, color='y', label='Betweenness')

	plt.xlabel('Centrality Measures')
	plt.ylabel('Influenced')
	plt.title('Influenced Comparison')
	plt.xticks([1.2,2.2,3.2], ('L', 'E', 'B'))

	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	plt.tight_layout()

	plt.savefig('ws.png')
	plt.show()
	'''
	print("program complete in {}".format(time.time() - start))