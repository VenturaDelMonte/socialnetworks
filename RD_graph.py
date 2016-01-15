from experiment import experiment
from directed_graph import DirectedGraph
from directed_graph import topk
import sys
import matplotlib.pyplot as plt
import random
from datetime import datetime
import concurrent.futures
import time
import pyprind

def worker_task(i):
	start_time = time.time()
	rounds = 10
	NODES = 7115
	min_edges = 75000
	max_edges = 125000
	incr = 0.001
	p = random.uniform(0.001, 0.0028) # probability
	seed = 100
	#print("# iteration %d" % (i+1))
	graph = None
	edges, avgc = -1, -1
	while p > 0:
		#print("generating graph with N={} p={}".format(NODES, p))
		graph = DirectedGraph.randomDirectedGraph(NODES, p)
		edges = graph.size()[1]
		avgc = graph.average_clustering()
		#print("prob={} edges={}".format(p, edges))
		if edges >= min_edges and edges <= max_edges:
			break
			# avgc = graph.average_clustering()
			# print("** avgc={}".format(avgc))
			# if avgc >= 0.1 and avgc <= 0.2:
			# 	break
			# elif avgc > 0.2:
			# 	p += ((max_edges - edges) / min_edges * 10.0) * incr
			# else:
			# 	p += ((min_edges - edges) / min_edges * 10.0) * incr
		elif edges > max_edges:
			p += ((max_edges - edges) / min_edges) * incr
		else:
			p += ((min_edges - edges) / min_edges) * incr
		sys.stdout.flush()

	ret = experiment(graph, seed, rounds)
	elapsed = time.time() - start_time
	#print("# iteration %d done in %f" % (i+1, time.time() - start_time))

	#return [(lin_max_seed, max_lin_influenced), (eigenc_max_seed, max_eigenc_influenced), (bet_max_seed, max_bet_influenced)]
	return ret.append((edges, avgc, elapsed))

if __name__ == '__main__':
	start = time.time()
	random.seed(datetime.now())	
	seed = 100

	# lin_max_values = []
	# eigenc_max_values = []
	# bet_max_values = []
	result = []

	n = 100
	pbar = pyprind.ProgBar(n,stream=1)
	with concurrent.futures.ProcessPoolExecutor(4) as executor:
		futures = { executor.submit(worker_task, i) for i in range(n) }
		for future in concurrent.futures.as_completed(futures):
			ret = future.result()
			# lin_max_values.append(ret[0])
			# eigenc_max_values.append(ret[1])
			# bet_max_values.append(ret[2])
			result.append(ret)
			pbar.update()

	dill.dump(result, open('rd', 'wb'))
	'''
	print('# Lin\tEigenvector\tBetweenness')
	for x, y, z in zip(lin_max_values, eigenc_max_values, bet_max_values):
		print("{}-{} {}-{} {}-{}".format(x[0], x[1], y[0], y[1], z[0], z[1]))

	sys.stdout.flush()

	fig, ax = plt.subplots()

	bar_width = 0.35

	opacity = 0.4

	max_lin = sorted(lin_max_values, key=lambda t: t[1], reverse=True)[0][1]
	max_eigenc = sorted(eigenc_max_values, key=lambda t: t[1], reverse=True)[0][1]
	max_bet = sorted(bet_max_values, key=lambda t: t[1], reverse=True)[0][1]

	rects1 = plt.bar(1, max_lin, width=bar_width, alpha=opacity, color='b', label='Lin')

	rects2 = plt.bar(2, max_eigenc, width=bar_width, alpha=opacity, color='r', label='Eigenvector')

	rects3 = plt.bar(3, max_bet, width=bar_width, alpha=opacity, color='y', label='Betweenness')

	plt.xlabel('Centrality Measures')
	plt.ylabel('Influenced')
	plt.title('Influenced Comparison')
	plt.xticks([1.2,2.2,3.2], ('L', 'E', 'B'))

	plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	plt.tight_layout()

	plt.savefig('rd.png')
	plt.show()
	'''
	print("program complete in {}".format(time.time() - start))
