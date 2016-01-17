from experiment import experiment
from directed_graph import DirectedGraph
from directed_graph import topk
import sys
import psutil
import gc
#import matplotlib.pyplot as plt
import random
from datetime import datetime
import concurrent.futures
import time
import atexit
import dill
from process_affinity_pool import ProcessPoolExecutorWithAffinity as ProcessPoolExecutor
import logging

logger = None

def worker_task(i):
	global logger
	current_proc = psutil.Process()
	if logger is None:
		logging.basicConfig(format="%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
		fileHandler = logging.FileHandler('RD_log.log.{}'.format(current_proc.cpu_affinity()[0]),mode='a')
		logger=logging.getLogger()
		logger.addHandler(fileHandler)
		logger.setLevel(logging.DEBUG)
	random.seed(datetime.now())	
	start_time = time.time()
	rounds = 10
	NODES = 7115
	min_edges = 75000
	max_edges = 125000
	incr = 0.001
	p = random.uniform(0.001, 0.0025) # probability
	seed = 100
	logger.info("# iteration {} on proc {}".format(i+1, current_proc.cpu_affinity()))
	graph = None
	edges, avgc = -1, -1
	while p > 0:
		logger.info("generating graph with N={} p={}".format(NODES, p))
		graph = DirectedGraph.randomDirectedGraph(NODES, p)
		edges = graph.size()[1]
		avgc = graph.toUndirect().average_clustering()
		logger.info("prob={} edges={}".format(p, edges))
		if edges >= min_edges and edges <= max_edges:
			break
			# avgc = graph.average_clustering()
			# logger.info("** avgc={}".format(avgc))
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

	graph.setLogger(logger)
	ret = experiment(graph, seed, rounds)
	elapsed = time.time() - start_time
	logger.info("# iteration %d done in %f" % (i+1, time.time() - start_time))
	gc.collect()
	#return [(lin_max_seed, max_lin_influenced), (eigenc_max_seed, max_eigenc_influenced), (bet_max_seed, max_bet_influenced)]
	return ret.append((edges, avgc, elapsed))



result = []

def on_shutdown():
	global result
	dill.dump(result, open('rd1', 'wb'))

if __name__ == '__main__':
	atexit.register(on_shutdown)

	# logFormatter = logging.Formatter("%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
	# logger = logging.getLogger()
	# logger.setLevel(logging.DEBUG)
	# fileHandler = logging.FileHandler('RD_log.log',mode='w')
	# fileHandler.setFormatter(logFormatter)
	# logger.addHandler(fileHandler)
	# consoleHandler = logging.StreamHandler()
	# consoleHandler.setFormatter(logFormatter)
	# logger.addHandler(consoleHandler)
	# install_mp_handler(logger)

	logging.basicConfig(format="%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
	logger=logging.getLogger()
	logger.setLevel(logging.DEBUG)
	logger.info("starting...")

	start = time.time()
	random.seed(datetime.now())	
	seed = 100

	# lin_max_values = []
	# eigenc_max_values = []
	# bet_max_values = []
	
	n = 100
	#pbar = pyprind.ProgBar(n,stream=1)
	with ProcessPoolExecutor(5) as executor:
		futures = { executor.submit(worker_task, i) for i in range(n) }
		for future in concurrent.futures.as_completed(futures):
			ret = future.result()
			# lin_max_values.append(ret[0])
			# eigenc_max_values.append(ret[1])
			# bet_max_values.append(ret[2])
			result.append(ret)
			#pbar.update()

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
	logger.info("program complete in {}".format(time.time() - start))

