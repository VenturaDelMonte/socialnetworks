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
#from process_affinity_pool import ProcessPoolExecutorWithAffinity as ProcessPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import logging
import os

logger = None

def worker_task(i):
	global logger
	if logger is None:
		logging.basicConfig(format="%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
		fileHandler = logging.FileHandler('PA_log.log.{}'.format(os.getpid()),mode='w')
		logger=logging.getLogger()
		#fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
		logger.addHandler(fileHandler)
		logger.setLevel(logging.DEBUG)
	random.seed(datetime.now().timestamp() * i)
	start_time = time.time()
	rounds = 10
	NODES = 7115
	min_edges = 75000
	max_edges = 125000
	incr = 0.001
	p = random.uniform(0.35, 0.45) # probability
	seed = 100
	d = int(random.randint(min_edges, max_edges) / NODES)
	ret = None
	avgc = 0
	edges = 0
	with DirectedGraph.preferentialAttachment(NODES, d, p) as graph:
		edges = graph.size()[1]
		avgc = graph.toUndirect().average_clustering()
		ret = experiment(graph, seed, rounds)
#		print("# iteration %d done in %f" % (i+1, time.time() - start_time))

	elapsed = time.time() - start_time
	ret.append((edges, avgc, elapsed, p, d))
	logger.info("# iteration %d done in %f" % (i+1, elapsed))
	logger.info("# {}".format(ret))
	sys.stdout.flush()
	return ret


if __name__ == '__main__':
	logging.basicConfig(format="%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
	logger=logging.getLogger()
	logger.setLevel(logging.DEBUG)
	fileHandler = logging.FileHandler('PA_log.log',mode='w')
	#fileHandler.setFormatter(logging.Formatter("%(asctime)s [%(process)-4.4s--%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"))
	logger=logging.getLogger()
	logger.addHandler(fileHandler)
	logger.info("starting...")

	start = time.time()
	random.seed(datetime.now())	
	
	seed = 100
	'''
	lin_max_values = []
	eigenc_max_values = []
	bet_max_values = []
	'''
	
	result = []

	n = 64
	with concurrent.futures.ProcessPoolExecutor() as executor:
		futures = { executor.submit(worker_task, i) for i in range(n) }
		for future in concurrent.futures.as_completed(futures):
			ret = future.result()
			# lin_max_values.append(ret[0])
			# eigenc_max_values.append(ret[1])
			# bet_max_values.append(ret[2])
			result.append(ret)
			logger.info("{}+{}".format(result,ret))
			print("{}+{}".format(result,ret))
			sys.stdout.flush()

	dill.dump(result, open('pa', 'wb'))


	'''
	print('# Lin\tEigenvector\tBetweenness')
	for x, y, z in zip(lin_max_values, eigenc_max_values, bet_max_values):
		print("{}-{} {}-{} {}-{}".format(x[0], x[1], y[0], y[1], z[0], z[1]))

	sys.stdout.flush()

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

	plt.savefig('pa.png')
	plt.show()
	'''
	logger.info("program complete in {}".format(time.time() - start))