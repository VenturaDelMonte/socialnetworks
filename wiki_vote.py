from directed_graph import DirectedGraph
from directed_graph import topk
import sys
import matplotlib.pyplot as plt
from experiment import take_measures
from experiment import simulate
import dill
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import logging

def worker_task(graph, top_eigenc, top_bet, top_lin, seed, rounds):
	l, e, b = simulate(graph, top_eigenc, top_bet, top_lin, seed, rounds)
	return [l, e, b]

result = []
if __name__ == '__main__':
		
	rounds = 10
	seed = 100

	with DirectedGraph.from_filename('wiki-Vote.txt') as graph:
		top_eigenc, top_bet, top_lin = take_measures(graph, seed)
		n = 100
		#lin_max_values, eigenc_max_values, bet_max_values = experiment(graph, seed, rounds)
		with ProcessPoolExecutor() as executor:
			futures = { executor.submit(worker_task, graph, top_eigenc, top_bet, top_lin, seed, rounds) for i in range(n) }
			for future in concurrent.futures.as_completed(futures):
				ret = future.result()
				# lin_max_values.append(ret[0])
				# eigenc_max_values.append(ret[1])
				# bet_max_values.append(ret[2])
				print(ret)
				result.append(ret)


		dill.dump(result, open('ww', 'wb'))

		# print('# Lin\tEigenvector\tBetweenness')
		# for x, y, z in zip(lin_max_values, eigenc_max_values, bet_max_values):
		# 	print("{}::{}::{}".format(x,y,z))

		# sys.stdout.flush()

		# fig, ax = plt.subplots()

		# bar_width = 0.35

		# opacity = 0.4

		# rects1 = plt.bar(1, max_lin_influenced, width=bar_width, alpha=opacity, color='b', label='Lin')

		# rects2 = plt.bar(2, max_eigenc_influenced, width=bar_width, alpha=opacity, color='r', label='Eigenvector')

		# rects3 = plt.bar(3, max_bet_influenced, width=bar_width, alpha=opacity, color='y', label='Betweenness')

		# plt.xlabel('Centrality Measures')
		# plt.ylabel('Influenced')
		# plt.title('Influenced Comparison')
		# plt.xticks([1.2,2.2,3.2], ('L', 'E', 'B'))

		# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

		# plt.tight_layout()

		# plt.savefig('test.png')
		# plt.show()