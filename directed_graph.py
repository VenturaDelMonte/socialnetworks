#!/usr/bin/python

#author: Ventura Del Monte
#purpose: social networks class project
#last edit: 02/01/2016

from __future__ import division
from collections import defaultdict
import random
import itertools
import math
import sys
import time


def euclidean_distance(a, b):
	return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

def topk(cb, k):
	return sorted(cb.items(), key = lambda x : x[1], reverse=True)[:k]

def lastk(cb, k):
	return sorted(cb.items(), key = lambda x : x[1], reverse=True)[-k:]

def cb_max(cb):
	node = -1
	maxvalue = -1
	for v in cb:
		if cb[v] > maxvalue:
			maxvalue = cb[v]
			node = v
	return (node, maxvalue)


'''
	base class for implementing undirected/directed (optionally weighted) graphs
'''
class DirectedGraph: 
	
	def __init__(self, graph = {}):
		self.nodes = graph
		self.__edgesCount = 0
		self.__isWS = False
		self.__isDirected = True

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		self.nodes = None
		self.__edgesCount = None
		self.__isWS = None

	def setWS(self, mode):
		self.__isWS = mode

	def isWS(self):
		return self.__isWS

	def setDirected(self, mode):
		self.__isDirected = mode

	def isDirected(self):
		return self.__isDirected

	def add_vertex(self, vid):
		if not vid in self.nodes:
			self.nodes[vid] = dict()
			self.nodes[vid]['list'] = set()

	def add_edge(self, fromId, toId):
		self.add_vertex(fromId)
		self.add_vertex(toId)
		if not self.has_edge(fromId, toId):
			self.nodes[fromId]['list'].add(toId)
			self.__edgesCount += 1

	def countTriangles(self):
		triangles = 0
		for t in itertools.combinations(self.nodes.keys(), 3):
			for x in itertools.permutations(t):
				#if x[1] in self.nodes[x[0]] and x[2] in self.nodes[x[1]] and x[2] in self.nodes[x[0]]:
				if self.has_edge(x[0], x[1]) and self.has_edge(x[1], x[2]) and self.has_edge(x[0], x[2]):
					triangles += 1
					break
		return triangles

	def countEdges(self):
		return self.__edgesCount

	def size(self):
		return len(self.nodes), self.__edgesCount

	def diameter(self):
		n = len(self.nodes)
		diameter = -1
		for u in self.nodes: #check the max distance for every node u through BFS
			visited = [] # keep a list of visited nodes to check if the graph is connected
			max_distance = 0 # keep the max distance from u
			queue = [u]
			distances = {u : 0}
			while len(queue) > 0:
				s = queue.pop(0)
				visited.append(s)
				for v in self.nodes[s]['list']:
					if not v in distances:
						queue.append(v)
						distances[v] = distances[s] + 1
						if distances[v] > max_distance:
							max_distance = distances[v]

			if len(visited) < n: # graph is not connected
				break
			elif max_distance > diameter:
				diameter = max_distance

		return diameter	

	def has_edge(self, u, v):
		return v in self.nodes[u]['list']

	def average_clustering(self):
		n = len(self.nodes)
		total = 0.0
		for i in self.nodes:
			triangles = 0.0
			neighbors = self.nodes[i]['list']
			neighbors_count = len(neighbors)
			pairs = float(neighbors_count * (neighbors_count - 1) / 2) #number of pairs of neighbors of node i
			if pairs > 0:
				for t in itertools.permutations(neighbors, 2):
					#number of pairs of neighbors of node i that are adjacent
					if self.has_edge(t[0], t[1]):
						triangles += 1 
				if self.isDirected():
					total += triangles / pairs  
				else:
					total += triangles / (2 * pairs)   # triangles / neighbors is the individual clustering of node i

		return total / n #the average clustering is the average of individual clusterings

	def toUndirect(self):
		ret = DirectedGraph()
		ret.setDirected(False)
		for vertex in self.nodes:
			ret.add_vertex(vertex)
			for oth in self.nodes[vertex]['list']:
				ret.add_vertex(oth)
				ret.nodes[vertex]['list'].add(oth)
				ret.nodes[oth]['list'].add(vertex)
		return ret


	def betweenness(self, verbose = False):
		'''
			Girvan-Newman algorithm
			####
			this is an indicator of a node's centrality in a network.
			It is equal to the number of shortest paths from all vertices to all others
			that pass through that node. A node with high betweenness centrality has
			a large influence on the transfer of items through the network, 
			under the assumption that item transfer follows the shortest paths.
		'''
		cb = {i : 0 for i in self.nodes} #betweenness centrality

		for i in self.nodes: #check the max distance for every node i
			visited = [] #keep a list of visited nodes to check if the graph is connected
			P = {}
			sigma = {}
			distances = {}

			# for j in self.nodes:
			# 	P[j] = [] # j's parents list
			# 	sigma[j] = 0
			# 	distances[j] = -1

			P = defaultdict(lambda: [])	
			sigma = defaultdict(lambda: 0)	
			distances = defaultdict(lambda: -1)	 

			sigma[i] = 1
			queue = [i]
			distances[i] = 0

			# BFS
			while len(queue) > 0:
				s = queue.pop(0)
				visited.append(s)
				for j in self.nodes[s]['list']: # foreach neighbor j of s do
					# j found for the first time?
					if distances[j] < 0:
						queue.append(j)
						distances[j] = distances[s] + 1
					# shortest path to j via s?
					if distances[j] == distances[s] + 1:
						sigma[j] += sigma[s]
						P[j].append(s)
			
			# bottom-up
			delta = defaultdict(lambda: 0) #{j: 0 for j in self.nodes}
			# visited returns vertices in order of non-increasing distance from s
			while len(visited) > 0:
				w = visited.pop()
				if len(P[w]) > 0:
					for v in P[w]:
						delta[v] += (float(sigma[v]) / sigma[w]) * (1.0 + delta[w])
				if w != i:
					cb[w] += delta[w]
		return cb


	def eigenvector_centrality(self, epsilon = 1e-03):
		'''
			Eigenvector centrality is a measure of the influence of a node in a network. 
			It assigns relative scores to all nodes in the network based on the concept 
			that connections to high-scoring nodes contribute more to the score of the node
			in question than equal connections to low-scoring nodes.
		'''
		diff_sum = float('+inf')
		keys = self.nodes.keys()
		cscores = dict.fromkeys(keys, 1.0)

		#i = 0
		while (diff_sum >= epsilon):
			# print i, '. eigenvector_centrality: current error = ', diff_sum
			old_cscores_summation = sum(cscores.values())
			# print 'old_cscores_summation', old_cscores_summation
			tmp = {}
			for node in keys:
				tmp[node] = cscores[node]

				for neigh in self.nodes[node]['list']:
					tmp[node] += cscores[neigh]

			max_neigh_cscore = max(tmp.values())

			# print tmp, max_neigh_cscore
			for node in keys:
				cscores[node] = tmp[node] / max_neigh_cscore

			diff_sum = abs(sum(cscores.values()) - old_cscores_summation)
			#i += 1

		return diff_sum, cscores

	def distance(self, u, v):
		queue = [u]
		distances = {u : 0}
		while len(queue) > 0:
			s = queue.pop(0)
			d = distances[s] + 1
			if len(self.nodes[s]['list']) <= 0:
				continue
			for j in self.nodes[s]['list']:
				if not j in distances:
					queue.append(j)
					distances[j] = d
				if j == v:
					return distances[j]
		return float('+inf')


	def lin_index(self):
		'''
			it retuns lin's index, which weights closeness using the 
			square of the number of coreachable nodes.
			Nodes with an empty coreachable set have centrality 1 by definition.
			The rationale behind this definitions is the following: first,
			we consider closeness not the inverse of a sum of distances,
			but rather the inverse of the average distance, which entails a
			first multiplication by the number of coreachable nodes. This
			change normalizes closeness across the graph. Now, however,
			we want nodes with a larger coreachable set to be more
			important, given that the average distance is the same, so we
			multiply again by the number of coreachable nodes.
			Lin’s index was somewhat surprisingly ignored in the following
			literature. Nonetheless, it seems to provide a reasonable
			solution for the problems caused by the definition of closeness.
			----------------------------------------------------------------
			The intuition behind closeness is that nodes with a large sum of
			distances are peripheral. By reciprocating the sum, nodes with
			a smaller denominator obtain a larger centrality. We remark
			that for the above definition to make sense, the graph needs
			be strongly connected. Lacking that condition, some of the
			denominators will be 1, resulting in a rank of zero for all
			nodes which cannot coreach the whole graph. One further assumes
			that nodes with an empty coreachable set have centrality 0 by definition.
			These apparently innocuous adjustments, however, introduce a
			strong bias toward nodes with a small coreachable set.
			----------------------------------------------------------------
			the trick to speed up this method consists of storing the
			distances between v and every node reachable by this latter
			using a BFS visit. 
		'''
		L = {}
		D = {} # D[u][v] - dist between u and v

		for v in self.nodes:
			D[v] = {v : 0}
			queue = [v]
			while len(queue) > 0:
				curr = queue.pop(0)
				d = D[v][curr] + 1
				if len(self.nodes[curr]['list']) <= 0:
					continue
				for next in self.nodes[curr]['list']:
					if not next in D[v]:
						queue.append(next)
						D[v][next] = d
					elif D[v][next] > d:
						D[v][next] = d

		for u in self.nodes:
			reachable = 0.0
			closeness = 0.0
			for v in self.nodes:
				if v != u and u in D[v]:
					reachable += 1
					closeness += D[v][u]
			if closeness <= 0:
				L[u] = 1
			else:
				L[u] = (reachable ** 2) / closeness
			self.nodes[u]['lin'] = L[u]
			self.nodes[u]['closeness'] = closeness
			#print("node {} c={} l={} r={}".format(u, closeness, L[u], reachable))
		return L



	# def lin_indexFW(self):
	# 	# it includes Floyd-Warshall to compute all pairs distances
	# 	max = float('+inf')
	# 	lin = {}
		
	# 	distances = {}
	# 	for i in self.nodes:
	# 		for j in self.nodes:
	# 			if not i in distances:
	# 				distances[i] = defaultdict(lambda: max)
	# 				distances[i][i] = 0
	# 			if not j in distances:
	# 				distances[j] = defaultdict(lambda: max)
	# 				distances[j][j] = 0
	# 			if self.has_edge(i, j):
	# 				distances[i][j] = 1
	# 			if self.has_edge(j, i):
	# 				distances[j][i] = 1

		
	# 	for k in self.nodes:
	# 		if len(self.nodes[k]['list']) == 0:
	# 			continue
	# 		for i in self.nodes:
	# 			for j in self.nodes:
	# 				if i == j or j in self.nodes[i]['list']:
	# 					continue
	# 				target_dist = distances[i][k] + distances[k][j]
	# 				if distances[i][j] > target_dist:
	# 					distances[i][j] = target_dist

		
	# 	for u in self.nodes:
	# 		num = den = 0
	# 		for v in self.nodes: 
	# 			if u == v or len(self.nodes[v]['list']) <= 0:
	# 				continue
	# 			vu = distances[v][u] # self.distance(v, u) #
	# 			if vu < max:
	# 				num += 1
	# 				den += vu

	# 		lin[u] = (num ** 2) / den if den > 0 else 1.0
	# 		self.nodes[u]['lin'] = lin[u]

	#	return lin

	def __ltm_round(self, A, neg_A):
		activated = set()
		for v in neg_A:
			sum = 0.0
			for j in self.nodes[v]['list']:
				if j in A:
					sum += self.nodes[j]['LTMthreshold']
			if sum > self.nodes[v]['LTMthreshold']:
				activated.add(v)
		return activated


	def ltm(self, seeds, rounds = 1):
		'''
		Granovetter - Threshold models of collective behavior.
		'''
		vertices = len(self.nodes)
		for u in self.nodes:
			self.nodes[u]['LTMthreshold'] = random.uniform(0.0, 1.0)

		A = set(seeds)
		neg_A = set(self.nodes.keys()) - A

		round = 0
		while round < rounds and len(A) < vertices:
			activated = self.__ltm_round(A, neg_A)
			if len(activated) <= 0:
				break
			A |= activated
			neg_A -= activated
			round += 1

		return A, neg_A, round



	def __str__(self):
		ret = ""
		for vertex in self.nodes:
			ret += 'Node {}: {}\n'.format(vertex, [x for x in self.nodes[vertex]['list']])
		return ret

	def __getitem__(self, key):
		return self.nodes[key]['list'] if not self.__isWS else self.nodes[key]

	@staticmethod
	def randomDirectedGraph(n, p):
		graph = DirectedGraph({k : {'list': set()} for k in range(n)})
		'''
		for i in range(n):
			for j in range(n):
				if i == j:
					continue
				if random.random() <= p:
					graph.add_edge(i, j)
				if random.random() <= p:
					graph.add_edge(j, i)
		'''
		for edge in itertools.permutations(graph.nodes, 2):
			if random.random() <= p:
				graph.add_edge(edge[0], edge[1])
		return graph

	@staticmethod
	def randomDirectedBalancedGraph(n, p, edges):
		graph = DirectedGraph({k : {'list': set()} for k in range(n)})
		while edges > 0:
			while True:
				u = random.randint(0, n - 1)
				v = random.randint(0, n - 1)
				if (u != v) and graph.has_edge(u, v):
					break
			# random graph property
			if random.random() <= p:
				graph.add_edge(u, v)
				edges -= 1
		return graph	

	@staticmethod	
	def GenWSGridGraph(n, m, r, k, q=2):
		'''
		@param n: number of nodes
		@param m: number of edges
		@param r : radius of each node (a node u is connected with each other node at distance at most r) - strong ties
		@param k : number of random edges for each node u - weak ties [list] e.g. [0, 5, 10]
		'''
		line = int(math.sqrt(n))
		#Initialization
		graph = DirectedGraph({v : {'list': set()} for v in range(n)})

		while m > 0:
			i = random.randint(0, line-1)
			#j = random.randint(0, line-1)
			#For each node u, we add an edge to each node at distance at most r from u
			for j in range(line):
				for x in range(r+1): # x is the horizontal offset
					for y in range(r+1-x): # y is the vertical offset. The sum of offsets must be at most r
						if x + y > 0: # The sum of offsets must be at least 1
							target = i * line + j
							if i + x < line:
								dest = (i + x) * line
								destp = dest + (j + y)
								if j + y < line and destp not in graph[target]:
									graph[target].add(destp)
									m -= 1
									if m == 0:
										return graph
								destm = dest + (j - y)
								if j - y >= 0 and destm not in graph[target]:
									graph[target].add(destm)
									m -= 1
									if m == 0:
										return graph
							if i - x >= 0:
								dest = (i - x) * line
								destp = dest + (j + y)
								if j + y < line and destp not in graph[target]:
									graph[target].add(destp)
									m -= 1
									if m == 0:
										return graph
								destm = dest + (j - y)
								if j - y >= 0 and destm not in graph[target]:
									graph[target].add(destm)
									m -= 1
									if m == 0:
										return graph

			#For each node u, we add a node to k randomly chosen nodes
			weak_ties = random.choice(k)
			while weak_ties > 0:
				xt = random.randint(0, line - 1)
				yt = random.randint(0, line - 1)
				target = i * line + j
				dest = xt * line + yt
				if xt * line + yt > n - 1:
					continue 
				if xt != i and yt != j and dest not in graph[target] and random.random() <= (1.0 / (euclidean_distance((xt, yt), (i,j)) ** q)):
					graph[target].add(dest)
					m -= 1
					weak_ties -= 1
					if m == 0:
						return graph
		return graph

	@staticmethod
	def preferentialAttachment(n, d, p):
		'''
		In a Prefential Attachment Graph when a node j is created, its out-links are produced in the following way:
    		- with a probability p, choose a node i uniformly at random and creates an edge from j to i
    		- with a probability 1-p, choose a node l with a proability that is proportional to the l's current in-degree and creates an edge from j to l.
    
    	This is the idea behind the 'Rich get Richer' phenomen

		'''
		graph = DirectedGraph()
		for v in range(n):
			graph.add_vertex(v)
		I = []
		for v in range(n):
			for i in range(d):
				if random.random() <= p:
					j = random.randint(0, n - 1)
					if v == j:
						continue
					graph.add_edge(v, j)
					I.append(j)
				elif len(I) > 1:
					l = v
					while True:
						l = random.choice(I)
						if l != v:
							break
					graph.add_edge(v, l)
					I.append(l)
		return graph

	@staticmethod		
	def WS2DGraph(n, m, r, k):
		'''
		@param n: number of nodes
		@param m: number of edges
		@param r: radius
		@param k: number of random edges [list] e.g. [0, 5, 10]
		'''
		line = int(math.sqrt(n))
		graph = DirectedGraph({v : {'list': set()} for v in range(n)})
		graph.setWS(True)
		#Initialization
		for i in range(n):
			x = random.random()
			y = random.random()
			graph[i]["x"] = x * line
			graph[i]["y"] = y * line

		#For each node u, we add an edge to each node at distance at most r from u
		tot = m
		while m > 0:
			i = random.randint(0, n - 1)
			if len(graph[i]['list']) <= 0:
				for j in range(n):
					#dist = math.sqrt((graph[i]["x"] - graph[j]["x"])**2 + (graph[i]["y"] - graph[j]["y"])**2) # Euclidean distance between i and j
					dist = euclidean_distance((graph[i]["x"], graph[i]["y"]), (graph[j]["x"], graph[j]["y"]))
					if dist <= r and i != j and j not in graph[i]['list']:
						graph[i]['list'].add(j)
						m -= 1
						if m == 0:
							break

				#For each node u, we add a node to k randomly chosen nodes
				weak_ties = random.choice(k)
				while weak_ties > 0:
					s = random.randint(0, n - 1)
					if s == i or s in graph[i]['list']:
						continue
					graph[i]['list'].add(s)
					m -= 1
					weak_ties -= 1
					if m == 0:
						break

		graph.__edgesCount = tot - m
		return graph

	@staticmethod	
	def from_filename(filename, mode = 'd'):
		graph = DirectedGraph()
		if mode == "u":
			graph.setDirected(False) 
		with open(filename) as fp:
			for line in fp:
				if '#' not in line:
					u, v = map(int, line.split())
					graph.add_edge(u, v)
					if mode == "u":
						graph.add_edge(v, u)
		return graph