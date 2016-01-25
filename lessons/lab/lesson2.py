#!/usr/bin/python

import random
import math

# n is the number of nodes
# r is the radius of each node (a node u is connected with each other node at distance at most r) - strong ties
# k is the number of random edges for each node u - weak ties
def WSGridGraph(n, r, k):
	line = int(math.sqrt(n))
	graph = dict()
	#Initialization
	for i in range(line): #i represents the grid row
		for j in range(line): #j represents the grid coloumn
			graph[i*line+j] = set() #Each node is identified by a number in [0, n - 1]

	#For each node u, we add an edge to each node at distance at most r from u
	for i in range(line):
		for j in range(line):
			for x in range(r+1): # x is the horizontal offset
				for y in range(r+1-x): # y is the vertical offset. The sum of offsets must be at most r
					if x + y > 0: # The sum of offsets must be at least 1
						if i + x < line and j + y < line:
							graph[i*line+j].add((i+x)*line+(j+y))
							graph[(i+x)*line+(j+y)].add(i*line+j)
							# We do not consider(i+x,j-y) (i-x,j+y) and (i-x,j-y) since the edge between these nodes and (i,j) has been already added at previous iterations

			#For each node u, we add a node to k randomly chosen nodes
			for h in range(k):
				s = random.randint(0,n-1)
				if s != i*line+j:
					graph[i*line+j].add(s)
					graph[s].add(i*line+j)
	return graph

# n is the number of nodes
# r is the radius
# k is the number of random edges
def WS2DGraph(n, r, k):
	line = int(math.sqrt(n))
	graph = dict()
	#Initialization
	for i in range(n):
		x = random.random()
		y = random.random()
		graph[i] = dict()
		graph[i]["x"] = x*line
		graph[i]["y"] = y*line
		graph[i]["list"] = set()

	#For each node u, we add an edge to each node at distance at most r from u
	for i in range(n):
		for j in range(i+1,n):
			dist=math.sqrt((graph[i]["x"]-graph[j]["x"])**2 + (graph[i]["y"]-graph[j]["y"])**2) # Euclidean distance between i and j
			if dist <= r:
				graph[i]["list"].add(j)
				graph[j]["list"].add(i)

		#For each node u, we add a node to k randomly chosen nodes
		for h in range(k):
			s = random.randint(0,n-1)
			if s != i:
				graph[i]["list"].add(s)
				graph[s]["list"].add(i)
	return graph

def countEdges(graph):
	edges = 0
	for k in graph.keys():
		edges += len(graph[k])
	return int(edges/2)