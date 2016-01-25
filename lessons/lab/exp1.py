#!/usr/bin/python

import sys
from lesson1 import randomGraph, diameter, averageClustering

rep = int(sys.argv[1]) #number of repetitions
nodes = int(sys.argv[2]) #number of nodes
sep = float(sys.argv[3]) #experimets for p = sep, 2sep, 3sep, ..., 1
p = sep
while p < 1:
	diam = 0
	aclust = 0
	not_connected = 0
	for i in range(rep): # for each p we make rep simulations
		graph = randomGraph(nodes,p)
		tmp_diam=diameter(graph)
		if tmp_diam > 0: #if diameter is -1, then the graph is not connected
			diam += tmp_diam
		else:
			not_connected += 1
		aclust += averageClustering(graph)
	print(p, not_connected, diam/(rep-not_connected), aclust/rep) #we show average values
	p += sep