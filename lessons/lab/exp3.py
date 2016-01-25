#!/usr/bin/python

import sys
from lesson3 import readGraph, CounterUtility, diameter

graph = readGraph(sys.argv[1])
nodes = len(graph)
print("Nodes:",nodes)
edges, triangles, paths, aclust = CounterUtility(graph)
print("Edges:",edges)
print("Triangles:",triangles)
print("Average Clustering:",aclust)
print("Fraction of closed triangles:",float(triangles)/(paths-2*triangles))
cnodes,cedges,diam = diameter(graph)
print("Nodes in the largest component:",cnodes)
print("Edges in the largest component:",cedges)
print("Diameter:",diam)