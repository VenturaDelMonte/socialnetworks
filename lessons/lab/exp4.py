#!/usr/bin/python

import sys
from lesson3 import readGraph
from lesson4 import top

simple = dict()
simple[0] = {1}
simple[1] = {0,2}
simple[2] = {1,3}
simple[3] = {2,4}
simple[4] = {3}
stop, svalues = top(simple,1)
print(stop, svalues)

graph = readGraph(sys.argv[1])
print(top(graph,sys.argv[2]))