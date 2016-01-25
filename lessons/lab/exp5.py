#!/usr/bin/python

import sys
from lesson3 import readGraph
from lesson5 import top

simple = dict()
simple[0] = {1}
simple[1] = {0,2}
simple[2] = {1,3}
simple[3] = {2,4}
simple[4] = {3}
sbtop, sbvalues = top(simple,5,"b",0.001)
print(sbtop, sbvalues)
setop, sevalues = top(simple,5,"e",0.001)
print(setop, sevalues)


graph = readGraph(sys.argv[1])
btop, bvalues = top(graph,sys.argv[2],"b",sys.argv[3])
print(btop, bvalues)
etop, evalues = top(graph,sys.argv[2],"e",sys.argv[3])
print(etop, evalues)