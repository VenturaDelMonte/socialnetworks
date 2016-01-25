#!/usr/bin/python

import sys
from lesson6 import simulate_game

graph1 = dict()
graph1['r'] = {'s','v','w'}
graph1['s'] = {'r','w','u'}
graph1['v'] = {'r','w','t'}
graph1['w'] = {'r','s','v','u','t'}
graph1['u'] = {'s','w','t'}
graph1['t'] = {'w','v','u'}
print(simulate_game(graph1, 3, 2, {'v','w'}))

graph2 = dict()
graph2[1] = {2,3}
graph2[2] = {1,3,6}
graph2[3] = {1,2}
graph2[4] = {5,6,7}
graph2[5] = {4,7,8}
graph2[6] = {2,4,9}
graph2[7] = {4,5,8,9,10}
graph2[8] = {5,7,10,14}
graph2[9] = {6,7,10,11}
graph2[10] = {7,8,9,12}
graph2[11] = {9,12,15}
graph2[12] = {10,11,13,15,16}
graph2[13] = {12,14,16,17}
graph2[14] = {8,13,17}
graph2[15] = {11,12,16}
graph2[16] = {12,13,15,17}
graph2[17] = {13,14,16}
print(simulate_game(graph2, 3, 2, {7,8}))

graph3 = dict()
graph3['z'] = {'x'}
graph3['x'] = {'z','v'}
graph3['v'] = {'x','r'}
graph3['r'] = {'v','s'}
graph3['s'] = {'r','u'}
graph3['u'] = {'s','w'}
graph3['w'] = {'u','y'}
graph3['y'] = {'w'}
print(simulate_game(graph3, 2, 3, {'r','s'}))