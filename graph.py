# DELINIATOR DATA

# DELINIATOR:
#   DATAPOINT
#   BOUNDINGBOX

# DATA:
#   DATAPOINT:
#       X Y Z
#   BOUNDINGBOX:
#       MINX MINY MINZ MAXX MAXY MAXZ


import sys
if len(sys.argv) < 2:
    print("Invalid filename")

import random
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

points = []
lines = []
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


file = open(sys.argv[1], 'r')
for line in file:
    values = line.rstrip().split(' ')
    if values[0] == 'DATAPOINT':
        points.append(tuple(values[1:]))
    if values[0] == 'BOUNDINGBOX':
        xmin, ymin, zmin, xmax, ymax, zmax = values[1:]
        a = [(xmin, ymin, zmin), (xmax, ymin, zmin)]
        b = [(xmin, ymax, zmin), (xmin, ymin, zmin)]
        c = [(xmin, ymax, zmin), (xmax, ymax, zmin)]
        d = [(xmax, ymax, zmin), (xmax, ymin, zmin)]
        e = [(xmin, ymin, zmax), (xmax, ymin, zmax)]
        f = [(xmin, ymax, zmax), (xmin, ymin, zmax)]
        g = [(xmin, ymax, zmax), (xmax, ymax, zmax)]
        h = [(xmax, ymax, zmax), (xmax, ymin, zmax)]
        i = [(xmin, ymax, zmin), (xmin, ymax, zmax)]
        j = [(xmax, ymax, zmin), (xmax, ymax, zmax)]
        k = [(xmax, ymin, zmin), (xmax, ymin, zmax)]
        l = [(xmin, ymin, zmin), (xmin, ymin, zmax)]
        if a not in lines:
            lines.append(a)
        if b not in lines:
            lines.append(b)
        if c not in lines:
            lines.append(c)
        if d not in lines:
            lines.append(d)
        if e not in lines:
            lines.append(e)
        if f not in lines:
            lines.append(f)
        if g not in lines:
            lines.append(g)
        if h not in lines:
            lines.append(h)
        if i not in lines:
            lines.append(i)
        if j not in lines:
            lines.append(j)
        if k not in lines:
            lines.append(k)
        if l not in lines:
            lines.append(l)
        

px = [float(x[0]) for x in points]
py = [float(x[1]) for x in points]
pz = [float(x[2]) for x in points]

# Draw points
ax.scatter3D(px, py, pz, c=pz)

# Draw bounding boxes
#color = colors[random.randint(0, len(colors)-1)]
#num_lines = 0
for line in lines:
    #num_lines += 1
    #if num_lines == 12:
    #    num_lines = 0
    #    color = colors[random.randint(0, len(colors)-1)]
    x = [float(x[0]) for x in line]
    y = [float(x[1]) for x in line]
    z = [float(x[2]) for x in line]
    ax.plot3D(x, y, z, 'gray') 

plt.show()
