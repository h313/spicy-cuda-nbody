# DELINIATOR DATA

# DELINIATOR:
#   DATAPOINT
#   BOUNDINGBOX

# DATA:
#   DATAPOINT:
#       X Y Z
#   BOUNDINGBOX:
#       MINX MINY MINZ MAXX MAXY MAXZ


from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

points = []
lines = []

file = open('output.txt', 'r')
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
        lines.append(a)
        lines.append(b)
        lines.append(c)
        lines.append(d)
        lines.append(e)
        lines.append(f)
        lines.append(g)
        lines.append(h)
        lines.append(i)
        lines.append(j)
        lines.append(k)
        lines.append(l)


print(lines)

#points = np.asarray(points)

px = [float(x[0]) for x in points]
py = [float(x[1]) for x in points]
pz = [float(x[2]) for x in points]

# Draw points
ax.scatter3D(px, py, pz, c=pz)

# Draw bounding boxes
for line in lines:
    x = [float(x[0]) for x in line]
    y = [float(x[1]) for x in line]
    z = [float(x[2]) for x in line]
    ax.plot3D(x, y, z, 'gray') 



plt.show()
