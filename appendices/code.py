from numpy import genfromtxt, mean, linalg, allclose, diag, matrix, array, cos, sin, pi
from numpy.random import normal, random, uniform
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.stats import special_ortho_group

# Set cardinality
n = 100
# Generate set T
target = random((n, 3))
target = array(target)
# Generate rotation matrix
R = special_ortho_group.rvs(3)
R = matrix(R)
# Generate shift vector
b = random((3, 1))
# Generate random noise
xi = normal(0, .001, target.shape)
# Transform set T into set S
source = array((R.dot(data.T)).T  + b.T + xi)

# Function for plotting sets
def visualize(source, result):
    fig = pyplot.figure(figsize=(10,10))
    ax = Axes3D(fig)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.scatter(-source[:,0], -source[:,2], source[:,1], c='b', s=20, marker='o')
    ax.scatter(-result[:,0], -result[:,2], result[:,1], c='r', s=20, marker='^')
    pyplot.show()

# Function for finding labeling
tree = cKDTree(target)
def find_labeling(target, source):
    return target[tree.query(source)[1]]

# Function for finding transformation
def find_transformation(nearest_neighbours, source):
    centroid_target = mean(nearest_neighbours, axis=0)
    centroid_source = mean(source, axis=0)
    H = ((source - centroid_source).T).dot(nearest_neighbours - centroid_target)
    U, S, V = linalg.svd(H)
    R = ((V.T).dot(diag([1, 1, linalg.det((V.T).dot(U.T))]))).dot(U.T)
    t = centroid_target - R.dot(centroid_source.T).T
    return R.dot(source.T).T + t

# Function with ICP algorithms
def icp(target, source, max_iterations=1000):
    labelings = []
    transformations = []
    labelings.append(find_labeling(target, source))
    transformations.append(find_transformation(labelings[0], source))
    i = 1
    print "Iteration ", i
    visualize(data, transformations[-1])
    while (len(labelings) < 2 or not allclose(labelings[-1], labelings[-2])) and i < max_iterations:
        i += 1
        print "Iteration ", i
        labelings.append(find_labeling(target, transformations[-1]))
        transformations.append(find_transformation(labelings[-1], source))
        visualize(data, transformations[-1])
    return transformations

# Running the algorithm
result = icp(data, source)[-1]

# Visualizing the result
visualize(data, result)
