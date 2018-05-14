from numpy.random import random, normal
from numpy import array, matrix, mean, allclose, ones, diag
from scipy.stats import special_ortho_group
from scipy.spatial import cKDTree
from numpy import linalg
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# Set cardinality
n = 100
# Generate set T
target = random((n, 3))
target = array(target)
# Generate rotation matrix
R = special_ortho_group.rvs(3)
R = matrix(R)
# Generate shift vector
b = random((3, 1)) * 2 - 1
# Generate random noise
xi = normal(0, .005, target.shape)
# Transform set T into set S
source = array((R * target.T).T + b.T + xi)

# Plot sets
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(-target[:,0], -target[:,2], target[:,1], c='b')
ax.scatter(-source[:,0], -source[:,2], source[:,1], c='r')
pyplot.show()

# Function for finding labeling
tree = cKDTree(target)
def find_labeling(target, source):
    return target[tree.query(source)[1]]

# Function for finding transformation
def find_transformation(nearest_neighbours, source):
    centroid_target = mean(nearest_neighbours, axis=0)
    centroid_source = mean(source, axis=0)
    H = ((nearest_neighbours - centroid_target).T).dot(source - centroid_source)
    U, S, V = linalg.svd(H)
    R = ((U.T).dot(diag([1, 1, linalg.det(U.T.dot(V.T))]))).dot(V.T)
    t = centroid_target - R.dot(centroid_source.T).T
    return R.dot(source.T).T + t

# Function with ICP algorithms
def icp(target, source, max_iterations=400):
    labelings = []
    transformations = []
    labelings.append(find_labeling(target, source))
    transformations.append(find_transformation(labelings[0], source))
    i = 1
    while (len(labelings) < 2 or not allclose(labelings[-1], labelings[-2])) and i < max_iterations:
        i += 1
        labelings.append(find_labeling(target, transformations[-1]))
        transformations.append(find_transformation(labelings[-1], source))
    print 'Number of iterations:', i
    return transformations

# Running the algorithm
result = icp(data, source)[-1]

# Plot result
fig = pyplot.figure()
ax = Axes3D(fig)
ax.scatter(-data[:,0], -data[:,2], data[:,1], c='b')
ax.scatter(-result[:,0], -result[:,2], result[:,1], c='r')
pyplot.show()
