import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
 
    self.X_train = X
    self.y_train = y

  def predict(self, X, k=1, num_loops=0):
  
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)



  def compute_distances_no_loops(self, X):

    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))

    M = np.dot(X, self.X_train.T)
    te = np.square(X).sum(axis = 1)
    tr = np.square(self.X_train).sum(axis = 1)
    dists = np.sqrt(-2*M+tr+np.matrix(te).T)
  
    return dists

  def predict_labels(self, dists, k=1):
 
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
 
      labels = self.y_train[np.argsort(dists[i,:])].flatten()
      #print 'labels shape:',labels.shape

      closest_y = labels[:k]

      u, indices = np.unique(closest_y,return_inverse = True)
      y_pred[i] = u[np.argmax(np.bincount(indices))]

    return y_pred

