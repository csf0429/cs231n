import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  for i in xrange(num_train):
      scores = X[i].dot(W)
      correct_class_score = scores[y[i]]
      f = scores - np.max(scores)
      # print "f",f.shape
      p = np.exp(f[y[i]])/np.sum(np.exp(f))
      #print "p.shape:",p.shape
      loss += -np.log(p)
      for j in xrange(num_classes):
          dW[:,j] += np.exp(f[j])/np.sum(np.exp(f))*X[i].T - (j==y[i])*X[i].T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  f = scores - np.max(scores)
  #print scores
  #print f
  correct_class_score = f[np.arange(num_train),y]
  p = np.exp(f)/np.sum(np.exp(f),axis=1,keepdims = True)
  #print "p shape:",p.shape
  
  loss = np.sum(-np.log(p[np.arange(num_train),y]))
  
  ind = np.zeros_like(p)
  ind[np.arange(num_train),y] = 1
  
  dW = X.T.dot(p - ind)
  loss /= num_train
  loss += 0.5 *reg *np.sum(W*W)
  
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

