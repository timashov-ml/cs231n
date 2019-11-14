from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in range(num_train):
        # loss
        scores = X[i, :].dot(W)
        correct_class_score = scores[y[i]]
        loss += (-correct_class_score + np.log(np.sum(np.exp(scores))))
        
        # dW
        d_correct_class_margin = np.zeros_like(dW)
        d_correct_class_margin[:, y[i]] = X[i, :]
        
        d_sum_margin = np.zeros_like(dW)
        for j in range(num_classes):
            d_sum_margin[:,j] = X[i, :] * np.exp(scores[j]) # it's kind of sum. different columns assigning
        dW += (d_sum_margin / np.sum(np.exp(scores)) - d_correct_class_margin)

    # deviding by number of guys
    loss /= num_train
    dW /= num_train
    
    # regularization
    dW += 2 * reg * W
    loss += reg * np.sum(W * W)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    
    # loss
    loss = (-np.sum(correct_class_scores) + np.sum(np.log(np.sum(np.exp(scores), axis = 1)), axis = 0))
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # gradient
    df_dscores = (np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1, 1))
    df_dscores[range(num_train), y] -= 1
    # compute softmax gradients w.r.t. to weights W, shape (num_features, num_classes)
    dW = X.T.dot(df_dscores)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
