from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N = x.shape[0]
    D = np.prod(np.array(x.shape)[1:])
    out = x.reshape((N, D)).dot(w) + b.reshape((1, -1))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, D = cache[0].shape[0], np.prod(np.array(cache[0].shape)[1:])
    
    dx = dout.dot(cache[1].T).reshape(x.shape)
    dw = cache[0].reshape((N,D)).T.dot(dout)
    db = np.sum(dout, axis = 0)
    
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = dout * (x > 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # 1. sample mean and variance
        sample_mean = np.mean(x, axis = 0, keepdims = True)
        sample_var = np.mean((x - sample_mean) ** 2, axis = 0, keepdims = True) # np.var(x, axis = 0)
        
        # 2. standart deviation
        std = np.sqrt(sample_var + eps)
        
        # 3. normalization
        x_normed = (x - sample_mean) / std # .reshape(1, -1) <- is not necessary
        
        # 4. rescale and shift
        out = gamma * x_normed + beta
        
        # 5. cache and running statistics (for test)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #cache = x, x_normed, gamma, eps, sample_mean, sample_var
        cache = sample_mean, sample_var, x_normed, gamma, eps
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        running_std = np.sqrt(running_var + eps)
        x_normed = (x - running_mean) / running_std
        out = gamma * x_normed + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # x, x_normed, gamma, eps, mean, var = cache
    # 1. get data from cache
    mean, var, x_normed, gamma, eps = cache
    M, D = x_normed.shape   
    std = np.sqrt(var + eps)
    
    # 2. simple derivatives
    dgamma = np.sum(dout * x_normed, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    # 3. x_normed derivatives
    dx_normed = dout * gamma
    
    # 4. statistics derivatives
    dvar = - np.sum(dx_normed * x_normed , axis = 0, keepdims = True) / (var + eps) / 2
    dmean = -np.sum(dx_normed, axis = 0, keepdims = True) / std - 2 * dvar * np.sum(x_normed, axis = 0) * std / M

    # 5. derivative at x
    dx = 2 * dvar * x_normed * std / M + dmean / M + dx_normed / std 
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. unpack cache
    mean, var, x_normed, gamma, eps = cache
    M, D = x_normed.shape
    std = np.sqrt(var + eps)
    
    # 2. get simple derivatives
    dgamma = np.sum(dout * x_normed, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    # 3. get derivative out of x (dx)
    dx = M * dout - np.sum(dout, axis = 0) - (x_normed / std) * np.sum(dout * x_normed * std, axis = 0)
    dx *= (1 / M) * gamma / std
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
    # 1. sample mean and variance
    mean = np.mean(x, axis = 1, keepdims = True)
    var = np.mean((x - mean) ** 2, axis = 1, keepdims=True)
        
    # 2. standart deviation
    std = np.sqrt(var + eps)
        
    # 3. normalization
    x_normed = (x - mean) / std # .reshape(1, -1) <- is not necessary
        
    # 4. rescale and shift
    out = gamma * x_normed + beta
        
    # 5. cache and running statistics (for test)
    cache = mean, var, x_normed, gamma, eps

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 1. getting data from cache
    mean, var, x_normed, gamma, eps,  = cache
    M, D = x_normed.shape
    std = np.sqrt(var + eps)
    
    # 2. getting simple derivative
    dgamma = np.sum(dout * x_normed, axis = 0)
    dbeta = np.sum(dout, axis = 0)
    
    # 3. getting derivative with respect of x_normed
    dx_normed = dout * gamma
    
    # 4. getting derivative with respect of statistics
    dvar = -0.5 * np.sum(dx_normed * x_normed, axis = 1, keepdims = True) / (var + eps)
    dmean = -np.sum(dx_normed, axis = 1, keepdims = True) / std
    dmean -= 2 * dvar * np.sum(x_normed, axis = 1, keepdims = True) * std / D

    # 5. getting derivative with respect of x
    dx = dx_normed / std + dmean / D + dvar * 2 * x_normed * std / D
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 0. stride and padding to vars
    stride, pad = conv_param['stride'], conv_param['pad']
    
    # 1. padding of x
    pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_p = np.pad(x, pad_width = pad_width, mode = 'constant', constant_values = 0)
    
    # 2. helper variables
    N, C, H, W = x_p.shape
    F, _, HH, WW = w.shape
    if (H - HH) % stride != 0 or (W - WW) % stride != 0: print("ERROR:stride and pad aren't eligible")
    H_out, W_out = 1 + (H - HH) // stride, 1 + (W - WW) // stride
    out = np.zeros((N, F, H_out, W_out))    
    
    # 3. super baseline approach: 4 loops
#     for n in range(N):
#         for f in range(F):
#             for h_out in range(H_out):
#                 for w_out in range(W_out):
#                     h_from, h_to = h_out * stride, h_out * stride + HH
#                     w_from, w_to = w_out * stride, w_out * stride + WW
#                     x_slice = x_p[n, :, h_from : h_to, w_from : w_to]
#                     out[n, f, h_out, w_out] = np.sum(x_slice * w[f, :, :, :]) + b[f]
                    
    # 4. baseline approach: 2 loops (*motivated by Yorko)
    w_flat = w.reshape((F, -1))
    for h_out in range(H_out):
        for w_out in range(W_out):
            h_from, h_to = h_out * stride, h_out * stride + HH
            w_from, w_to = w_out * stride, w_out * stride + WW
            x_slice = x_p[:, :, h_from : h_to, w_from : w_to]
            x_slice_flat = x_slice.reshape((N, -1))
            out[:, :, h_out, w_out] = x_slice_flat.dot(w_flat.T) + b
    
                    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_p, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. get vars from cache
    x_p, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']
    F, C, HH, WW = w.shape
    
    # 2. get b derivative
    db = np.sum(dout, axis = (0,2,3))
    
    # 3. initialize derivatives
    dx_p, dw = np.zeros(x_p.shape), np.zeros(w.shape)
    
    # 4. get dx and dw
    n, f, hout, wout = dout.shape
    for i in range(hout):
        for j in range(wout):
            # a. calculate indexes
            h_from, h_to = i * stride, i * stride + HH
            w_from, w_to = j * stride, j * stride + WW
            
            # b. calculate dx
            dout_slice = dout[:, :,i, j].reshape(n, f,  1, 1, 1)
            dx_slice = dout_slice * w.reshape((1,) + w.shape) 
            dx_p[:, :, h_from:h_to, w_from:w_to] += np.sum(dx_slice, axis = 1)
            
            # c. calculate dw 
            dout_slice = dout[:, :,i, j].T.reshape(f, n, 1, 1, 1)
            x_slice = x_p[:, :, h_from:h_to, w_from:w_to].reshape(1, n, C, HH, WW)
            dw += np.sum(dout_slice * x_slice, axis = 1)
    # 4c. remove padding        
    dx = dx_p[:,:,pad:-pad, pad:-pad]   

    # 5. calculate db
    db = np.sum(dout, axis = (0,2,3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # 1. cache out
    pool_h, pool_w, stride =  pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    n, c, h, w = x.shape
    h_out, w_out = 1 + (h - pool_h) // stride, 1 + (w - pool_w) // stride 
    
    # 2. initialize out
    out = np.zeros((n, c, h_out, w_out))
    
    # 3. calculate out
    for i in range(h_out):
        for j in range(w_out):
            # indexes
            h_from, h_to = i * stride, i * stride + pool_h
            w_from, w_to = j * stride, j * stride + pool_w
            
            # out
            out[:, :, i, j] = x[:, :, h_from:h_to, w_from:w_to].max(axis = (2, 3))    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. cache out
    x, pool_param = cache
    pool_h, pool_w, stride =  pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    n, c, h_out, w_out = dout.shape
    _, _, h, w = x.shape
    
    # 2. initialize dx
    dx = np.zeros(x.shape)
    # 3. calculate out
    for i in range(h_out):
        for j in range(w_out):
            # indexes
            h_from, h_to = i * stride, i * stride + pool_h
            w_from, w_to = j * stride, j * stride + pool_w
            
            # max args
            args = x[:, :, h_from:h_to, w_from:w_to].reshape(n, c, -1).argmax(axis = 2)
            args_h = (args // pool_w) 
            args_w = (args - args_h * pool_w).reshape(-1)
            args_h = args_h.reshape(-1)
            
            a_n = (np.arange(n).reshape(n,1) * np.ones((1, c))).reshape(-1).astype(int)
            a_c = (np.arange(c).reshape(1,c) * np.ones((n, 1))).reshape(-1).astype(int)
            a_h = args_h.reshape(-1) + i * stride
            a_w = args_w.reshape(-1) + j * stride
            dx[a_n, a_c, a_h, a_w] += dout[:, :, i,j].reshape(-1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. dimensions og the input
    N, C, H, W = x.shape
    
    # 2. change dims to be able to use vanilla version
    x_flat = np.transpose(x, (0,2,3,1)).reshape(N * H * W, C)

    # 3. apply vanilla version
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    
    # 4. change dims of output to x.shape size
    out = np.transpose(out_flat.reshape(N, H, W, C), (0, 3, 1, 2))    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. dimensions og the input
    N, C, H, W = dout.shape
    
    # 2. change dims to be able to use vanilla version
    dout_flat = np.transpose(dout, (0,2,3,1)).reshape(N * H * W, C)
   
    # 3. apply vanilla version
    dx_flat, dgamma, dbeta = batchnorm_backward_alt(dout_flat, cache)
    
    # 4. change dims of output to x.shape size
    dx = np.transpose(dx_flat.reshape(N, H, W, C), (0, 3, 1, 2))  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. get dimensions
    N, C, H, W = x.shape
    
    # 2. transform dimensions
    x_new = x.reshape((N, G, C // G, H, W))
    
    # 3. sample mean and variance
    mean = np.mean(x_new, axis = (2, 3, 4), keepdims = True)
       
    var = np.mean((x_new - mean) ** 2, axis = (2, 3, 4), keepdims = True)
        
    # 4. standart deviation
    std = np.sqrt(var + eps)
        
    # 5. normalization
    x_normed = ((x_new - mean) / std).reshape(x.shape) # .reshape(1, -1) <- is not necessary
        
    # 6. rescale and shift
    out = gamma * x_normed + beta
        
    # 7. cache and running statistics (for test)
    cache = mean, var, x_normed, gamma, eps, G

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # 1. getting data from cache
    mean, var, x_normed, gamma, eps, G = cache
    N, C, H, W = x_normed.shape
    std = np.sqrt(var + eps)
    
    # 2. getting simple derivative (dgamma, dbeta)
    dgamma = np.sum(dout * x_normed, axis = (0, 2 ,3), keepdims=True)
    dbeta = np.sum(dout, axis = (0, 2 ,3), keepdims=True)
    
    # 3. getting derivative with respect of x_normed
    dx_normed = dout * gamma
    
    # 4. getting derivative with respect of statistics
    dx_5 = dx_normed.reshape((N, G, C // G, H, W))
    x_5 = x_normed.reshape((N, G, C // G, H, W))
    D = C // G * H * W
    
    dvar = -0.5 * np.sum(dx_5 * x_5, axis = (2, 3, 4), keepdims = True) / (var + eps)
    dmean = -np.sum(dx_5, axis = (2, 3, 4), keepdims = True) / std
    dmean -= 2 * dvar * np.sum(x_5, axis = (2, 3, 4), keepdims = True) * std / D

    # 5. getting derivative with respect of x
    dx_new = dx_5 / std + dmean / D + dvar * 2 * x_5 * std / D
    dx = dx_new.reshape((N, C, H, W))


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
