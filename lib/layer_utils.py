import numpy as np


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


class RNN(object):
    def __init__(self, *args):
        """
        RNN Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        layer_cnt = 0
        for layer in args:
            for n, v in layer.params.items():
                if v is None:
                    continue
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)
            layer_cnt += 1
        layer_cnt = 0

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.iteritems():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.iteritems():
                self.grads[n] = v

    def load(self, pretrained):
        """ 
        Load a pretrained model by names 
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.iteritems():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))


class VanillaRNN(object):
    def __init__(self, input_dim, h_dim, init_scale=0.02, name='vanilla_rnn'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - h_dim: hidden state dimension
        - meta: to store the forward pass activations for computing backpropagation 
        """
        self.name = name
        self.wx_name = name + "_wx"
        self.wh_name = name + "_wh"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.params = {}
        self.grads = {}
        self.params[self.wx_name] = init_scale * np.random.randn(input_dim, h_dim)
        self.params[self.wh_name] = init_scale * np.random.randn(h_dim, h_dim)
        self.params[self.b_name] = np.zeros(h_dim)
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        self.meta = None
        
    def step_forward(self, x, prev_h):
        """
        x: input feature (N, D)
        prev_h: hidden state from the previous timestep (N, H)

        meta: variables needed for the backward pass
        """
        next_h, meta = None, None
        assert np.prod(x.shape[1:]) == self.input_dim, "But got {} and {}".format(
            np.prod(x.shape[1:]), self.input_dim)
        ############################################################################
        # TODO: implement forward pass of a single timestep of a vanilla RNN.      #
        # Store the results in the variable output provided above as well as       #
        # values needed for the backward pass.                                     #
        ############################################################################
        pass
    
        #np.longdouble
        
        Wx = self.params[self.wx_name]
        Wh = self.params[self.wh_name]
        B = self.params[self.b_name]
        #print(np.matmul(x, Wx))
        #print(np.matmul(prev_h, Wh)) 
        if self.meta is None:
            self.meta = []
        
        meta = []
     
        h_raw = np.matmul(prev_h, Wh) + np.matmul(x, Wx) + B
        meta.append(h_raw)
        #print("h_raw:",h_raw)
        next_h = np.tanh(h_raw)
        meta.append(prev_h)
        meta.append(x)

        self.meta.append(meta)
 
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return next_h, meta

    def step_backward(self, dnext_h, meta):
        """
        dnext_h: gradient w.r.t. next hidden state
        meta: variables needed for the backward pass

        dx: gradients of input feature (N, D)
        dprev_h: gradients of previous hiddel state (N, H)
        dWh: gradients w.r.t. feature-to-hidden weights (D, H)
        dWx: gradients w.r.t. hidden-to-hidden weights (H, H)
        db: gradients w.r.t bias (H,)
        """
        dx, dprev_h, dWx, dWh, db = None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass of a single timestep of a vanilla RNN.  #
        # Store the computed gradients for current layer in self.grads with         #
        # corresponding name.                                                       # 
        #############################################################################
        pass
        
        Wx = self.params[self.wx_name]
        Wh = self.params[self.wh_name]
        B = self.params[self.b_name]
        
        #print("meta0:", meta[0].shape)
        
        h_raw = meta[0]
        h_prev = meta[1]
        X = meta[2]
   
        #print("hraw:",h_raw)
        #print("tanh_hraw:",np.tanh(h_raw))
        #print("tanh_sqr_hraw:",np.power(np.tanh(h_raw), 2))
        #print("modified_ones",(1- np.power(np.tanh(h_raw), 2)))
        
        #print(h_raw)
        t = np.tanh(h_raw)
        #print(t)
        
        t = np.power(t,2)
        dh_raw = np.multiply((1-(t)),dnext_h)
        
  
        dx = np.matmul((dh_raw), (Wx).T)
        dprev_h = (np.matmul(dh_raw, np.transpose(Wh)))
        
        dWx = np.matmul((X).T, dh_raw)
        
        #dh_raw_ = np.sum(dh_raw, axis=0)
        #h_prev_ = np.sum(h_prev, axis=0)
   
        dWh = np.matmul((h_prev).T, (dh_raw))
        
        db = np.sum(dh_raw, axis=0)
    
        if self.grads[self.wx_name] is None:
            self.grads[self.wx_name] = dWx
        else:
            self.grads[self.wx_name] = self.grads[self.wx_name] + dWx
        if self.grads[self.wh_name] is None:
            self.grads[self.wh_name] = dWh
        else:
            self.grads[self.wh_name] = self.grads[self.wh_name] + dWh
        if self.grads[self.b_name] is None:
            self.grads[self.b_name] = db
        else:
            self.grads[self.b_name] = self.grads[self.b_name] + db
            
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx, dprev_h, dWx, dWh, db

    def forward(self, x, h0):
        """
        x: input feature for the entire timeseries (N, T, D)
        h0: initial hidden state (N, H)
        """
        h = None
        self.meta = []
        ##############################################################################
        # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
        # input data. You should use the step_forward function that you defined      #
        # above. You can use a for loop to help compute the forward pass.            #
        ##############################################################################
        pass
    
        N, T, D = x.shape
        N, H = h0.shape
        
        #print(x.shape)
        #print(x[:,0,:].shape)
        #print(h0.shape)
        next_h, meta = self.step_forward(x[:,0,:],h0)
        h  = np.linspace(-0.3, 0.1, num=N*T*H).reshape(N, T, H)
        #print(h.shape)
        #print(next_h.shape)
        h[:,0,:] = next_h
        for i in range(1,T):
            next_h, meta = self.step_forward(x[:,i,:],next_h)
            h[:,i,:] = next_h
    
    
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return h

    def backward(self, dh):
        """
        dh: gradients of hidden states for the entire timeseries (N, T, H)

        dx: gradient of inputs (N, T, D)
        dh0: gradient w.r.t. initial hidden state (N, H)
        self.grads[self.wx_name]: gradient of input-to-hidden weights (D, H)
        self.grads[self.wh_name]: gradient of hidden-to-hidden weights (H, H)
        self.grads[self.b_name]: gradient of biases (H,)
        """
        dx, dh0 = None, None
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        ##############################################################################
        # TODO: Implement the backward pass for a vanilla RNN running an entire      #
        # sequence of data. You should use the rnn_step_backward function that you   #
        # defined above. You can use a for loop to help compute the backward pass.   #
        # HINT: Gradients of hidden states come from two sources                     #
        ##############################################################################
        pass
    
        N, T, H = dh.shape
        Meta = self.meta.pop()
        x_single = Meta[2]
        N, D = x_single.shape
        #print(x.shape)
        #print(x[:,0,:].shape)
        #print(h0.shape)
        #dnext_h, meta
        d_x, d_prev_h, d_Wx, d_Wh, d_b = self.step_backward(dh[:,T-1,:],Meta)
        dx = np.linspace(-0.3, 0.1, num=N*T*D).reshape(N, T, D)
        dh0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
        dx[:,T-1,:] = d_x
        #dx  = np.linspace(-0.3, 0.1, num=N*T*D).reshape(N, T, D)
        #print(h.shape)
        #print(next_h.shape)
        #dx[:,0,:] = next_h
        for i in reversed(range(T-1)):
            Meta = self.meta.pop()
            d_x, d_prev_h, d_Wx, d_Wh, d_b = self.step_backward(dh[:,i,:]+d_prev_h,Meta)
            dx[:,i,:] = d_x
            dh0 = d_prev_h
    
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        self.meta = []
        return dx, dh0


class LSTM(object):
    def __init__(self, input_dim, h_dim, init_scale=0.02, name='lstm'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - h_dim: hidden state dimension
        - meta: to store the forward pass activations for computing backpropagation 
        """
        self.name = name
        self.wx_name = name + "_wx"
        self.wh_name = name + "_wh"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.params = {}
        self.grads = {}
        self.params[self.wx_name] = init_scale * np.random.randn(input_dim, 4*h_dim)
        self.params[self.wh_name] = init_scale * np.random.randn(h_dim, 4*h_dim)
        self.params[self.b_name] = np.zeros(4*h_dim)
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        self.meta = None
        
    def step_forward(self, x, prev_h, prev_c):
        """
        x: input feature (N, D)
        prev_h: hidden state from the previous timestep (N, H)

        meta: variables needed for the backward pass
        """
        next_h, next_c, meta = None, None, None
        #############################################################################
        # TODO: Implement the forward pass for a single timestep of an LSTM.        #
        # You may want to use the numerically stable sigmoid implementation above.  #
        #############################################################################
        pass
    
        #prev_c
        
        Wx = self.params[self.wx_name]
        Wh = self.params[self.wh_name]
        B = self.params[self.b_name]
        #print(np.matmul(x, Wx))
        #print(np.matmul(prev_h, Wh)) 
        if self.meta is None:
            self.meta = []
        
        meta = []
     
        h_raw = np.matmul(prev_h, Wh) + np.matmul(x, Wx) + B
        D, H4 = h_raw.shape
        H = int(H4/4)
        a_i = h_raw[:,0:H:1]
        a_f = h_raw[:,H:2*H:1]
        a_o = h_raw[:,2*H:3*H:1]
        a_g = h_raw[:,3*H:4*H:1]
        
        i = sigmoid(a_i)
        f = sigmoid(a_f)
        o = sigmoid(a_o)
        g = np.tanh(a_g)
        
        next_c = np.multiply(f, prev_c) + np.multiply(i, g)
        
        meta.append(h_raw)
        #print("h_raw:",h_raw)
        next_h = np.tanh(h_raw)
        
        
        meta.append(prev_h)
        meta.append(x)
        
        activations = [a_i, a_f, a_o, a_g]
        ifog = [i,f,o,g]
        meta.append(prev_c)        
        meta.append(activations)
        meta.append(ifog)
        meta.append(next_c)

        self.meta.append(meta)
        
        next_h = np.multiply(o, np.tanh(next_c))
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################
        return next_h, next_c, meta
        
    def step_backward(self, dnext_h, dnext_c, meta):
        """
        dnext_h: gradient w.r.t. next hidden state
        meta: variables needed for the backward pass

        dx: gradients of input feature (N, D)
        dprev_h: gradients of previous hiddel state (N, H)
        dWh: gradients w.r.t. feature-to-hidden weights (D, H)
        dWx: gradients w.r.t. hidden-to-hidden weights (H, H)
        db: gradients w.r.t bias (H,)
        """
        dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
        #############################################################################
        # TODO: Implement the backward pass for a single timestep of an LSTM.       #
        #                                                                           #
        # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
        # the output value from the nonlinearity.                                   #
        #############################################################################
        pass
    
        def dev_elementaryWise(f, x):
            return np.divide(f,x)
        def dev_tanh(x):
            return 1-(np.tanh(x)**2)
        def dev_sig(x):
            sig = sigmoid(x)
            return np.multiply((1-sig), sig)
 
    
        Wx = self.params[self.wx_name]
        Wh = self.params[self.wh_name]
        B = self.params[self.b_name]
        
        #print("meta0:", meta[0].shape)
        
        h_raw = meta[0]
        h_prev = meta[1]
        X = meta[2]
        
        prev_c = meta[3]
        activations = meta[4]
        ifog = meta[5]
        Ct = meta[6]
        
        a_i = activations[0] # N * H 
        a_f = activations[1]
        a_o = activations[2]
        a_g = activations[3]
        
        i = ifog[0] # N * H
        f = ifog[1]
        o = ifog[2]
        g = ifog[3]
        
        dCt = dnext_c
        dht = dnext_h
        # N * H 
        
        
        # N * H
        dCt = np.multiply(np.multiply(o, dev_tanh(Ct)), dht) + dCt
        #dht o dev_tanh(Ct)   + dCt
        dCPrev = np.multiply(f, dCt)
        dprev_c = dCPrev
        N, H = dCPrev.shape
        
        db = np.zeros(N*4*H).reshape(N, 4*H)
        db[:, 0:H] = np.multiply(np.multiply(g, dCt), dev_sig(a_i))
        
        #np.sum(dh_raw, axis=0)
        #1H
        # N H   N H 
        db[:,H:2*H] = np.multiply(np.multiply(prev_c, dCt), dev_sig(a_f))
        db[:,2*H:3*H] = np.multiply(np.multiply(np.tanh(Ct), dht), dev_sig(a_o))
        db[:,3*H:4*H] = np.multiply(np.multiply(i, dCt), dev_tanh(a_g))
        
        #N D N 4H
        dWx = np.matmul((X).T, db) # D * 4H
        dWh = np.matmul((h_prev).T, (db)) # H * 4H
        dx = np.matmul((db), (Wx).T) # N * D
        dprev_h = (np.matmul(db, np.transpose(Wh)))
        
        db = np.sum(db, axis=0)
        
    
        if self.grads[self.wx_name] is None:
            self.grads[self.wx_name] = dWx
        else:
            self.grads[self.wx_name] = self.grads[self.wx_name] + dWx
        if self.grads[self.wh_name] is None:
            self.grads[self.wh_name] = dWh
        else:
            self.grads[self.wh_name] = self.grads[self.wh_name] + dWh
        if self.grads[self.b_name] is None:
            self.grads[self.b_name] = db
        else:
            self.grads[self.b_name] = self.grads[self.b_name] + db
            
            
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################

        return dx, dprev_h, dprev_c, dWx, dWh, db

    def forward(self, x, h0):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns a tuple of:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
        - cache: Values needed for the backward pass.
        """
        h = None
        self.meta = []
        #############################################################################
        # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
        # You should use the lstm_step_forward function that you just defined.      #
        #############################################################################
        pass
    
        N, T, D = x.shape
        N, H = h0.shape
        
        #print(x.shape)
        #print(x[:,0,:].shape)
        #print(h0.shape)
        next_h, next_c, meta = self.step_forward(x[:,0,:],h0, 0)
 
        h  = np.linspace(-0.3, 0.1, num=N*T*H).reshape(N, T, H)
        #print(h.shape)
        #print(next_h.shape)
        h[:,0,:] = next_h
        for i in range(1,T):
            next_h, next_c, meta = self.step_forward(x[:,i,:],next_h, next_c)
            h[:,i,:] = next_h
    
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################
        return h

    def backward(self, dh):
        """
        Backward pass for an LSTM over an entire sequence of data.]

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)
        - cache: Values from the forward pass

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0 = None, None
        #############################################################################
        # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
        # You should use the lstm_step_backward function that you just defined.     #
        #############################################################################
        pass
    
        #dx, dprev_h, dprev_c, dWx, dWh, db
        
        N, T, H = dh.shape
        Meta = self.meta.pop()
        x_single = Meta[2]
        N, D = x_single.shape
  
        d_x, d_prev_h, dprev_c, d_Wx, d_Wh, d_b = self.step_backward(dh[:,T-1,:], 0, Meta)
        dx = np.zeros(N*T*D).reshape(N, T, D)
        dh0 = np.zeros(N*H).reshape(N, H)
        dx[:,T-1,:] = d_x

        for i in reversed(range(T-1)):
            Meta = self.meta.pop()
            d_x, d_prev_h, dprev_c, d_Wx, d_Wh, d_b = self.step_backward(dh[:,i,:]+d_prev_h, dprev_c, Meta)
            dx[:,i,:] = d_x
            dh0 = d_prev_h
    
    
        #############################################################################
        #                               END OF YOUR CODE                            #
        #############################################################################
        self.meta = []
        return dx, dh0
            
        
class word_embedding(object):
    def __init__(self, voc_dim, vec_dim, name="we"):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - v_dim: words size
        - output_dim: vector dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.w_name = name + "_w"
        self.voc_dim = voc_dim
        self.vec_dim = vec_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = np.random.randn(voc_dim, vec_dim)
        self.grads[self.w_name] = None
        self.meta = None
        
    def forward(self, x):
        """
        Forward pass for word embeddings. We operate on minibatches of size N where
        each sequence has length T. We assume a vocabulary of V words, assigning each
        to a vector of dimension D.

        Inputs:
        - x: Integer array of shape (N, T) giving indices of words. Each element idx
          of x muxt be in the range 0 <= idx < V.
        - W: Weight matrix of shape (V, D) giving word vectors for all words.

        Returns a tuple of:
        - out: Array of shape (N, T, D) giving word vectors for all input words.
        - meta: Values needed for the backward pass
        """
        out, self.meta = None, None
        ##############################################################################
        # TODO: Implement the forward pass for word embeddings.                      #
        #                                                                            #
        # HINT: This can be done in one line using NumPy's array indexing.           #
        ##############################################################################
        pass
        
        W = self.params[self.w_name]
        V, D = W.shape
        N, T = x.shape
        
        X = np.zeros(N*T*V).reshape(N, T, V)
        X_for_back = np.zeros(N*V*T).reshape(N, V, T)
        for i in range(N):
            for j in range(T):
                for k in range(V):
                    if k == x[i,j]:
                        X[i,j,k] = 1
                        X_for_back[i,k,j] = 1
                        
        out = np.matmul(X,W)
        self.meta = X_for_back
        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return out
        
    def backward(self, dout):
        """
        Backward pass for word embeddings. We cannot back-propagate into the words
        since they are integers, so we only return gradient for the word embedding
        matrix.

        HINT: Look up the function np.add.at

        Inputs:
        - dout: Upstream gradients of shape (N, T, D)
        - cache: Values from the forward pass

        Returns:
        - dW: Gradient of word embedding matrix, of shape (V, D).
        """
        self.grads[self.w_name] = None
        ##############################################################################
        # TODO: Implement the backward pass for word embeddings.                     #
        # Note that Words can appear more than once in a sequence.                   #
        # HINT: Look up the function np.add.at                                       #
        ##############################################################################
        pass
    
        X = self.meta
        N,V,T = ((X).shape)
        single_wd = np.matmul(X[0,:,:], dout[0,:,:])
        for i in range(1, N):
            single_wd = single_wd + np.matmul(X[i,:,:], dout[i,:,:])
        self.grads[self.w_name] = single_wd

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################


class temporal_fc(object):
    def __init__(self, input_dim, output_dim, init_scale=0.02, name='t_fc'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - output_dim: output dimension
        - meta: to store the forward pass activations for computing backpropagation 
        """
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(input_dim, output_dim)
        self.params[self.b_name] = np.zeros(output_dim)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
        
    def forward(self, x):
        """
        Forward pass for a temporal fc layer. The input is a set of D-dimensional
        vectors arranged into a minibatch of N timeseries, each of length T. We use
        an affine function to transform each of those vectors into a new vector of
        dimension M.

        Inputs:
        - x: Input data of shape (N, T, D)
        - w: Weights of shape (D, M)
        - b: Biases of shape (M,)

        Returns a tuple of:
        - out: Output data of shape (N, T, M)
        - cache: Values needed for the backward pass
        """
        N, T, D = x.shape
        M = self.params[self.b_name].shape[0]
        out = x.reshape(N * T, D).dot(self.params[self.w_name]).reshape(N, T, M) + self.params[self.b_name]
        self.meta = [x, out]
        return out

    def backward(self, dout):
        """
        Backward pass for temporal fc layer.

        Input:
        - dout: Upstream gradients of shape (N, T, M)
        - cache: Values from forward pass

        Returns a tuple of:
        - dx: Gradient of input, of shape (N, T, D)
        - dw: Gradient of weights, of shape (D, M)
        - db: Gradient of biases, of shape (M,)
        """
        x, out = self.meta
        N, T, D = x.shape
        M = self.params[self.b_name].shape[0]

        dx = dout.reshape(N * T, M).dot(self.params[self.w_name].T).reshape(N, T, D)
        self.grads[self.w_name] = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
        self.grads[self.b_name] = dout.sum(axis=(0, 1))

        return dx


class temporal_softmax_loss(object):
    def __init__(self, dim_average=True):
        """
        - dim_average: if dividing by the input dimension or not
        - dLoss: intermediate variables to store the scores
        - label: Ground truth label for classification task
        """
        self.dim_average = dim_average  # if average w.r.t. the total number of features
        self.dLoss = None
        self.label = None

    def forward(self, feat, label, mask):
        """ Some comments """
        loss = None
        N, T, V = feat.shape

        feat_flat = feat.reshape(N * T, V)
        label_flat = label.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        probs = np.exp(feat_flat - np.max(feat_flat, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), label_flat]))
        if self.dim_average:
            loss /= N

        self.dLoss = probs.copy()
        self.label = label
        self.mask = mask
        
        return loss

    def backward(self):
        N, T = self.label.shape
        dLoss = self.dLoss
        if dLoss is None:
            raise ValueError("No forward function called before for this module!")
        dLoss[np.arange(dLoss.shape[0]), self.label.reshape(N * T)] -= 1.0
        if self.dim_average:
            dLoss /= N
        dLoss *= self.mask.reshape(N * T)[:, None]
        self.dLoss = dLoss
        
        return dLoss.reshape(N, T, -1)
