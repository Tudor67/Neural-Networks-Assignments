import numpy as np

from tqdm import tqdm


class FCNet(object):
    """
    A two-layer fully-connected neural network with Tanh nonlinearity and
    softmax loss.

    The architecure is fc - tanh - fc - softmax.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=300, num_classes=10):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        """
        self.params = {}
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) / np.sqrt(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)   
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.best_val_epoch = 0
        self.best_val_acc = 0
    

    def forward(self, X, y=None):
        """
        The forward pass for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        - loss: If y is not None return a scalar value giving the loss,
                otherwise return None.
        - cache: A list of useful info for the backward pass.
        """
        a1, l1_cache = FCNet.__fc_tanh_forward__(X, self.params['W1'], self.params['b1'])
        scores, l2_cache = FCNet.__fc_forward__(a1, self.params['W2'], self.params['b2'])
        
        if y is None:
            return scores, None, None
        
        loss, dscores = FCNet.__softmax_loss__(scores, y)
        cache = (l1_cache, l2_cache, dscores)
        
        return scores, loss, cache
        

    def backward(self, cache):
        """
        The backward pass for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns: 
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        (l1_cache, l2_cache, dscores) = cache
        grads = {}
        
        (da1, grads['W2'], grads['b2']) = FCNet.__fc_backward__(dscores, l2_cache)
        (_, grads['W1'], grads['b1']) = FCNet.__fc_tanh_backward__(da1, l1_cache)
        
        return grads
    
    
    def __optimization_step__(self, X, y,
                              update_rule, lr,
                              momentum, velocity,
                              grads_cache, decay_rate,
                              beta1, beta2, iteration):
        grads = None
        grads_ahead = None
        if update_rule in ['sgd', 'momentum', 'adagrad', 'rmsprop', 'adam']:
            _, loss, cache = self.forward(X, y)
            grads = self.backward(cache)
        elif update_rule in ['nesterov']:
            # save initial/current params/weights
            initial_params = {}
            for param_name in self.params.keys():
                initial_params[param_name] = self.params[param_name].copy()
            
            # lookahead params/weights
            for param_name in self.params.keys():
                self.params[param_name] += momentum * velocity[param_name]
                
            # compute grads_ahead
            _, loss_ahead, cache_ahead = self.forward(X, y)
            grads_ahead = self.backward(cache_ahead)
            
            # go back to initial/current params/weights
            for param_name in self.params.keys():
                self.params[param_name] = initial_params[param_name]
        
        for param_name in self.params.keys():
            if update_rule == 'sgd':
                self.params[param_name] -= lr * grads[param_name]
            elif update_rule == 'momentum':
                velocity[param_name] = momentum * velocity[param_name] - lr * grads[param_name]
                self.params[param_name] += velocity[param_name]
            elif update_rule == 'nesterov':
                velocity[param_name] = momentum * velocity[param_name] - lr * grads_ahead[param_name]
                self.params[param_name] += velocity[param_name]
            elif update_rule == 'adagrad':
                grads_cache[param_name] += grads[param_name] ** 2
                self.params[param_name] -= lr * grads[param_name] / (1e-8 + np.sqrt(grads_cache[param_name]))
            elif update_rule == 'rmsprop':
                grads_cache[param_name] = decay_rate * grads_cache[param_name] + (1 - decay_rate) * grads[param_name] ** 2
                self.params[param_name] -= lr * grads[param_name] / (1e-8 + np.sqrt(grads_cache[param_name]))
            elif update_rule == 'adam':
                # adam with bias correction
                velocity[param_name] = beta1 * velocity[param_name] + (1 - beta1) * grads[param_name]
                velocity_hat = velocity[param_name] / (1 - beta1 ** iteration)
                grads_cache[param_name] = beta2 * grads_cache[param_name] + (1 - beta2) * grads[param_name] ** 2
                grads_cache_hat = grads_cache[param_name] / (1 - beta2 ** iteration)
                self.params[param_name] -= lr * velocity_hat / (1e-8 + np.sqrt(grads_cache_hat))
    
    
    def optimize(self, X_train, y_train,
                 X_val=None, y_val=None,
                 batch_size=128, epochs=10,
                 update_rule='sgd',
                 update_params={}):
        
        lr = update_params.get('lr', 1e-3)
        momentum = update_params.get('momentum', 0.9)
        decay_rate = update_params.get('decay_rate', 0.9)
        beta1 = update_params.get('beta1', 0.9)
        beta2 = update_params.get('beta2', 0.999)
        iteration = 0
        velocity = {}
        grads_cache = {}
        
        for param_name in self.params.keys():
            if update_rule in ['momentum', 'nesterov', 'adam']:
                velocity[param_name] = np.zeros_like(self.params[param_name])
            if update_rule in ['adagrad', 'rmsprop', 'adam']:
                grads_cache[param_name] = np.zeros_like(self.params[param_name])
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        best_val_params = {}
        
        N = len(y_train)
        start_indices = list(range(0, N, batch_size))
        end_indices = list(range(batch_size, N, batch_size))
        if end_indices[-1] != N:
            end_indices.append(N)
        
        for epoch in tqdm(range(epochs)):
            train_batch_losses = []
            train_batch_acc = []  
            
            for (start_idx, end_idx) in zip(start_indices, end_indices):
                iteration += 1
                X_train_batch = X_train[start_idx:end_idx]
                y_train_batch = y_train[start_idx:end_idx]
                self.__optimization_step__(X_train_batch, y_train_batch, 
                                           update_rule, lr,
                                           momentum, velocity,
                                           grads_cache, decay_rate,
                                           beta1, beta2, iteration)
                
                # train_batch loss and accuracy
                scores, loss, _ = self.forward(X_train_batch, y_train_batch)
                y_pred = np.argmax(scores, axis=1)
                acc = FCNet.get_acc(y_pred, y_train_batch)
                train_batch_losses.append(loss)
                train_batch_acc.append(acc)
            
            # train loss and acc history
            self.train_loss_history.append(np.mean(train_batch_losses))
            self.train_acc_history.append(np.mean(train_batch_acc))
            
            # val loss and acc history
            if X_val is not None:
                scores, loss, _ = self.forward(X_val, y_val)
                y_pred = np.argmax(scores, axis=1)
                acc = FCNet.get_acc(y_pred, y_val)
                self.val_loss_history.append(loss)
                self.val_acc_history.append(acc)
                
                # save the weights of the best model until now
                if self.best_val_acc < acc:
                    self.best_val_acc = acc
                    self.best_val_epoch = epoch + 1
                    # copy the weights carefully :) !
                    for param_name in self.params.keys():
                        best_val_params[param_name] = self.params[param_name].copy()
        
        # save the weights of the best model
        for param_name in best_val_params.keys():
            self.params[param_name] = best_val_params[param_name]
    
    
    def predict(self, X):
        scores, _, _ = self.forward(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
    
    
    def get_train_loss_history(self):
        return self.train_loss_history
    
    
    def get_train_acc_history(self):
        return self.train_acc_history
    
    
    def get_val_loss_history(self):
        return self.val_loss_history
    
    
    def get_val_acc_history(self):
        return self.val_acc_history
    
    
    def get_best_val_epoch(self):
        return self.best_val_epoch
    
    
    def get_best_val_acc(self):
        return self.best_val_acc
    
    
    def get_acc(y_pred, y):
        return np.sum(y_pred == y) / np.float64(y.shape[0])

    
    def __tanh_forward__(x):
        """
        Computes the forward pass for a layer of tanh units.

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = np.tanh(x)
        cache = x
        return out, cache
    
    
    def __tanh_backward__(dout, cache):
        """
        Computes the backward pass for a layer of tanh units.

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        x = cache
        dx = (1 - np.tanh(x) ** 2) * dout
        return dx
      
    
    def __fc_forward__(x, w, b):
        """
        Computes the forward pass for an fc layer.

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
        N = x.shape[0]
        out = np.dot(x.reshape(N, -1), w) + b
        cache = (x, w, b)
        return out, cache


    def __fc_backward__(dout, cache):
        """
        Computes the backward pass for an fc layer.

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
        N = x.shape[0]
        dx = np.dot(dout, w.T).reshape(x.shape)
        dw = np.dot(dout.T, x.reshape(N, -1)).T
        db = np.sum(dout, axis=0)
        return dx, dw, db

    
    def __fc_tanh_forward__(x, w, b):
        """
        Convenience layer composed of a fc layer followed by a tanh

        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer

        Returns a tuple of:
        - out: Output from the tanh
        - cache: Object to give to the backward pass
        """
        a, fc_cache = FCNet.__fc_forward__(x, w, b)
        out, tanh_cache = FCNet.__tanh_forward__(a)
        cache = (fc_cache, tanh_cache)
        return out, cache
    
    
    def __fc_tanh_backward__(dout, cache):
        """
        Backward pass for the fc-tanh convenience layer
        """
        fc_cache, tanh_cache = cache
        da = FCNet.__tanh_backward__(dout, tanh_cache)
        dx, dw, db = FCNet.__fc_backward__(da, fc_cache)
        return dx, dw, db
    
    
    def __softmax_loss__(x, y):
        """
        Computes the loss and gradient for softmax classification.

        Inputs:
        - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
             class for the ith input.
        - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
             0 <= y[i] < C.

        Returns a tuple of:
        - loss: Scalar giving the loss
        - dx: Gradient of the loss with respect to x
        """
        N = x.shape[0]
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(Z)
        probs = np.exp(log_probs)
        loss = -np.sum(log_probs[np.arange(N), y]) / N
        
        dx = probs.copy()
        dx[np.arange(N), y] -= 1
        dx /= N
        
        return loss, dx