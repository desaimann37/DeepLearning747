import numpy as np 
 
class Softmax(): 
    def __init__(self): 
        """ 
        Initialises Softmax classifier with initializing  
        weights, alpha(learning rate), number of epochs 
        and regularization constant. 
        """ 
        self.w = None 
        self.alpha = 0.5 
        self.epochs = 100 
        self.reg_const = 0.05 
     
    def calc_gradient(self, X_train, y_train): 
        """ 
        Calculate gradient of the softmax loss 
           
        Inputs have dimension D, there are C classes, and we operate on minibatches 
        of N examples. 
 
        Inputs: 
        - X_train: A numpy array of shape (N, D) containing a minibatch of data. 
        - y_train: A numpy array of shape (N,) containing training labels; y[i] = c means 
          that X[i] has label c, where 0 <= c < C. 
 
        Returns: 
        - gradient with respect to weights W; an array of same shape as W 
        """ 
        N = X_train.shape[0]
        scores = X_train @ self.w

        scores -= np.max(scores, axis=1, keepdims=True)

        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # One-hot subtract
        probs[np.arange(N), y_train] -= 1

        # Gradient
        grad_w = (X_train.T @ probs) / N

        # Add L2 regularization
        grad_w += self.reg_const * self.w

        return grad_w 
     
    def train(self, X_train, y_train): 
        """ 
        Train Softmax classifier using stochastic gradient descent. 
 
        Inputs: 
        - X_train: A numpy array of shape (N, D) containing training data; 
        N examples with D dimensions 
        - y_train: A numpy array of shape (N,) containing training labels; 
         
        Hint : Operate with Minibatches of the data for SGD 
        """ 

        N, D = X_train.shape
        C = np.max(y_train) + 1

        # Initialize weights
        self.w = 0.001 * np.random.randn(D, C)

        batch_size = 200

        for epoch in range(self.epochs):

            # Shuffle data
            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            # Learning rate decay
            lr = self.alpha / (1 + 0.01 * epoch)

            for i in range(0, N, batch_size):

                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                grad = self.calc_gradient(X_batch, y_batch)

                # Update weights
                self.w -= lr * grad

            print(f"Epoch {epoch+1}/{self.epochs} | lr={lr:.5f}")
     
    def predict(self, X_test): 
        """ 
        Use the trained weights of softmax classifier to predict labels for 
        data points. 
 
        Inputs: 
        - X_test: A numpy array of shape (N, D) containing training data; there are N 
          training samples each of dimension D. 
 
        Returns: 
        - pred: Predicted labels for the data in X_test. pred is a 1-dimensional 
          array of length N, and each element is an integer giving the predicted 
          class. 
        """ 

        scores = X_test @ self.w
        pred = np.argmax(scores, axis=1)

        return pred
