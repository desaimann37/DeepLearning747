import numpy as np


class Perceptron:
    def __init__(self, alpha=0.01, epochs=40, decay=0.01, reg=1e-4):
        """
        alpha  : initial learning rate
        epochs : number of passes over data
        decay  : learning rate decay factor
        reg    : L2 regularization strength (Penalize large weights to prevent overfitting - I am using it to improve performance of gthe model on testing data)
        """
        self.alpha = alpha
        self.epochs = epochs
        self.decay = decay # As mentioned in the Notebook file to use decay. 
        self.reg = reg

        self.w = None
        self.b = None
        self.mean = None
        self.std = None

    # Normalization: 
    def normalize_fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return (X - self.mean) / self.std

    def normalize_transform(self, X):
        return (X - self.mean) / self.std


    def train(self, X_train, y_train):

        X_train = self.normalize_fit(X_train)

        N, D = X_train.shape
        C = np.max(y_train) + 1

        # small random initialization (I am using it to improve accuracy)
        self.w = 0.001 * np.random.randn(C, D)
        self.b = np.zeros(C)

        for epoch in range(self.epochs):

            # Shuffle each epoch
            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            lr = self.alpha / (1 + self.decay * epoch)

            for i in range(N):
                x = X_train[i]
                y_true = y_train[i]

                scores = self.w @ x + self.b
                y_pred = np.argmax(scores)

                if y_pred != y_true:
                    self.w[y_true] += lr * x
                    self.w[y_pred] -= lr * x
                    self.b[y_true] += lr
                    self.b[y_pred] -= lr

                # L2 regularization
                self.w -= lr * self.reg * self.w

            print(f"Epoch {epoch+1}/{self.epochs} | lr={lr:.5f}")


    def predict(self, X):
        X = self.normalize_transform(X)
        scores = X @ self.w.T + self.b
        return np.argmax(scores, axis=1)
