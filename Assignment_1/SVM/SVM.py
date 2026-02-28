import numpy as np


class SVM:
    def __init__(self, alpha=0.01, epochs=50, reg_const=0.1, batch_size=200, decay=0.01):

        self.alpha = alpha
        self.epochs = epochs
        self.reg_const = reg_const
        self.batch_size = batch_size
        self.decay = decay

        self.w = None
        self.mean = None
        self.std = None

    #Normalization:
    def normalize_fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        return (X - self.mean) / self.std

    def normalize_transform(self, X):
        return (X - self.mean) / self.std

    def calc_gradient(self, X_batch, y_batch):

        N = X_batch.shape[0]

        scores = X_batch @ self.w         
        correct_class_scores = scores[np.arange(N), y_batch].reshape(-1, 1)

        margins = np.maximum(0, scores - correct_class_scores + 1)
        margins[np.arange(N), y_batch] = 0

        # Gradient calculation
        binary = margins > 0
        binary = binary.astype(float)

        row_sum = np.sum(binary, axis=1)
        binary[np.arange(N), y_batch] = -row_sum

        grad_w = (X_batch.T @ binary) / N

        # Add regularization gradient
        grad_w += self.reg_const * self.w

        return grad_w

    def train(self, X_train, y_train):

        X_train = self.normalize_fit(X_train)

        N, D = X_train.shape
        C = np.max(y_train) + 1

        self.w = 0.001 * np.random.randn(D, C)

        for epoch in range(self.epochs):

            indices = np.random.permutation(N)
            X_train = X_train[indices]
            y_train = y_train[indices]

            lr = self.alpha / (1 + self.decay * epoch)

            for i in range(0, N, self.batch_size):

                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                grad = self.calc_gradient(X_batch, y_batch)

                self.w -= lr * grad

            print(f"Epoch {epoch+1}/{self.epochs} | lr={lr:.5f}")

    def predict(self, X_test):

        X_test = self.normalize_transform(X_test)
        scores = X_test @ self.w
        pred = np.argmax(scores, axis=1)

        return pred
