import numpy as np
from scipy.spatial.distance import pdist, squareform


class GaussianSNE:
    """Educational exact SNE using symmetric Gaussian affinities and gradient descent."""

    def __init__(self, n_components=2, perplexity=30, n_iter=500, learning_rate=100, random_state=42):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _probabilities(self, X):
        D = squareform(pdist(X, 'sqeuclidean'))
        n = len(X)
        P = np.zeros((n, n))
        target = np.log(self.perplexity)

        for i in range(n):
            lo, hi = -20.0, 20.0
            for _ in range(40):
                beta = np.exp((lo + hi) / 2)
                p = np.exp(-D[i] * beta)
                p[i] = 0
                p /= p.sum() + 1e-12
                H = -np.sum(p * np.log(p + 1e-12))
                if H > target:
                    lo = np.log(beta)
                else:
                    hi = np.log(beta)
            P[i] = p

        P = (P + P.T) / (2 * n)
        return np.maximum(P, 1e-12)

    def fit_transform(self, X):
        P = self._probabilities(X)
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(0, 1e-4, (len(X), self.n_components))
        velocity = np.zeros_like(Y)
        self.loss_curve_ = []

        for it in range(self.n_iter):
            D = squareform(pdist(Y, 'sqeuclidean'))
            Q = np.exp(-D)
            np.fill_diagonal(Q, 0)
            Q /= Q.sum() + 1e-12
            Q = np.maximum(Q, 1e-12)

            grad = np.zeros_like(Y)
            for i in range(len(Y)):
                grad[i] = 4 * np.sum(((P[i] - Q[i])[:, None]) * (Y[i] - Y), axis=0)

            momentum = 0.5 if it < 100 else 0.8
            velocity = momentum * velocity - self.learning_rate * grad
            Y += velocity
            Y -= Y.mean(0)

            if it % 10 == 0 or it == self.n_iter - 1:
                self.loss_curve_.append(float(np.sum(P * np.log(P / Q))))

        self.kl_divergence_ = self.loss_curve_[-1]
        return Y
