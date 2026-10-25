import numpy as np
from scipy.spatial.distance import pdist, squareform


class GaussianSNE:
    """Educational exact SNE using symmetric Gaussian affinities and gradient descent.

    Unlike t-SNE, the low-dimensional similarities Q are Gaussian too (not Student-t),
    which makes SNE suffer from the "crowding problem". It is O(n²) per iteration
    and cannot embed new points (no transform), so it is used transductively.
    """

    def __init__(self, n_components=2, perplexity=30, n_iter=500, learning_rate='auto', random_state=42):
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

        # Binary-search a per-point bandwidth beta_i so each row's entropy matches log(perplexity).
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
        n = len(X)
        lr = max(n / 12.0, 50.0) if self.learning_rate == 'auto' else self.learning_rate
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(0, 1e-4, (n, self.n_components))
        velocity = np.zeros_like(Y)
        self.loss_curve_ = []

        for it in range(self.n_iter):
            D = squareform(pdist(Y, 'sqeuclidean'))
            Q = np.exp(-D)
            np.fill_diagonal(Q, 0)
            Q /= Q.sum() + 1e-12
            Q = np.maximum(Q, 1e-12)

            # grad_i = 4 Σ_j (p_ij − q_ij)(y_i − y_j), written as matrix products.
            W = P - Q
            grad = 4 * (W.sum(axis=1)[:, None] * Y - W @ Y)

            momentum = 0.5 if it < 100 else 0.8
            velocity = momentum * velocity - lr * grad
            Y += velocity
            Y -= Y.mean(0)

            if it % 10 == 0 or it == self.n_iter - 1:
                self.loss_curve_.append(float(np.sum(P * np.log(P / Q))))

        self.kl_divergence_ = self.loss_curve_[-1]
        return Y
