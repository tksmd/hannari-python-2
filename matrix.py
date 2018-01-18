import numpy as np
from typing import Tuple, Callable, Optional


def soft_thresh(X: np.ndarray, thresh: float) -> np.ndarray:
    return np.where(np.abs(X) <= thresh, 0, X - thresh * np.sign(X))


def generate_random_matrix(size: Tuple[int, int], rank: int, random_state: Optional[int] = None):
    m, n = size
    np.random.seed(random_state)
    U = np.random.randint(1, m, m * rank).reshape(m, rank)
    V = np.random.randint(1, n, rank * n).reshape(rank, n)
    A = U.dot(V) / (m * n)
    return A


def drop_values(X: np.ndarray, ratio: float = 0.2, missing_value: float = -1) -> np.ndarray:
    m, n = X.shape
    drop_indices = np.random.choice(m * n, int(m * n * ratio))
    flattened = np.ravel(X.copy())
    flattened[drop_indices] = missing_value
    return flattened.reshape(m, n)


class LowrankReconstruction:

    def __init__(self, alpha: float = 2.0, max_iter: int = 1000, missing_value: float = -1,
                 random_state: Optional[int] = None):
        self.alpha = alpha
        self.max_iter = max_iter
        self.missing_value = missing_value
        np.random.seed(random_state)

    def transform(self, X: np.ndarray) -> np.ndarray:
        refill = self.get_refill(X)

        # initial guess
        X_trans = np.random.rand(*X.shape)
        # retrieve existing values
        refill(X_trans)

        for t in range(self.max_iter):
            # update threshold in each iteration
            thresh = self.alpha * (1 - t / self.max_iter)
            # apply SVD
            U, S, V = np.linalg.svd(X_trans, full_matrices=False)
            # apply soft thresholding function to singular values
            S = soft_thresh(S, thresh)
            S = np.diag(S)
            # reconstruct X
            X_trans = U.dot(S).dot(V)
            refill(X_trans)

        return X_trans

    def get_refill(self, src) -> Callable[[np.ndarray], None]:
        # keep position of existing values
        pos = np.where(src != self.missing_value)

        def _refill(target: np.ndarray):
            for i, j in zip(pos[0], pos[1]):
                target[i][j] = src[i][j]

        return _refill
