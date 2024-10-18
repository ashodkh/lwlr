import numpy as np
from sklearn.neighbors import KDTree


class LWLR:
    def __init__(self, weight_type='inverse_distance'):
        weight_funcs = {
            'constant': lambda d: np.ones_like(d),
            'inverse_distance': lambda d: 1 / d,
            'inverse_distance_squared': lambda d: 1 / (d ** 2)
        }

        if weight_type not in weight_funcs:
            raise ValueError("Invalid weight. Choose from 'constant', 'inverse_distance', 'inverse_distance_squared'.")
        self.weight_func = weight_funcs[weight_type]

    def predict(self, x, x_train, y_train, nn, extra_weights=None):
        n_query = x.shape[0]

        # Add bias terms (column of ones) to input matrices
        x = np.hstack([np.ones((n_query, 1)), x])
        x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])

        # Find nearest neighbors using KDTree (excluding self-match at index 0)
        tree = KDTree(x_train)
        dist, ind = tree.query(x, k=nn + 1)  # Get k + 1 neighbors (including self)
        dist, ind = dist[:, 1:], ind[:, 1:]  # Exclude self-match

        # Handle extra weights (if provided)
        extra_weights = np.ones(x_train.shape[0]) if extra_weights is None else extra_weights

        # Pre-compute weights for all queries and neighbors
        weights = self.weight_func(dist) * extra_weights[ind]  # Shape: (n_query, nn)

        # Prepare arrays for batch computations
        predictions = np.zeros(n_query)

        # Loop over each query point (small loop, optimized with batch operations)
        for i in range(n_query):
            X_neighbors = x_train[ind[i]]  # Nearest neighbor features, shape: (nn, n_features + 1)
            Y_neighbors = y_train[ind[i]]  # Nearest neighbor targets, shape: (nn,)

            # Weighted least squares solution: theta = (X^T W X)^-1 X^T W Y
            W_sqrt = np.sqrt(np.diag(weights[i]))  # Efficient diagonal weight matrix
            X_weighted = W_sqrt @ X_neighbors
            Y_weighted = W_sqrt @ Y_neighbors

            # Solve for theta using np.linalg.solve (faster and more stable)
            theta = np.linalg.solve(X_weighted.T @ X_weighted, X_weighted.T @ Y_weighted)

            # Predict outcome for the current query point
            predictions[i] = x[i] @ theta

        return predictions
