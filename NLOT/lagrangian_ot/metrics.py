# Copyright (c) Meta Platforms, Inc. and affiliates

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from flax.training import train_state
from sklearn.cluster import KMeans

import copy

from typing import Tuple, Optional

from dataclasses import dataclass, field

from abc import ABC, abstractmethod

plot_cache = {}

@dataclass
class MetricBase(nn.Module):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


@dataclass
class EuclideanMetric(MetricBase):
    def __call__(self, x):
        assert x.ndim == 1
        D = x.shape[0]
        return jnp.eye(D)


@dataclass
class ScarvelisMetric(MetricBase, ABC):
    div_eps: float = 1e-6
    metric_eps: float = 1e-3

    def __call__(self, x):
        assert x.ndim == 1
        D = x.shape[0]
        v_vals = self.v(x)
        return jnp.eye(D) - (1-self.metric_eps)*jnp.outer(v_vals, v_vals)

    @abstractmethod
    def v(self, x):
        raise NotImplementedError


@dataclass
class CircleMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        norm = jnp.clip(jnp.linalg.norm(x), a_min=self.div_eps)
        return jnp.array([-x[1], x[0]]) / norm

@dataclass
class VeeMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        sign_y = jnp.sign(x[1])
        return jnp.array([1./jnp.sqrt(2), sign_y/jnp.sqrt(2)])

@dataclass
class XMetric(ScarvelisMetric):
    def v(self, x):
        assert x.ndim == 1 and x.shape[0] == 2
        a = 1.25 * jax.nn.tanh(jax.nn.relu(x[0]*x[1]))
        b = -1.25 * jax.nn.tanh(jax.nn.relu(-x[0]*x[1]))
        v1 = jnp.array([1./jnp.sqrt(2), 1./jnp.sqrt(2)])
        v2 = jnp.array([1./jnp.sqrt(2), -1./jnp.sqrt(2)])
        return (a*v1 + b*v2) / 1.25

@dataclass
class NeuralNetMetric(MetricBase):
    D: int = 2
    C: int = 0
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4

    def setup(self):
        assert self.D == 2
        self.net = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(2)
        ])

    def __call__(self, x):
        assert x.ndim == 1
        
        if self.categorical:
            assert x.shape[0] == self.D + 1, f"Expected input shape ({self.D + 1},), got {x.shape}"
            x_ambient = x[:self.D]
            category_index = x[self.D].astype(jnp.int32) # Ensure integer type
            category_one_hot = jax.nn.one_hot(category_index, num_classes=self.num_categories)
            net_input = jnp.concatenate([x_ambient, category_one_hot])
        else:
            assert x.shape[0] == self.D + self.C, f"Expected input shape ({self.D + self.C},), got {x.shape}"
            net_input = x

        theta = jnp.arctan2(*self.net(net_input).squeeze())
        R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
        Q = jnp.array([[1., 0.],
                       [0., 0.1]])

        A = R.T @ Q @ R
        return A


@dataclass
class NeuralNetMetricDirect(MetricBase):
    D: int = 2
    C: int = 0
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4
    eta: Optional[float] = 1e-3

    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(128),
            nn.leaky_relu,
            nn.Dense(self.D * self.D)
        ])

    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D + self.C

        if self.categorical:
            x_ambient = x[:self.D]
            category_index = x[self.D].astype(jnp.int32)
            category_one_hot = jax.nn.one_hot(category_index, num_classes=self.num_categories)
            net_input = jnp.concatenate([x_ambient, category_one_hot])
        else:
            net_input = x
        
        nn_out = self.net(net_input)
        Q = jnp.reshape(nn_out, (self.D, self.D))
        A = Q.T @ Q + self.eta * jnp.eye(self.D)

        return A

@dataclass
class NeuralNetMetricEig(MetricBase):
    D: int = 2 #ambient dimension
    C: int = 0 #conditional dimension
    hidden_dim: int = 128
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4
    min_eigenvalue: float = 0.1
    max_eigenvalue: float = 1
    temperature: float = 1.0
    total_budget: Optional[float] = None  # Total eigenvalue budget, defaults to D if None
    use_film: Optional[bool] = True # Add a flag to enable/disable FiLM

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1

        # Determine output size based on D
        if self.D == 2:
            output_size = self.D + 2
        else:
            output_size = self.D + (self.D * (self.D - 1)) // 2

        if self.categorical:
            assert x.shape[0] == self.D + 1, f"Expected categorical input shape ({self.D + 1},), got {x.shape}"
            x_ambient = x[:self.D]
            category_index = x[self.D].astype(jnp.int32)
            category_one_hot = jax.nn.one_hot(category_index, num_classes=self.num_categories)
            # Embed category and concatenate with spatial features
            # Adjust embedding_dim as needed
            embedding_dim = max(16, self.hidden_dim // 4)
            cat_embedding = nn.Embed(num_embeddings=self.num_categories, features=embedding_dim)(category_one_hot)
            
            # Process spatial features separately first
            h_spatial = nn.Dense(self.hidden_dim, name="spatial_dense_0")(x_ambient)
            h_spatial = nn.leaky_relu(h_spatial)

            # Concatenate processed spatial features and embedding
            h = jnp.concatenate([h_spatial, cat_embedding], axis=-1)
            # Add a dense layer to combine them to the main hidden dim
            h = nn.Dense(self.hidden_dim, name="combine_dense")(h)
            h = nn.leaky_relu(h)
            current_hidden_idx = 1

        elif self.use_film and not self.categorical:
            assert x.shape[0] == self.D + self.C, f"Expected continuous conditional input shape ({self.D + self.C},), got {x.shape}"
            x_ambient = x[:self.D]
            c = x[self.D:]

            # --- FiLM Implementation ---
            # Process spatial features (first hidden layer)
            h_spatial = nn.Dense(self.hidden_dim, name="spatial_dense_0")(x_ambient)

            # FiLM generator network
            film_hidden_dim = max(16, self.hidden_dim // 4)
            film_params = nn.Dense(film_hidden_dim, name="film_dense_0")(c)
            film_params = nn.leaky_relu(film_params)
            # Output size is 2 * target activation size (gamma and beta)
            film_params = nn.Dense(2 * self.hidden_dim, name="film_dense_1")(film_params)

            gamma = film_params[:self.hidden_dim]
            beta = film_params[self.hidden_dim:]

            # Apply FiLM
            h = gamma * h_spatial + beta
            h = nn.leaky_relu(h)
            # --- End FiLM ---
            current_hidden_idx = 1

        else: #just simple concatenation for condition
            assert x.shape[0] == self.D + self.C, f"Expected concatenated input shape ({self.D + self.C},), got {x.shape}"
            net_input = x
            h = nn.Dense(self.hidden_dim, name="dense_0")(net_input)
            h = nn.leaky_relu(h)
            current_hidden_idx = 1


        h = nn.Dense(self.hidden_dim, name=f"dense_{current_hidden_idx}")(h)
        h = nn.leaky_relu(h)
        nn_out = nn.Dense(output_size, name="output_dense")(h)


        # --- Eigenvalue and Rotation Calculation ---
        raw_eigenvalues = nn_out[:self.D]

        # Allocation of eigenvalue budget using softmax
        eigenvalue_weights = jax.nn.softmax(raw_eigenvalues * self.temperature)

        # Set total budget to D if not specified
        budget = self.D if self.total_budget is None else self.total_budget

        # Compute eigenvalues: ensure minimum values while allocating the budget
        # Ensure budget_range calculation avoids issues if max=min
        budget_range = jnp.maximum(0.0, self.max_eigenvalue - self.min_eigenvalue)
        # Scale weights by budget relative to default (D) and the available range
        scaled_weights = eigenvalue_weights * (budget / self.D) * budget_range
        eigenvalues = self.min_eigenvalue + scaled_weights
        eigenvalues = jnp.clip(eigenvalues, self.min_eigenvalue, self.max_eigenvalue)


        rotation = self._create_rotation_matrix(nn_out[self.D:])
        diagonal = jnp.diag(eigenvalues)
        A = rotation.T @ diagonal @ rotation

        return A

    def _create_rotation_matrix(self, params):
        """
        Create a rotation matrix from parameters.
        For D=2: uses arctan2 from 2D vector like original implementation
        For D=3: 3 angle parameters (Euler angles)
        For D>3: D(D-1)/2 parameters for generalized rotation
        """
        if self.D == 2:
            theta = jnp.arctan2(params[1], params[0])
            rotation = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)]
            ])
            return rotation
        else:
            # For higher dimensions, use a series of Givens rotations
            # This is one approach to parameterize SO(n) with D(D-1)/2 parameters
            rotation = jnp.eye(self.D)
            param_idx = 0
            
            for i in range(self.D):
                for j in range(i+1, self.D):
                    # Create a Givens rotation in the (i,j) plane
                    angle = params[param_idx]
                    param_idx += 1
                    
                    # Create rotation matrix for this plane
                    givens = jnp.eye(self.D)
                    givens = givens.at[i, i].set(jnp.cos(angle))
                    givens = givens.at[j, j].set(jnp.cos(angle))
                    givens = givens.at[i, j].set(-jnp.sin(angle))
                    givens = givens.at[j, i].set(jnp.sin(angle))
                    
                    rotation = rotation @ givens
                    
            return rotation

@dataclass
class LANDMetric(MetricBase):
    D: int = 2
    C: int = 0
    samples: jnp.ndarray = None
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4

    # --- LAND Parameters ---
    gamma: float = 0.2 # Width parameter for the weighting kernel
    rho: float = 0.001  # Regularization parameter
    alpha: float = 1.0  # Power for the metric
    
    
    def setup(self):
        if self.samples is None:
            raise ValueError("The 'samples' parameter must be provided for LANDMetric")
        
    
    def _get_processed_samples(self):
        # Reshape paired samples if they have shape e.g. (2,100,2) to (200,2)
        if len(self.samples.shape) > 2:
            return jnp.reshape(self.samples, (-1, self.D))
        return self.samples
    
    def _weighting_function(self, x, samples):
        pairwise_sq_diff = (x - samples) ** 2
        pairwise_sq_dist = jnp.sum(pairwise_sq_diff, axis=-1)
        weights = jnp.exp(-pairwise_sq_dist / (2 * self.gamma**2))
        return weights
    
    def __call__(self, x):
        assert x.ndim == 1 and x.shape[0] == self.D
        processed_samples = self._get_processed_samples()
        
        # Compute weights for each sample based on distance to x
        weights = self._weighting_function(x, processed_samples)
        
        # Compute the weighted sum of squared differences for each dimension
        differences = processed_samples - x
        squared_differences = differences**2
        M_diag = jnp.zeros(self.D)
        for d in range(self.D):
            M_diag = M_diag.at[d].set(
                jnp.sum(weights * squared_differences[:, d]) + self.rho
            )
        
        M_diag = M_diag ** self.alpha
        M_diag = 1/M_diag
        M = jnp.diag(M_diag)
        return M
    

def assign_to_centers(samples: jnp.ndarray, centers: jnp.ndarray) -> jnp.ndarray:
    """
    Assigns each sample to the index of the nearest center using manual
    squared Euclidean distance calculation with broadcasting.

    Args:
        samples: Array of samples, shape (N, D).
        centers: Array of centers, shape (K, D).

    Returns:
        Array of labels (indices of nearest centers), shape (N,).
    """
    # Ensure inputs are JAX arrays
    samples = jnp.asarray(samples)
    centers = jnp.asarray(centers)

    # Calculate squared norms for samples and centers
    # sum(samples**2, axis=1) -> shape (N,)
    samples_sq_norms = jnp.sum(samples**2, axis=1)
    # sum(centers**2, axis=1) -> shape (K,)
    centers_sq_norms = jnp.sum(centers**2, axis=1)

    # Calculate dot products: samples @ centers.T -> shape (N, K)
    dot_products = jnp.dot(samples, centers.T)

    # Calculate squared Euclidean distances using broadcasting:
    # ||a - b||^2 = ||a||^2 - 2 * a^T * b + ||b||^2
    # samples_sq_norms[:, None] -> shape (N, 1)
    # centers_sq_norms[None, :] -> shape (1, K)
    # The subtraction and addition broadcast correctly to (N, K)
    dist_matrix_sq = samples_sq_norms[:, None] - 2 * dot_products + centers_sq_norms[None, :]

    # Ensure distances are non-negative (numerical precision might cause small negatives)
    dist_matrix_sq = jnp.maximum(dist_matrix_sq, 0.0)

    # Find the index of the minimum distance for each sample: shape (N,)
    labels = jnp.argmin(dist_matrix_sq, axis=1)
    return labels

def compute_sigmas(samples: jnp.ndarray, centers: jnp.ndarray, labels: jnp.ndarray, K: int) -> jnp.ndarray:
    """Computes the standard deviation (sigma) for each cluster."""
    D = samples.shape[1]
    sigmas = jnp.zeros(K)
    samples = jnp.asarray(samples) # Ensure JAX array
    centers = jnp.asarray(centers) # Ensure JAX array

    for k in range(K):
        points_in_cluster = samples[labels == k]
        num_points = points_in_cluster.shape[0]
        if num_points > 0:
            center_k = centers[k]
            # Use squared distance for variance calc
            sq_distances_from_center = jnp.sum((points_in_cluster - center_k)**2, axis=1)
            # Mean squared distance (variance estimate)
            variance_k = jnp.mean(sq_distances_from_center)
            sigma_k = jnp.sqrt(variance_k)
            sigmas = sigmas.at[k].set(jnp.maximum(sigma_k, 1e-6)) # Avoid sigma=0
        else:
            # Handle empty clusters - assign a default sigma (e.g., 1.0 or average)
             # print(f"Warning: Cluster {k} is empty during sigma calculation.") # Optional warning
             sigmas = sigmas.at[k].set(1.0) # Placeholder default sigma

    # Post-process: fill remaining zeros (from empty clusters) with average sigma
    # This prevents lambda from becoming huge/infinite for empty clusters
    non_empty_sigmas = sigmas[sigmas > 1e-6]
    avg_sigma = jnp.mean(non_empty_sigmas) if non_empty_sigmas.size > 0 else 1.0
    # Ensure avg_sigma is at least epsilon to avoid issues if all clusters somehow had sigma=0
    avg_sigma = jnp.maximum(avg_sigma, 1e-6)
    sigmas = jnp.where(sigmas < 1e-6, avg_sigma, sigmas)

    return sigmas

@dataclass
class RBFMetric(MetricBase):
    """
    RBF Metric with internal methods to calculate centers and train weights.

    Workflow:
    1. Instantiate (provide K - num centers, samples).
    2. Call `calculate_centers()` (requires sklearn).
    3. Call `init()` to run setup (computes lambdas) and get params structure.
    4. Call `calculate_and_set_weights()`.
    5. Use via `apply()`.
    """
    D: int = 2
    C: int = 0
    samples: jnp.ndarray = None
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4

    # --- RBF Parameters ---
    K: int = 2                   # Number of centers to compute
    kappa: float = 1.0
    alpha: float = 1.0
    epsilon: float = 1e-2
    weight_initializer: nn.initializers.Initializer = nn.initializers.uniform(scale=0.1)
    kmeans_n_init: int = 10       # Hyperparameter for KMeans stability
    kmeans_random_state: Optional[int] = 0 # For reproducible KMeans

    # --- Internal fields ---
    # Centers are now computed internally, not passed at init
    _computed_centers: Optional[jnp.ndarray] = field(init=False, default=None)
    lambdas: jnp.ndarray = field(init=False, default=None)
    _trained_weights: Optional[jnp.ndarray] = field(init=False, default=None)
    _initialized: bool = field(init=False, default=False) # Tracks if setup ran

    def calculate_centers(self, verbose: bool = True):
        """
        Calculates RBF centers using KMeans on the stored samples.
        Requires scikit-learn to be installed.
        Stores the result in `self._computed_centers`.
        MUST be called before `init()`.
        """

        if self.samples is None or self.samples.shape[0] == 0:
            raise ValueError("Cannot calculate centers: 'samples' attribute is not set or is empty.")
        if self.K <= 0:
             raise ValueError("Cannot calculate centers: 'K' (number of centers) must be positive.")

        if verbose:
            print(f"Calculating {self.K} centers using KMeans (n_init={self.kmeans_n_init})...")

        # Convert samples to NumPy for sklearn
        samples_np = np.array(self.samples)

        # Run KMeans
        kmeans = KMeans(
            n_clusters=self.K,
            random_state=self.kmeans_random_state,
            n_init=self.kmeans_n_init
        )
        kmeans.fit(samples_np)

        # Store centers as JAX array
        self._computed_centers = jnp.array(kmeans.cluster_centers_)

        if verbose:
            print(f"Center calculation finished. Computed centers shape: {self._computed_centers.shape}")

    def setup(self):
        """
        Validates inputs, calculates fixed lambdas using pre-calculated centers,
        and defines initial learnable weights.
        Requires `calculate_centers()` to have been called before this runs (via init).
        """
        if self._initialized: return

        # Critical check: Ensure centers have been computed
        if self._computed_centers is None:
             raise RuntimeError("Centers have not been calculated. Call `calculate_centers()` before `init()`.")
        if self._computed_centers.shape != (self.K, self.D):
             # This might happen if K changed after calculate_centers was called but before init
             raise ValueError(f"Internal error: Computed centers shape {self._computed_centers.shape} "
                              f"does not match K={self.K} and D={self.D}.")

        if not (self.samples.ndim == 2 and self.samples.shape[1] == self.D):
            raise ValueError(f"Expected samples shape (N, {self.D}), got {self.samples.shape}")

        jnp_samples = jnp.asarray(self.samples)
        # Use the internally computed centers
        jnp_centers = self._computed_centers

        # Compute and store lambdas
        labels = assign_to_centers(jnp_samples, jnp_centers)
        sigmas = compute_sigmas(jnp_samples, jnp_centers, labels, self.K)
        denominator = (self.kappa * sigmas)**2
        self.lambdas = 0.5 / jnp.maximum(denominator, 1e-8)
        self.lambdas = jnp.asarray(self.lambdas)

        # Define the parameter structure (placeholder for Flax)
        self.learned_weights_param = self.param('learned_weights', self.weight_initializer, (self.K,))
        self._trained_weights = None
        self._initialized = True

    def _compute_h_internal(self, weights: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Internal helper to compute h(x) given specific weights and using computed centers."""
        if self._computed_centers is None:
             # Should ideally not happen if setup ran correctly
             raise RuntimeError("Internal error: Centers not available for h(x) calculation.")

        assert x.ndim == 1 and x.shape[0] == self.D
        x = jnp.asarray(x)
        # Use the computed centers
        centers = self._computed_centers
        x_sq_norm = jnp.sum(x**2)
        centers_sq_norms = jnp.sum(centers**2, axis=1)
        dot_prods = jnp.dot(centers, x)
        dist_sq = centers_sq_norms - 2 * dot_prods + x_sq_norm
        dist_sq = jnp.maximum(dist_sq, 0.0)

        phi_x = jnp.exp(-0.5 * self.lambdas * dist_sq)
        h_x = jnp.sum(weights * phi_x)
        return h_x

    def calculate_and_set_weights(self,
                                  key: jax.random.PRNGKey,
                                  learning_rate: float = 1e-2,
                                  epochs: int = 100,
                                  batch_size: int = 64,
                                  verbose: bool = True):
        """Trains the RBF weights using the stored samples and updates the internal state."""
        if not self._initialized:
             raise RuntimeError("RBFMetric setup has not run. Call `init()` first.")
        # Ensure centers are available (they should be if setup ran)
        if self._computed_centers is None:
             raise RuntimeError("Internal error: Centers not available for weight training.")

        initial_weights = self.weight_initializer(key, (self.K,), dtype=jnp.float32)
        initial_weights = jnp.maximum(initial_weights, 0.0001)

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(initial_weights)
        current_weights = initial_weights

        compute_h_batch = jax.vmap(self._compute_h_internal, in_axes=(None, 0))

        def loss_fn(weights, batch):
            h_values = compute_h_batch(weights, batch)
            loss = jnp.mean((1.0 - h_values)**2)
            return loss

        @jax.jit
        def train_step(weights, opt_state, batch):
            loss, grads = jax.value_and_grad(loss_fn)(weights, batch)
            updates, new_opt_state = optimizer.update(grads, opt_state, weights)
            new_weights = optax.apply_updates(weights, updates)
            new_weights = jnp.maximum(new_weights, 0.0001)
            return loss, new_weights, new_opt_state

        num_samples = self.samples.shape[0]
        steps_per_epoch = max(1, num_samples // batch_size)
        jnp_samples = jnp.asarray(self.samples)

        if verbose: print(f"\nStarting RBF weight training ({epochs} epochs)...")
        for epoch in range(epochs):
            key, subkey = jax.random.split(key)
            perms = jax.random.permutation(subkey, num_samples)
            epoch_loss = 0.0
            processed_samples = 0
            for i in range(steps_per_epoch):
                 batch_start = i * batch_size
                 batch_end = batch_start + batch_size
                 batch_indices = perms[batch_start:batch_end]
                 if batch_indices.size == 0: continue
                 batch = jnp_samples[batch_indices]
                 loss_val, current_weights, opt_state = train_step(current_weights, opt_state, batch)
                 epoch_loss += loss_val * batch.shape[0]
                 processed_samples += batch.shape[0]

            avg_loss = epoch_loss / processed_samples if processed_samples > 0 else 0.0
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.6f}")

        self._trained_weights = current_weights
        if verbose: print("RBF weight training finished.")

    def compute_h(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the RBF network output h(x) using the TRAINED weights."""
        if self._trained_weights is None:
            raise RuntimeError("RBF weights have not been calculated. Call `calculate_and_set_weights` first.")
        # Check if setup ran (implicitly checks if centers are available)
        if not self._initialized:
             raise RuntimeError("RBFMetric setup has not run. Call `init()` first.")
        return self._compute_h_internal(self._trained_weights, x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the diagonal RBF metric tensor M(x) using computed centers and trained weights."""
        if not self._initialized:
             raise RuntimeError("RBFMetric setup has not run. Call `init()` first.")
        if self._trained_weights is None:
             raise RuntimeError("RBF weights have not been calculated. Call `calculate_and_set_weights` first.")

        h_x = self.compute_h(x)

        denominator = (h_x + self.epsilon)**self.alpha
        M_scalar = 1.0 / jnp.maximum(denominator, 1e-8)
        A = M_scalar * jnp.eye(self.D)
        return A