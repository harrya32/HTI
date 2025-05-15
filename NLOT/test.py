import jax.numpy as jnp
import torch
import numpy as np
import pickle

#read in the samples
samples = torch.load("scarvelis_data/reward_weighting_data_0_10.pt").cpu().numpy()
D = 2
C = 7

# current shape is (3,1000,9), reshape to (3000,3)
N = samples.shape[0] * samples.shape[1]
samples = samples.reshape(N, -1)
samples = jnp.array(samples)
all_samples_ambient = samples[:, :C]
sigma_per_coord = jnp.std(all_samples_ambient, axis=0, ddof=1)
sigma_iso = jnp.mean(sigma_per_coord)
factor = (4.0 / (D + 2)) ** (1.0 / (D + 4))
h0 = factor * N ** (-1.0 / (D + 4)) * sigma_iso
ks = jnp.linspace(-2, 2, 9)
h_grid = h0 * (10.0 ** ks)

print("σ per coord:", sigma_per_coord)
print("σ isotropic:", sigma_iso)
print("h0 (Silverman):", h0)
print("candidate h values:", h_grid)