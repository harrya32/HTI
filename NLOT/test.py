import jax.numpy as jnp
import torch
import numpy as np
import pickle

#read in the samples
"""samples = torch.load("scarvelis_data/conditional_circles.pt").cpu().numpy()[:-1]
D = 2
C = 1

# current shape is (4,400,3), reshape to (1600,3)
N = samples.shape[0] * samples.shape[1]


samples = samples.reshape(N, -1)
N=400
samples = samples[:N]

#convert to jax
samples = jnp.array(samples)

# assume samples.shape = (N, D+C)
all_samples_ambient = samples[:, :D]

# 1) per‐coord σ, then isotropic σ
sigma_per_coord = jnp.std(all_samples_ambient, axis=0, ddof=1)
sigma_iso = jnp.mean(sigma_per_coord)

# 2) compute rule-of-thumb h0
factor = (4.0 / (D + 2)) ** (1.0 / (D + 4))
h0 = factor * N ** (-1.0 / (D + 4)) * sigma_iso

# 3) build a log‐grid around h0
ks = jnp.linspace(-2, 2, 9)
h_grid = h0 * (10.0 ** ks)

print("σ per coord:", sigma_per_coord)
print("σ isotropic:", sigma_iso)
print("h0 (Silverman):", h0)
print("candidate h values:", h_grid)"""


#read in conditional circles data, and make it a fake testing file, by creating a list of tuples of (time, samples) for each time point in the data, and saving as a pkl

samples = torch.load("scarvelis_data/conditional_circles.pt").cpu().numpy()

samples_0 = (0, jnp.array(samples[0]))
samples_1 = (0.25, jnp.array(samples[1]))
samples_2 = (0.5, jnp.array(samples[2]))
samples_3 = (0.75, jnp.array(samples[3]))
samples_4 = (1.0, jnp.array(samples[4]))

samples_list = [samples_0, samples_1, samples_2, samples_3, samples_4]

with open("test_data.pkl", "wb") as f:
    pickle.dump(samples_list, f)


