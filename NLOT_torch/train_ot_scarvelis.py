#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse
import math
import functools

import csv
import time
import json
# import jax
# import jax.numpy as jnp
import torch
import numpy as np
import os
# import optax
import torch.optim as optim
import cloudpickle as pkl
from copy import copy

import dataclasses
from typing import Iterator

import hydra
from omegaconf import OmegaConf

# from lagrangian_ot import models, neuraldual, metrics, geodesics, geometries, data
# TODO: Import PyTorch equivalents or implementations later

import matplotlib.pyplot as plt
plt.style.use('bmh')

import sys
# from IPython.core import ultratb # JAX specific debugging
# sys.excepthook = ultratb.FormattedTB(
#     mode='Plain', color_scheme='Neutral', call_pdb=1)

# import wandb # Temporarily comment out WandB

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        # Initialize wandb
        # wandb.init(
        #     project=cfg.get("wandb_project", "NLOT-Scarvelis_" + self.cfg.geometry), # Default project name
        #     entity=cfg.get("wandb_entity", None), # Replace with your entity if needed
        #     config=OmegaConf.to_container(cfg, resolve=True), # Convert Hydra cfg to dict
        #     name=cfg.get("run_name", None), # Optional run name
        #     # mode="disabled" if cfg.get("debug", False) else "online", # Optionally disable for debugging
        # )

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        torch.manual_seed(self.cfg.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.cfg.seed)

        # self.key = jax.random.PRNGKey(self.cfg.seed) # Use torch.manual_seed
        self.elapsed_time = 0.

        # --- Geometry Section (Requires PyTorch Adaptation) ---
        # self.geometry = geometries.get(
        #     self.cfg.geometry, self.cfg.geometry_kwargs)
        #
        # if 'euclidean' in self.cfg.geometry or 'neural' in self.cfg.geometry:
        #     if self.cfg.data is None:
        #         raise ValueError(
        #             'data must be specified for euclidean and neural geometries')
        #
        # self.has_reference_geometry = 'neural' in self.cfg.geometry
        # if self.has_reference_geometry:
        #     self.reference_geometry = geometries.get(
        #         self.cfg.data, self.cfg.geometry_kwargs)
        # --- End Geometry Section ---

        if self.cfg.data is None:
            self.cfg.data = self.cfg.geometry # Assuming geometry name is used as data key if data is None
        # Pass the requested number of pairs from config to the sampler function
        # Store samplers as instance variable
        # TODO: Adapt data loading for PyTorch
        # self.samplers = data.get_samplers_scarvelis(
        #     self.cfg.data,
        #     num_pairs_requested=self.cfg.get("num_pairs", None) # Use .get for safety
        # )
        # self.geometry.bounds, self.geometry.xbounds, self.geometry.ybounds = data.get_bounds(self.cfg.data)

        # Placeholder for samplers and data bounds until data loading is adapted
        print("Warning: Data loading needs adaptation for PyTorch.")
        self.samplers = [] # Placeholder
        self.num_pairs = 0 # Placeholder
        self.eval_samples = [] # Placeholder
        # self.geometry.bounds, self.geometry.xbounds, self.geometry.ybounds = (None, None, None) # Placeholder

        # Update num_pairs based on the actual length of the loaded samplers
        # self.num_pairs = len(self.samplers) - 1
        # print(f'training on {self.num_pairs} pairs')
        # Use self.samplers for eval samples
        # self.eval_samples = [torch.tensor(next(s), dtype=torch.float32).to(self.device) for s in self.samplers] # Adapted eval samples

        # --- Optimizer and Model State Initialization (Requires PyTorch Adaptation) ---
        # self.optimizer_target_potential = optax.adamw(
        #     learning_rate=self.cfg.potential_lr)
        # self.optimizer_source_map = self.optimizer_target_potential
        # self.optimizer_geom = optax.adamw(
        #     learning_rate=self.cfg.metric.lr)

        # Placeholder optimizers
        print("Warning: Optimizers need adaptation for PyTorch.")
        self.optimizer_target_potential = None # Placeholder
        self.optimizer_source_map = None # Placeholder
        self.optimizer_geom = None # Placeholder


        # k1, self.key = jax.random.split(self.key) # Use torch.manual_seed
        # self.params_geometry = self.geometry.init(
        #     k1, self.eval_samples[0][0], self.eval_samples[1][0],
        #     method=self.geometry.cost
        # ).get('params', {})
        # self.state_geometry = self.optimizer_geom.init(self.params_geometry) # Needs PyTorch state handling

        print("Warning: Model/Potential/Map initialization needs adaptation for PyTorch.")
        # target_potential = models.MLP( # Needs PyTorch Model definition
        #     dim_hidden=self.cfg.target_potential_dim_hidden,
        #     is_potential=True)
        # source_map = models.MLP( # Needs PyTorch Model definition
        #     dim_hidden=self.cfg.source_map_dim_hidden,
        #     is_potential=False)
        # ctransform_solver = hydra.utils.instantiate(self.cfg.ctransform_solver) # Might need adaptation
        # self.neural_dual_solver = neuraldual.ManifoldW2NeuralDual( # Needs PyTorch NeuralDual implementation
        #     geometry=self.geometry,
        #     target_potential=target_potential,
        #     source_map=source_map,
        #     ctransform_solver=ctransform_solver,
        # )
        self.target_potential_model = None # Placeholder
        self.source_map_model = None # Placeholder
        self.neural_dual_solver = None # Placeholder


        # init_key, self.key = jax.random.split(self.key) # Use torch.manual_seed
        # state_target_potential, state_source_map = self.neural_dual_solver.initialize_states(
        #     self.optimizer_target_potential, self.optimizer_source_map,
        #     init_key, self.eval_samples[0], self.eval_samples[1]) # Needs PyTorch state init
        # self.state_target_potentials = [state_target_potential] # Placeholder list of models/states
        # self.state_source_maps = [state_source_map] # Placeholder list of models/states
        self.target_potential_models = [] # Placeholder
        self.source_map_models = [] # Placeholder
        self.target_potential_optimizers = [] # Placeholder
        self.source_map_optimizers = [] # Placeholder

        # --- Spline Amortizer (Requires PyTorch Adaptation) ---
        # if 'spline_model' in self.params_geometry:
            # Pass self.samplers to fit_spline_amortizer
            # self.fit_spline_amortizer(self.samplers, init=True)
        # --- End Spline Amortizer ---

        self.train_step = 0


    # --- Function Definitions Requiring PyTorch Adaptation ---

    # def fit_spline_amortizer(self, samplers, init):
    #     num_iters = self.cfg.spline.init_train_iters if init else self.cfg.spline.train_iters
    #
    #     if init:
    #         def sampler(key): # Needs torch random state handling
    #             # sample from random pairs of source and target
    #             t = 0
    #             while True:
    #                 source_samples = torch.tensor(next(samplers[t]), dtype=torch.float32).to(self.device)
    #                 target_samples = torch.tensor(next(samplers[t+1]), dtype=torch.float32).to(self.device)
    #                 all_samples = torch.cat([source_samples, target_samples], dim=0)
    #                 # k1, key = jax.random.split(key) # Use torch.randperm
    #                 perm_indices = torch.randperm(all_samples.size(0))
    #                 all_samples = all_samples[perm_indices]
    #                 t = (t + 1) % self.num_pairs
    #                 yield all_samples
    #
    #         # k1, k2, self.key = jax.random.split(self.key, 3) # Use torch seeds
    #         xsampler = iter(sampler(None)) # Key not needed with torch seeds
    #         ysampler = iter(sampler(None)) # Key not needed with torch seeds
    #     else:
    #         def xsampler():
    #             # key = jax.random.PRNGKey(0) # Use torch seeds
    #             t = 0
    #             while True:
    #                 source_samples = torch.tensor(next(samplers[t]), dtype=torch.float32).to(self.device)
    #                 t = (t + 1) % self.num_pairs
    #                 yield source_samples
    #
    #         def ysampler():
    #             # key = jax.random.PRNGKey(0) # Use torch seeds
    #             t = 0
    #             while True:
    #                 source_samples = torch.tensor(next(samplers[t]), dtype=torch.float32).to(self.device)
    #                 # transported_samples = self.neural_dual_solver.source_map_apply_jit( # Needs PyTorch model application
    #                 #     {'params': self.state_source_maps[t].params}, source_samples)
    #                 # Placeholder for transported samples
    #                 transported_samples = source_samples # Replace with actual map application
    #                 if self.cfg.spline.noise > 0.:
    #                     # k1, key = jax.random.split(key) # Use torch.randn
    #                     noise = torch.randn_like(transported_samples) * self.cfg.spline.noise
    #                     transported_samples += noise.to(self.device)
    #
    #                 t = (t + 1) % self.num_pairs
    #                 yield transported_samples
    #
    #         xsampler = iter(xsampler())
    #         ysampler = iter(ysampler())
    #
    #
    #     # self.params_geometry = self.geometry.spline_amortizer.train( # Needs PyTorch training loop
    #     #     self.params_geometry,
    #     #     xsampler, ysampler,
    #     #     max_iter=num_iters,
    #     #     grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
    #     # )
    #     print("Warning: fit_spline_amortizer needs PyTorch adaptation.")


    # def update_all_states(self, state_target_potentials, state_source_maps, batches):
    #     # This function needs significant rework for PyTorch's training loop structure
    #     # (e.g., iterating through models/optimizers, zeroing grads, loss calculation, backward pass, optimizer step)
    #     print("Warning: update_all_states needs PyTorch adaptation.")
    #     # out = []
    #     # for t in range(self.num_pairs):
    #     #     out_t = self.neural_dual_solver.update_fn_jit( # Needs PyTorch update function
    #     #         state_target_potentials[t if self.train_step > 0 else 0],
    #     #         state_source_maps[t if self.train_step > 0 else 0],
    #     #         self.params_geometry,
    #     #         batches[t],
    #     #     )
    #     #     out.append(out_t)
    #     #
    #     #     if self.cfg.spline.update_on_conjugates \
    #     #             and 'spline_model' in self.params_geometry:
    #     #         _, info = out_t
    #     #         # Update spline amortizer (needs PyTorch adaptation)
    #     #         # self.params_geometry = self.geometry.spline_amortizer.train_single(
    #     #         #     self.params_geometry,
    #     #         #     batches[t]['source'], info.target_hat,
    #     #         #     verbose=False,
    #     #         # )
    #
    #     # new_states, infos = zip(*out)
    #     # new_states = zip(*new_states)
    #     # mean_info = type(infos[0])(
    #     #     *[jnp.array(x).mean() for x in list(zip(*infos))]) # Needs torch.mean
    #     # return new_states, mean_info
    #     return None, None # Placeholder

    def sample_all_batches(self, samplers):
        batches = []
        for t in range(self.num_pairs):
            # Convert numpy arrays from sampler to PyTorch tensors
            source_np = next(samplers[t])
            target_np = next(samplers[t+1])
            batches.append({
                "source": torch.tensor(source_np, dtype=torch.float32).to(self.device),
                "target": torch.tensor(target_np, dtype=torch.float32).to(self.device),
            })
        return batches


    # def geometry_loss(self, params_geometry,
    #                   state_target_potentials, state_source_maps,
    #                   batches, key): # Key not needed for PyTorch
    #     # Needs PyTorch adaptation: model forward passes, loss calculation
    #     print("Warning: geometry_loss needs PyTorch adaptation.")
    #     # metric = lambda x: self.geometry.apply( # Needs PyTorch geometry apply
    #     #     {'params': params_geometry},
    #     #     x, method=self.geometry.metric)
    #     # metric_vmap = jax.vmap(metric) # Use torch operations directly
    #     # metric_jac_vmap = jax.vmap(jax.jacfwd(metric)) # Use torch.autograd.functional.jacobian
    #
    #     # dual_losses = []
    #     # for t in range(self.num_pairs):
    #     #     batch = batches[t]
    #     #     _, info_t = self.neural_dual_solver.loss_fn( # Needs PyTorch loss function
    #     #         state_target_potentials[t].params,
    #     #         state_source_maps[t].params,
    #     #         params_geometry, batch)
    #     #     dual_losses.append(-info_t.dual_loss)
    #
    #     # mean_dual_loss = torch.mean(torch.stack(dual_losses)) # Adapted mean calculation
    #     # total_loss = mean_dual_loss
    #     # return total_loss
    #     return torch.tensor(0.0, requires_grad=True) # Placeholder loss

    # @functools.partial(jax.jit, static_argnums=[0]) # Remove JAX jit
    # def update_geometry(self, params_geometry, state_geometry,
    #                     state_target_potentials, state_source_maps,
    #                     batches, key): # Key not needed
    #     # Needs PyTorch adaptation: loss calculation, backward pass, optimizer step
    #     print("Warning: update_geometry needs PyTorch adaptation.")
    #     # geometry_grad_fn = jax.value_and_grad(self.geometry_loss) # Use loss.backward()
    #     # loss, grads = geometry_grad_fn(
    #     #     params_geometry,
    #     #     state_target_potentials, state_source_maps,
    #     #     batches, key)
    #
    #     # TODO: could remove 'spline_model' from updates
    #
    #     # updates, new_opt_state = self.optimizer_geom.update( # Use optimizer.step()
    #     #     grads, state_geometry, params_geometry)
    #     # new_params = optax.apply_updates(params_geometry, updates) # Updates applied by optimizer.step()
    #     # return new_params, new_opt_state, loss
    #     return None, None, None # Placeholder

    def run(self):
        print("Starting training...")
        start_time = time.time()

        # Placeholder for actual PyTorch models and optimizers
        # We need to instantiate them properly before the loop
        # Example:
        # self.target_potential_models = [YourPotentialModel(...).to(self.device) for _ in range(self.num_pairs)]
        # self.source_map_models = [YourMapModel(...).to(self.device) for _ in range(self.num_pairs)]
        # self.target_potential_optimizers = [optim.AdamW(model.parameters(), lr=self.cfg.potential_lr) for model in self.target_potential_models]
        # self.source_map_optimizers = [optim.AdamW(model.parameters(), lr=self.cfg.potential_lr) for model in self.source_map_models]
        # if self.geometry_model: # Assuming a geometry model exists
        #      self.geom_optimizer = optim.AdamW(self.geometry_model.parameters(), lr=self.cfg.metric.lr)

        # --- Main Training Loop (Needs PyTorch Adaptation) ---
        for step in range(self.cfg.train_iters):
            self.train_step = step
            # batches = self.sample_all_batches(self.samplers) # Assumes samplers are adapted

            # --- Update Potentials and Maps ---
            # This needs a PyTorch training step implementation:
            # 1. Zero gradients (model.zero_grad() or optimizer.zero_grad())
            # 2. Forward pass to compute loss (using models and batches)
            # 3. Backward pass (loss.backward())
            # 4. Optimizer step (optimizer.step())

            # Placeholder for state updates
            # new_states, info = self.update_all_states(
            #     self.state_target_potentials, self.state_source_maps, batches)
            # self.state_target_potentials, self.state_source_maps = new_states
            print(f"Step {step}: Warning - Training step logic requires PyTorch adaptation.")


            # --- Update Geometry ---
            if step % self.cfg.metric.update_every == 0 and step > self.cfg.metric.warmup:
                # Needs PyTorch geometry update step:
                # 1. Zero geometry optimizer gradients
                # 2. Calculate geometry loss (using current potentials/maps)
                # 3. Backward pass on geometry loss
                # 4. Geometry optimizer step
                # update_key, self.key = jax.random.split(self.key) # Not needed
                # self.params_geometry, self.state_geometry, geom_loss = self.update_geometry(
                #     self.params_geometry, self.state_geometry,
                #     self.state_target_potentials, self.state_source_maps,
                #     batches, update_key)
                print(f"Step {step}: Warning - Geometry update logic requires PyTorch adaptation.")
                geom_loss = torch.tensor(0.0) # Placeholder

            # --- Spline Amortizer Update ---
            # if \'spline_model\' in self.params_geometry and not self.cfg.spline.update_on_conjugates:
            #     if step % self.cfg.spline.update_every == 0 and step > self.cfg.spline.warmup:
            #         self.fit_spline_amortizer(self.samplers, init=False)

            # --- Logging ---
            if step % self.cfg.logging_freq == 0 or step == self.cfg.train_iters - 1:
                elapsed_time = time.time() - start_time
                log_data = {
                    "step": step,
                    "elapsed_time": elapsed_time,
                    # "dual_loss": float(info.dual_loss) if info else 0.0, # Adapt based on PyTorch loss calc
                    # "grad_norm_target": float(info.grad_norm_target) if info else 0.0, # Adapt
                    # "grad_norm_source": float(info.grad_norm_source) if info else 0.0, # Adapt
                    # "geom_loss": float(geom_loss) if geom_loss is not None else 0.0, # Adapt
                }
                # if self.has_reference_geometry:
                #     # log_data[\"alignment\"] = self.eval_alignment() # Needs PyTorch adaptation
                #     pass
                print(f"Step: {step}, Elapsed Time: {elapsed_time:.2f}s") # Basic logging
                # wandb.log(log_data) # Log to wandb if enabled


            # --- Evaluation & Plotting ---
            if step % self.cfg.eval_freq == 0 or step == self.cfg.train_iters - 1:
                # self.plot() # Needs PyTorch adaptation
                # self.plot_all_pairs() # Needs PyTorch adaptation
                print(f"Step {step}: Warning - Plotting requires PyTorch adaptation.")
                pass

            # --- Saving ---
            if step % self.cfg.save_freq == 0 or step == self.cfg.train_iters - 1:
                # self.save(tag=f"step_{step}") # Needs PyTorch adaptation for saving models/states
                print(f"Step {step}: Warning - Saving requires PyTorch adaptation.")
                pass

        self.elapsed_time = time.time() - start_time
        print(f"Training finished in {self.elapsed_time:.2f} seconds")
        # self.save() # Final save
        # wandb.finish() # Finish wandb run

    # --- Evaluation and Plotting Functions (Need PyTorch Adaptation) ---

    # def eval_alignment(self):
    #     print("Warning: eval_alignment needs PyTorch adaptation.")
    #     # metric = self.geometry.metric
    #     # ref_metric = self.reference_geometry.metric
    #     # params_geometry = self.params_geometry
    #     #
    #     # @functools.partial(jax.jit, static_argnums=[0])
    #     # @functools.partial(jax.vmap, in_axes=(None, None, 0))
    #     # def eigvecs(geometry, params_geometry, x):
    #     #     m = geometry.metric({'params': params_geometry}, x)
    #     #     eigvals, eigvecs = jnp.linalg.eigh(m)
    #     #     return eigvecs
    #     #
    #     # batch = next(self.eval_batcher)
    #     # learned_eigvecs = eigvecs(self.geometry, params_geometry, batch['source'])
    #     # reference_eigvecs = eigvecs(self.reference_geometry, {}, batch['source'])
    #     #
    #     # alignment = metrics.principal_angles(learned_eigvecs, reference_eigvecs)
    #     # return float(alignment)
    #     return 0.0

    # def _init_logging(self):
    #     # Logging initialization might differ with PyTorch (e.g., TensorBoard)
    #     print("Warning: _init_logging might need changes.")
    #     log_headers = ['step', 'dual_loss', 'grad_norm_target', 'grad_norm_source', 'geom_loss']
    #     if self.has_reference_geometry:
    #         log_headers.append('alignment')
    #     self.csv_writer.writerow(log_headers)
    #     self.log_file.flush()

    # def plot(self):
    #     print("Warning: plot needs PyTorch adaptation.")
        # self.plot_pushforward()

    # def plot_all_pairs(self):
    #     print("Warning: plot_all_pairs needs PyTorch adaptation.")
    #     # fig, axs = plt.subplots(1, self.num_pairs, figsize=(5 * self.num_pairs, 5))
    #     # if self.num_pairs == 1:
    #     #     axs = [axs] # Ensure axs is iterable
    #     # for t in range(self.num_pairs):
    #     #     ax = axs[t]
    #     #     source_samples = self.eval_samples[t]['source']
    #     #     target_samples = self.eval_samples[t+1]['target']
    #     #     # Apply source map (needs PyTorch model application)
    #     #     # transported_samples = self.neural_dual_solver.source_map_apply_jit(
    #     #     #     {'params': self.state_source_maps[t].params}, source_samples
    #     #     # )
    #     #     # Placeholder: Use source samples if map not available
    #     #     transported_samples = source_samples.cpu().numpy() if isinstance(source_samples, torch.Tensor) else source_samples
    #     #     source_samples_np = source_samples.cpu().numpy() if isinstance(source_samples, torch.Tensor) else source_samples
    #     #     target_samples_np = target_samples.cpu().numpy() if isinstance(target_samples, torch.Tensor) else target_samples
    #
    #     #     ax.scatter(target_samples_np[:, 0], target_samples_np[:, 1], s=10, alpha=0.5, label='Target (t+1)')
    #     #     ax.scatter(transported_samples[:, 0], transported_samples[:, 1], s=10, alpha=0.5, label='Transported Source (T(x_t))')
    #     #     ax.set_title(f'Pair {t} -> {t+1}')
    #     #     self._setup_ax(ax)
    #     #     # Add arrows
    #     #     if source_samples_np.shape[0] == transported_samples.shape[0]:
    #     #         for i in range(min(50, source_samples_np.shape[0])): # Plot limited arrows
    #     #             ax.arrow(source_samples_np[i, 0], source_samples_np[i, 1],
    #     #                      transported_samples[i, 0] - source_samples_np[i, 0],
    #     #                      transported_samples[i, 1] - source_samples_np[i, 1],
    #     #                      head_width=0.03, head_length=0.05, fc='gray', ec='gray', alpha=0.3)
    #     # fig.tight_layout()
    #     # plt.savefig(f'all_pairs_transport_step_{self.train_step}.png')
    #     # plt.close(fig)
    #     pass


    def _setup_ax(self, ax):
        # This should work fine with matplotlib regardless of backend
        if hasattr(self, 'geometry') and hasattr(self.geometry, 'bounds'):
            ax.set_xlim(self.geometry.bounds[0])
            ax.set_ylim(self.geometry.bounds[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend()

    def _clean_axis(self, ax):
        # This should work fine with matplotlib
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect('equal', adjustable='box')


    # def plot_pushforward(self, ax=None, fname='pushforward.png', num_samples=100):
    #     print("Warning: plot_pushforward needs PyTorch adaptation.")
    #     # Assumes plotting for the first pair (t=0)
    #     # t = 0
    #     # if not self.state_source_maps: # Check if source maps exist
    #     #     print("Cannot plot pushforward: No source map state available.")
    #     #     return
    #     #
    #     # close_fig = ax is None
    #     # if ax is None:
    #     #     fig, ax = plt.subplots(figsize=(6, 6))
    #     #
    #     # source_batch = self.eval_samples[t]['source'][:num_samples]
    #     # target_batch = self.eval_samples[t+1]['target'][:num_samples]
    #     #
    #     # # Apply source map (needs PyTorch adaptation)
    #     # # transported_batch = self.neural_dual_solver.source_map_apply_jit(
    #     # #     {'params': self.state_source_maps[t].params}, source_batch
    #     # # )
    #     # # Placeholder:
    #     # transported_batch = source_batch.cpu().numpy() if isinstance(source_batch, torch.Tensor) else source_batch
    #     # source_batch_np = source_batch.cpu().numpy() if isinstance(source_batch, torch.Tensor) else source_batch
    #     # target_batch_np = target_batch.cpu().numpy() if isinstance(target_batch, torch.Tensor) else target_batch
    #     #
    #     # ax.scatter(target_batch_np[:, 0], target_batch_np[:, 1], label='Target (t+1)', alpha=0.5, s=10)
    #     # ax.scatter(transported_batch[:, 0], transported_batch[:, 1], label='Transported Source (T(x_t))', alpha=0.5, s=10)
    #     # self._setup_ax(ax)
    #     #
    #     # # Plot arrows
    #     # if source_batch_np.shape[0] == transported_batch.shape[0]:
    #     #     for i in range(source_batch_np.shape[0]):
    #     #         ax.arrow(source_batch_np[i, 0], source_batch_np[i, 1],
    #     #                     transported_batch[i, 0] - source_batch_np[i, 0],
    #     #                     transported_batch[i, 1] - source_batch_np[i, 1],
    #     #                     head_width=0.03, head_length=0.05, fc='gray', ec='gray', alpha=0.3)
    #     #
    #     # if close_fig:
    #     #     plt.title(f'Pushforward Step {self.train_step}')
    #     #     plt.savefig(fname)
    #     #     plt.close(fig)
    #     pass

    def save(self, tag="latest"):
        print(f"Warning: Saving logic needs PyTorch adaptation (tag: {tag}).")
        # # Create directory for saving checkpoints
        # save_dir = os.path.join(self.work_dir, "checkpoints")
        # os.makedirs(save_dir, exist_ok=True)
        #
        # # --- Save Configuration ---
        # config_path = os.path.join(save_dir, f"config_{tag}.yaml")
        # with open(config_path, 'w') as f:
        #     OmegaConf.save(config=self.cfg, f=f.name)
        #
        # # --- Save Models and Optimizer States (PyTorch way) ---
        # checkpoint = {
        #     'step': self.train_step,
        #     'elapsed_time': self.elapsed_time,
        #     'target_potential_models_state_dict': [model.state_dict() for model in self.target_potential_models],
        #     'source_map_models_state_dict': [model.state_dict() for model in self.source_map_models],
        #     'target_potential_optimizers_state_dict': [opt.state_dict() for opt in self.target_potential_optimizers],
        #     'source_map_optimizers_state_dict': [opt.state_dict() for opt in self.source_map_optimizers],
        #     # Add geometry model/optimizer state if applicable
        #     # 'geometry_model_state_dict': self.geometry_model.state_dict() if self.geometry_model else None,
        #     # 'geom_optimizer_state_dict': self.geom_optimizer.state_dict() if self.geom_optimizer else None,
        #     # Add other necessary states like random states if needed
        # }
        #
        # checkpoint_path = os.path.join(save_dir, f"checkpoint_{tag}.pt")
        # torch.save(checkpoint, checkpoint_path)
        #
        # print(f"Saved checkpoint and config to {save_dir} with tag '{tag}'")


@hydra.main(config_path=".", config_name="train_ot_scarvelis.yaml", version_base="1.1")
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
