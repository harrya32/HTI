#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates

import argparse
# Removed jax imports
import numpy as np
import os
# Removed optax import
import cloudpickle as pkl
from copy import copy
# Removed flax import
import json
import time
import shutil
import csv
import torch as th # Use th alias
import torch.nn as nn
import torch.optim as optim
import functools
import wandb

from geomloss import SamplesLoss

import dataclasses
from typing import Iterator, Dict, Any, List, Literal, Tuple, Optional, TYPE_CHECKING

# Removed ott imports (or ensure they work with torch)

import hydra
from omegaconf import DictConfig, OmegaConf # For Hydra config typing

# Import converted modules
from lagrangian_ot import models, neuraldual, metrics, geodesics, geometries, data, spline_amortizer, meters # Added meters

import matplotlib.pyplot as plt
plt.style.use('bmh')

import sys
# Removed IPython import/excepthook

if TYPE_CHECKING:
    from torch.optim.optimizer import Optimizer
    from .geometries import GeometryBase
    from .models import ModelBase
    from .ctransform_solvers import CTransformSolver

class Workspace:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        # Set random seed for reproducibility
        th.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        # Add cuda seed if using GPU
        if th.cuda.is_available() and not cfg.get("cpu_only", False):
            th.cuda.manual_seed_all(self.cfg.seed)
            self.device = th.device("cuda")
        else:
            self.device = th.device("cpu")
        print(f"Using device: {self.device}")

        # Initialize wandb
        wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            project=cfg.get("wandb_project", "NLOT_torch"), # Updated project name
            entity=cfg.get("wandb_entity", None),
            config=wandb_config, # Log hydra config
            name=cfg.get("run_name", None),
            mode="disabled" if cfg.get("debug", False) else "online", # Use debug flag
        )

        if self.cfg.save_all_plots:
            self.plot_dir = os.path.join(self.work_dir, 'plots')
            os.makedirs(self.plot_dir, exist_ok=True)
        else:
             self.plot_dir = None

        self.train_step = 0
        # Removed JAX key

        # --- Geometry Initialization ---
        geometry_kwargs = OmegaConf.to_container(self.cfg.geometry_kwargs, resolve=True)
        self.geometry = geometries.get(
            self.cfg.geometry, geometry_kwargs)
        if isinstance(self.geometry, nn.Module):
            self.geometry.to(self.device)

        # --- Data Loading --- #
        if self.cfg.data is None:
            self.cfg.data = self.cfg.geometry
        # Use integer seed, assumes get_samplers yields torch tensors
        self.source_sampler_train, self.target_sampler_train = data.get_samplers(
            self.cfg.data, self.cfg.batch_size, seed=self.cfg.seed)

        if isinstance(self.geometry, geometries.SqEuclidean):
            bounds, xbounds, ybounds = data.get_bounds(self.cfg.data)
            # Convert bounds to tensors if needed
            if isinstance(bounds[0], (int, float, np.ndarray)):
                self.geometry.bounds = tuple(th.as_tensor(b, device=self.device, dtype=th.float32) for b in bounds)
            elif isinstance(bounds[0], th.Tensor):
                self.geometry.bounds = tuple(b.to(self.device) for b in bounds)
            else: # Keep as is if format is unexpected (e.g., list of tensors already)
                 self.geometry.bounds = bounds
            self.geometry.xbounds = xbounds
            self.geometry.ybounds = ybounds

        num_eval_samples = 1024 if self.cfg.data in ['gsb_gmm', 'gmm'] else 128
        source_sampler_eval, target_sampler_eval = data.get_samplers(
            self.cfg.data, num_eval_samples, seed=self.cfg.seed + 1)
        self.source_samples_eval = next(source_sampler_eval).to(self.device)
        self.target_samples_eval = next(target_sampler_eval).to(self.device)

        # --- Model Initialization --- #
        D_input = self.source_samples_eval.shape[-1]
        target_potential_kwargs = OmegaConf.to_container(self.cfg.target_potential_kwargs, resolve=True)
        self.target_potential = models.MLP(
            dim_input=D_input,
            is_potential=True, **target_potential_kwargs,
        ).to(self.device)
        source_map_kwargs = OmegaConf.to_container(self.cfg.source_map_kwargs, resolve=True)
        self.source_map = models.MLP(
            dim_input=D_input,
            is_potential=False, **source_map_kwargs,
        ).to(self.device)

        # --- C-Transform Solver --- #
        self.ctransform_solver = hydra.utils.instantiate(self.cfg.ctransform_solver)

        # --- Neural Dual Solver --- #
        self.neural_dual_solver = neuraldual.ManifoldW2NeuralDual(
            geometry=self.geometry,
            target_potential=self.target_potential,
            source_map=self.source_map,
            ctransform_solver=self.ctransform_solver,
            device=self.device,
            amortization_loss=self.cfg.get('amortization_loss', 'regression')
        )

        # --- Optimizers --- #
        self.optimizer_target_potential = get_opt(
            self.target_potential.parameters(), self.cfg.num_train_iters, self.cfg.target_potential_opt
        )
        self.optimizer_source_map = get_opt(
            self.source_map.parameters(), self.cfg.num_train_iters, self.cfg.source_map_opt
        )
        self.optimizer_geometry = None
        if isinstance(self.geometry, nn.Module) and list(self.geometry.parameters()):
             geometry_opt_cfg = self.cfg.get('geometry_opt') # Get geometry opt config safely
             if geometry_opt_cfg:
                 self.optimizer_geometry = get_opt(
                     self.geometry.parameters(), self.cfg.num_train_iters, geometry_opt_cfg
                 )
             else:
                  print("Warning: Geometry is nn.Module but no geometry_opt config found.")

        # Removed Flax TrainState initialization

        # --- Parameter Annealing & Spline Fit --- #
        self.geometry_has_annealing = hasattr(self.geometry, 'lagrangian_potential_module') and self.geometry.lagrangian_potential_module is not None
        self.anneal_params_geometry() # Initial call
        # Fit spline amortizer (if needed) after models and geometry are ready
        if isinstance(self.geometry, geometries.MetricManifold) and hasattr(self.geometry, 'spline_model'):
            self.fit_spline_amortizer(
                source_sampler=self.source_sampler_train,
                target_sampler=self.target_sampler_train,
                init=True
            )

        self.elapsed_time = 0.
        self.best_marginal_w2 = np.inf

    def fit_spline_amortizer(self, source_sampler, target_sampler, init):
        # Assumes geometry has attribute 'spline_amortizer' of the converted type
        if not hasattr(self.geometry, 'spline_amortizer') or \
           not isinstance(self.geometry.spline_amortizer, spline_amortizer.SplineAmortizer):
            print("Geometry does not have a compatible SplineAmortizer. Skipping fit.")
            return

        num_iters = self.cfg.spline.init_train_iters if init else self.cfg.spline.train_iters
        lr = self.cfg.spline.get('lr', 1e-4) # Get LR from config

        if init:
            # Initial fitting uses the provided source/target samplers directly
            print("Performing initial fit for spline amortizer...")
            self.geometry.spline_amortizer.train(
                source_sampler, target_sampler, # Samplers must yield torch tensors
                max_iter=num_iters,
                lr=lr,
                grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
            )
        else:
            # Subsequent fitting uses source samples and their current mappings
            def mapped_target_sampler():
                self.source_map.eval() # Ensure source map is in eval mode
                while True:
                    # Ensure sampler yields tensors on the correct device
                    source_samples = next(source_sampler).to(self.device)
                    with th.no_grad():
                        transported_samples = self.source_map(source_samples)
                    # Apply noise if configured
                    if self.cfg.spline.get('noise', 0.0) > 0.: # Safely get noise value
                        noise = self.cfg.spline.noise * th.randn_like(transported_samples)
                        transported_samples += noise
                    yield transported_samples.detach() # Yield detached tensors

            print("Refitting spline amortizer based on current source map...")
            # Use the generator directly with iter()
            self.geometry.spline_amortizer.train(
                source_sampler, # Sampler yielding source tensors
                iter(mapped_target_sampler()), # Sampler yielding mapped targets
                max_iter=num_iters,
                lr=lr,
                grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
            )

    def anneal_params_geometry(self):
        current_t = self.train_step
        max_t = self.cfg.get('anneal_geometry_steps', 0) # Default to 0 if not set

        if max_t <= 0 or not self.geometry_has_annealing or max_t < current_t:
            return

        t = current_t / max_t
        assert 0. <= t <= 1.

        # Anneal parameters using the method defined in LagrangianPotentialBase
        target_module = None
        if isinstance(self.geometry, nn.Module) and hasattr(self.geometry, 'set_annealed_params') and callable(self.geometry.set_annealed_params):
            # If geometry itself handles it
            self.geometry.set_annealed_params(t)
            target_module = self.geometry # For logging
        elif hasattr(self.geometry, 'lagrangian_potential_module') and \
             self.geometry.lagrangian_potential_module is not None and \
             hasattr(self.geometry.lagrangian_potential_module, 'set_annealed_params'):
            # If the potential module handles it
            module = self.geometry.lagrangian_potential_module
            module.set_annealed_params(t)
            target_module = module # For logging
        else:
            print("Warning: Geometry/Potential Module does not have a compatible set_annealed_params method.")
            return

        # Log annealed params if possible
        if target_module and hasattr(target_module, 'M') and hasattr(target_module, 'temp'):
            try:
                m_val = target_module.M.item()
                temp_val = target_module.temp.item()
                print(f'Updated Lagrangian params: M: {m_val:.2e}, temp: {temp_val:.2e}')
                wandb.log({'anneal/M': m_val, 'anneal/temp': temp_val}, step=self.train_step)
            except Exception as e:
                 print(f"Could not log annealed params: {e}")

    def run(self):
        # Ensure data samplers yield PyTorch tensors
        source_sampler, target_sampler = self.source_sampler_train, self.target_sampler_train
        # Get optional schedulers
        scheduler_target = self.optimizer_target_potential[1] if isinstance(self.optimizer_target_potential, tuple) else None
        scheduler_source = self.optimizer_source_map[1] if isinstance(self.optimizer_source_map, tuple) else None
        scheduler_geometry = self.optimizer_geometry[1] if self.optimizer_geometry and isinstance(self.optimizer_geometry, tuple) else None
        # Extract only optimizers for the steps
        opt_target = self.optimizer_target_potential[0] if isinstance(self.optimizer_target_potential, tuple) else self.optimizer_target_potential
        opt_source = self.optimizer_source_map[0] if isinstance(self.optimizer_source_map, tuple) else self.optimizer_source_map
        opt_geometry = self.optimizer_geometry[0] if self.optimizer_geometry and isinstance(self.optimizer_geometry, tuple) else self.optimizer_geometry

        # Initial plot
        self.plot()

        logf, writer = self._init_logging()

        # --- Training Loop --- #
        for self.train_step in range(self.cfg.num_train_iters):
            start_time = time.time()

            # Get batch and move to device
            try:
                train_batch = {
                    "source": next(source_sampler).to(self.device),
                    "target": next(target_sampler).to(self.device),
                }
            except StopIteration:
                print("Samplers exhausted. Restarting...")
                source_sampler, target_sampler = data.get_samplers(
                    self.cfg.data, self.cfg.batch_size, seed=self.cfg.seed + self.train_step)
                self.source_sampler_train, self.target_sampler_train = source_sampler, target_sampler # Update instance samplers
                train_batch = {
                    "source": next(source_sampler).to(self.device),
                    "target": next(target_sampler).to(self.device),
                }

            # Anneal geometry parameters if applicable
            self.anneal_params_geometry()

            # Perform training step using the converted method
            # Set models to train mode
            self.target_potential.train()
            self.source_map.train()
            if isinstance(self.geometry, nn.Module):
                 self.geometry.train()

            # Zero gradients for all optimizers involved
            opt_target.zero_grad()
            opt_source.zero_grad()
            if opt_geometry:
                 opt_geometry.zero_grad()

            # Calculate loss
            loss, info = self.neural_dual_solver.loss_fn(train_batch)

            # Backpropagate
            loss.backward()

            # Apply gradient clipping (example, adjust norm value if needed)
            grad_clip_val = self.cfg.get("grad_clip") # Check if grad clipping is in config
            if grad_clip_val:
                 nn.utils.clip_grad_norm_(self.target_potential.parameters(), grad_clip_val)
                 nn.utils.clip_grad_norm_(self.source_map.parameters(), grad_clip_val)
                 if isinstance(self.geometry, nn.Module) and list(self.geometry.parameters()):
                      nn.utils.clip_grad_norm_(self.geometry.parameters(), grad_clip_val)

            # Step optimizers
            opt_target.step()
            opt_source.step()
            if opt_geometry:
                 opt_geometry.step()

            # Step schedulers (if they exist)
            if scheduler_target: scheduler_target.step()
            if scheduler_source: scheduler_source.step()
            if scheduler_geometry: scheduler_geometry.step()

            update_step_time = time.time() - start_time
            self.elapsed_time += update_step_time

            # --- Logging --- #
            if self.train_step % self.cfg.print_every == 0:
                print(
                    f'step: {self.train_step}/{self.cfg.num_train_iters} '
                    f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                    f'num_ctransform_iter: {info.num_ctransform_iter:.2f} '
                    f'update_step_time: {update_step_time:.2f}s '
                )
            log_data = {
                "train/dual_loss": info.dual_loss,
                "train/amor_loss": info.amor_loss,
                "train/num_ctransform_iter": info.num_ctransform_iter,
                "train/update_step_time": update_step_time,
                "train/elapsed_time": self.elapsed_time,
                # Get current LR from scheduler or optimizer
                "lr/target_potential": scheduler_target.get_last_lr()[0] if scheduler_target else opt_target.param_groups[0]['lr'],
                "lr/source_map": scheduler_source.get_last_lr()[0] if scheduler_source else opt_source.param_groups[0]['lr'],
            }
            if opt_geometry:
                 log_data["lr/geometry"] = scheduler_geometry.get_last_lr()[0] if scheduler_geometry else opt_geometry.param_groups[0]['lr']
            wandb.log(log_data, step=self.train_step)

            if writer is not None:
                try:
                     writer.writerow([self.train_step, info.dual_loss, info.amor_loss, info.num_ctransform_iter, update_step_time, self.elapsed_time])
                     logf.flush()
                except Exception as e:
                     print(f"Error writing to log file: {e}")

            # --- Evaluation & Plotting --- #
            if self.train_step > 0 and self.train_step % self.cfg.eval_every == 0:
                eval_start_time = time.time()
                w2_dist = self.eval_marginal_W2() # Call converted eval
                eval_time = time.time() - eval_start_time
                print(f'step: {self.train_step} W2: {w2_dist:.4f} eval_time: {eval_time:.2f}s')
                wandb.log({
                    "eval/W2_distance": w2_dist,
                    "eval/eval_time": eval_time
                }, step=self.train_step)

                if w2_dist < self.best_marginal_w2:
                    print(f"New best W2 distance: {w2_dist:.4f}")
                    self.best_marginal_w2 = w2_dist
                    wandb.log({"eval/best_W2_distance": self.best_marginal_w2}, step=self.train_step)
                    self.save(tag="best") # Call converted save

                self.plot() # Call converted plot

            # --- Spline Amortizer Refitting --- #
            if self.cfg.spline.refit_every is not None and \
               self.train_step > 0 and \
               self.train_step % self.cfg.spline.refit_every == 0 and \
               isinstance(self.geometry, geometries.MetricManifold) and \
               hasattr(self.geometry, 'spline_model'):
                print(f'Refitting spline amortizer at step {self.train_step}')
                # Use training samplers for refitting, get new instances
                current_source_sampler, current_target_sampler = data.get_samplers(
                     self.cfg.data, self.cfg.batch_size, seed=self.cfg.seed + self.train_step + 100)
                self.fit_spline_amortizer(
                    source_sampler=current_source_sampler,
                    target_sampler=current_target_sampler, # Note: target needs adaptation
                    init=False # Indicate refitting
                )

            # --- Checkpointing --- #
            if self.train_step > 0 and self.train_step % self.cfg.save_every == 0:
                self.save(tag="latest") # Call converted save

            # self.train_step is incremented by the loop

        # Final save and close logs
        self.save(tag="final")
        if logf is not None: logf.close()
        wandb.finish()

    @th.no_grad()
    def eval_marginal_W2(self):
        """Evaluate W2 distance between target and transported source samples."""
        # Set models to evaluation mode
        self.source_map.eval()
        self.target_potential.eval() # Needed for pushforward solve
        if isinstance(self.geometry, nn.Module):
             self.geometry.eval()

        # Use eval samples
        source_samples = self.source_samples_eval.to(self.device)
        target_samples = self.target_samples_eval.to(self.device)

        # Compute transported samples (pushforward)
        # Using the loop approach defined in neuraldual's loss_fn structure
        transported_samples_list = []
        batch_size_eval = 128 # Process in smaller batches if needed
        num_batches = (source_samples.shape[0] + batch_size_eval - 1) // batch_size_eval

        for i in range(num_batches):
            start_idx = i * batch_size_eval
            end_idx = min((i + 1) * batch_size_eval, source_samples.shape[0])
            source_batch = source_samples[start_idx:end_idx]

            # Get initial map
            init_target_hat_batch = self.neural_dual_solver.source_map(source_batch)

            # Refine with c-transform solver (looping per point within batch)
            for j in range(source_batch.shape[0]):
                 out = self.neural_dual_solver.ctransform_solver.solve(
                     self.neural_dual_solver.geometry,
                     lambda t: self.neural_dual_solver.target_potential(t),
                     source_batch[j],
                     target_init=init_target_hat_batch[j]
                 )
                 transported_samples_list.append(out.solution)

        if not transported_samples_list:
             print("Warning: Transported samples list is empty in eval_marginal_W2.")
             return np.inf # Return infinity if transport failed

        transported_samples = th.stack(transported_samples_list)

        # Compute W2 using Sinkhorn (geomloss or OTT if adapted)
        w2_dist = self.sinkhorn_cost(transported_samples, target_samples)
        return w2_dist.item() # Return scalar value

    def plot(self):
        """Generate and save plots using PyTorch models."""
        if self.plot_dir is None and not self.cfg.get("show_plots", False): # Check show_plots flag
             return # Skip plotting if not saving and not showing

        # Use the plotting methods from the NeuralDual solver (already converted)
        try:
            fig_fm, ax_fm = self.neural_dual_solver.plot_forward_map(
                self.source_samples_eval, self.target_samples_eval)
            fig_pot, ax_pot = self.neural_dual_solver.plot_target_potential(
                 self.source_samples_eval, self.target_samples_eval)

            # Plot geometry background if available
            if hasattr(self.geometry, 'add_plot_background') and callable(self.geometry.add_plot_background):
                # Get bounds (handle potential Tensor bounds)
                xbounds = self.geometry.xbounds if hasattr(self.geometry, 'xbounds') else (-6, 6)
                ybounds = self.geometry.ybounds if hasattr(self.geometry, 'ybounds') else (-6, 6)
                # Convert tensor bounds to tuple for plotting if needed
                if isinstance(xbounds, th.Tensor): xbounds = tuple(xbounds.cpu().numpy())
                if isinstance(ybounds, th.Tensor): ybounds = tuple(ybounds.cpu().numpy())

                self.geometry.add_plot_background(ax_fm, xlims=xbounds, ylims=ybounds)
                self.geometry.add_plot_background(ax_pot, xlims=xbounds, ylims=ybounds, alpha=0.3)

            if self.plot_dir:
                fm_path = os.path.join(self.plot_dir, f'forward_map_{self.train_step:06d}.png')
                pot_path = os.path.join(self.plot_dir, f'potential_{self.train_step:06d}.png')
                fig_fm.savefig(fm_path)
                fig_pot.savefig(pot_path)
                print(f"Saved plots to {fm_path} and {pot_path}")
                # Log the latest plot to wandb
                try:
                    wandb.log({"plots/forward_map": wandb.Image(fm_path),
                               "plots/potential": wandb.Image(pot_path)},
                              step=self.train_step)
                except Exception as e:
                    print(f"Wandb logging failed for plots: {e}")

            if self.cfg.get("show_plots", False):
                 plt.show()

            plt.close(fig_fm)
            plt.close(fig_pot)

        except Exception as e:
             print(f"Plotting failed: {e}")
             # Close any potentially open figures
             plt.close("all")

    def save(self, tag="latest"):
        """Save models and optimizers using torch.save."""
        save_path = os.path.join(self.work_dir, f"checkpoint_{tag}.pth")
        # Extract optimizers if they are tuples with schedulers
        opt_target = self.optimizer_target_potential[0] if isinstance(self.optimizer_target_potential, tuple) else self.optimizer_target_potential
        opt_source = self.optimizer_source_map[0] if isinstance(self.optimizer_source_map, tuple) else self.optimizer_source_map
        opt_geometry = self.optimizer_geometry[0] if self.optimizer_geometry and isinstance(self.optimizer_geometry, tuple) else self.optimizer_geometry

        save_dict = {
            'train_step': self.train_step,
            'target_potential_state_dict': self.target_potential.state_dict(),
            'source_map_state_dict': self.source_map.state_dict(),
            'optimizer_target_state_dict': opt_target.state_dict(),
            'optimizer_source_state_dict': opt_source.state_dict(),
            'elapsed_time': self.elapsed_time,
            'best_marginal_w2': self.best_marginal_w2,
            'cfg': OmegaConf.to_container(self.cfg, resolve=True) # Save config
        }
        if isinstance(self.geometry, nn.Module):
             save_dict['geometry_state_dict'] = self.geometry.state_dict()
        if opt_geometry:
             save_dict['optimizer_geometry_state_dict'] = opt_geometry.state_dict()

        # Save schedulers if they exist
        if isinstance(self.optimizer_target_potential, tuple) and self.optimizer_target_potential[1] is not None:
             save_dict['scheduler_target_state_dict'] = self.optimizer_target_potential[1].state_dict()
        if isinstance(self.optimizer_source_map, tuple) and self.optimizer_source_map[1] is not None:
             save_dict['scheduler_source_state_dict'] = self.optimizer_source_map[1].state_dict()
        if self.optimizer_geometry and isinstance(self.optimizer_geometry, tuple) and self.optimizer_geometry[1] is not None:
             save_dict['scheduler_geometry_state_dict'] = self.optimizer_geometry[1].state_dict()

        try:
             th.save(save_dict, save_path)
             print(f"Saved checkpoint to {save_path}")
        except Exception as e:
             print(f"Error saving checkpoint {save_path}: {e}")

    def load(self, path):
        """Load models and optimizers from PyTorch checkpoint."""
        if not os.path.exists(path):
            print(f"Checkpoint file not found: {path}")
            return
        try:
            checkpoint = th.load(path, map_location=self.device)

            # Load models
            self.target_potential.load_state_dict(checkpoint['target_potential_state_dict'])
            self.source_map.load_state_dict(checkpoint['source_map_state_dict'])
            if isinstance(self.geometry, nn.Module) and 'geometry_state_dict' in checkpoint:
                self.geometry.load_state_dict(checkpoint['geometry_state_dict'])

            # Load optimizers (extract optimizer part if saved as tuple)
            opt_target = self.optimizer_target_potential[0] if isinstance(self.optimizer_target_potential, tuple) else self.optimizer_target_potential
            opt_source = self.optimizer_source_map[0] if isinstance(self.optimizer_source_map, tuple) else self.optimizer_source_map
            opt_geometry = self.optimizer_geometry[0] if self.optimizer_geometry and isinstance(self.optimizer_geometry, tuple) else self.optimizer_geometry
            opt_target.load_state_dict(checkpoint['optimizer_target_state_dict'])
            opt_source.load_state_dict(checkpoint['optimizer_source_state_dict'])
            if opt_geometry and 'optimizer_geometry_state_dict' in checkpoint:
                opt_geometry.load_state_dict(checkpoint['optimizer_geometry_state_dict'])

            # Load schedulers
            if isinstance(self.optimizer_target_potential, tuple) and self.optimizer_target_potential[1] is not None and 'scheduler_target_state_dict' in checkpoint:
                 self.optimizer_target_potential[1].load_state_dict(checkpoint['scheduler_target_state_dict'])
            if isinstance(self.optimizer_source_map, tuple) and self.optimizer_source_map[1] is not None and 'scheduler_source_state_dict' in checkpoint:
                 self.optimizer_source_map[1].load_state_dict(checkpoint['scheduler_source_state_dict'])
            if self.optimizer_geometry and isinstance(self.optimizer_geometry, tuple) and self.optimizer_geometry[1] is not None and 'scheduler_geometry_state_dict' in checkpoint:
                 self.optimizer_geometry[1].load_state_dict(checkpoint['scheduler_geometry_state_dict'])

            # Load training state
            self.train_step = checkpoint['train_step']
            self.elapsed_time = checkpoint.get('elapsed_time', 0.)
            self.best_marginal_w2 = checkpoint.get('best_marginal_w2', np.inf)
            print(f"Loaded checkpoint from {path} at step {self.train_step}")

        except Exception as e:
            print(f"Error loading checkpoint {path}: {e}")

    def _init_logging(self):
        log_path = os.path.join(self.work_dir, 'train_log.csv')
        try:
            # Open in append mode ('a') if resuming, write mode ('w') otherwise
            mode = 'a' if self.train_step > 0 else 'w'
            logf = open(log_path, mode, newline='') # Add newline='' for csv
            writer = csv.writer(logf)
            # Write header only if file is new (or opened in write mode)
            if mode == 'w' or os.stat(log_path).st_size == 0:
                writer.writerow(['step', 'dual_loss', 'amor_loss', 'num_ctransform_iter', 'update_time', 'elapsed_time'])
                logf.flush()
            return logf, writer
        except IOError as e:
             print(f"Error opening log file {log_path}: {e}")
             return None, None

    # Remove JAX specific sinkhorn wrapper

    def sinkhorn_cost_geomloss(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        # Use previously converted version
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        # Geomloss needs [N, D] or [B, N, D]
        if x.ndim == 1: x = x.unsqueeze(0)
        if y.ndim == 1: y = y.unsqueeze(0)
        if x.ndim == 2: x = x.unsqueeze(0) # Add batch dim if needed
        if y.ndim == 2: y = y.unsqueeze(0)

        loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
        try:
            # Geomloss can be sensitive to input shapes
            dist = loss_fn(x.squeeze(0), y.squeeze(0)) # Try [N, D]
        except (RuntimeError, ValueError):
             try:
                 dist = loss_fn(x, y) # Try [B, N, D]
             except Exception as e:
                 print(f"Geomloss Sinkhorn failed: {e}")
                 return th.tensor(float('inf'), device=self.device)
        return dist.mean() # Return scalar mean over batch if applicable

    def sinkhorn_cost(self, x: th.Tensor, y: th.Tensor) -> th.Tensor:
        # Default to geomloss version
        return self.sinkhorn_cost_geomloss(x, y)


def get_opt(params: Iterator[nn.Parameter], num_train_iters: int, cfg: DictConfig) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    # Use previously converted version
    opt_name = cfg.name
    lr = float(cfg.lr)
    schedule_type = cfg.get('schedule', None)
    optimizer_kwargs = {k: v for k, v in cfg.items() if k not in ['name', 'lr', 'schedule', 'alpha', 'grad_clip']}

    if opt_name.lower() == 'adam':
        optimizer = optim.Adam(params, lr=lr, **optimizer_kwargs)
    elif opt_name.lower() == 'adamw':
         optimizer = optim.AdamW(params, lr=lr, **optimizer_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    scheduler = None
    if schedule_type == 'cosine_decay':
        eta_min = lr * cfg.get('alpha', 0.0)
        t_max = max(1, num_train_iters)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif schedule_type is not None:
        warnings.warn(f"Scheduler '{schedule_type}' not implemented, using constant LR.")

    return optimizer, scheduler


@hydra.main(config_path=".", config_name="train_ot.yaml", version_base="1.1")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    workspace = Workspace(cfg)

    # Use the PyTorch load method
    load_path = cfg.get("load_checkpoint")
    if load_path:
        workspace.load(load_path)
        # If loading, start training from the loaded step
        print(f"Resuming training from step {workspace.train_step}")

    workspace.run()

if __name__ == "__main__":
    main()
