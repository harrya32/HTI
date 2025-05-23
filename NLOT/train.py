import argparse
import math
import functools
import csv
import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
import cloudpickle as pkl
from copy import copy
from flax.core import FrozenDict
from scipy.optimize import linear_sum_assignment
import dataclasses
from typing import Iterator
import hydra
from omegaconf import OmegaConf
from lagrangian_ot import models, neuraldual, metrics, geodesics, geometries, data, lagrangian_potentials
from generate_synth_data import log_likelihood_conditional_semicircle
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('bmh')
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Plain', color_scheme='Neutral', call_pdb=1)
import wandb
import ot
from scipy.spatial.distance import cdist # Add this import

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        wandb.init(
            project=cfg.get("wandb_project", "NLOT-Scarvelis_" + self.cfg.geometry), 
            config=OmegaConf.to_container(cfg, resolve=True), 
            name=cfg.get("run_name", None),
            mode="disabled" if not cfg.get("wandb", True) else "online",
        )
        self.data = self.cfg.dataset 

        self.key = jax.random.PRNGKey(self.cfg.seed)
        self.elapsed_time = 0.
        self.frobenius_weight = self.cfg.metric.get('frobenius_reg_weight', 0.0)
        self.samplers = data.get_samplers_scarvelis(self.data, num_pairs_requested=self.cfg.get('num_pairs', None))
        self.all_samples = jnp.concatenate([next(s) for s in self.samplers], axis=0)
        
        print(f"all_samples shape: {self.all_samples.shape}")

        if self.cfg.get('include_inverse_potential', False):
            if self.data == "conditional_circles":
                lagrangian_potential_initializer_fn = lagrangian_potentials.InverseDensityPotential(
                    D=self.cfg.get('D', 2),
                    C=self.cfg.get('C', 0),
                    samples=self.all_samples,
                    bandwidth=self.cfg.get('bandwidth', 1.0),
                    lambda_repel=self.cfg.get('lambda', 0.01),
                )
            else:
                lagrangian_potential_initializer_fn = lagrangian_potentials.InverseDensityPotentialNW(
                    D=self.cfg.get('D', 2),
                    C=self.cfg.get('C', 0),
                    samples=self.all_samples,
                    bandwidth=self.cfg.get('bandwidth', 1.0),
                    conditional_bandwidth=self.cfg.get('conditional_bandwidth', 1.0),
                    lambda_repel=self.cfg.get('lambda', 0.01),
                )
        else:
            lagrangian_potential_initializer_fn = None

        self.geometry = geometries.get(
            self.cfg.geometry, 
            self.cfg.get('geometry_kwargs', {}),
            self.cfg.get('land_kwargs', {}),
            self.cfg.get('rbf_kwargs', {}),
            samples=self.all_samples,
            D=self.cfg.get('D', 2),
            C=self.cfg.get('C', 0),
            categorical=self.cfg.get('categorical', False),
            num_categories=self.cfg.get('num_categories', 0),
            lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn
        )

        if 'euclidean' in self.cfg.geometry or 'neural' in self.cfg.geometry or 'land' in self.cfg.geometry:
            if self.data is None:
                raise ValueError('data must be specified for euclidean and neural geometries')

        self.has_reference_geometry = 'neural' in self.cfg.geometry or 'land' in self.cfg.geometry
        if self.has_reference_geometry:
            self.reference_geometry = geometries.get(
                self.cfg.geometry, 
                self.cfg.get('geometry_kwargs', {}),
                self.cfg.get('land_kwargs', {}),
                self.cfg.get('rbf_kwargs', {}),
                samples=self.all_samples,
                D=self.cfg.get('D', 2),
                C=self.cfg.get('C', 0),
                categorical=self.cfg.get('categorical', False),
                num_categories=self.cfg.get('num_categories', 0),
                lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn
            )

        if self.data is None:
            self.data = self.cfg.geometry

        self.geometry.bounds, self.geometry.xbounds, self.geometry.ybounds = data.get_bounds(self.data)

        self.num_pairs = len(self.samplers) - 1
        self.time_points = cfg.get('time_points', np.linspace(0, 1, self.num_pairs + 1))

        print(f'training on {self.num_pairs} pairs at times {self.time_points}')
        self.eval_samples = [next(s) for s in self.samplers]
        self.optimizer_target_potential = optax.adamw(learning_rate=self.cfg.potential_lr)
        self.optimizer_source_map = self.optimizer_target_potential
        self.optimizer_geom = optax.adamw(learning_rate=self.cfg.metric.lr)
        k1, self.key = jax.random.split(self.key)
        self.params_geometry = self.geometry.init(
            k1, self.eval_samples[0][0], self.eval_samples[1][0],
            method=self.geometry.cost
        ).get('params', {})
        self.params_geometry = FrozenDict(self.params_geometry)
        self.state_geometry = self.optimizer_geom.init(self.params_geometry)

        target_potential = models.MLP(
            dim_hidden=self.cfg.target_potential_dim_hidden,
            is_potential=True,
            D=self.cfg.get('D', 2),
            C=self.cfg.get('C', 0),
            categorical=self.cfg.get('categorical', False),
            num_categories=self.cfg.get('num_categories', 0),
        )
        
        source_map = models.MLP(
            dim_hidden=self.cfg.source_map_dim_hidden,
            is_potential=False,
            D=self.cfg.get('D', 2),
            C=self.cfg.get('C', 0),
            categorical=self.cfg.get('categorical', False),
            num_categories=self.cfg.get('num_categories', 0),
        )
        
        ctransform_solver = hydra.utils.instantiate(self.cfg.ctransform_solver)

        self.neural_dual_solver = neuraldual.ManifoldW2NeuralDual(
            geometry=self.geometry,
            target_potential=target_potential,
            source_map=source_map,
            ctransform_solver=ctransform_solver,
        )
        
        init_key, self.key = jax.random.split(self.key)
        state_target_potential, state_source_map = self.neural_dual_solver.initialize_states(
            self.optimizer_target_potential, self.optimizer_source_map,
            init_key, self.eval_samples[0], self.eval_samples[1])
        self.state_target_potentials = [state_target_potential]
        self.state_source_maps = [state_source_map]

        if 'spline_model' in self.params_geometry:
            self.fit_spline_amortizer(self.samplers, init=True)

        self.train_step = 0

    def fit_spline_amortizer(self, samplers, init):
        num_iters = self.cfg.spline.init_train_iters if init else self.cfg.spline.train_iters

        if init:
            def sampler(key):
                # sample from random pairs of source and target
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    target_samples = next(samplers[t+1])
                    all_samples = jnp.concatenate([source_samples, target_samples], axis=0)
                    k1, key = jax.random.split(key)
                    all_samples = jax.random.permutation(k1, all_samples)
                    t = (t + 1) % self.num_pairs
                    yield all_samples

            k1, k2, self.key = jax.random.split(self.key, 3)
            xsampler = iter(sampler(k1))
            ysampler = iter(sampler(k2))
        else:
            def xsampler():
                key = jax.random.PRNGKey(0)
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    t = (t + 1) % self.num_pairs
                    yield source_samples

            def ysampler():
                key = jax.random.PRNGKey(0)
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    transported_samples = self.neural_dual_solver.source_map_apply_jit(
                        {'params': self.state_source_maps[t].params}, source_samples)
                    if self.cfg.spline.noise > 0.:
                        k1, key = jax.random.split(key)
                        transported_samples += self.cfg.spline.noise * jax.random.normal(
                            key, transported_samples.shape)
                    t = (t + 1) % self.num_pairs
                    yield transported_samples

            xsampler = iter(xsampler())
            ysampler = iter(ysampler())


        self.params_geometry = self.geometry.spline_amortizer.train(
            self.params_geometry,
            xsampler, ysampler,
            max_iter=num_iters,
            grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
        )

    def update_all_states(self, state_target_potentials, state_source_maps, batches):
        out = []
        for t in range(self.num_pairs):
            out_t = self.neural_dual_solver.update_fn_jit(
                state_target_potentials[t if self.train_step > 0 else 0],
                state_source_maps[t if self.train_step > 0 else 0],
                self.params_geometry,
                batches[t],
            )
            out.append(out_t)

            if self.cfg.spline.update_on_conjugates \
                    and 'spline_model' in self.params_geometry:
                _, info = out_t
                self.params_geometry = self.geometry.spline_amortizer.train_single(
                    self.params_geometry,
                    batches[t]['source'], info.target_hat,
                    verbose=False,
                )

        new_states, infos = zip(*out)
        new_states = zip(*new_states)
        mean_info = type(infos[0])(*[jnp.array(x).mean() for x in list(zip(*infos))])
        return new_states, mean_info

    def sample_all_batches(self, samplers):
        batches = []
        for t in range(self.num_pairs):
            batches.append({
                "source": jnp.asarray(next(samplers[t])),
                "target": jnp.asarray(next(samplers[t+1])),
            })
        return batches

    def geometry_loss(self, params_geometry, state_target_potentials, state_source_maps, batches, key):
        metric_fn = lambda x: self.geometry.apply(
            {'params': params_geometry},
            x, method=self.geometry.metric)
        metric_vmap = jax.vmap(metric_fn)
        inv_metric_fn = lambda x: jnp.linalg.inv(metric_fn(x))
        inv_metric_vmap = jax.vmap(inv_metric_fn)

        dual_losses = []
        frobenius_regs = [] # List to store Frobenius regularization term for each pair

        reg_key, key = jax.random.split(key) # Split key for regularization sampling

        for t in range(self.num_pairs):
            batch = batches[t]
            _, info_t = self.neural_dual_solver.loss_fn(
                state_target_potentials[t].params,
                state_source_maps[t].params,
                params_geometry, 
                batch
            )
            dual_losses.append(-info_t.dual_loss)
            
            if self.frobenius_weight:
                # Scarvelis regularization
                source_points = batch['source']
                target_points = batch['target'] 

                batch_size = source_points.shape[0]
                t_key, reg_key = jax.random.split(reg_key)
                times = jax.random.uniform(t_key, shape=(batch_size, 1))

                # Calculate points along straight line: sigma(t') = (1-t')*x0 + t'*x1
                path_points = (1.0 - times) * source_points + times * target_points

                # Evaluate inverse metric G^-1(sigma(t'))
                inv_metrics_at_path = inv_metric_vmap(path_points)

                # Calculate squared Frobenius norm ||G^-1(sigma(t'))||^2_F = Tr((G^-1)^T G^-1) = Tr(G^-2)
                frobenius_sq_norms = jax.vmap(lambda m: jnp.trace(m @ m))(inv_metrics_at_path)
                mean_frobenius_reg = jnp.mean(frobenius_sq_norms)
                frobenius_regs.append(mean_frobenius_reg)

        mean_dual_loss = jnp.mean(jnp.stack(dual_losses)) 

        if self.frobenius_weight:
            mean_frobenius_reg = jnp.mean(jnp.stack(frobenius_regs)) # Average over pairs
            total_loss = mean_dual_loss + self.frobenius_weight * mean_frobenius_reg
        else:
            total_loss = mean_dual_loss

        return total_loss

    @functools.partial(jax.jit, static_argnums=[0])
    def update_geometry(self, params_geometry, state_geometry, state_target_potentials, state_source_maps, batches, key):
        geometry_grad_fn = jax.value_and_grad(self.geometry_loss)
        loss, grads = geometry_grad_fn(
            params_geometry,
            state_target_potentials, 
            state_source_maps,
            batches, 
            key
        )

        # TODO: could remove 'spline_model' from updates
        # (currently grads are all zero)
        #updates, new_state_geometry = self.optimizer_geom.update(
        #    grads, state_geometry, params=params_geometry)
        
        updates, new_state_geometry = self.optimizer_geom.update(
            grads,
            state_geometry,
            FrozenDict(params_geometry) # Ensure params are passed as FrozenDict
        )

        new_params_geometry = optax.apply_updates(params_geometry, updates)

        return new_params_geometry, new_state_geometry, loss

    def run(self):

        logf, writer = self._init_logging()
        dual_loss = -1.

        while self.train_step < self.cfg.num_train_iters:
            start = time.time()
            batches = self.sample_all_batches(self.samplers)


            new_states, info = self.update_all_states(
                self.state_target_potentials,
                self.state_source_maps,
                batches
            )
            self.state_target_potentials, self.state_source_maps = new_states

            update_step_time = time.time() - start
            self.elapsed_time += update_step_time            

            if self.train_step % self.cfg.metric.update_frequency == 0:
                start = time.time()
                k1, self.key = jax.random.split(self.key)
                new_params_geometry, new_state_geometry, geom_loss = self.update_geometry(
                    self.params_geometry, 
                    self.state_geometry,
                    self.state_target_potentials, 
                    self.state_source_maps,
                    batches, 
                    k1
                )
                self.params_geometry, self.state_geometry = new_params_geometry, new_state_geometry
                update_metric_time = time.time() - start
                self.elapsed_time += update_metric_time
                print(
                    f'step: {self.train_step}/{self.cfg.num_train_iters} '
                    f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                    f'geom_loss: {geom_loss:.2e} '
                    f'update_step_time: {update_step_time:.2f}s '
                    f'update_metric_time: {update_metric_time:.2f}s '
                )
                wandb.log({
                    "train/dual_loss": info.dual_loss,
                    "train/amor_loss": info.amor_loss,
                    "train/geom_loss": geom_loss,
                    "train/elapsed_time": self.elapsed_time,
                    "train/mean_potential_target": info.mean_potential_target,
                    "train/min_potential_target": info.min_potential_target,
                    "train/max_potential_target": info.max_potential_target,
                    "train/mean_potential_target_hat": info.mean_potential_target_hat,
                    "train/min_potential_target_hat": info.min_potential_target_hat,
                    "train/max_potential_target_hat": info.max_potential_target_hat,
                }, step=self.train_step)

            else:
                print(
                    f'step: {self.train_step}/{self.cfg.num_train_iters} '
                    f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                    f'update_step_time: {update_step_time:.2f}s '
                )
                wandb.log({
                    "train/dual_loss": info.dual_loss,
                    "train/amor_loss": info.amor_loss,
                    "train/elapsed_time": self.elapsed_time,
                    "train/mean_potential_target": info.mean_potential_target,
                    "train/min_potential_target": info.min_potential_target,
                    "train/max_potential_target": info.max_potential_target,
                    "train/mean_potential_target_hat": info.mean_potential_target_hat,
                    "train/min_potential_target_hat": info.min_potential_target_hat,
                    "train/max_potential_target_hat": info.max_potential_target_hat,
                 }, step=self.train_step)


            if self.train_step % self.cfg.spline.update_frequency == 0 and 'spline_model' in self.params_geometry and self.train_step < self.cfg.num_train_iters:
                self.fit_spline_amortizer(samplers=self.samplers, init=False)

            if not self.cfg.plotting.get('disable', False):
                if self.train_step % self.cfg.plot_frequency == 0:
                    #self.plot_all_pairs()
                    #self.plot_pushforward()
                    #self.plot_assignment_paths()

                    #marginals eval
                    if 'circles' in self.data or 'reward' in self.data or 'ett' in self.data:
                        path = os.path.dirname(os.path.realpath(__file__)) + "/eval_data/"

                        if self.data == 'conditional_circles':
                            test_path = path + 'eval_marginals.pkl'
                        elif self.data == 'conditional_circles_normal':
                            test_path = path + 'eval_marginals_normal.pkl'
                        elif self.data == 'conditional_semicircles':
                            test_path = path + 'eval_marginals_semicircle.pkl'
                        else:
                            test_path = path + 'eval_marginals.pkl'
                        
                        test_data = jnp.load(test_path, allow_pickle=True)
                        time_0_points = test_data[0][1]

                        if self.data == 'reward_weighting_data':
                            path = os.path.dirname(os.path.realpath(__file__)) + "/data/"
                            test_path = path + 'reward_weighting_data_0_10.pt'
                            test_data = torch.load(test_path)[[0,1,2,3,4,6,7,8,9], :1000, :self.cfg.D + self.cfg.C].cpu().numpy()
                            test_data = jnp.array(test_data)
                            time_0_points = test_data[0]
                            test_data = [(0, test_data[0]), (0.1, test_data[1]), (0.2, test_data[2]), (0.3, test_data[3]), (0.4, test_data[4]), (0.6, test_data[5]), (0.7, test_data[6]), (0.8, test_data[7]), (0.9, test_data[8])]
                        
                        if self.data == 'ett_forecasts':
                            path = os.path.dirname(os.path.realpath(__file__)) + "/../data/"
                            test_path = path + 'ett_forecasts_more_noise.pt'
                            test_data = torch.load(test_path)[[0,1,3], :, :self.cfg.D + self.cfg.C].cpu().numpy()
                            test_data = jnp.array(test_data)
                            test_data_ambient = test_data[:, :, self.cfg.C:]
                            test_data_conditioning = test_data[:, :, :self.cfg.C]
                            test_data = jnp.concatenate((test_data_ambient, test_data_conditioning), axis=2)
                            time_0_points = test_data[0]
                            test_data = [(0, test_data[0]), (0.25, test_data[1]), (0.75, test_data[2])]


                        print("Evaluating marginals")
                        self.evaluate_marginals(time_0_points, test_data[1:], plot_results=False, verbose=False)

                    

            writer.writerow({
                'iter': self.train_step,
                'ot_cost': -info.dual_loss,
                'elapsed_time': self.elapsed_time,
            })
            logf.flush()

            self.train_step += 1
            if self.train_step % self.cfg.save_frequency == 0:
                self.save()

    def eval_alignment(self):
        xflat, x1, x2 = geometries._get_grid(
            self.geometry.xbounds, self.geometry.ybounds, 100)
        xflat = jnp.asarray(xflat)

        if self.cfg.C > 0:
            # assume default condition is 0 for alignment evaluation
            default_condition = jnp.zeros((xflat.shape[0], self.cfg.C))
            x_eval = jnp.concatenate([xflat, default_condition], axis=-1)
        else:
            x_eval = xflat

        if not hasattr(self, 'true_eigvecs') or not hasattr(self, 'learned_eigvecs'):
            # create separate functions for each geometry to avoid comparison issues
            @functools.partial(jax.jit)
            @functools.partial(jax.vmap, in_axes=(None, 0))
            def true_eigvecs(params_geometry, x):
                A = self.reference_geometry.apply(
                    {'params': params_geometry},
                    x, method=self.reference_geometry.metric)
                vals, vecs = jnp.linalg.eigh(A)
                return vecs.T, vals
            
            @functools.partial(jax.jit)
            @functools.partial(jax.vmap, in_axes=(None, 0))
            def learned_eigvecs(params_geometry, x):
                A = self.geometry.apply(
                    {'params': params_geometry},
                    x, method=self.geometry.metric)
                vals, vecs = jnp.linalg.eigh(A)
                return vecs.T, vals
                
            self.true_eigvecs = true_eigvecs
            self.learned_eigvecs = learned_eigvecs

        true_A_evecs, true_eigen_vals = self.true_eigvecs(
            self.params_geometry, x_eval)
        learned_A_evecs, learned_eigen_vals = self.learned_eigvecs(
            self.params_geometry, x_eval)
        alignment = jnp.abs(
            (true_A_evecs * learned_A_evecs).sum(axis=2)).mean().item()
        return alignment, true_eigen_vals, learned_eigen_vals

    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'ot_cost', 'elapsed_time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer


    def plot(self):
        if self.cfg.plotting.get('disable', False):
            print('--- plotting disabled by config')
            return
        print('--- plotting')
        self.plot_all_pairs()
        self.plot_pushforward()

    def plot_all_pairs(self):
        print('plotting')
        rows = math.ceil(math.sqrt(self.num_pairs))
        cols = math.ceil(self.num_pairs / rows)
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        
        
        if self.num_pairs == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        num_plot_points = self.cfg.plotting.get('num_pairs_plot', 100) 
        plot_key = jax.random.PRNGKey(self.cfg.seed + 1) 

        for t in range(self.num_pairs):
            self._setup_ax(axs[t])
            
            source_samples = self.eval_samples[t]
            target_samples = self.eval_samples[t+1]

            if source_samples.shape[0] > num_plot_points:
                k1, plot_key = jax.random.split(plot_key) 
                num_to_sample = min(num_plot_points, source_samples.shape[0], target_samples.shape[0])
                indices = jax.random.choice(k1, source_samples.shape[0], shape=(num_to_sample,), replace=False)
                source_samples_plot = source_samples[indices]
                target_samples_plot = target_samples[indices] 
            else:
                source_samples_plot = source_samples
                target_samples_plot = target_samples

            self.neural_dual_solver.plot_forward_map(
                source_samples_plot, 
                target_samples_plot, 
                self.state_source_maps[t],
                self.state_target_potentials[t],
                self.params_geometry,
                ax=axs[t],
            )

        for i in range(self.num_pairs, len(axs)):
            axs[i].axis('off')

        fig.tight_layout()
        fig.savefig('all_pairs.png')
        wandb.log({"plots/all_pairs": wandb.Image('all_pairs.png')}, step=self.train_step)
        plt.close(fig)

    def _setup_ax(self, ax, condition = None): 
        if hasattr(self.geometry, 'xbounds'):
            xlims = self.geometry.xbounds
            ylims = self.geometry.ybounds
        else:
            xlims = ylims = self.geometry.bounds

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        self.geometry.add_plot_background(
            self.params_geometry, ax, xlims=xlims, ylims=ylims, condition=condition)

        if 'neural' in self.cfg.geometry or 'land' in self.cfg.geometry:
            if self.data in ['scarvelis_xpath','scarvelis_vee','scarvelis_circle','scarvelis_arch']:
                self.reference_geometry.add_plot_background(
                    self.params_geometry, ax, xlims=xlims, ylims=ylims,
                    alpha=0.5,
                )

    def _clean_axis(self, ax):
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color('k')
            sp.set_linewidth(3)

    def plot_pushforward(self, num_samples=100):
        all_init_xs = jax.random.choice(
            jax.random.PRNGKey(0), self.eval_samples[0], shape=(num_samples,), replace=False)

        if self.cfg.C > 0 and self.cfg.categorical:
            condition_vectors = all_init_xs[:, self.cfg.D:]
            unique_conditions = jnp.unique(condition_vectors, axis=0)
            num_conditions = unique_conditions.shape[0]
        else:
            unique_conditions = None 
            num_conditions = 1

        cmap = plt.get_cmap('viridis', num_conditions)
        norm = mpl.colors.Normalize(vmin=0, vmax=num_conditions - 1)

        for c_idx in range(num_conditions):

            fig, ax = plt.subplots(figsize=(8, 4))
            
            if unique_conditions is not None:
                current_condition = unique_conditions[c_idx]
                condition_matches = jnp.all(condition_vectors == current_condition, axis=1)
                condition_indices = jnp.where(condition_matches)[0]
                self._setup_ax(ax, condition=current_condition)
            else:
                condition_indices = jnp.arange(all_init_xs.shape[0])
                self._setup_ax(ax)

            plot_color = cmap(norm(c_idx))

            init_xs_condition = all_init_xs[condition_indices]

            for i in range(init_xs_condition.shape[0]):
                init_x = init_xs_condition[i]
                ax.scatter([init_x[0]], [init_x[1]], s=20, alpha=1,
                            zorder=10, c=[plot_color])

                x = init_x
                for t in range(self.num_pairs):
                    prev_x = x

                    x = self.neural_dual_solver.pushforward_jit(
                        self.state_source_maps[t].params,
                        self.state_target_potentials[t].params,
                        self.params_geometry,
                        x
                    ).solution

                    path = self.neural_dual_solver.path_jit(
                        self.params_geometry, prev_x, x)

                    ax.plot(
                        path[:, 0], path[:, 1],
                        color=plot_color, 
                        alpha=0.5,
                        lw=3,
                    )

            self._clean_axis(ax)
            fname = f'pushforward_condition_{c_idx}.png' 
            print(f'saving to {fname}')
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)
            wandb.log({f"plots/pushforward_condition_{c_idx}": wandb.Image(fname)}, step=self.train_step)
            plt.close(fig)

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        print(f"Saving to {path}")

        # Temporarily remove non-picklable samplers
        samplers_backup = self.samplers
        self.samplers = None

        try:
            with open(path, "wb") as f:
                pkl.dump(self, f)
        finally:
            # Restore samplers
            self.samplers = samplers_backup

#uncert. quant.
    def _compute_geodesic_distance(self, x, y):
        """
        Computes geodesic distance d(x, y) using the learned metric.
        """
        cost_val = self.geometry.apply(
            {'params': self.params_geometry},
            x,
            y,
            method=self.geometry.cost
        )
        distance = jnp.sqrt(2 * cost_val)
        return distance

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_pair_uncertainty(self, x_i, x_i_plus_1):
        """
        Computes uncertainty heuristic along the path between a single pair (x_i, x_{i+1}).
        """
        # 1. Compute geodesic path
        path = self.neural_dual_solver.path_jit(
            self.params_geometry, x_i, x_i_plus_1
        )
        num_path_points = path.shape[0]
        if num_path_points < 3:
             # Cannot find an intermediate point, return 0 uncertainty, s*=0.5 (midpoint)
             return 0.5, 0.0

        dist_to_start_fn = jax.vmap(lambda p: self._compute_geodesic_distance(p, x_i))
        dist_to_end_fn = jax.vmap(lambda p: self._compute_geodesic_distance(p, x_i_plus_1))

        # 3. Compute distances for all intermediate points
        intermediate_path = path[1:-1] # Exclude exact start and end points
        dists_to_start = dist_to_start_fn(intermediate_path)
        dists_to_end = dist_to_end_fn(intermediate_path)

        # 4. Find point maximizing min(dist_to_start, dist_to_end)
        min_dists = jnp.minimum(dists_to_start, dists_to_end)
        max_min_dist_idx = jnp.argmax(min_dists)
        max_min_dist_val = min_dists[max_min_dist_idx] # This is U_pair

        # 5. Calculate corresponding s* (time along the path)
        s_star = (max_min_dist_idx + 1) / (num_path_points - 1)

        return s_star, max_min_dist_val

    def find_most_uncertain_time(self, num_pairs_per_interval=100, key=None):
        """
        Finds the time point t in [0, 1] estimated to be most uncertain based on
        the heuristic of maximizing the minimum geodesic distance to endpoints along paths.

        Args:
            num_pairs_per_interval: Number of random pairs to sample per interval/condition.
            key: JAX PRNGKey for sampling.

        Returns:
            A tuple (most_uncertain_time, max_average_uncertainty).
            Returns (None, 0.0) if computation fails.
        """
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))

        vmapped_pair_uncertainty = jax.vmap(self._compute_pair_uncertainty)

        all_interval_results = [] # Stores (interval_idx, avg_s_star, avg_uncertainty, condition_idx)

        # Assume time points are equally spaced in [0, 1] (WILL NEED TO CHANGE ONCE CONDUCTING SELECTION)
        #time_points = jnp.linspace(0, 1, self.num_pairs + 1)
        #time_step_duration = 1.0 / self.num_pairs if self.num_pairs > 0 else 1.0

        for t in range(self.num_pairs):
            interval_start_time = self.time_points[t]
            interval_end_time = self.time_points[t+1]
            interval_duration = interval_end_time - interval_start_time
            interval_results = []
            print(f"Analyzing interval {t} -> {t+1} (Time {interval_start_time:.2f} -> {interval_end_time:.2f})...")
            X_i = self.eval_samples[t]
            X_i_plus_1 = self.eval_samples[t+1]

            # --- Handle Conditions ---
            if self.cfg.C > 0:
                conditions_i = X_i[:, self.cfg.D:]
                # Find unique condition vectors and map samples to them
                unique_conditions, condition_map_i = jnp.unique(conditions_i, axis=0, return_inverse=True)
                num_conditions = unique_conditions.shape[0]
                print(f"  Found {num_conditions} unique conditions.")
            else:
                unique_conditions = [None] # Placeholder
                num_conditions = 1
                condition_map_i = jnp.zeros(X_i.shape[0], dtype=int) # All samples belong to condition 0

            for c_idx in range(num_conditions):
                current_condition_vector = unique_conditions[c_idx]
                if self.cfg.C > 0:
                    indices_i = jnp.where(condition_map_i == c_idx)[0]
                    # Find matching samples in X_{i+1} based on the condition vector
                    condition_matches_i_plus_1 = jnp.all(X_i_plus_1[:, self.cfg.D:] == current_condition_vector, axis=1)
                    indices_i_plus_1 = jnp.where(condition_matches_i_plus_1)[0]
                else:
                    indices_i = jnp.arange(X_i.shape[0])
                    indices_i_plus_1 = jnp.arange(X_i_plus_1.shape[0])

                if len(indices_i) == 0 or len(indices_i_plus_1) == 0:
                    print(f"    Skipping condition {c_idx}: No samples found in one or both time points.")
                    continue

                X_i_c = X_i[indices_i]
                X_i_plus_1_c = X_i_plus_1[indices_i_plus_1]

                # --- Sample Pairs ---
                n_samples_i = X_i_c.shape[0]
                n_samples_i_plus_1 = X_i_plus_1_c.shape[0]
                num_to_sample = min(num_pairs_per_interval, n_samples_i, n_samples_i_plus_1)

                if num_to_sample == 0:
                     print(f"    Skipping condition {c_idx}: Not enough samples ({n_samples_i}, {n_samples_i_plus_1}) to form pairs.")
                     continue

                key, subkey1, subkey2 = jax.random.split(key, 3)
                idx1 = jax.random.choice(subkey1, n_samples_i, shape=(num_to_sample,), replace=False)
                idx2 = jax.random.choice(subkey2, n_samples_i_plus_1, shape=(num_to_sample,), replace=False)

                sampled_x_i = X_i_c[idx1]
                sampled_x_i_plus_1 = X_i_plus_1_c[idx2]

                # --- Compute Uncertainty for Sampled Pairs ---
                s_stars, uncertainties = vmapped_pair_uncertainty(sampled_x_i, sampled_x_i_plus_1)
                avg_s_star = jnp.mean(s_stars)
                avg_uncertainty = jnp.mean(uncertainties)

                print(f"    Avg s*: {avg_s_star:.4f}, Avg Uncertainty: {avg_uncertainty:.4f} ({num_to_sample} pairs)")
                interval_results.append((t, avg_s_star, avg_uncertainty, c_idx))

            # average across all conditions
            interval_avg = [sum(x) / len(interval_results) for x in zip(*interval_results)]
            all_interval_results.append(interval_avg) 

        # Find the maximum average uncertainty
        best_result_idx = jnp.argmax(jnp.array([res[2] for res in all_interval_results]))
        best_interval_t_idx, best_s_star, max_uncertainty, best_condition_idx = all_interval_results[best_result_idx]
        best_interval_t_idx = int(best_interval_t_idx)

        # Calculate the final uncertain time t \in [0, 1]
        interval_start_time = self.time_points[best_interval_t_idx]
        interval_end_time = self.time_points[best_interval_t_idx + 1]
        interval_duration = interval_end_time - interval_start_time
        most_uncertain_time = interval_start_time + best_s_star * interval_duration

        print(f"\nMost uncertain time estimated at t = {most_uncertain_time:.4f}")
        print(f"  (Based on interval {best_interval_t_idx} -> {best_interval_t_idx+1}, Condition Index {best_condition_idx})")
        print(f"  Max average uncertainty value: {max_uncertainty:.4f}")
        print(f"  Corresponding average s* for max uncertainty interval/condition: {best_s_star:.4f}")

        return most_uncertain_time, max_uncertainty

    def predictor_map_for_assignment(self, x_batch, t = 0):
        
        params_source_map = self.state_source_maps[t].params

        return self.neural_dual_solver.source_map_apply_jit(
            {'params': params_source_map},
            x_batch
        )

    @staticmethod
    @jax.jit
    def _compute_cost_matrix(y_hats, ys):
        """
        Computes the squared Euclidean distance cost matrix.
        """
        # y_hats: (N, D), ys: (N, D)
        # Expand dims for broadcasting: (N, 1, D) and (1, N, D)
        diff = y_hats[:, None, :] - ys[None, :, :] # Shape (N, N, D)
        costs_sq = jnp.sum(diff**2, axis=2) # Shape (N, N)
        return costs_sq

    def assignment_coupling(self, xs, ys, t=0):
        """
        Solves the assignment problem between samples xs and ys, potentially
        separating by condition if conditional dimensions exist (C > 0).

        1. If C > 0, separates samples by unique condition vectors.
        2. For each condition (or for all samples if C=0):
           a. Predicts destinations y_hat_i = predictor_map(x_i) for source samples x_i.
           b. Computes the cost matrix C_ij = ||y_hat_i - y_j||^2 between predicted
              and true target samples within the condition.
           c. Solves the assignment problem to find the permutation minimizing sum C_i, sigma(i).
        3. Returns the combined list of assigned pairs (x_i, y_sigma(i)) across all conditions.

        Args:
            xs (jax.numpy.ndarray): Source samples, shape (N, D+C).
            ys (jax.numpy.ndarray): Target samples, shape (N, D+C).
            t (int): The time interval index for selecting the correct predictor map.

        Returns:
            list[tuple[jax.numpy.ndarray, jax.numpy.ndarray]]: A list of N pairs (x_i, y_j)
                                    representing the optimal assignment.
                                    Returns an empty list if N=0.
        """
        num_samples = xs.shape[0]
        if num_samples == 0:
            return []

        if num_samples != ys.shape[0]:
            raise ValueError(f"xs ({num_samples}) and ys ({ys.shape[0]}) must have the same number of samples.")
        if xs.shape[1] != ys.shape[1]:
            raise ValueError(f"xs ({xs.shape[1]}) and ys ({ys.shape[1]}) must have the same dimension D+C.")
        if len(xs.shape) != 2 or len(ys.shape) != 2:
            raise ValueError("xs and ys must be 2D arrays (N, D+C).")

        all_assigned_pairs = []

        if self.cfg.C > 0:
            # --- Conditional Assignment ---
            conditions_xs = xs[:, self.cfg.D:]
            conditions_ys = ys[:, self.cfg.D:] # Assuming conditions are consistent

            # Find unique condition vectors present in xs (assuming they match ys)
            unique_conditions, inverse_map_xs = jnp.unique(conditions_xs, axis=0, return_inverse=True)
            num_conditions = unique_conditions.shape[0]
            # print(f"Assignment coupling (t={t}): Found {num_conditions} unique conditions.")

            for c_idx in range(num_conditions):
                current_condition = unique_conditions[c_idx]
                # print(f"  Processing condition {c_idx}: {current_condition}")

                # Get indices for samples matching the current condition
                indices_xs = jnp.where(inverse_map_xs == c_idx)[0]
                # Find matching indices in ys (assuming conditions align perfectly)
                condition_matches_ys = jnp.all(conditions_ys == current_condition, axis=1)
                indices_ys = jnp.where(condition_matches_ys)[0]

                xs_c = xs[indices_xs]
                ys_c = ys[indices_ys]
                num_samples_c = xs_c.shape[0]

                if num_samples_c == 0:
                    # print(f"    Skipping condition {c_idx}: No samples.")
                    continue
                if xs_c.shape[0] != ys_c.shape[0]:
                     # This case is assumed not to happen based on the prompt
                     raise ValueError(f"Condition {c_idx}: Mismatch in sample count for condition {current_condition}. xs: {xs_c.shape[0]}, ys: {ys_c.shape[0]}. This violates the assumption.")

                # Predict destinations for source samples of this condition
                try:
                    y_hats_c = self.predictor_map_for_assignment(xs_c, t=t)
                    if y_hats_c.shape != xs_c.shape:
                         raise ValueError(f"Predictor output shape {y_hats_c.shape} does not match input shape {xs_c.shape} for condition {c_idx}")
                except Exception as e:
                    print(f"Error calling predictor_map_for_assignment for condition {c_idx}: {e}")
                    raise e

                # Compute cost matrix using only spatial dimensions (D) for distance
                cost_matrix_jax_c = self._compute_cost_matrix(y_hats_c[:, :self.cfg.D], ys_c[:, :self.cfg.D])
                cost_matrix_np_c = np.array(cost_matrix_jax_c)

                # Solve assignment for this condition
                row_ind_c, col_ind_c = linear_sum_assignment(cost_matrix_np_c)

                # Store pairs using original full samples (including condition dims)
                condition_pairs = []
                assignment_c = sorted(zip(row_ind_c, col_ind_c))
                for r, c in assignment_c:
                    # r is index within xs_c, c is index within ys_c
                    original_x = xs_c[r]
                    original_y = ys_c[c]
                    condition_pairs.append((original_x, original_y))

                all_assigned_pairs.extend(condition_pairs)
                # print(f"    Assigned {len(condition_pairs)} pairs for condition {c_idx}.")

        else:
            # --- Unconditional Assignment (Original Logic) ---
            # print(f"Assignment coupling (t={t}): No conditions (C=0).")
            try:
                y_hats = self.predictor_map_for_assignment(xs, t=t)
                if y_hats.shape != xs.shape:
                    raise ValueError(f"Predictor output shape {y_hats.shape} does not match input shape {xs.shape}")
            except Exception as e:
                print(f"Error calling predictor_map_for_assignment: {e}")
                raise e

            # Compute cost matrix (implicitly uses all dimensions if C=0)
            cost_matrix_jax = self._compute_cost_matrix(y_hats, ys)
            cost_matrix_np = np.array(cost_matrix_jax)

            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix_np)

            # Store pairs
            assignment = sorted(zip(row_ind, col_ind))
            for r, c in assignment:
                all_assigned_pairs.append((xs[r], ys[c]))
            # print(f"  Assigned {len(all_assigned_pairs)} pairs.")

        if len(all_assigned_pairs) != num_samples:
             print(f"Warning: Number of assigned pairs ({len(all_assigned_pairs)}) does not match input sample size ({num_samples}). Check conditional logic.")

        return all_assigned_pairs

    def plot_assignment_paths(self, num_samples=100):
        """
        Computes the optimal transport assignment between consecutive time points
        based on the learned source map predictor (handling conditions internally)
        and plots the geodesic paths between the assigned pairs.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        self._setup_ax(ax) # Setup axis once for the whole plot
        colors = plt.cm.viridis(np.linspace(0, 1, self.num_pairs))

        plot_key = jax.random.PRNGKey(self.cfg.seed + 42) # Use a consistent key for sampling

        for t in range(self.num_pairs):
            source_samples_full = self.eval_samples[t]
            target_samples_full = self.eval_samples[t+1]

            # --- Sampling Logic (Consistent across conditions if C>0) ---
            num_available = source_samples_full.shape[0]
            num_to_sample = min(num_samples, num_available)

            if num_available > num_samples:
                plot_key, subkey = jax.random.split(plot_key)
                # Sample indices once, these indices will be used for both source and target
                # This maintains the implicit pairing assumption if data is ordered by condition
                indices = jax.random.choice(subkey, num_available, shape=(num_to_sample,), replace=False)
                source_samples_plot = source_samples_full[indices]
                target_samples_plot = target_samples_full[indices] # Use same indices for target
            else:
                source_samples_plot = source_samples_full
                target_samples_plot = target_samples_full

            # --- Get Assigned Pairs (Handles conditions internally) ---
            # assignment_coupling now returns pairs for *all* conditions for this time t
            pairs_to_plot = self.assignment_coupling(source_samples_plot, target_samples_plot, t=t)

            if not pairs_to_plot:
                print(f"Warning: No pairs returned from assignment_coupling for t={t}. Skipping plot for this interval.")
                continue

            # --- Plotting ---
            print(f"Plotting {len(pairs_to_plot)} assigned paths for interval t={t}")
            for i, (x_i, y_j) in enumerate(pairs_to_plot):
                # Compute path using only spatial dimensions for geometry
                path = self.neural_dual_solver.path_jit(
                    self.params_geometry, x_i, y_j # Use only spatial dims for path
                )

                # Plot path
                ax.plot(
                    path[:, 0], path[:, 1],
                    color=colors[t],
                    alpha=0.5,
                    lw=0.8
                )
                # Plot start point (spatial dims)
                ax.scatter(x_i[0], x_i[1], color=colors[t], s=10, alpha=0.7, zorder=5)
                # Plot assigned end point (spatial dims) - maybe use a different marker/color
                ax.scatter(y_j[0], y_j[1], color='red', marker='x', s=10, alpha=0.7, zorder=5)

        ax.set_title(f"Geodesic Paths from OT Assignment (Step {self.train_step})")
        self._clean_axis(ax)
        fig.tight_layout()

        fname = 'assignment_paths.png'
        print(f"Saving assignment paths plot to {fname}")
        fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)
        wandb.log({"plots/assignment_paths": wandb.Image(fname)}, step=self.train_step)
        plt.close(fig)


    def compute_wasserstein_distance(self, samples1, samples2):
        """
        Compute Wasserstein distance between two sets of samples using POT library.
        
        Args:
            samples1: JAX array of shape (n_samples1, dim)
            samples2: JAX array of shape (n_samples2, dim)
            
        Returns:
            float: The Wasserstein distance
        """
        # Convert from JAX arrays to numpy for POT library compatibility
        samples1_np = np.array(samples1)
        samples2_np = np.array(samples2)
        

        M = ot.dist(samples1_np, samples2_np)
        a = np.ones(samples1_np.shape[0]) / samples1_np.shape[0]  # uniform weights
        b = np.ones(samples2_np.shape[0]) / samples2_np.shape[0]  # uniform weights
            
        return float(ot.emd2(a, b, M))
    
    def _rbf_kernel(self, X, Y, gamma=1.0):
        """
        Computes the RBF kernel matrix between X and Y.
        K(x, y) = exp(-gamma * ||x-y||^2)
        """
        XY_sqdist = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * XY_sqdist)

    def compute_mmd_rbf(self, samples1_np, samples2_np, gamma=None):
        """
        Computes MMD^2 (squared) between two sets of samples using an RBF kernel.
        Assumes samples1_np and samples2_np are numpy arrays.
        """
        if samples1_np.ndim == 1:
            samples1_np = samples1_np[:, np.newaxis]
        if samples2_np.ndim == 1:
            samples2_np = samples2_np[:, np.newaxis]

        if gamma is None:
            # Heuristic for gamma: median of pairwise squared distances
            all_samples = np.vstack([samples1_np, samples2_np])
            pairwise_sq_dists = cdist(all_samples, all_samples, 'sqeuclidean')
            # Use only upper triangle to avoid diagonal zeros and duplicates
            median_sq_dist = np.median(pairwise_sq_dists[np.triu_indices_from(pairwise_sq_dists, k=1)])
            if median_sq_dist <= 0: # Handle case where all points are identical or very close
                median_sq_dist = 1e-6
            gamma = 1.0 / (2 * median_sq_dist)


        K_XX = self._rbf_kernel(samples1_np, samples1_np, gamma)
        K_YY = self._rbf_kernel(samples2_np, samples2_np, gamma)
        K_XY = self._rbf_kernel(samples1_np, samples2_np, gamma)

        n = K_XX.shape[0]
        m = K_YY.shape[0]

        # Biased estimator of MMD^2
        mmd_sq = (np.sum(K_XX) / (n * n) +
                  np.sum(K_YY) / (m * m) -
                  2 * np.sum(K_XY) / (n * m))
        
        # Ensure non-negativity due to potential floating point issues for very similar distributions
        return max(0, mmd_sq)
        
    def evaluate_marginals(self, initial_samples_at_t0, evaluation_points, plot_results=False, verbose=False):
        """
        Evaluates model by transporting initial samples through learned maps and
        geodesic paths, comparing with ground truth time marginals.

        For each interval [T_k, T_{k+1}], samples are transported from T_k.
        If an `eval_time` falls within this interval:
        - At T_k or T_{k+1}: uses directly transported samples.
        - Between T_k and T_{k+1}: interpolates along the geodesic path for time (eval_time - T_k) / (T_{k+1} - T_k).

        Args:
            initial_samples_at_t0 (jax.numpy.ndarray): Samples at self.time_points[0].
            evaluation_points (list[tuple[float, jax.numpy.ndarray]]):
                List of (eval_time, ground_truth_samples_at_eval_time).
            plot_results (bool): If True, logs plots of comparisons.

        Returns:
            dict: Discrepancy metrics for each evaluation point.
        """
        evaluation_points.sort(key=lambda x: x[0])
        min_train_time = self.time_points[0]
        max_train_time = self.time_points[self.num_pairs]

        valid_evaluation_points = []
        for t_eval, samples in evaluation_points:
            if not (min_train_time <= t_eval <= max_train_time):
                print(f"Warning: Evaluation time {t_eval:.4f} is outside the trained range "
                      f"[{min_train_time:.4f}, {max_train_time:.4f}]. Skipping.")
                continue
            valid_evaluation_points.append((t_eval, samples))
        
        evaluation_points = valid_evaluation_points
        
        metrics_log = {}
        current_transported_samples = initial_samples_at_t0
        eval_point_idx = 0
        plot_key = jax.random.PRNGKey(self.cfg.seed + 200)

        if verbose:
            print(f"\n--- Evaluating at {len(evaluation_points)} points ---")

        all_wasserstein_distances = []
        all_circle_distances = []
        all_log_likelihoods = []
        all_mmd_scores = []
        for k in range(self.num_pairs):
            T_k = self.time_points[k]
            T_k_plus_1 = self.time_points[k+1]

            if verbose:
                print(f"Processing interval {k}: [{T_k:.4f}, {T_k_plus_1:.4f}]")

            params_source_map_k = self.state_source_maps[k].params
            end_samples_pred_at_Tk_plus_1 = self.neural_dual_solver.source_map_apply_jit(
                {'params': params_source_map_k},
                current_transported_samples
            )
            end_samples_pred_at_Tk_plus_1 = self.geometry.batch_project(end_samples_pred_at_Tk_plus_1)

            @jax.jit
            def interpolate_batch_in_interval(current_geom_params, start_samples_batch, end_samples_pred_for_batch, s_fraction_val):
                return jax.vmap(
                    lambda x_start, y_end, s_f: self.geometry.apply(
                        {'params': current_geom_params},
                        x_start,
                        y_end,
                        s_f,
                        method=self.geometry.point_on_path
                    ), 
                    in_axes=(0, 0, None)
                )(start_samples_batch, end_samples_pred_for_batch, s_fraction_val)

            while eval_point_idx < len(evaluation_points):
                eval_time, true_eval_samples = evaluation_points[eval_point_idx]
                
                if eval_time > T_k_plus_1:
                    break 
                
                predicted_eval_samples = None
                desc = ""

                if eval_time == T_k:
                    predicted_eval_samples = current_transported_samples
                    desc = f"at T_k={T_k:.4f} (start of interval {k})"
                elif eval_time == T_k_plus_1:
                    predicted_eval_samples = end_samples_pred_at_Tk_plus_1
                    desc = f"at T_k+1={T_k_plus_1:.4f} (end of interval {k}, full transport)"
                else:
                    s_fraction = (eval_time - T_k) / (T_k_plus_1 - T_k)
                    predicted_eval_samples = interpolate_batch_in_interval(
                        self.params_geometry,
                        current_transported_samples,
                        end_samples_pred_at_Tk_plus_1,
                        s_fraction
                    )
                    desc = f"interpolated at s={s_fraction:.3f} in [{T_k:.4f}, {T_k_plus_1:.4f}]"
                
                predicted_eval_samples = self.geometry.batch_project(predicted_eval_samples)
                predicted_spatial_overall = predicted_eval_samples[:, :self.cfg.D]
                actual_spatial_overall = true_eval_samples[:, :self.cfg.D]

                #Wasserstein distance
                if self.cfg.C and self.data != "ett_forecasts":
                    true_conditions_all = true_eval_samples[:, self.cfg.D:]
                    predicted_conditions_all = predicted_eval_samples[:, self.cfg.D:]
                    unique_true_conditions = jnp.unique(true_conditions_all, axis=0)

                    per_condition_wasserstein = []
                    for cond_idx, true_cond_vec in enumerate(unique_true_conditions):
                        true_cond_mask = jnp.all(true_conditions_all == true_cond_vec, axis=1)
                        pred_cond_mask = jnp.all(predicted_conditions_all == true_cond_vec, axis=1)

                        true_samples_for_cond = true_eval_samples[true_cond_mask][:, :self.cfg.D]
                        pred_samples_for_cond = predicted_eval_samples[pred_cond_mask][:, :self.cfg.D]

                        if true_samples_for_cond.shape[0] > 0 and pred_samples_for_cond.shape[0] > 0:
                            wasserstein_dist = self.compute_wasserstein_distance(pred_samples_for_cond, true_samples_for_cond)
                            per_condition_wasserstein.append(wasserstein_dist)
                            metrics_log[f"time_{eval_time:.4f}_cond_{cond_idx}_wass"] = float(wasserstein_dist)

                    if per_condition_wasserstein:
                        avg_wasserstein = jnp.mean(jnp.array(per_condition_wasserstein))
                        metrics_log[f"time_{eval_time:.4f}_avg_wass"] = float(avg_wasserstein)
                        all_wasserstein_distances.extend(per_condition_wasserstein)
                        if verbose:
                            print(f"Avg Wasserstein Across Conditions: {avg_wasserstein:.4e}")

                else: 
                    wasserstein_dist = self.compute_wasserstein_distance(predicted_spatial_overall, actual_spatial_overall)
                    metrics_log[f"time_{eval_time:.4f}_wass"] = float(wasserstein_dist)
                    all_wasserstein_distances.append(wasserstein_dist)
                    metric_val_str = f"Wasserstein: {wasserstein_dist:.4e} (Pred N={predicted_spatial_overall.shape[0]}, Actual N={actual_spatial_overall.shape[0]})"

                    mmd_score = np.nan
                    if predicted_spatial_overall.shape[0] > 0 and actual_spatial_overall.shape[0] > 0:
                        predicted_np = np.array(predicted_spatial_overall)
                        actual_np = np.array(actual_spatial_overall)
                        mmd_score = self.compute_mmd_rbf(predicted_np, actual_np)
                    
                    metrics_log[f"time_{eval_time:.4f}_mmd_rbf"] = float(mmd_score)
                    all_mmd_scores.append(mmd_score)
                    metric_val_str += f", MMD_RBF: {mmd_score:.4e}"
                    if verbose:
                        print(f"Evaluated at time {eval_time:.4f} ({desc}): {metric_val_str}")

                # Circle distance
                if "circle" in self.data:
                    circle_centers = {0: jnp.array([-1.0, 0.0]), 1: jnp.array([-1.0, 0.0]), 2: jnp.array([1.0, 0.0]), 3: jnp.array([1.0, 0.0])}
                    circle_radius = 1.0
                    per_condition_circle_dist = []

                    for cond_idx, true_cond_vec in enumerate(unique_true_conditions):
                        true_cond_mask = jnp.all(true_conditions_all == true_cond_vec, axis=1)
                        pred_cond_mask = jnp.all(predicted_conditions_all == true_cond_vec, axis=1)

                        pred_samples_for_cond = predicted_spatial_overall[pred_cond_mask]

                        if pred_samples_for_cond.shape[0] > 0:
                            center = circle_centers[cond_idx]
                            distances = jnp.abs(jnp.linalg.norm(pred_samples_for_cond - center, axis=1) - circle_radius)
                            avg_distance = jnp.mean(distances)
                            per_condition_circle_dist.append(avg_distance)
                            metrics_log[f"time_{eval_time:.4f}_cond_{cond_idx}_circle_dist"] = float(avg_distance)

                    if per_condition_circle_dist:
                        avg_circle_dist = jnp.mean(jnp.array(per_condition_circle_dist))
                        metrics_log[f"time_{eval_time:.4f}_avg_circle_dist"] = float(avg_circle_dist)
                        all_circle_distances.extend(per_condition_circle_dist)
                        if verbose:
                            print(f"Avg Circle Distance Across Conditions: {avg_circle_dist:.4e}")

                if "semicircle" in self.data and self.cfg.C > 0:
                    if predicted_eval_samples.shape[1] == self.cfg.D + self.cfg.C:
                        predicted_data_torch = torch.from_numpy(np.array(predicted_eval_samples))
                        try:
                            log_likelihood_val = -log_likelihood_conditional_semicircle(
                                data=predicted_data_torch,
                                time=eval_time
                            ) / predicted_data_torch.shape[0]
                            metrics_log[f"time_{eval_time:.4f}_log_likelihood"] = float(log_likelihood_val)
                            all_log_likelihoods.append(log_likelihood_val)
                            if verbose:
                                print(f"Log Likelihood at t={eval_time:.4f}: {log_likelihood_val:.4e}")
                        except Exception as e:
                            if verbose:
                                print(f"Error computing log-likelihood for semicircles at t={eval_time:.4f}: {e}")
                    else:
                        if verbose:
                            print(f"Skipping log-likelihood for semicircles at t={eval_time:.4f}: predicted_eval_samples shape {predicted_eval_samples.shape} incorrect for conditional data.")

                # Plotting
                if plot_results and self.cfg.D >= 2:
                    fig_comp, ax_comp = plt.subplots(figsize=(8, 4))
                    #with the 100th condition for background
                    self._setup_ax(ax_comp, condition = true_eval_samples[100, self.cfg.D:])
                    num_plot = self.cfg.plotting.get('num_eval_plot', 200)
                    pk1, pk2, plot_key = jax.random.split(plot_key, 3)

                    if self.cfg.C:
                        true_conditions_all = true_eval_samples[:, self.cfg.D:]
                        predicted_conditions_all = predicted_eval_samples[:, self.cfg.D:]
                        unique_conditions = jnp.unique(true_conditions_all, axis=0)
                        cmap = plt.cm.get_cmap('nipy_spectral', len(unique_conditions)) 

                        for cond_idx, cond_vec in enumerate(unique_conditions):
                            cond_mask_true = jnp.all(true_conditions_all == cond_vec, axis=1)
                            cond_mask_pred = jnp.all(predicted_conditions_all == cond_vec, axis=1)
                            true_cond_samples = actual_spatial_overall[cond_mask_true]
                            pred_cond_samples = predicted_spatial_overall[cond_mask_pred]
                            idx_pred = jax.random.choice(pk1, pred_cond_samples.shape[0], shape=(min(num_plot, pred_cond_samples.shape[0]),), replace=False)
                            idx_actual = jax.random.choice(pk2, true_cond_samples.shape[0], shape=(min(num_plot, true_cond_samples.shape[0]),), replace=False)
                            ax_comp.scatter(pred_cond_samples[idx_pred, 0], pred_cond_samples[idx_pred, 1], alpha=0.6, s=20, color=cmap(cond_idx), edgecolors='black', linewidths=0.5)
                            ax_comp.scatter(true_cond_samples[idx_actual, 0], true_cond_samples[idx_actual, 1], alpha=0.3, s=20, color=cmap(cond_idx), marker='x')
                       
                        if "circle" in self.data:
                            circle1 = plt.Circle((-1, 0), 1, color='black', fill=False, linestyle='--', lw=0.5)
                            circle2 = plt.Circle((1, 0), 1, color='black', fill=False, linestyle='--', lw=0.5)
                            ax_comp.add_artist(circle1)
                            ax_comp.add_artist(circle2)
                    else: 
                        idx_pred = jax.random.choice(pk1, predicted_spatial_overall.shape[0], shape=(min(num_plot, predicted_spatial_overall.shape[0]),), replace=False)
                        idx_actual = jax.random.choice(pk2, actual_spatial_overall.shape[0], shape=(min(num_plot, actual_spatial_overall.shape[0]),), replace=False)
                        ax_comp.scatter(predicted_spatial_overall[idx_pred, 0], predicted_spatial_overall[idx_pred, 1], alpha=0.6, label=f'Predicted (t={eval_time:.3f})', s=20, color='blue')
                        ax_comp.scatter(actual_spatial_overall[idx_actual, 0], actual_spatial_overall[idx_actual, 1], alpha=0.3, label=f'Actual Test (t={eval_time:.3f})', s=20, color='red', marker='x')

                    ax_comp.set_title(f'Test Eval: t={eval_time:.3f} (Interval {k}')
                    comp_fname = f'test_eval_time_{eval_time:.3f}.png'
                    fig_comp.savefig(comp_fname)
                    if wandb.run is not None:
                        wandb.log({f"plots/test/time_{eval_time:.3f}": wandb.Image(comp_fname)}, step=self.train_step)
                    plt.close(fig_comp)

                eval_point_idx += 1

            current_transported_samples = end_samples_pred_at_Tk_plus_1

        if all_wasserstein_distances:
            overall_avg_wasserstein = jnp.mean(jnp.array(all_wasserstein_distances))
            metrics_log["overall_avg_wass"] = float(overall_avg_wasserstein)
            if verbose:
                print(f"Overall Avg Wasserstein Across All Conditions and Time Points: {overall_avg_wasserstein:.4e}")

        if all_mmd_scores:
            overall_avg_mmd = np.nanmean(np.array(all_mmd_scores))
            metrics_log["overall_avg_mmd_rbf"] = float(overall_avg_mmd)
            if verbose:
                print(f"Overall Avg MMD (RBF) Across All Time Points: {overall_avg_mmd:.4e}")

        if all_circle_distances:
            overall_avg_circle_dist = jnp.mean(jnp.array(all_circle_distances))
            metrics_log["overall_avg_circle_dist"] = float(overall_avg_circle_dist)
            if verbose:
                print(f"Overall Avg Circle Distance Across All Conditions and Time Points: {overall_avg_circle_dist:.4e}")

        if all_log_likelihoods:
            overall_avg_log_likelihood = np.mean(np.array(all_log_likelihoods))
            metrics_log["overall_avg_log_likelihood"] = float(overall_avg_log_likelihood)
            if verbose:
                print(f"Overall Avg Log Likelihood Across All Time Points: {overall_avg_log_likelihood:.4e}")
        
        if verbose:
            print("--- Finished Marginal Evaluation ---")
        if wandb.run is not None and metrics_log:
            wandb.log({"test": metrics_log}, step=self.train_step)

        return metrics_log



@hydra.main(config_path=".", config_name="train.yaml", version_base="1.1")
def main(cfg):
    from train import Workspace as W

    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        print(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
        workspace.samplers = data.get_samplers_scarvelis(
            workspace.cfg.data,
            num_pairs_requested=workspace.cfg.get("num_pairs", None)
        )
        print(f"Re-initialized samplers for loaded workspace (num_pairs={len(workspace.samplers)-1})")
    else:
        workspace = W(cfg)

    workspace.run()

if __name__ == '__main__':
    main()
