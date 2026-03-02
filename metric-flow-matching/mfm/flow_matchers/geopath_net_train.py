import torch
import pytorch_lightning as pl


from mfm.flow_matchers.ema import EMA


class GeoPathNetTrain(pl.LightningModule):
    def __init__(
        self,
        flow_matcher,
        args,
        skipped_time_points: list = None,
        ot_sampler=None,
        data_manifold_metric=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.flow_matcher = flow_matcher
        self.geopath_net = flow_matcher.geopath_net
        self.ot_sampler = ot_sampler
        self.skipped_time_points = skipped_time_points if skipped_time_points else []
        self.optimizer_name = args.geopath_optimizer
        self.lr = args.geopath_lr
        self.weight_decay = args.geopath_weight_decay
        self.args = args
        self.data_manifold_metric = data_manifold_metric
        self.multiply_validation = 4

        # Class conditioning
        self.class_conditioning = getattr(args, 'class_conditioning', False)
        self.categorical = getattr(args, 'categorical', True)
        self.ambient_dim = args.dim

        self.first_loss = None
        self.timesteps = None
        self.computing_reference_loss = False

    def forward(self, x0, x1, t):
        return self.geopath_net(x0, x1, t)

    def on_train_start(self):
        self.first_loss = self.compute_initial_loss()

    def compute_initial_loss(self):
        self.geopath_net.train(mode=False)
        total_loss = 0
        total_count = 0
        with torch.enable_grad():
            self.t_val = []
            for i in range(
                self.trainer.datamodule.num_timesteps - len(self.skipped_time_points)
            ):
                self.t_val.append(
                    torch.rand(
                        self.trainer.datamodule.batch_size * self.multiply_validation,
                        requires_grad=True,
                    )
                )
        self.computing_reference_loss = True
        with torch.no_grad():
            old_alpha = self.flow_matcher.alpha
            self.flow_matcher.alpha = 0
            for batch in self.trainer.datamodule.train_dataloader():
                self.timesteps = torch.linspace(
                    0.0, 1.0, len(batch[0]["train_samples"][0])
                )
                loss = self._compute_loss(
                    batch[0]["train_samples"][0],
                    batch[0]["metric_samples"][0],
                )
                total_loss += loss.item()
                total_count += 1
            self.flow_matcher.alpha = old_alpha
        self.computing_reference_loss = False
        self.geopath_net.train(mode=True)
        return total_loss / total_count if total_count > 0 else 1.0

    def _compute_loss(self, main_batch, metric_samples_batch):
        main_batch_filtered = [
            x.to(self.device)
            for i, x in enumerate(main_batch)
            if i not in self.skipped_time_points
        ]
        metric_samples_batch_filtered = [
            x.to(self.device)
            for i, x in enumerate(metric_samples_batch)
            if i not in self.skipped_time_points
        ]

        if self.class_conditioning:
            ambient_dim = self.ambient_dim
            # Split spatial and condition for main batch
            spatials = [x[:, :ambient_dim] for x in main_batch_filtered]
            conds = [x[:, ambient_dim:] for x in main_batch_filtered]
            if self.categorical:
                conds = [c.squeeze(-1) for c in conds]
            x0s, x1s = spatials[:-1], spatials[1:]
            c0s, c1s = conds[:-1], conds[1:]

            # Use spatial-only for metric samples
            metric_spatials = [x[:, :ambient_dim] for x in metric_samples_batch_filtered]
            samples0, samples1 = metric_spatials[:-1], metric_spatials[1:]

            ts, xts, uts, ct = self._process_flow(x0s, x1s, c0s, c1s)
        else:
            x0s, x1s = main_batch_filtered[:-1], main_batch_filtered[1:]
            samples0, samples1 = (
                metric_samples_batch_filtered[:-1],
                metric_samples_batch_filtered[1:],
            )
            ts, xts, uts = self._process_flow(x0s, x1s)

        loss = 0
        velocities = []
        for i in range(len(ts)):
            samples = torch.cat([samples0[i], samples1[i]], dim=0)
            vel = self.data_manifold_metric.calculate_velocity(
                xts[i], uts[i], samples, i
            )
            velocities.append(vel)
        loss = torch.mean(torch.cat(velocities) ** 2)
        self.log(
            "GeoPathNet/mean_velocity_geopath",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def _process_flow(self, x0s, x1s, c0s=None, c1s=None):
        ts, xts, uts = [], [], []
        conds = [] if self.class_conditioning else None
        t_start = self.timesteps[0]
        i_start = 0

        for i, (x0, x1) in enumerate(zip(x0s, x1s)):
            x0, x1 = torch.squeeze(x0), torch.squeeze(x1)

            # Handle condition repeats for validation
            c0 = c0s[i] if c0s is not None else None
            c1 = c1s[i] if c1s is not None else None

            if self.trainer.validating or self.computing_reference_loss:
                repeat_tuple = (self.multiply_validation, 1) + (1,) * (
                    len(x0.shape) - 2
                )
                x0 = x0.repeat(repeat_tuple)
                x1 = x1.repeat(repeat_tuple)
                if c0 is not None:
                    cond_repeat = (self.multiply_validation,) + (1,) * (c0.dim() - 1)
                    c0 = c0.repeat(cond_repeat)
                    c1 = c1.repeat(cond_repeat)

            if self.skipped_time_points and i + 1 >= self.skipped_time_points[0]:
                t_start_next = self.timesteps[i + 2]
            else:
                t_start_next = self.timesteps[i + 1]

            t = None
            if self.trainer.validating or self.computing_reference_loss:
                t = self.t_val[i].to(x0.device)

            if self.class_conditioning and self.categorical and c0 is not None:
                # Group by class, process each class separately
                unique_classes = torch.unique(c0)
                batch_ts, batch_xts, batch_uts, batch_cs = [], [], [], []

                for cls in unique_classes:
                    mask0 = (c0 == cls)
                    mask1 = (c1 == cls)
                    x0_cls = x0[mask0]
                    x1_cls = x1[mask1]
                    min_n = min(len(x0_cls), len(x1_cls))
                    if min_n == 0:
                        continue
                    x0_cls = x0_cls[:min_n]
                    x1_cls = x1_cls[:min_n]
                    cond_cls = torch.full((min_n,), cls.item(), device=x0.device)

                    # Handle validation t
                    if t is not None:
                        t_cls = t[mask0][:min_n] if len(t) > min_n else t[:min_n]
                    else:
                        t_cls = None

                    t_out, xt_cls, ut_cls = self.flow_matcher.sample_location_and_conditional_flow(
                        x0_cls, x1_cls, t_start, t_start_next,
                        training_geopath_net=True, t=t_cls, cond=cond_cls
                    )
                    batch_ts.append(t_out)
                    batch_xts.append(xt_cls)
                    batch_uts.append(ut_cls)
                    batch_cs.append(cond_cls)

                ts.append(torch.cat(batch_ts))
                xts.append(torch.cat(batch_xts))
                uts.append(torch.cat(batch_uts))
                conds.append(torch.cat(batch_cs))

            elif self.class_conditioning and not self.categorical and c0 is not None:
                # Continuous conditioning: pair randomly, use x0's condition
                if self.ot_sampler is not None:
                    x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=True)
                t_out, xt_i, ut_i = self.flow_matcher.sample_location_and_conditional_flow(
                    x0, x1, t_start, t_start_next,
                    training_geopath_net=True, t=t, cond=c0
                )
                ts.append(t_out)
                xts.append(xt_i)
                uts.append(ut_i)
                conds.append(c0)

            else:
                if self.ot_sampler is not None:
                    x0, x1 = self.ot_sampler.sample_plan(
                        x0,
                        x1,
                        replace=True,
                    )
                t_out, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(
                    x0, x1, t_start, t_start_next, training_geopath_net=True, t=t
                )
                ts.append(t_out)
                xts.append(xt)
                uts.append(ut)

            t_start = t_start_next

        if self.class_conditioning:
            return ts, xts, uts, conds
        return ts, xts, uts

    def training_step(self, batch, batch_idx):
        main_batch = batch["train_samples"][0]
        metric_batch = batch["metric_samples"][0]
        tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)
        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss
        self.log(
            "GeoPathNet/mean_geopath_geopath",
            (self.flow_matcher.geopath_net_output.abs().mean()),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "GeoPathNet/train_loss_geopath",
            tangential_velocity_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss

    def validation_step(self, batch, batch_idx):
        main_batch = batch["val_samples"][0]
        metric_batch = batch["metric_samples"][0]
        tangential_velocity_loss = self._compute_loss(main_batch, metric_batch)
        if self.first_loss:
            tangential_velocity_loss = tangential_velocity_loss / self.first_loss
        self.log(
            "GeoPathNet/val_loss_geopath",
            tangential_velocity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return tangential_velocity_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if isinstance(self.geopath_net, EMA):
            self.geopath_net.update_ema()

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.geopath_net.parameters(),
                lr=self.lr,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.geopath_net.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        return optimizer
