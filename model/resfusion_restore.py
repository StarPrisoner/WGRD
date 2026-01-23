import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from pytorch_lightning.loggers import TensorBoardLogger
from typing import List, Union, Optional

from utils.dwt import DWTForward, DWTInverse
from .distributions import resfusion_x0_to_xt, ddpm_x0_to_xt
from variance_scheduler.abs_var_scheduler import Scheduler


class GaussianResfusion_Restore(pl.LightningModule):
    """
    Gaussian Prior Residual Noise embedded De-noising Diffusion Probabilistic Model (WGRD).
    Implementation enforces physics-guided structural priors via subband-weighted optimization
    while maintaining a standard Gaussian diffusion process for stability.
    """

    def __init__(self, denoising_module: pl.LightningModule, variance_scheduler: Scheduler,
                 mode='epsilon', loss_type='L2', optimizer_type='AdamW',
                 lr_scheduler_type='CosineAnnealingLR', **kwargs):
        """
        :param denoising_module: The neural network for denoising (UNet/WaveUNet).
        :param variance_scheduler: The noise schedule provider.
        :param mode: Prediction mode ('epsilon' or 'residual').
        """
        super().__init__()
        self.save_hyperparameters(ignore=['denoising_module', 'variance_scheduler'])

        # Configuration
        self.mode = mode
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.lr_scheduler_type = lr_scheduler_type

        # Initialize Models
        self.denoising_module = denoising_module

        # Wavelet Transforms (Haar)
        n_channels = kwargs.get('n_channels', 3)
        self.dwt = DWTForward()
        self.iwt = DWTInverse(in_channels=n_channels)

        # Metrics
        self.val_PSNR = PeakSignalNoiseRatio(dim=(1, 2, 3), data_range=(0, 1))
        self.test_PSNR = PeakSignalNoiseRatio(dim=(1, 2, 3), data_range=(0, 1))

        # Variance Scheduler
        self.var_scheduler = variance_scheduler
        # Register buffers to ensure they are moved to the correct device automatically
        self.register_buffer('alphas_hat', self.var_scheduler.get_alphas_hat())
        self.register_buffer('alphas', self.var_scheduler.get_alphas())
        self.register_buffer('betas', self.var_scheduler.get_betas())
        self.register_buffer('betas_hat', self.var_scheduler.get_betas_hat())
        self.register_buffer('alphas_hat_t_minus_1', self.var_scheduler.get_alphas_hat_t_minus_1())

        # Acceleration Point (for efficient inference)
        self.T_acc = self.get_acc_point(self.alphas_hat).item()
        print(f'Acceleration point (T_acc): {self.T_acc}')

    def get_acc_point(self, alphas_hat):
        abs_dist = torch.abs(torch.sqrt(alphas_hat) - 0.5)
        return abs_dist.argmin() + 1

    # ==========================================================
    # Physics-Guided Helpers
    # ==========================================================

    def compute_physics_weights(self, hf_tuple):
        """
        Compute physics-guided weighting coefficients based on subband energy.
        Corresponds to Eq. (7) in the paper: w = 1.0 + tanh(sqrt(E)).
        """
        weights = []
        for sb in hf_tuple:
            # Calculate energy per image in batch
            energy = torch.sqrt(torch.mean(sb ** 2, dim=[1, 2, 3], keepdim=True))
            w = 1.0 + torch.tanh(energy)
            weights.append(w)
        return weights

    def gradient_loss(self, pred, target):
        """Sobel Gradient Loss for structural sharpness."""
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        kernel_y = kernel_x.transpose(2, 3)

        grad_pred_x = F.conv2d(pred, kernel_x, padding=1, groups=3)
        grad_pred_y = F.conv2d(pred, kernel_y, padding=1, groups=3)
        grad_target_x = F.conv2d(target, kernel_x, padding=1, groups=3)
        grad_target_y = F.conv2d(target, kernel_y, padding=1, groups=3)

        return F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)

    def frequency_loss(self, pred, target):
        """Frequency Domain Loss with High-Frequency Masking."""
        orig_dtype = pred.dtype
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)

        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        pred_amp, pred_phase = torch.abs(pred_fft), torch.angle(pred_fft)
        target_amp, target_phase = torch.abs(target_fft), torch.angle(target_fft)

        # High-frequency suppression mask
        _, H, W = pred_amp.shape[-3:]
        mask = torch.ones((H, W), device=pred.device)
        center_y, center_x = H // 2, W // 2
        Y, X = torch.meshgrid(torch.arange(H, device=pred.device),
                              torch.arange(W, device=pred.device),
                              indexing='ij')
        dist = torch.sqrt((Y - center_y) ** 2 + (X - center_x) ** 2)
        mask[dist > min(center_y, center_x) * 0.7] = 0.8

        amp_loss = F.l1_loss(pred_amp * mask, target_amp * mask)
        phase_loss = F.l1_loss(pred_phase, target_phase)

        return (0.7 * amp_loss + 0.3 * phase_loss).to(orig_dtype)

    def forward(self, x_t: torch.FloatTensor, I_in: torch.FloatTensor, t: int) -> torch.Tensor:
        return self.denoising_module(x_t, I_in, t)

    # ==========================================================
    # Training Step
    # ==========================================================

    def training_step(self, batch, batch_idx: int):
        # 1. Unpack Data
        if len(batch) == 5:
            inputs, targets, _, _, _ = batch
        elif len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]

        X_0 = targets
        X_0_hat = inputs
        residual_term = X_0_hat - X_0  # Ground Truth Residual

        # 2. Physics-Guided Weights Calculation (Implicit Guidance)
        # Compute weights from the degraded input X_0_hat to guide the loss function.
        with torch.no_grad():
            _, hf_hat_tuple = self.dwt(X_0_hat)
            dyn_weights = self.compute_physics_weights(hf_hat_tuple)

        # 3. Standard Forward Diffusion
        # We use standard Gaussian noise here. The "physics" is enforced via the loss below.
        t = torch.randint(0, self.T_acc, (X_0.shape[0],), device=self.device)
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)

        noise = torch.randn_like(X_0)  # Standard Normal Distribution

        # Go from x_0 to x_t (Standard formulation)
        x_t = resfusion_x0_to_xt(X_0, alpha_hat, residual_term, noise)

        # 4. Prediction & Loss Calculation
        if self.mode == 'residual':
            # Predict Residual
            pred_residual_term = self.denoising_module(x=x_t, time=t, input_cond=X_0_hat)

            # --- Physics-Guided Wavelet Loss ---
            p_ll, p_hf_tuple = self.dwt(pred_residual_term)
            t_ll, t_hf_tuple = self.dwt(residual_term)

            # LL Subband Loss (L1)
            loss_ll = F.l1_loss(p_ll, t_ll)

            # HF Subband Loss (Weighted by Dynamic Weights from Eq. 7)
            loss_hf = 0
            # hf_tuple order: (LH, HL, HH)
            for i in range(3):
                # Apply weights derived from the forward physics model to the loss
                loss_hf += torch.mean(dyn_weights[i] * torch.abs(p_hf_tuple[i] - t_hf_tuple[i]))

            # --- Auxiliary Losses ---
            loss_grad = self.gradient_loss(pred_residual_term, residual_term)
            loss_freq = self.frequency_loss(pred_residual_term, residual_term)

            # --- Final Combined Loss ---
            loss = 1.0 * loss_ll + \
                   2.0 * loss_hf + \
                   1.0 * loss_grad + \
                   0.2 * loss_freq

        elif self.mode == 'epsilon':
            # Standard DDPM Loss (Optional fallback)
            alpha = self.alphas[t].reshape(-1, 1, 1, 1)
            beta = self.betas[t].reshape(-1, 1, 1, 1)

            resnoise = noise + (1 - torch.sqrt(alpha)) * torch.sqrt(1 - alpha_hat) / beta * residual_term
            pred_resnoise = self.denoising_module(x=x_t, time=t, input_cond=X_0_hat)

            if self.loss_type == 'L2':
                loss = F.mse_loss(input=pred_resnoise, target=resnoise)
            else:
                loss = F.smooth_l1_loss(input=pred_resnoise, target=resnoise)
        else:
            raise ValueError(f"Mode {self.mode} not supported.")

        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    # ==========================================================
    # Validation & Testing
    # ==========================================================

    def validation_step(self, batch, batch_idx: int):
        inputs, targets = batch[:2] if isinstance(batch, list) else batch
        X_0, X_0_hat = targets, inputs

        # Generate (Standard Inference)
        pred_x_0 = self.generate(X_0_hat)

        # Clamp & Rescale
        pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
        pred_x_0 = (pred_x_0 + 1) / 2

        # Visualization preparation
        vis_inputs = (inputs.clone().detach() + 1) / 2
        vis_targets = (targets.clone().detach() + 1) / 2
        vis_inputs = torch.clamp(vis_inputs, 0, 1)
        vis_targets = torch.clamp(vis_targets, 0, 1)

        # Metric Calculation
        self.val_PSNR(preds=pred_x_0, target=vis_targets)
        self.log('val_PSNR', self.val_PSNR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if batch_idx < 4:
            self.log_tb_images((vis_inputs, vis_targets, pred_x_0), batch_idx, self.current_epoch, save_all=False)

    def test_step(self, batch, batch_idx: int):
        inputs, targets = batch[:2] if isinstance(batch, list) else batch
        X_0, X_0_hat = targets, inputs

        pred_x_0 = self.generate(X_0_hat)
        pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
        pred_x_0 = (pred_x_0 + 1) / 2

        vis_inputs = (inputs.clone().detach() + 1) / 2
        vis_targets = (targets.clone().detach() + 1) / 2
        vis_inputs = torch.clamp(vis_inputs, 0, 1)
        vis_targets = torch.clamp(vis_targets, 0, 1)

        self.test_PSNR(preds=pred_x_0, target=vis_targets)
        self.log('test_PSNR', self.test_PSNR, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log_tb_images((vis_inputs, vis_targets, pred_x_0), batch_idx, self.current_epoch, save_all=True)

    # ==========================================================
    # Generation (Inference)
    # ==========================================================

    def generate(self, X_0_hat: torch.Tensor,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Standard 5-step Inference.
        Note: The network parameters have internalized the physics-guided weights during training,
        so explicit weight computation is not required here.
        """
        if get_intermediate_steps:
            steps = []

        # 1. Initialize with Standard Gaussian Noise
        alpha_hat = self.alphas_hat[self.T_acc - 1]
        noise = torch.randn_like(X_0_hat)  # Standard N(0, I)
        X_noise = ddpm_x0_to_xt(X_0_hat, alpha_hat, noise)

        # 2. Reverse Loop
        for t in range(self.T_acc - 1, -1, -1):
            alpha_t = self.alphas[t]
            alpha_hat_t = self.alphas_hat[t]
            beta_hat_t = self.betas_hat[t]
            alpha_hat_t_minus_1 = self.alphas_hat_t_minus_1[t]

            if get_intermediate_steps:
                steps.append(X_noise)

            t_tensor = torch.LongTensor([t]).to(self.device).expand(X_noise.shape[0])

            # Fixed Variance
            sigma = torch.sqrt(beta_hat_t)
            if t < 3:
                sigma = sigma * 0.5  # Trick to reduce noise at the end of generation
            z = torch.randn_like(X_noise) if t > 0 else 0

            if self.mode == 'residual':
                # Predict Residual
                # Conceptually: pred = epsilon_theta(x, t | w^s)
                # The network implicitly applies the learned physics guidance.
                pred_residual_term = self.denoising_module(x=X_noise, time=t_tensor, input_cond=X_0_hat)

                # Implicit clamping for stability
                pred_x_0 = X_0_hat - pred_residual_term
                pred_x_0 = torch.clamp(pred_x_0, min=-1, max=1)
                pred_residual_term = X_0_hat - pred_x_0

                # Standard Posterior Update (Formula 44 in ResFusion)
                if t == 0:
                    X_noise = pred_x_0
                else:
                    X_noise = (((torch.sqrt(alpha_t) * (1 - alpha_hat_t_minus_1)) * (X_noise - pred_residual_term)
                                + (torch.sqrt(alpha_hat_t_minus_1)) * (1 - alpha_t) * (pred_x_0 - pred_residual_term))
                               / (1 - alpha_hat_t)
                               + pred_residual_term + sigma * z)

            elif self.mode == 'epsilon':
                beta_t = self.betas[t]
                pred_resnoise = self.denoising_module(x=X_noise, time=t_tensor, input_cond=X_0_hat)
                if t == 0: z = 0
                X_noise = 1 / (torch.sqrt(alpha_t)) * \
                          (X_noise - (beta_t / torch.sqrt(1 - alpha_hat_t)) * pred_resnoise) + sigma * z

        if get_intermediate_steps:
            steps.append(X_noise)
            return steps

        return X_noise

    # ==========================================================
    # Utilities
    # ==========================================================

    def log_tb_images(self, viz_batch, batch_idx, current_epoch, save_all=False) -> None:
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None: return

        for img_idx, (image, y_true, y_pred) in enumerate(zip(*viz_batch)):
            if save_all:
                tb_logger.add_image(f"Image/{batch_idx:04d}_{img_idx:04d}", image, 0)
                tb_logger.add_image(f"GroundTruth/{batch_idx:04d}_{img_idx:04d}", y_true, 0)
                tb_logger.add_image(f"Prediction/{batch_idx:04d}_{img_idx:04d}", y_pred, current_epoch)
            else:
                if img_idx < 8:
                    tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
                    tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
                    tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, current_epoch)

    def configure_optimizers(self):
        # Effective batch size calculation for LR scaling
        blr = self.hparams.blr
        devices = getattr(self.hparams, 'devices', 1)
        num_nodes = getattr(self.hparams, 'num_nodes', 1)
        eff_batch_size = self.hparams.batch_size * self.hparams.accum_iter * devices * num_nodes
        lr = blr * eff_batch_size / 256

        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                                   weight_decay=self.hparams.weight_decay, amsgrad=True)
        elif self.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                                    weight_decay=self.hparams.weight_decay, amsgrad=True)
        elif self.optimizer_type == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr,
                                  weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        if self.lr_scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epochs,
                                                             eta_min=self.hparams.min_lr)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.lr_scheduler_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20,
                                                             cooldown=20, min_lr=self.hparams.min_lr)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train_loss_epoch"}}
        else:
            raise ValueError(f"Unsupported scheduler type: {self.lr_scheduler_type}")