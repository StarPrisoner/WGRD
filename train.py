""" Train the WGRD restoration module """
import os
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from argparse import ArgumentParser
from pytorch_lightning import Trainer

from datamodule import RaindropDataModule, Rain100HDataModule
from model.denoising_module import WGRD_Unet, WGRD_WaveUnet
from model import GaussianResfusion_Restore
from variance_scheduler import LinearProScheduler, CosineProScheduler

# Set matrix multiplication precision
torch.set_float32_matmul_precision('medium')


def load_callbacks(args):
    """Initialize Pytorch Lightning callbacks."""
    callbacks = []

    # Model Checkpoint: Save best model based on Validation PSNR
    callbacks.append(plc.ModelCheckpoint(
        monitor='val_PSNR',
        filename='best-{epoch:02d}-{val_PSNR:.3f}',
        mode='max',
        save_last=True,
        save_on_train_epoch_end=True,
        every_n_epochs=args.check_val_every_n_epoch
    ))

    # Monitor Learning Rate
    callbacks.append(plc.LearningRateMonitor(logging_interval='epoch'))

    # Early Stopping
    if args.early_stopping:
        callbacks.append(plc.EarlyStopping(monitor='val_PSNR', mode='max', patience=50))

    return callbacks


def main(args):
    # Windows-specific configuration for distributed backend
    if os.name == 'nt':
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        args.strategy = 'auto'

    # Precision settings
    if args.set_float32_matmul_precision_high:
        torch.set_float32_matmul_precision('high')
    if args.set_float32_matmul_precision_medium:
        torch.set_float32_matmul_precision('medium')

    pl.seed_everything(args.seed, workers=True)

    # 1. Initialize Data Module
    if args.dataset == 'Raindrop':
        data_module = RaindropDataModule(root_dir=args.data_dir, batch_size=args.batch_size, pin_mem=args.pin_mem,
                                         num_workers=args.num_workers)
    elif args.dataset == 'Rain100H':
        data_module = Rain100HDataModule(root_dir=args.data_dir, batch_size=args.batch_size, pin_mem=args.pin_mem,
                                         num_workers=args.num_workers)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset}")

    # 2. Initialize Variance Scheduler
    if args.noise_schedule == 'LinearPro':
        variance_scheduler = LinearProScheduler(T=args.T)
    elif args.noise_schedule == 'CosinePro':
        variance_scheduler = CosineProScheduler(T=args.T)
    else:
        raise ValueError(f"Unsupported variance scheduler: {args.noise_schedule}")

    # 3. Initialize Denoising Backbone
    if args.denoising_model == 'WGRD_Unet':
        denoising_model = WGRD_Unet(
            dim=args.dim,
            out_dim=args.n_channels,
            channels=args.n_channels,
            input_condition=True,
            input_condition_channels=args.n_channels,
            resnet_block_groups=args.resnet_block_groups
        )
    elif args.denoising_model == 'WGRD_WaveUnet':
        denoising_model = WGRD_WaveUnet(
            dim=args.dim,
            out_dim=args.n_channels,
            channels=args.n_channels,
            input_condition=True,
            input_condition_channels=args.n_channels,
            resnet_block_groups=args.resnet_block_groups
        )
    else:
        raise ValueError(f"Unsupported denoising model: {args.denoising_model}")

    # 4. Initialize Main Lightning Module (WGRD)
    resfusion_restore_model = GaussianResfusion_Restore(
        denoising_module=denoising_model,
        variance_scheduler=variance_scheduler,
        **vars(args)
    )

    # 5. Load Pretrained Weights (if provided)
    if args.pretrained_path:
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        # Clean state_dict keys (remove 'model.' prefix if present from previous saving methods)
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key[6:] if key.startswith('model.') else key
            new_state_dict[new_key] = value

        missing, unexpected = resfusion_restore_model.load_state_dict(new_state_dict, strict=False)

        # Critical: Warn if core denoising parameters are missing
        if len(missing) > 0:
            print(f"[WARNING] Missing keys: {len(missing)} items. Check if denoising_module is loaded correctly.")

        # --- Fine-tuning Configuration ---
        # Enable gradients for all parameters for full fine-tuning
        print(">>> Full fine-tuning mode enabled. All parameters requires_grad=True.")
        for param in resfusion_restore_model.parameters():
            param.requires_grad = True

    # 6. Initialize Trainer
    trainer = Trainer(
        log_every_n_steps=1,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum_iter,
        default_root_dir=args.log_dir,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        gradient_clip_val=args.gradient_clip,
        precision=args.precision,
        logger=True,
        callbacks=load_callbacks(args),
        deterministic='warn',
        strategy='auto' if args.devices == 1 else 'ddp',  # Auto for single GPU, DDP for multi-GPU
        enable_model_summary=False
    )

    # 7. Start Training
    trainer.fit(model=resfusion_restore_model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Train the WGRD module')

    # Accuracy control
    parser.add_argument('--set_float32_matmul_precision_high', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_high=False)
    parser.add_argument('--set_float32_matmul_precision_medium', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_medium=False)

    # Basic Training Control
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations for effective batch size')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--gradient_clip', default=1, type=float)
    parser.add_argument('--precision', default='32', type=str)  # Options: '32', '16-mixed'
    parser.add_argument('--early_stopping', action='store_true')
    parser.set_defaults(early_stopping=False)

    # Pretrained Model Path (Leave empty for training from scratch)
    parser.add_argument('--pretrained_path',
                        default='',
                        type=str, help='Path to the pretrained model checkpoint for fine-tuning')

    # Hyperparameters
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--noise_schedule', default='LinearPro', type=str)
    parser.add_argument('--T', default=12, type=int)
    parser.add_argument('--loss_type', default='fuse', type=str)
    parser.add_argument('--optimizer_type', default='AdamW', type=str)
    parser.add_argument('--lr_scheduler_type', default='CosineAnnealingLR', type=str)

    # Denoising Model Architecture
    parser.add_argument('--denoising_model', default='WGRD_WaveUnet', type=str)
    parser.add_argument('--mode', default='residual', type=str)

    # WGRD_Unet specific params
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--resnet_block_groups', default=16, type=int)

    # DiT / DDIM_Unet specific params
    parser.add_argument('--input_size', default=256, type=int)

    # Optimizer parameters
    parser.add_argument('--blr', default=8.8e-5, type=float)  # Base Learning Rate
    parser.add_argument('--min_lr', default=1e-6, type=float)  # Min Learning Rate
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    # Dataset Info
    parser.add_argument('--dataset', default='Rain100H', type=str)
    parser.add_argument('--data_dir', default='datasets/Rain100H', type=str)
    parser.add_argument('--log_dir', default='resfusion_restore_train', type=str)

    # Distributed Training Parameters
    parser.add_argument('--accelerator', default="gpu", type=str, help='Type of accelerator (gpu/cpu)')
    parser.add_argument('--devices', default=1, type=int, help='Number of devices')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')

    args = parser.parse_args()

    # Print Configuration
    print("\n" + "=" * 50)
    print("Starting Training/Fine-Tuning with Configuration:")
    print(f"  Pretrained model: {args.pretrained_path if args.pretrained_path else 'None (Training from scratch)'}")
    print(f"  Mode: {args.mode}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Learning rate: {args.blr}")
    print(f"  Min learning rate: {args.min_lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Log directory: {args.log_dir}")
    print("=" * 50 + "\n")

    main(args)