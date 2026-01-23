""" Test the WGRD restoration module """
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer

from datamodule import RaindropDataModule, Rain100HDataModule
from model.denoising_module import WGRD_Unet, WGRD_WaveUnet
from model import GaussianResfusion_Restore
from variance_scheduler import LinearProScheduler, CosineProScheduler
import torch


def main(args):
    # Set matrix multiplication precision for performance/accuracy trade-off
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

    # 4. Load Model from Checkpoint
    # strict=False allows loading even if some keys (like loss params) are missing in the checkpoint
    print(f"Loading checkpoint from: {args.model_ckpt}")
    resfusion_restore_model = GaussianResfusion_Restore.load_from_checkpoint(
        checkpoint_path=args.model_ckpt,
        strict=False,
        denoising_module=denoising_model,
        variance_scheduler=variance_scheduler,
        mode=args.mode
    )

    # 5. Initialize Trainer for Testing
    trainer = Trainer(
        devices=1,
        num_nodes=1,
        logger=True,
        default_root_dir=args.log_dir,
        deterministic='warn',
        precision=args.precision,
        enable_model_summary=False,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    # 6. Run Test
    trainer.test(model=resfusion_restore_model, datamodule=data_module)


if __name__ == '__main__':
    parser = ArgumentParser('Test the WGRD module')

    # Accuracy control
    parser.add_argument('--set_float32_matmul_precision_high', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_high=False)
    parser.add_argument('--set_float32_matmul_precision_medium', action='store_true')
    parser.set_defaults(set_float32_matmul_precision_medium=False)

    # Basic Test Control
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', default=True, type=bool)
    parser.add_argument('--seed', default=2024, type=int)
    parser.add_argument('--precision', default='32', type=str)

    # Hyperparameters
    parser.add_argument('--n_channels', default=3, type=int)
    parser.add_argument('--noise_schedule', default='LinearPro', type=str)
    parser.add_argument('--T', default=12, type=int)

    # Denoising Model Hyperparameters
    parser.add_argument('--denoising_model', default='WGRD_WaveUnet', type=str)
    parser.add_argument('--mode', default='residual', type=str)

    # Checkpoint Path (Update this to your best model path)
    parser.add_argument('--model_ckpt',
                        default='resfusion_restore_train/lightning_logs/version_70/checkpoints/best-epoch=558-val_PSNR=32.551.ckpt',
                        type=str)

    # WGRD_Unet specific params
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--resnet_block_groups', default=16, type=int)

    # DiT / DDIM_Unet specific params
    parser.add_argument('--input_size', default=256, type=int)

    # Test Info
    parser.add_argument('--dataset', default='Rain100H', type=str)
    parser.add_argument('--data_dir', default='datasets/Rain100H', type=str)
    parser.add_argument('--log_dir', default='resfusion_restore_test', type=str)

    args = parser.parse_args()

    # Print Configuration
    print("\n" + "=" * 50)
    print("Starting Testing with Configuration:")
    print(f"  Checkpoint: {args.model_ckpt}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Batch Size: {args.batch_size}")
    print("=" * 50 + "\n")

    main(args)