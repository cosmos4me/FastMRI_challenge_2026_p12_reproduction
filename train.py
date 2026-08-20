import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_m02_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument(
        '--extra-data-path-train', type=Path, nargs='*', default=[],
        help='Additional training roots, each containing image/ and kspace/',
    )
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    parser.add_argument(
        '--extra-data-path-val', type=Path, nargs='*', default=[],
        help='Additional validation roots, each containing image/ and kspace/',
    )
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes; use 4 on RTX 3090')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Batches prefetched by each worker')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Pin CPU batches for asynchronous GPU copies')
    parser.add_argument('--allow-tf32', action='store_true',
                        help='Allow TensorFloat-32 CUDA operations')
    parser.add_argument('--cudnn-benchmark', action='store_true',
                        help='Autotune cuDNN kernels for observed MRI widths')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument(
        '--model-type', choices=('p12_stable_unified_promptmr_plus',),
        default='p12_stable_unified_promptmr_plus',
        help='Fixed P12 reconstruction architecture',
    )
    parser.add_argument('--optimizer', choices=('adam', 'adamw'), default='adam')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument(
        '--conditioning-lr-scale', type=float, default=1.0,
        help='Learning-rate multiplier for P12-stable routing parameters',
    )
    parser.add_argument(
        '--scheduler',
        choices=('none', 'warmup_cosine', 'smartify_step', 'warmup_step'), default='none',
        help='Learning-rate schedule stepped after every optimizer update',
    )
    parser.add_argument('--warmup-epochs', type=float, default=0.0)
    parser.add_argument('--min-lr', type=float, default=0.0)
    parser.add_argument(
        '--scheduler-step-epochs', type=int, nargs='+', default=[16, 27],
        help='Epoch boundaries for smartify_step learning-rate decay',
    )
    parser.add_argument(
        '--scheduler-gamma', type=float, default=0.3,
        help='Multiplicative decay used by smartify_step',
    )

    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument(
        '--loss-mode', choices=('legacy', 'challenge'), default='legacy',
        help='legacy whole-image SSIM or exact challenge-aligned training loss',
    )
    parser.add_argument(
        '--cross-acc-remask', action='store_true',
        help='Apply exact provided acc4/acc8 masks to full train k-space per volume',
    )
    parser.add_argument(
        '--all-anatomy-acc8', action='store_true',
        help=(
            'Remask every fully sampled train slice with an official acc8 '
            'mask for acc8-specialist training'
        ),
    )
    parser.add_argument(
        '--all-anatomy-acc4', action='store_true',
        help=(
            'Remask every fully sampled train slice with an official acc4 '
            'mask for acc4-specialist training'
        ),
    )
    parser.add_argument(
        '--acc8-mask-offset-augmentation', action='store_true',
        help=(
            'Cycle each training volume through all eight official acc8 '
            'equispaced mask offsets; requires P07M and --all-anatomy-acc8'
        ),
    )
    parser.add_argument(
        '--balanced-acc-offset-cycle', action='store_true',
        help=(
            'Use one shared model with a deterministic 16-epoch cycle: '
            '50/50 acc4/acc8 and complete coverage of all mask offsets'
        ),
    )
    parser.add_argument(
        '--mri-augment', action='store_true',
        help='Apply scheduled k-space-consistent translation/phase augmentation',
    )
    parser.add_argument(
        '--mri-augment-start-epoch', type=int, default=5,
        help='Keep MRI augmentation disabled through this 1-based epoch',
    )
    parser.add_argument('--mri-augment-ramp-epochs', type=int, default=5)
    parser.add_argument('--mri-augment-max-prob', type=float, default=0.5)
    parser.add_argument('--mri-augment-max-shift', type=int, default=4)
    parser.add_argument(
        '--mri-augment-coil-phase', action='store_true',
        help='Randomize each coil global phase when MRI augmentation is selected',
    )
    parser.add_argument(
        '--max-boxes', type=int, default=8,
        help='Maximum annotation boxes allowed in one slice',
    )
    parser.add_argument(
        '--bbox-loss-weight', type=float, default=0.0,
        help='Weight in [0, 1] for bbox SSIM loss; 0 reproduces the stock loss',
    )
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument(
        '--foreground-loss-weight', type=float, default=0.0,
        help='Weight in [0, 1] for challenge foreground-masked SSIM loss',
    )
    parser.add_argument(
        '--foreground-l1-weight', type=float, default=0.0,
        help=(
            'Extra weight for max-normalized foreground L1 on top of the '
            'exact challenge loss'
        ),
    )
    parser.add_argument(
        '--missing-kspace-loss-weight', type=float, default=0.0,
        help=(
            'Training-only normalized complex L1 weight on unmeasured '
            'fully sampled k-space lines (H16 only)'
        ),
    )
    parser.add_argument(
        '--bbox-sample-weight', type=float, default=1.0,
        help='Relative sampling weight for slices containing bbox annotations',
    )
    parser.add_argument('--init-checkpoint', type=Path, default=None,
                        help='Initialize model weights from a checkpoint and reset optimizer')
    parser.add_argument('--allow-partial-init', action='store_true',
                        help='Allow only new architecture parameters to be missing from init checkpoint')
    parser.add_argument('--resume-checkpoint', type=Path, default=None,
                        help='Resume model, optimizer, epoch, and best validation loss')
    parser.add_argument('--save-every-epoch', action='store_true',
                        help='Keep epoch_XXX.pt checkpoints for averaging/model selection')
    parser.add_argument(
        '--keep-last-checkpoints', type=int, default=0,
        help=(
            'When --save-every-epoch is enabled, retain only the newest N '
            'epoch_XXX.pt files; 0 keeps every epoch checkpoint'
        ),
    )
    parser.add_argument(
        '--validation-first-epochs', type=int, default=0,
        help=(
            'Run validation in the first N epochs. When both validation '
            'window arguments are 0, validate every epoch'
        ),
    )
    parser.add_argument(
        '--validation-last-epochs', type=int, default=0,
        help='Run validation in the last N epochs',
    )
    parser.add_argument(
        '--disable-validation', action='store_true',
        help=(
            'Skip validation entirely while preserving training, optimizer, '
            'scheduler, and checkpoint-saving behavior'
        ),
    )
    parser.add_argument('--grad-accum-steps', type=int, default=1,
                        help='Accumulate gradients over this many slices')
    parser.add_argument('--grad-clip-norm', type=float, default=0.0,
                        help='Clip gradient norm before updates; 0 disables clipping')
    parser.add_argument(
        '--aux-fi-weight', type=float, default=0.0,
        help='Weight for the FI decoder foreground full-SSIM auxiliary loss',
    )
    parser.add_argument(
        '--aux-prompt-weight', type=float, default=0.0,
        help='Weight for the Prompt cascade-2 foreground full-SSIM auxiliary loss',
    )
    parser.add_argument(
        '--p11-low-aux-weight', type=float, default=0.0,
        help='Training-only PE low-frequency loss weight at PromptMR cascade 4',
    )
    parser.add_argument(
        '--p11-mid-aux-weight', type=float, default=0.0,
        help='Training-only PE low+mid-frequency loss weight at cascade 8',
    )
    parser.add_argument(
        '--aux-decay-start-epoch', type=int, default=0,
        help='Keep auxiliary weights through this epoch, then linearly decay to zero',
    )
    parser.add_argument(
        '--h12-adaptation-ramp-epochs', type=int, default=0,
        help='Ramp H12-only acc8 adaptation over this many training epochs',
    )

    args = parser.parse_args()
    if args.bbox_loss_weight < 0 or args.foreground_loss_weight < 0:
        parser.error('loss weights must be non-negative')
    if args.foreground_l1_weight < 0:
        parser.error('--foreground-l1-weight must be non-negative')
    if args.missing_kspace_loss_weight < 0:
        parser.error('--missing-kspace-loss-weight must be non-negative')
    if args.bbox_loss_weight + args.foreground_loss_weight > 1.0:
        parser.error('bbox and foreground loss weights must sum to at most 1')
    if args.bbox_sample_weight < 1.0:
        parser.error('--bbox-sample-weight must be at least 1')
    if args.init_checkpoint is not None and args.resume_checkpoint is not None:
        parser.error('--init-checkpoint and --resume-checkpoint are mutually exclusive')
    if args.allow_partial_init and args.init_checkpoint is None:
        parser.error('--allow-partial-init requires --init-checkpoint')
    if args.allow_partial_init and args.resume_checkpoint is not None:
        parser.error('--allow-partial-init cannot be used with --resume-checkpoint')
    if args.weight_decay < 0:
        parser.error('--weight-decay must be non-negative')
    if not 0 < args.conditioning_lr_scale <= 1:
        parser.error('--conditioning-lr-scale must be in (0, 1]')
    if args.warmup_epochs < 0:
        parser.error('--warmup-epochs must be non-negative')
    if args.min_lr < 0 or args.min_lr > args.lr:
        parser.error('--min-lr must be between 0 and --lr')
    if any(epoch <= 0 for epoch in args.scheduler_step_epochs):
        parser.error('--scheduler-step-epochs must contain positive epochs')
    if args.scheduler_step_epochs != sorted(set(args.scheduler_step_epochs)):
        parser.error('--scheduler-step-epochs must be unique and increasing')
    if not 0 < args.scheduler_gamma <= 1:
        parser.error('--scheduler-gamma must be in (0, 1]')
    if args.max_boxes < 1:
        parser.error('--max-boxes must be positive')
    if args.mri_augment_start_epoch < 0:
        parser.error('--mri-augment-start-epoch must be non-negative')
    if args.mri_augment_ramp_epochs < 0:
        parser.error('--mri-augment-ramp-epochs must be non-negative')
    if not 0.0 <= args.mri_augment_max_prob <= 1.0:
        parser.error('--mri-augment-max-prob must be in [0, 1]')
    if args.mri_augment_max_shift < 0:
        parser.error('--mri-augment-max-shift must be non-negative')
    if args.mri_augment_coil_phase and not args.mri_augment:
        parser.error('--mri-augment-coil-phase requires --mri-augment')
    if args.acc8_mask_offset_augmentation and not args.all_anatomy_acc8:
        parser.error(
            '--acc8-mask-offset-augmentation requires --all-anatomy-acc8'
        )
    if (
        args.model_type in {
            'p12_unified_acquisition_promptmr_plus',
            'p12_stable_unified_promptmr_plus',
        }
        and not args.balanced_acc_offset_cycle
    ):
        parser.error('P12-full requires --balanced-acc-offset-cycle')
    if (
        args.balanced_acc_offset_cycle
        and args.model_type not in {
            'p12_unified_acquisition_promptmr_plus',
            'p12_stable_unified_promptmr_plus',
        }
    ):
        parser.error('--balanced-acc-offset-cycle requires P12-full')
    if (
        args.acc8_mask_offset_augmentation
        and args.model_type not in {
            'p07m_acc8_multimask_promptmr_plus',
            'p11m_acc8_sampling_aware_promptmr_plus',
        }
    ):
        parser.error('--acc8-mask-offset-augmentation requires P07M/P11')
    if args.grad_accum_steps < 1:
        parser.error('--grad-accum-steps must be positive')
    if args.keep_last_checkpoints < 0:
        parser.error('--keep-last-checkpoints must be non-negative')
    if args.validation_first_epochs < 0 or args.validation_last_epochs < 0:
        parser.error('validation epoch windows must be non-negative')
    if args.num_workers < 0:
        parser.error('--num-workers must be non-negative')
    if args.prefetch_factor < 1:
        parser.error('--prefetch-factor must be positive')
    if args.grad_clip_norm < 0:
        parser.error('--grad-clip-norm must be non-negative')
    if args.aux_fi_weight < 0 or args.aux_prompt_weight < 0:
        parser.error('H11 auxiliary loss weights must be non-negative')
    if args.p11_low_aux_weight < 0 or args.p11_mid_aux_weight < 0:
        parser.error('P11 auxiliary loss weights must be non-negative')
    uses_h11_aux = args.aux_fi_weight > 0 or args.aux_prompt_weight > 0
    uses_p11_aux = (
        args.p11_low_aux_weight > 0 or args.p11_mid_aux_weight > 0
    )
    if uses_h11_aux and uses_p11_aux:
        parser.error('H11 and P11 auxiliary losses are mutually exclusive')
    if uses_h11_aux and args.model_type != 'h11_aux_supervised_varnet':
        parser.error('H11 auxiliary losses require H11 model type')
    if (
        uses_p11_aux
        and args.model_type not in {
            'p11m_acc8_sampling_aware_promptmr_plus',
            'p12_unified_acquisition_promptmr_plus',
            'p12_stable_unified_promptmr_plus',
        }
    ):
        parser.error('P11 band losses require P11-soft/P12 model type')
    uses_aux = uses_h11_aux or uses_p11_aux
    if uses_aux and args.loss_mode != 'challenge':
        parser.error('auxiliary losses require --loss-mode challenge')
    if args.aux_decay_start_epoch < 0:
        parser.error('--aux-decay-start-epoch must be non-negative')
    if uses_aux and args.aux_decay_start_epoch >= args.num_epochs:
        parser.error('--aux-decay-start-epoch must be below --num-epochs')
    if args.h12_adaptation_ramp_epochs < 0:
        parser.error('--h12-adaptation-ramp-epochs must be non-negative')
    h12_family = {
        'h12_alias_aware_varnet',
        'h13_cross_acc_aug_varnet',
    }
    if args.h12_adaptation_ramp_epochs and args.model_type not in h12_family:
        parser.error('--h12-adaptation-ramp-epochs requires H12/H13')
    if args.loss_mode == 'challenge' and args.bbox_sample_weight != 1.0:
        parser.error('challenge loss must use uniform slice sampling')
    if args.foreground_l1_weight and args.loss_mode != 'challenge':
        parser.error('--foreground-l1-weight requires --loss-mode challenge')
    if (
        args.missing_kspace_loss_weight
        and args.model_type != 'h16_adjacent_kspace_varnet'
    ):
        parser.error('--missing-kspace-loss-weight requires H16')
    if args.model_type == 'h16_adjacent_kspace_varnet' and args.mri_augment:
        parser.error(
            'H16 neighbor alignment currently requires --mri-augment off'
        )
    if args.all_anatomy_acc8 and args.model_type not in {
        'p07m_acc8_multimask_promptmr_plus',
        'p11m_acc8_sampling_aware_promptmr_plus',
    }:
        parser.error('--all-anatomy-acc8 requires P07M/P11')
    if (
        args.all_anatomy_acc4
        and args.model_type not in {
            'p01m_acc4_promptmr_plus',
            'p02m_acc4_promptmr_plus',
        }
    ):
        parser.error('--all-anatomy-acc4 requires a P01M/P02M acc4 specialist')
    specialist_flags = int(args.all_anatomy_acc4) + int(args.all_anatomy_acc8)
    if specialist_flags > 1:
        parser.error('all-anatomy acc4 and acc8 remasking are mutually exclusive')
    if specialist_flags and args.cross_acc_remask:
        parser.error(
            'all-anatomy specialist remasking and --cross-acc-remask are mutually exclusive'
        )
    if args.balanced_acc_offset_cycle and (
        specialist_flags
        or args.cross_acc_remask
        or args.acc8_mask_offset_augmentation
    ):
        parser.error(
            '--balanced-acc-offset-cycle cannot use specialist/remasking flags'
        )
    return args

if __name__ == '__main__':
    args = parse()
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
