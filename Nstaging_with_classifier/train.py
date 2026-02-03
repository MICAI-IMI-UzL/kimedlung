import os
from pathlib import Path
import wandb
import argparse
import json
from datetime import datetime
from monai.transforms import (
    Compose,
    RandGaussianNoised,
    RandAffined,
    Rand3DElasticd,
    ScaleIntensityd,
    Spacingd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ResampleToMatchd,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    CenterSpatialCropd,
)
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from Nstaging_with_classifier.datamodule import KIMedLungDataModule
from Nstaging_with_classifier.custom_transforms import ConcatChannels, ReadVOI, ReadSeriesVOIs, EncodeTab, SamplingLymphNodes
from Nstaging_with_classifier.trainer_lnlevel import Model as Model_lnlevel
from Nstaging_with_classifier.trainer_serieslevel import Model as Model_serieslevel
from Nstaging_with_classifier.loggingmodule import LogPredictionsCallback
from Nstaging_with_classifier.show_results import extract_text
import glob
import torch

def find_ckpt(ckpt_path:str):
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_path, f'*fold={fold}*.ckpt')))
    max_acc = 0
    max_epoch = 0
    for file in ckpt_files:
        val_acc = float(extract_text(Path(file).name, 'val_bal_acc_lnlevel=', '.ckpt'))
        epoch = int(extract_text(Path(file).name, 'epoch=', '_val'))
        if val_acc >= max_acc:
            max_acc = val_acc
            max_epoch = epoch
    ckpt = sorted(glob.glob(os.path.join(ckpt_path, f'*fold={fold}_epoch={max_epoch:02d}_val_bal_acc_lnlevel={max_acc:.2f}.ckpt')))[0]
    return ckpt

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--inference', type=boolean_string, default=False)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--modality', type=str, default='CT')
    parser.add_argument('--label_column', type=str, default='pN')
    parser.add_argument('--level', type=str, default='ln')
    parser.add_argument('--dataset', nargs='+', default=['UKSH']),
    parser.add_argument('--patch_sampling', type=str, default='weight_map'),
    parser.add_argument('--encode_ln_station', type=boolean_string, default=False),
    parser.add_argument('--mask_as_prior', type=boolean_string, default=False),
    parser.add_argument('--encode_pt_suv', type=boolean_string, default=False),
    parser.add_argument('--freeze_encoder', type=boolean_string, default=False),
    parser.add_argument('--share_weights', type=boolean_string, default=True)
    parser.add_argument('--GIN_IPA', type=boolean_string, default=False),
    parser.add_argument('--pseudo_loss', type=boolean_string, default=True),
    parser.add_argument('--ckpt', type=str, default=''),
    args = parser.parse_args()

    if args.level == 'ln':
        batch_size = 96
    elif args.level == 'series':
        batch_size = 16
    else:
        raise Exception('Unknown level.')

    CONFIG = dict(
        imagepath='/mnt/remote/MILLymphnodeSeg/subjects',
        voipath = '/mnt/remote/VOIs',
        resultspath = '/workspace/KIMedLung/results',
        level = args.level,
        modality=args.modality,
        label_column=args.label_column,
        seed=123,
        num_classes=1,
        lr=5e-4,
        num_epochs=150,
        batch_size=batch_size,
        lymphnode_names=['1L', '1R', '2L', '2R', '3A', '3P',
                         '4L', '4R', '5', '6', '7', '8', '9',
                         '10L', '10R', '11L', '11R'],
        datasets=args.dataset,
        spatial_size= (64, 64, 64),
        spatial_size_extract= (75, 75, 75),
        sample_size=10,
        patch_sampling=args.patch_sampling,
        encode_lnstation=args.encode_ln_station,
        mask_as_prior=args.mask_as_prior,
        encode_pt_suv=args.encode_pt_suv,
        freeze_encoder=args.freeze_encoder,
        pseudo_loss=True,
        lambdas=[1, 0.2, 0.02],
        fold=[0, 1, 2, 3, 4],
        GIN_IPA=args.GIN_IPA,
        share_weights=args.share_weights
    )
    if CONFIG['spatial_size'] == (32, 32, 32):
        CONFIG['ln_name'] = f'CV_sampl={args.patch_sampling}_size=32_enclns={args.encode_ln_station}_maskprior={args.mask_as_prior}_encptsuv={args.encode_pt_suv}_GINIPA={args.GIN_IPA}_newnet'
        CONFIG['series_name']= f'CV_size=32_freezeencoder={args.freeze_encoder}_shareweights={args.share_weights}_pseudo={args.pseudo_loss}_newnet'
    else:
        CONFIG['ln_name'] = f'CV_sampl={args.patch_sampling}_size=64_enclns={args.encode_ln_station}_maskprior={args.mask_as_prior}_encptsuv={args.encode_pt_suv}_GINIPA={args.GIN_IPA}_newnet'
        CONFIG['series_name']= f'CV_size=64_freezeencoder={args.freeze_encoder}_shareweights={args.share_weights}_pseudo={args.pseudo_loss}_newnet'

    augmentation_transforms = [
        RandGaussianNoised(keys=['img'], prob=0.5, mean=0, std=0.1, allow_missing_keys=True),
        RandAffined(keys=['img', 'pet', 'mask'], prob=0.5, rotate_range=[0.2, 0.2, 0.2], shear_range=[0.05, 0.05, 0.05],
                    padding_mode='zeros', mode=('bilinear', 'bilinear', 'nearest'),
                    allow_missing_keys=True),
        Rand3DElasticd(keys=['img', 'pet', 'mask'], prob=0.5, sigma_range=(2, 4), magnitude_range=(1, 4),
                       padding_mode='zeros', mode=('bilinear', 'bilinear', 'nearest'),
                       allow_missing_keys=True),
        ScaleIntensityd(keys=['img'], allow_missing_keys=True),
    ]

    after_aug_transforms = [
        ConcatChannels(config=CONFIG),
        EncodeTab(config=CONFIG)
    ]

    if CONFIG['patch_sampling'] == 'mask':
        before_aug_transforms_lnlevel = [
            ReadVOI(config=CONFIG),
            Spacingd(keys=['img'], pixdim=(1, 1, 1), mode='bilinear', allow_missing_keys=True),
            ResampleToMatchd(keys=['pet'], key_dst='img', mode='bilinear', allow_missing_keys=True),
            ScaleIntensityRanged(keys=['img'], a_min=-200, a_max=400, b_min=0, b_max=1, clip=True,
                                 allow_missing_keys=True),
            ScaleIntensityRanged(keys=['pet'], a_min=0, a_max=4, b_min=0, b_max=1, allow_missing_keys=True),
            ResizeWithPadOrCropd(keys=['img', 'pet'], spatial_size=CONFIG['spatial_size'], allow_missing_keys=True)
        ]
        if CONFIG['level'] == 'ln':
            before_aug_transforms = before_aug_transforms_lnlevel
        elif CONFIG['level'] == 'series':
            before_aug_transforms = [
                ReadSeriesVOIs(config=CONFIG, lnlevel_transforms=before_aug_transforms_lnlevel)
            ]
        else:
            raise Exception(f'Unknown level: {CONFIG["level"]}.')
    elif CONFIG['patch_sampling'] == 'weight_map':
        CONFIG['save_patches_transforms'] = Compose([
            LoadImaged(keys=['img', 'pet', 'mask'], allow_missing_keys=True),
            EnsureChannelFirstd(keys=['img', 'pet', 'mask'], allow_missing_keys=True),
            Orientationd(keys=['img', 'pet', 'mask'], axcodes='RPI', allow_missing_keys=True),
            Spacingd(keys=['img'], pixdim=(1, 1, 1), mode=('bilinear'), allow_missing_keys=True),
            ResampleToMatchd(keys=['pet', 'mask'], key_dst='img', mode=('bilinear', 'nearest'),
                             allow_missing_keys=True),
            ScaleIntensityRanged(keys=['img'], a_min=-200, a_max=400, b_min=0, b_max=1, clip=True,
                                 allow_missing_keys=True),
            ScaleIntensityRanged(keys=['pet'], a_min=0, a_max=4, b_min=0, b_max=1, allow_missing_keys=True)
        ])
        before_aug_transforms = [
            SamplingLymphNodes(config=CONFIG)
        ]
        after_aug_transforms.append(CenterSpatialCropd(keys=['img'], roi_size=CONFIG['spatial_size']))
    else:
        raise Exception(f'Unknown patch sampling: {CONFIG["patch_sampling"]}.')

    CONFIG['train_transforms'] = Compose(before_aug_transforms + augmentation_transforms + after_aug_transforms)
    CONFIG['test_transforms'] = Compose(before_aug_transforms + after_aug_transforms)

    # seed everything
    pl.seed_everything(CONFIG['seed'])

    # os.environ['WANDB_MODE'] = 'disabled'
    # os.environ['WANDB_CONSOLE'] = 'off'

    for fold in CONFIG['fold']:
        if CONFIG['level'] == 'ln':
            monitor = 'val_loss_lnlevel'
            run_name = f'train_{CONFIG["level"]}level_mod={CONFIG["modality"]}_label={CONFIG["label_column"]}_fold={str(fold)}'
            task = CONFIG['ln_name']
        elif CONFIG['level'] == 'series':
            monitor = 'val_loss_nstage'
            run_name = f'train_{CONFIG["level"]}level_mod={CONFIG["modality"]}_label={CONFIG["label_column"]}_fold={str(fold)}'
            run_name_lnlevel = run_name.replace('serieslevel', 'lnlevel')
            task = CONFIG['series_name']
        else:
            raise Exception(f'Unknown level: {CONFIG["level"]}.')
        # setup callbacks
        wandb_logger = WandbLogger(project='KIMedLung', name=run_name, config=CONFIG)
        log_predictions_callback = LogPredictionsCallback(wandb_logger)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stop_callback = EarlyStopping(monitor=monitor, min_delta=0.000, patience=30, verbose=False, mode="min")
        checkpoint_dir = f'./checkpoints/{"_".join(run_name.split("_")[:-1])}/{task}'
        Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
        timestamp = '{:%d%m%Y_%H%M%S}'.format(datetime.now())
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                                              filename=timestamp + '_' + run_name + '_{epoch:02d}_{val_bal_acc_lnlevel:.2f}',
                                              monitor=monitor, mode='min',
                                              save_top_k=3)

        # setup data
        data = KIMedLungDataModule(CONFIG, fold)

        # setup model
        if CONFIG['level'] == 'ln':
            Model = Model_lnlevel
        elif CONFIG['level'] == 'series':
            Model = Model_serieslevel
        else:
            raise Exception('Unknown level.')
        model = Model(CONFIG)

        # fit the model
        trainer = pl.Trainer(
            logger=wandb_logger,  # W&B integration
            callbacks=[log_predictions_callback, lr_monitor, checkpoint_callback, early_stop_callback],
            log_every_n_steps=5,  # set the logging frequency
            accelerator='gpu',
            devices=[args.gpu],
            max_epochs=CONFIG['num_epochs'],  # number of epochs
            num_sanity_val_steps=0
            # deterministic=True      # keep it deterministic
        )
        if args.inference:
            data.prepare_data()
            data.setup()
            ckpt_path = find_ckpt(checkpoint_dir)
            # if os.path.exists(os.path.join(checkpoint_dir, 'results.json')):
            #     with open(os.path.join(checkpoint_dir, 'results.json'), 'r') as outfile:
            #         result_read = json.load(outfile)
            #     if 'ckpt' in result_read[str(fold)][0].keys():
            #         ckpt_path = result_read[str(fold)][0]['ckpt']
            loaded_model = Model.load_from_checkpoint(
                ckpt_path,
                CONFIG = CONFIG)
            trainer.test(loaded_model, data.test_dataloader())
        else:
            if CONFIG['level'] == 'ln':
                trainer.fit(model, data)
            elif CONFIG['level'] == 'series':
                modality_name = "_".join(run_name_lnlevel.split("_")[:-1])
                if modality_name == 'train_lnlevel_mod=PET_label=pN':
                    task_name = 'CV_sampl=weight_map_size=64_enclns=True_maskprior=False_encptsuv=False_GINIPA=True_newnet'
                elif modality_name == 'train_lnlevel_mod=PET_label=cN':
                    task_name = 'CV_sampl=weight_map_size=64_enclns=True_maskprior=False_encptsuv=False_GINIPA=False_newnet'
                elif modality_name == 'train_lnlevel_mod=CT_label=pN':
                    task_name = 'CV_sampl=weight_map_size=64_enclns=True_maskprior=True_encptsuv=False_GINIPA=False_newnet'
                else:
                    raise Exception
                with open(f'./checkpoints/{modality_name}/{task_name}/results_fold={fold}.json', 'r') as openfile:
                    results = json.load(openfile)
                    ckpt_path = results['ckpt_path']
                #ckpt_path = find_ckpt(f'./checkpoints/{"_".join(run_name_lnlevel.split("_")[:-1])}/{CONFIG["ln_name"]}')
                print(f'Loading checkpoint from: {ckpt_path}')
                checkpoint = torch.load(ckpt_path)
                for key in list(checkpoint['state_dict'].keys()):
                    checkpoint['state_dict'][key.replace('model.', '')] = checkpoint['state_dict'].pop(key)
                if CONFIG['share_weights']:
                    model.model.enc.load_state_dict(checkpoint['state_dict'])
                else:
                    for enc in range(len(CONFIG['lymphnode_names'])):
                        model.model.enc_list[enc].load_state_dict(checkpoint['state_dict'])
                trainer.fit(model, data)
            else:
                raise Exception('Unknown level.')
            # evaluate the model on a test set
            trainer.test(datamodule=data, ckpt_path='best')  # uses last-saved model
        wandb.finish()
