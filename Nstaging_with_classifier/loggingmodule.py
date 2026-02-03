import pandas as pd
from pytorch_lightning.callbacks import Callback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from preprocessing.preprocess_utils import fill_ln_counts


def create_confusion_matrix(labels_gt: list, labels_pred: list, axis: list, title: str):
    """Displays a confusion matrix."""
    cm = confusion_matrix(y_true=labels_gt, y_pred=labels_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax = ax, colorbar=False)
    ax.set_title(title)
    ax.set_xlabel(axis[1])
    ax.set_ylabel(axis[0])
    plt.close(fig)
    return fig

def flatten_lists(list_of_lists: list):
    result = []
    for l in list_of_lists:
        result.extend(l)
    return result


class LogPredictionsCallback(Callback):

    def __init__(self, wandb_logger):
        super().__init__()
        self.wandb_logger = wandb_logger

    def plot_input(self, pl_module, batch, outputs):
        if pl_module.config['level'] == 'ln':
            preds_name = 'lnlevel_preds'
            pet_idx = 1
        elif pl_module.config['level'] == 'series':
            preds_name = 'nstage_preds'
            pet_idx = len(pl_module.config['lymphnode_names'])
        x, y = batch['img'], batch['label']
        captions = [f'Ground Truth: {y_i} - Prediction: {int(y_pred.cpu().item())}' for y_i, y_pred in
                    zip(y, outputs['preds'][preds_name])]
        if pl_module.config['modality'] == 'PET':
            list_images = []
            for i in range(x.shape[0]):
                fig, ax = plt.subplots()
                ax.imshow(x[i, 0, x.shape[2] // 2, :, :].cpu().numpy(), cmap='gray')
                ax.imshow(x[i, pet_idx, x.shape[2] // 2, :, :].cpu().numpy(), alpha=0.5, cmap=plt.hot())
                ax.axis('off')
                list_images.append(fig)
                plt.close(fig)
        else:
            list_images = [x[i, 0, x.shape[2] // 2, :, :].cpu().numpy() for i in range(x.shape[0])]
        return list_images, captions

    def plot_ln_found(self, tab: pd.DataFrame, lymphnode_names: list):
        TP = np.unique(tab[(tab['gt_lnlevel'] == 1) & (tab['pred_lnlevel'] == 1)]['lymph_node'], return_counts=True)
        TN = np.unique(tab[(tab['gt_lnlevel'] == 0) & (tab['pred_lnlevel'] == 0)]['lymph_node'], return_counts=True)
        FP = np.unique(tab[(tab['gt_lnlevel'] == 0) & (tab['pred_lnlevel'] == 1)]['lymph_node'], return_counts=True)
        FN = np.unique(tab[(tab['gt_lnlevel'] == 1) & (tab['pred_lnlevel'] == 0)]['lymph_node'], return_counts=True)
        TP_fill = fill_ln_counts(lymphnode_names, TP[0].tolist(), TP[1].tolist())
        TN_fill = fill_ln_counts(lymphnode_names, TN[0].tolist(), TN[1].tolist())
        FP_fill = fill_ln_counts(lymphnode_names, FP[0].tolist(), FP[1].tolist())
        FN_fill = fill_ln_counts(lymphnode_names, FN[0].tolist(), FN[1].tolist())
        fig, ax = plt.subplots()
        ax.bar(lymphnode_names, FP_fill, color='blue')
        ax.bar(lymphnode_names, FN_fill, bottom=FP_fill, color='green')
        ax.bar(lymphnode_names, TP_fill, bottom=[i + j for i, j in zip(FP_fill, FN_fill)], color='red')
        ax.bar(lymphnode_names, TN_fill, bottom=[i + j + k for i, j, k in zip(FP_fill, FN_fill, TP_fill)], color='orange')
        ax.set_xlabel('Lymph node stations')
        ax.set_ylabel('# lymph nodes')
        ax.legend(['FP', 'FN', 'TP', 'TN'])
        plt.close(fig)
        return fig

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx == 0:
            list_images, captions = self.plot_input(pl_module, batch, outputs)
            self.wandb_logger.log_image(key='val_image', images=list_images, caption=captions)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            list_images, captions = self.plot_input(pl_module, batch, outputs)
            self.wandb_logger.log_image(key='train_image', images=list_images, caption=captions)

    def on_validation_end(self, trainer, pl_module) -> None:
        if pl_module.config['level'] == 'ln':
            disp = create_confusion_matrix(list(pl_module.val_tab['gt_lnlevel'].values), list(pl_module.val_tab['pred_lnlevel'].values),
                                    ['true label', 'predicted label'], 'lnlevel')
            self.wandb_logger.log_image(key='val_lnlevel_confm', images=[disp])
            disp = create_confusion_matrix(pl_module.val_confm['gt_nstage'], pl_module.val_confm['pred_nstage'],
                                    ['true label', 'predicted label'], 'nstage')
            self.wandb_logger.log_image(key='val_nstage_confm', images=[disp])
            fig = self.plot_ln_found(pl_module.val_tab, pl_module.config['lymphnode_names'])
            self.wandb_logger.log_image(key='val_ln_found', images=[fig])
        elif pl_module.config['level'] == 'series':
            disp = create_confusion_matrix(list(pl_module.val_tab['gt_nstage'].values), list(pl_module.val_tab['pred_nstage'].values),
                                    ['true label', 'predicted label'], 'nstage')
            self.wandb_logger.log_image(key='val_nstage_confm', images=[disp])
            predictions = []
            gts = []
            for prediction, gt in zip(flatten_lists(pl_module.val_tab['pred_lnlevel'].values), flatten_lists(pl_module.val_tab['gt_lnlevel'].values)):
                if gt != 99:
                    predictions.append(prediction)
                    gts.append(gt)
            if len(predictions) > 0:
                disp = create_confusion_matrix(gts, predictions, ['true label', 'predicted label'], 'lnlevel')
                self.wandb_logger.log_image(key='val_lnlevel_confm', images=[disp])
            predictions = []
            gts = []
            for prediction, gt in zip(flatten_lists(pl_module.val_tab['pred_pseudo'].values), flatten_lists(pl_module.val_tab['gt_pseudo'].values)):
                if gt != 99:
                    predictions.append(prediction)
                    gts.append(gt)
            disp = create_confusion_matrix(gts, predictions, ['true label', 'predicted label'], 'pseudo')
            self.wandb_logger.log_image(key='val_pseudo_confm', images=[disp])

    def on_test_end(self, trainer, pl_module) -> None:
        if pl_module.config['level'] == 'ln':
            disp = create_confusion_matrix(list(pl_module.test_tab['gt_lnlevel'].values), list(pl_module.test_tab['pred_lnlevel'].values),
                                    ['true label', 'predicted label'], 'lnlevel')
            self.wandb_logger.log_image(key='test_lnlevel_confm', images=[disp])
            disp = create_confusion_matrix(pl_module.test_confm['gt_nstage'], pl_module.test_confm['pred_nstage'],
                                    ['true label', 'predicted label'], 'nstage')
            self.wandb_logger.log_image(key='test_nstage_confm', images=[disp])
            fig = self.plot_ln_found(pl_module.test_tab, pl_module.config['lymphnode_names'])
            self.wandb_logger.log_image(key='test_ln_found', images=[fig])
        elif pl_module.config['level'] == 'series':
            disp = create_confusion_matrix(list(pl_module.test_tab['gt_nstage'].values), list(pl_module.test_tab['pred_nstage'].values),
                                    ['true label', 'predicted label'], 'nstage')
            self.wandb_logger.log_image(key='test_nstage_confm', images=[disp])
            predictions = []
            gts = []
            for prediction, gt in zip(flatten_lists(pl_module.test_tab['pred_lnlevel'].values), flatten_lists(pl_module.test_tab['gt_lnlevel'].values)):
                if gt != 99:
                    predictions.append(prediction)
                    gts.append(gt)
            if len(predictions) > 0:
                disp = create_confusion_matrix(gts, predictions, ['true label', 'predicted label'], 'lnlevel')
                self.wandb_logger.log_image(key='test_lnlevel_confm', images=[disp])
            predictions = []
            gts = []
            for prediction, gt in zip(flatten_lists(pl_module.test_tab['pred_pseudo'].values), flatten_lists(pl_module.test_tab['gt_pseudo'].values)):
                if gt != 99:
                    predictions.append(prediction)
                    gts.append(gt)
            disp = create_confusion_matrix(gts, predictions, ['true label', 'predicted label'], 'pseudo')
            self.wandb_logger.log_image(key='test_pseudo_confm', images=[disp])
