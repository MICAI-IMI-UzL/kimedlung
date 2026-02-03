import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pytorch_lightning as pl
import pandas as pd
import os
import json
from Nstaging_with_classifier.models.CNN_for_size_64 import CNN as CNN_64
from Nstaging_with_classifier.models.CNN_for_size_32 import CNN as CNN_32
from Nstaging_with_classifier.rulebased_nstaging import from_station_to_N, calc_sens_spec_binclass, calc_sens_spec_multiclass
from Nstaging_with_classifier.GIN_IPA.GIN_IPA import GIN_IPA


class Model(pl.LightningModule):

    def __init__(self, CONFIG):
        super().__init__()
        self.config = CONFIG
        if self.config['spatial_size'] == (32, 32, 32):
            CNN = CNN_32
        else:
            CNN = CNN_64
        num_channels = 1
        if self.config['mask_as_prior']:
            num_channels += 1
        if self.config['modality'] == 'PET':
            num_channels += 1
        self.model = CNN(self.config, in_channels=num_channels, n_feat=4)
        self.save_hyperparameters()
        self.GIN_IPA_transform = GIN_IPA()

    def loss(self, x, y, ln_idx, pt_suv):
        logits = self.model(x, ln_idx, pt_suv)
        loss_func = nn.BCEWithLogitsLoss()
        loss = loss_func(logits.squeeze(dim=1), y.float())
        return logits, loss

    def bal_acc(self, preds, y):
        acc = balanced_accuracy_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return acc

    def acc(self, preds, y):
        acc = accuracy_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return acc

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, cooldown=10, patience=10,
                                                                    min_lr=self.config['lr']/100)
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, "monitor": 'train_loss_lnlevel'}

    def training_step(self, batch, batch_idx):
        probs, preds, loss, acc, bal_acc, sens, spec = self._get_preds_loss_accuracy(batch)
        self.log('train_loss_lnlevel', loss.item())
        self.log('train_acc_lnlevel', acc)
        self.log('train_bal_acc_lnlevel', bal_acc)
        self.log('train_sensitivity', sens)
        self.log('train_specificity', spec)
        return {'preds': {'lnlevel_preds': preds}, 'loss': loss}

    def validation_step(self, batch, batch_idx):
        probs, preds, loss, acc, bal_acc, sens, spec = self._get_preds_loss_accuracy(batch)
        to_append = {'patient_id': batch['patient_id'], 'study': batch['study'], 'series': batch['series_names'], 'lymph_node': batch['lymph_node'],
                     'PT_Location': batch['PT_Location'], 'gt_nstage': list(batch['N'].long().cpu().numpy()),
                     'gt_lnlevel': list(batch['label'].cpu().numpy()), 'pred_lnlevel': list(preds.view(-1).long().cpu().numpy())}
        self.val_tab = pd.concat([self.val_tab, pd.DataFrame(to_append)])
        self.log('val_loss_lnlevel', loss.item())
        self.log('val_acc_lnlevel', acc)
        self.log('val_bal_acc_lnlevel', bal_acc)
        self.log('val_sensitivity', sens)
        self.log('val_specificity', spec)
        return {'preds': {'lnlevel_preds': preds}, 'loss': loss}

    def test_step(self, batch, batch_idx):
        probs, preds, loss, acc, bal_acc, sens, spec = self._get_preds_loss_accuracy(batch)
        to_append = {'patient_id': batch['patient_id'], 'study': batch['study'], 'series': batch['series_names'], 'lymph_node': batch['lymph_node'],
                     'PT_Location': batch['PT_Location'], 'gt_nstage': list(batch['N'].cpu().numpy()),
                     'gt_lnlevel': list(batch['label'].cpu().numpy()), 'pred_lnlevel': list(preds.view(-1).long().cpu().numpy()),
                     'prob_lnlevel': list(probs.view(-1).cpu().numpy())}
        self.test_tab = pd.concat([self.test_tab, pd.DataFrame(to_append)])

    def _get_preds_loss_accuracy(self, batch):
        x, y, ln, pt_suv = batch['img'], batch['label'], batch['ln_idx'], batch['pt_suv']
        if self.config['GIN_IPA'] and self.training:
            x = self.GIN_IPA_transform(x)
        logits, loss = self.loss(x, y, ln, pt_suv)
        probs = torch.sigmoid(logits)
        preds = torch.round(probs)
        bal_acc = self.bal_acc(preds, y)
        acc = self.acc(preds, y)
        sens, spec = calc_sens_spec_binclass(list(preds.squeeze().detach().cpu().numpy()), list(y.detach().cpu().numpy()))
        return probs, preds, loss, acc, bal_acc, sens, spec

    def on_test_start(self) -> None:
        self.test_tab = pd.DataFrame(columns=['patient_id', 'study', 'series', 'lymph_node', 'PT_Location', 'gt_lnlevel', 'gt_nstage', 'pred_lnlevel', 'prob_lnlevel'])

    def on_test_epoch_end(self) -> None:
        #lnlevel
        preds_lnlevel = list(self.test_tab['pred_lnlevel'].values)
        gt_lnlevel = list(self.test_tab['gt_lnlevel'].values)
        acc_lnlevel = accuracy_score(gt_lnlevel, preds_lnlevel)
        bal_acc_lnlevel = balanced_accuracy_score(gt_lnlevel, preds_lnlevel)
        sens_lnlevel, spec_lnlevel = calc_sens_spec_binclass(preds_lnlevel, gt_lnlevel)
        self.log('test_acc_lnlevel', acc_lnlevel)
        self.log('test_bal_acc_lnlevel', bal_acc_lnlevel)
        self.log('test_sensitivity_lnlevel', sens_lnlevel)
        self.log('test_specificity_lnlevel', spec_lnlevel)
        # nstage
        preds_nstage, gt_nstage = self.calc_rulebased_nstage('test')
        self.test_confm = {'pred_nstage': preds_nstage, 'gt_nstage': gt_nstage}
        acc_nstage = accuracy_score(gt_nstage, preds_nstage)
        bal_acc_nstage = balanced_accuracy_score(gt_nstage, preds_nstage)
        sens_nstage, spec_nstage = calc_sens_spec_multiclass(preds_nstage, gt_nstage)
        self.log('test_acc_nstage', acc_nstage)
        self.log('test_bal_acc_nstage', bal_acc_nstage)
        self.log('test_sensitivity_nstage', sens_nstage)
        self.log('test_specificity_nstage', spec_nstage)
        with open(os.path.join(self.trainer.checkpoint_callback.dirpath, f'results_fold={self.config["current_fold"]}.json'), 'w') as outfile:
            json.dump({'ckpt_path' : self.trainer.ckpt_path,
                       'test_acc_lnlevel': acc_lnlevel, 'test_bal_acc_lnlevel': bal_acc_lnlevel,
                       'test_sensitivity_lnlevel': sens_lnlevel, 'test_specificity_lnlevel': spec_lnlevel,
                       'test_acc_nstage': acc_lnlevel, 'test_bal_acc_nstage': bal_acc_lnlevel,
                       'test_sensitivity_nstage': sens_lnlevel, 'test_specificity_nstage': spec_lnlevel,
                       }, outfile)
        self.test_tab.to_csv(os.path.join(self.trainer.checkpoint_callback.dirpath, f'test_confm_fold={self.config["current_fold"]}.csv'), index=False)

    def on_validation_start(self) -> None:
        self.val_tab = pd.DataFrame(columns=['patient_id', 'study', 'series', 'lymph_node', 'PT_Location', 'gt_lnlevel', 'gt_nstage', 'pred_lnlevel'])

    def on_validation_epoch_end(self) -> None:
        preds_list, gt_list = self.calc_rulebased_nstage('val')
        self.val_confm = {'pred_nstage': preds_list, 'gt_nstage': gt_list}
        acc = accuracy_score(preds_list, gt_list)
        bal_acc = balanced_accuracy_score(preds_list, gt_list)
        self.log('val_acc_nstage', acc)
        self.log('val_bal_acc_nstage', bal_acc)

    def calc_rulebased_nstage(self, stage):
        if stage == 'val':
            tab = self.val_tab
        elif stage == 'test':
            tab = self.test_tab
        preds_list = []
        gt_list = []
        for patient_id in tab['patient_id'].drop_duplicates():
            for study in tab[tab['patient_id'] == patient_id]['study'].drop_duplicates():
                for series in tab[(tab['patient_id'] == patient_id) & (tab['study'] == study)]['series'].drop_duplicates():
                    row = tab[(tab['patient_id'] == patient_id) & (tab['study'] == study) & (tab['series'] == series)]
                    assert len(row) == len(self.config['lymphnode_names'])
                    if row['PT_Location'].drop_duplicates().values[0] == '-':
                        continue
                    pathological_ln = []
                    for ln in self.config['lymphnode_names']:
                        if len(row) == len(self.config['lymphnode_names']):
                            if row[row['lymph_node'] == ln]['pred_lnlevel'].values[0] == 1:
                                pathological_ln.append(ln)
                        else:
                            continue
                    pred_n = from_station_to_N(pathological_ln, row['PT_Location'].drop_duplicates().values[0])
                    preds_list.append(pred_n)
                    gt_list.append(row['gt_nstage'].drop_duplicates().values[0])
                    tab.loc[(tab['patient_id'] == patient_id) & (tab['study'] == study) & (tab['series'] == series), 'pred_nstage'] = [int(pred_n)] * len(row)
        tab['pred_nstage'] = tab['pred_nstage'].astype('Int64')
        return preds_list, gt_list
