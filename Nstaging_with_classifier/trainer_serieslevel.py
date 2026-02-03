import pandas as pd
import torch
import os
from torch import nn
import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import pytorch_lightning as pl
from Nstaging_with_classifier.models.multi_encoder_model import MultiEncoder
from Nstaging_with_classifier.rulebased_nstaging import calc_sens_spec_binclass, calc_sens_spec_multiclass
from Nstaging_with_classifier.loggingmodule import flatten_lists

mapping_TL = {3: ['1R', '2R', '4R', '10R', '11R'],
              2: ['1L', '2L', '3A', '3P', '4L', '5', '6', '7', '8', '9'],
              1: ['10L', '11L'],
              }
mapping_TR = {3: ['1L', '2L', '4L', '10L', '11L'],
              2: ['1R', '2R', '3A', '3P', '4R', '5', '6', '7', '8', '9'],
              1: ['10R', '11R'],
              }
mapping_gt = {3: torch.Tensor([1, 99, 99]),
              2: torch.Tensor([0, 1, 99]),
              1: torch.Tensor([0, 0, 1]),
              0: torch.Tensor([0, 0, 0])
              }

class Model(pl.LightningModule):

    def __init__(self, CONFIG):
        super().__init__()
        self.config = CONFIG
        self.model = MultiEncoder(self.config)
        if self.config['freeze_encoder']:
            for param in self.model.enc.parameters():
                param.requires_grad = False
        self.save_hyperparameters()

    def mc_loss(self, logits, y):  # multi-class
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        loss = ce_loss(logits, y) + mse_loss(preds.float(), y.float())
        return preds, loss

    def bc_loss(self, logits, y): # binary
        loss_func = nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_func(logits, y)
        if len(loss[y!=99]) > 0:
            loss = loss[y!=99].mean()
        else:
            loss = torch.Tensor([0]).to(self.device)
        preds = torch.round(torch.sigmoid(logits))
        return preds, loss

    def generate_pseudo_labels(self, logits_lnlevel, y_nstage, pt_loc):
        probs = torch.sigmoid(logits_lnlevel).view(y_nstage.shape[0], -1)
        # print(f'Min:{logits_lnlevel.min()}, Max:{logits_lnlevel.max()}')
        # print(f'Min:{probs.min()}, Max:{probs.max()}')
        index_dict = {idx: ln for idx, ln in enumerate(self.config['lymphnode_names'])}
        pseudo_prob_list = []
        pseudo_gt_list = []
        for instance in range(y_nstage.shape[0]):
            nstage = y_nstage[instance]
            side = pt_loc[instance]
            if side == 0:
                mapping = mapping_TR
            else:
                mapping = mapping_TL
            pseudo_prob = []
            for key in mapping.keys():
                idx_patho_ln = [True if index_dict[index] in mapping[key] else False for index in list(range(len(self.config['lymphnode_names'])))]
                pseudo_prob.append(probs[instance, idx_patho_ln].max().unsqueeze(dim=0))
            pseudo_gt = mapping_gt[nstage.item()].to(self.device)
            pseudo_prob_list.append(torch.concat(pseudo_prob, dim=0))
            pseudo_gt_list.append(pseudo_gt)
        pseudo_prob_concat = torch.concat(pseudo_prob_list, dim=0)
        pseudo_gt_concat = torch.concat(pseudo_gt_list, dim=0)
        return pseudo_prob_concat, pseudo_gt_concat

    def pseudo_loss(self, pseudo_prob, pseudo_gt):
        mask = pseudo_gt[pseudo_gt != 99]
        differences = abs(pseudo_gt[pseudo_gt != 99] - pseudo_prob[pseudo_gt != 99])
        values, counts = torch.unique(mask, return_counts=True)
        weights = counts.sum() / (counts * len(counts))
        loss = 0
        for value, weight in zip(values, weights):
            loss += differences[mask == value].sum() * weight
        return loss

    def loss(self, x, y_lnlevel, y_nstage, pt_loc, pt_suv):
        logits_lnlevel, logits_nstage = self.model(x, pt_loc, pt_suv)
        preds_lnlevel, loss_lnlevel = self.bc_loss(logits_lnlevel.squeeze(0), y_lnlevel)
        preds_nstage, loss_nstage = self.mc_loss(logits_nstage, y_nstage)
        loss_dict = {'lnlevel': loss_lnlevel, 'nstage': loss_nstage}
        preds_dict = {'lnlevel_logits': logits_lnlevel, 'nstage_logits': logits_nstage,
                      'lnlevel_probs': torch.sigmoid(logits_lnlevel), 'nstage_probs': torch.softmax(logits_nstage, dim=1),
                      'lnlevel_preds': preds_lnlevel, 'nstage_preds': preds_nstage}
        return preds_dict, loss_dict

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
        # Debug pseudo loss only
        # return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, "monitor": 'train_loss_pseudo'}
        return {'optimizer': self.optimizer, 'lr_scheduler': self.scheduler, "monitor": 'train_loss_nstage'}

    def training_step(self, batch, batch_idx):
        preds_dict, loss_dict, acc_dict, _ = self._get_preds_loss_accuracy(batch)
        self.log('train_loss_lnlevel', self.config['lambdas'][0] * loss_dict['lnlevel'].item())
        self.log('train_loss_nstage', self.config['lambdas'][1] * loss_dict['nstage'].item())
        if self.config['pseudo_loss']:
            self.log('train_loss_pseudo', self.config['lambdas'][2] * loss_dict['pseudo'].item())
            self.log('train_acc_pseudo', acc_dict['acc_pseudo'])
            self.log('train_bal_acc_pseudo', acc_dict['bal_acc_pseudo'])
        if acc_dict['acc_lnlevel'] != 0:
            self.log('train_acc_lnlevel', acc_dict['acc_lnlevel'])
        self.log('train_acc_nstage', acc_dict['acc_nstage'])
        if acc_dict['bal_acc_lnlevel'] != 0:
            self.log('train_bal_acc_lnlevel', acc_dict['bal_acc_lnlevel'])
        self.log('train_bal_acc_nstage', acc_dict['bal_acc_nstage'])
        if self.config['pseudo_loss']:
            loss = self.config['lambdas'][0]*loss_dict['lnlevel'] + self.config['lambdas'][1]*loss_dict['nstage'] + \
                   self.config['lambdas'][2]*loss_dict['pseudo']
        else:
            loss = self.config['lambdas'][0]*loss_dict['lnlevel'] + self.config['lambdas'][1]*loss_dict['nstage']
        # Debug pseudo loss only
        # loss = self.config['lambdas'][2]*loss_dict['pseudo']
        return {'preds': preds_dict, 'loss': loss}

    def validation_step(self, batch, batch_idx):
        preds_dict, loss_dict, acc_dict, pseudo_gt = self._get_preds_loss_accuracy(batch)
        batch_size = preds_dict['nstage_logits'].shape[0]
        to_append = {'patient_id': batch['patient_id'], 'study': batch['study'], 'series': batch['series_names'],
                     'lymph_node': [self.config['lymphnode_names'] for i in range(batch_size)],
                     'pred_lnlevel': preds_dict['lnlevel_preds'].view(batch_size, len(self.config['lymphnode_names'])).squeeze().long().detach().cpu().tolist(),
                     'prob_lnlevel': preds_dict['lnlevel_probs'].view(batch_size, len(self.config['lymphnode_names'])).squeeze().long().detach().cpu().tolist(),
                     'pred_nstage': preds_dict['nstage_preds'].squeeze().long().detach().cpu().tolist(),
                     'gt_nstage': batch['label'].squeeze().long().detach().cpu().tolist(),
                     'gt_lnlevel': batch['label_lnlevel'].squeeze().long().detach().cpu().tolist(),
                     }
        self.log('val_loss_lnlevel', self.config['lambdas'][0] * loss_dict['lnlevel'].item())
        self.log('val_loss_nstage', self.config['lambdas'][1] * loss_dict['nstage'].item())
        if acc_dict['acc_lnlevel'] != 0:
            self.log('val_acc_lnlevel', acc_dict['acc_lnlevel'])
        self.log('val_acc_nstage', acc_dict['acc_nstage'])
        if acc_dict['bal_acc_lnlevel'] != 0:
            self.log('val_bal_acc_lnlevel', acc_dict['bal_acc_lnlevel'])
        self.log('val_bal_acc_nstage', acc_dict['bal_acc_nstage'])
        if self.config['pseudo_loss']:
            to_append['pred_pseudo'] = preds_dict['pseudo_preds'].view(batch_size, 3).squeeze().long().detach().cpu().tolist()
            to_append['gt_pseudo'] = pseudo_gt.squeeze().view(batch_size, 3).long().detach().cpu().tolist()
            self.log('val_loss_pseudo', self.config['lambdas'][2] * loss_dict['pseudo'].item())
            self.log('val_acc_pseudo', acc_dict['acc_pseudo'])
            self.log('val_bal_acc_pseudo', acc_dict['bal_acc_pseudo'])
        if self.config['pseudo_loss']:
            loss = self.config['lambdas'][0]*loss_dict['lnlevel'] + self.config['lambdas'][1]*loss_dict['nstage'] + \
                   self.config['lambdas'][2]*loss_dict['pseudo']
        else:
            loss = self.config['lambdas'][0]*loss_dict['lnlevel'] + self.config['lambdas'][1]*loss_dict['nstage']
        # Debug pseudo loss only
        # loss = self.config['lambdas'][2]*loss_dict['pseudo']
        self.val_tab = pd.concat([self.val_tab, pd.DataFrame(to_append)])
        return {'preds': preds_dict, 'loss': loss}

    def test_step(self, batch, batch_idx):
        preds_dict, loss_dict, acc_dict, pseudo_gt = self._get_preds_loss_accuracy(batch)
        batch_size = preds_dict['nstage_logits'].shape[0]
        to_append = {'patient_id': batch['patient_id'], 'study': batch['study'], 'series': batch['series_names'],
                     'lymph_node': [self.config['lymphnode_names'] for i in range(batch_size)],
                     'pred_lnlevel': preds_dict['lnlevel_preds'].view(batch_size, len(self.config['lymphnode_names'])).squeeze().long().detach().cpu().tolist(),
                     'prob_lnlevel': preds_dict['lnlevel_probs'].view(batch_size, len(self.config['lymphnode_names'])).squeeze().detach().cpu().tolist(),
                     'pred_nstage': preds_dict['nstage_preds'].squeeze().long().detach().cpu().tolist(),
                     'gt_nstage': batch['label'].squeeze().long().detach().cpu().tolist(),
                     'gt_lnlevel': batch['label_lnlevel'].squeeze().long().detach().cpu().tolist(),
                     }
        if self.config['pseudo_loss']:
            to_append['pred_pseudo'] = preds_dict['pseudo_preds'].view(batch_size, 3).squeeze().long().detach().cpu().tolist()
            to_append['gt_pseudo'] = pseudo_gt.squeeze().view(batch_size, 3).long().detach().cpu().tolist()
        if len(to_append['patient_id']) == 1:
            to_append['pred_lnlevel'] = [to_append['pred_lnlevel']]
            to_append['prob_lnlevel'] = [to_append['prob_lnlevel']]
            to_append['gt_lnlevel'] = [to_append['gt_lnlevel']]
            to_append['pred_pseudo'] = [to_append['pred_pseudo']]
        self.test_tab = pd.concat([self.test_tab, pd.DataFrame(to_append)])

    def _get_preds_loss_accuracy(self, batch):
        acc_dict = {}
        x, y_nstage, y_lnlevel, pt_loc, pt_suv  = batch['img'], batch['label'], batch['label_lnlevel'].view(-1), batch['PT_Location'], batch['pt_suv']
        preds_dict, loss_dict = self.loss(x, y_lnlevel, y_nstage, pt_loc, pt_suv)
        # lnlevel
        preds = preds_dict['lnlevel_preds'][y_lnlevel != 99]
        y = y_lnlevel[y_lnlevel != 99]
        if len(y) > 0:
            acc_dict['acc_lnlevel'] = self.acc(preds, y)
            acc_dict['bal_acc_lnlevel'] = self.bal_acc(preds, y)
            acc_dict['sens_lnlevel'], acc_dict['spec_lnlevel'] = calc_sens_spec_binclass(list(preds.squeeze().detach().cpu().numpy()),
                                                                                         list(y.detach().cpu().numpy()))
        else:
            acc_dict['acc_lnlevel'] = 0
            acc_dict['bal_acc_lnlevel'] = 0
            acc_dict['sens_lnlevel'], acc_dict['spec_lnlevel'] = 0, 0
        # N-staging
        acc_dict['acc_nstage'] = self.acc(preds_dict['nstage_preds'], y_nstage)
        acc_dict['bal_acc_nstage'] = self.bal_acc(preds_dict['nstage_preds'], y_nstage)
        acc_dict['sens_nstage'], acc_dict['spec_nstage'] = calc_sens_spec_multiclass(preds_dict['nstage_preds'], y_nstage)
        if self.config['pseudo_loss']:
            pseudo_prob, pseudo_gt = self.generate_pseudo_labels(preds_dict['lnlevel_logits'], y_nstage, pt_loc)
            loss_dict['pseudo'] = self.pseudo_loss(pseudo_prob, pseudo_gt)
            preds_dict['pseudo_preds'] = torch.round(pseudo_prob)
            acc_dict['acc_pseudo']  = self.acc(preds_dict['pseudo_preds'][pseudo_gt != 99], pseudo_gt[pseudo_gt != 99])
            acc_dict['bal_acc_pseudo']  = self.bal_acc(preds_dict['pseudo_preds'][pseudo_gt != 99], pseudo_gt[pseudo_gt != 99])
        else:
            pseudo_prob, pseudo_gt = None, None
        return preds_dict, loss_dict, acc_dict, pseudo_gt


    def on_validation_start(self) -> None:
        self.val_tab = pd.DataFrame(columns=['patient_id', 'study', 'series', 'lymph_node', 'pred_lnlevel', 'prob_lnlevel', 'pred_nstage',
                                              'gt_lnlevel', 'gt_nstage', 'pred_pseudo', 'gt_pseudo'])

    def on_test_start(self) -> None:
        self.test_tab = pd.DataFrame(columns=['patient_id', 'study', 'series', 'lymph_node', 'pred_lnlevel', 'prob_lnlevel', 'pred_nstage',
                                              'gt_lnlevel', 'gt_nstage', 'pred_pseudo', 'gt_pseudo'])

    def on_test_epoch_end(self) -> None:
        # lnlevel
        preds = []
        gts = []
        for pred, gt in zip(flatten_lists(self.test_tab['pred_lnlevel'].values), flatten_lists(self.test_tab['gt_lnlevel'].values)):
            if gt != 99:
                preds.append(pred)
                gts.append(gt)
        acc_lnlevel = accuracy_score(gts, preds)
        bal_acc_lnlevel = balanced_accuracy_score(gts, preds)
        sens_lnlevel, spec_lnlevel = calc_sens_spec_binclass(preds, gts)
        self.log('test_acc_lnlevel', acc_lnlevel)
        self.log('test_bal_acc_lnlevel', bal_acc_lnlevel)
        self.log('test_sensitivity_lnlevel', sens_lnlevel)
        self.log('test_specificity_lnlevel', spec_lnlevel)
        # nstage
        acc_nstage = accuracy_score(list(self.test_tab['gt_nstage'].values), list(self.test_tab['pred_nstage'].values))
        bal_acc_nstage = balanced_accuracy_score(list(self.test_tab['gt_nstage'].values), list(self.test_tab['pred_nstage'].values))
        sens_nstage, spec_nstage = calc_sens_spec_multiclass(list(self.test_tab['pred_nstage'].values), list(self.test_tab['gt_nstage'].values))
        self.log('test_acc_nstage', acc_nstage)
        self.log('test_bal_acc_nstage', bal_acc_nstage)
        self.log('test_sensitivity_nstage', sens_nstage)
        self.log('test_specificity_nstage', spec_nstage)
        with open(os.path.join(self.trainer.checkpoint_callback.dirpath, f'results_fold={self.config["current_fold"]}.json'), 'w') as outfile:
            json.dump({'ckpt_path' : self.trainer.ckpt_path,
                       'test_acc_lnlevel': acc_lnlevel, 'test_bal_acc_lnlevel': bal_acc_lnlevel,
                       'test_sensitivity_lnlevel': sens_lnlevel, 'test_specificity_lnlevel': spec_lnlevel,
                       'test_acc_nstage': acc_nstage, 'test_bal_acc_nstage': bal_acc_nstage,
                       'test_sensitivity_nstage': sens_nstage, 'test_specificity_nstage': spec_nstage,
                       }, outfile)
        self.test_tab.to_csv(os.path.join(self.trainer.checkpoint_callback.dirpath, f'test_confm_fold={self.config["current_fold"]}.csv'), index=False)

