import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import pytorch_lightning as pl
from monai.data import Dataset
import ast
import json


class StratifiedKFold3(StratifiedKFold):
    """Source: https://stackoverflow.com/questions/45221940/creating-train-test-val-split-with-stratifiedkfold"""
    def split(self, X, y, groups=None):
        s = super().split(X, y, groups)
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            train_indxs, cv_indxs = train_test_split(train_indxs,stratify=y_train, test_size=(1 / (self.n_splits - 1)))
            yield train_indxs, cv_indxs, test_indxs


class KIMedLungDataModule(pl.LightningDataModule):

    def __init__(self, config: dict, fold: int = None):
        super().__init__()
        self.config = config
        self.image_path = config['imagepath']
        self.voi_path = config['voipath']
        self.results_path = config['resultspath']
        self.fold = fold
        self.config['current_fold'] = fold

    def do_train_test_split(self):
        split_path = os.path.join(self.results_path, f'split_ds={self.config["datasets"]}_level=pat_mod={self.config["modality"]}_label={self.config["label_column"]}.json')
        if not os.path.exists(split_path):
            assert self.config['encode_lnstation'] == False and self.config['mask_as_prior'] == False and self.config['encode_pt_suv'] == False
            self.create_N_sumLN()
            skf = StratifiedKFold3(n_splits=5).split(X=self.patient_tab, y=self.patient_tab['N_sumLN'])
            self.splits = {}
            for indices, fold in zip(skf, list(range(5))):
                train_idx, val_idx, test_idx = indices
                self.splits[str(fold)] = {'train': train_idx.tolist(), 'val': val_idx.tolist(), 'test': test_idx.tolist()}
            with open(split_path, 'w') as outfile:
                json.dump(self.splits, outfile)
        else:
            with open(split_path, 'r') as openfile:
                self.splits = json.load(openfile)

    def create_N_sumLN(self):
        self.patient_tab['N_sumLN'] = self.patient_tab.apply(self.sum_lns, axis=1)
        values, counts = np.unique(self.patient_tab['N_sumLN'], return_counts=True)
        while 1 in counts or 2 in counts:
            prev_value = '0_0'
            for value, count in zip(values, counts):
                if count == 1 or count == 2:
                    self.patient_tab = self.patient_tab.replace(value, str(prev_value))
                else:
                    prev_value = value
            values, counts = np.unique(self.patient_tab['N_sumLN'], return_counts=True)

    def sum_lns(self, row):
        ln_sum = np.sum(ast.literal_eval(self.data_table[self.data_table['patient_id'] == row['patient_id']]['label_lnlevel'].iloc[0]))
        return f'{row["N"]}_{ln_sum}'

    def prepare_data(self) -> None:
        self.patient_tab = pd.DataFrame({})
        self.data_table = pd.DataFrame({})
        for dataset in self.config['datasets']:
            self.patient_tab = pd.concat([self.patient_tab,
                                          pd.read_csv(os.path.join(self.results_path, f'ds={dataset}_level=pat_label={self.config["label_column"]}.csv'))])
            self.data_table =  pd.concat([self.data_table,
                                          pd.read_csv(os.path.join(self.results_path, f'ds={dataset}_mod={self.config["modality"]}_'
                                                                                      f'level={self.config["level"]}_label={self.config["label_column"]}.csv'))])
        self.patient_tab = self.patient_tab.reset_index()
        self.data_table = self.data_table.reset_index()
        self.data_table['dataset'] = [entry[:entry.find('_pat')] for entry in self.data_table['patient_id'].to_list()]
        self.data_table['study'] = self.data_table['study'].astype(str)
        if self.config['patch_sampling'] == 'weight_map':
            self.data_table['ln_path'] = self.data_table['ln_path'].astype(str)
        self.clean_data()
        self.do_train_test_split()
        self.clean_data_postsplit()

    def clean_data(self):
        # Remove patients from patient_tab if not in data_table
        mask_pat = [False if patient_id not in list(self.data_table['patient_id'].drop_duplicates()) else True for patient_id in self.patient_tab['patient_id']]
        self.patient_tab = self.patient_tab[mask_pat].reset_index(drop=True)
        # Remove patients from data_table if not in patient_tab
        mask_pat = [False if patient_id not in list(self.patient_tab['patient_id']) else True for patient_id in self.data_table['patient_id']]
        self.data_table = self.data_table[mask_pat].reset_index(drop=True)

    def clean_data_postsplit(self):
        if self.config['level'] == 'series':
            self.patient_tab = self.patient_tab[self.patient_tab['PT_Location'] != '-']
        if self.config['encode_pt_suv']:
            self.patient_tab = self.patient_tab[-pd.isnull(self.patient_tab['PT_SUV'])]
        # Depending on config, some patients are removed, these need to be removed from the splits too
        for keys, values in self.splits.items():
            values_updated = values.copy()
            for state, idxs in values.items():
                idxs_updated = idxs.copy()
                for idx in idxs:
                    if idx not in self.patient_tab.index:
                        idxs_updated.remove(idx)
                values_updated[state] = idxs_updated
            self.splits[keys] = values_updated

    def setup(self, stage=None):
        train_patients = self.patient_tab.loc[self.splits[str(self.fold)]['train']]
        val_patients = self.patient_tab.loc[self.splits[str(self.fold)]['val']]
        test_patients = self.patient_tab.loc[self.splits[str(self.fold)]['test']]
        train_table = self.data_table.loc[self.data_table['patient_id'].isin(train_patients['patient_id'])]
        val_table = self.data_table.loc[self.data_table['patient_id'].isin(val_patients['patient_id'])]
        test_table = self.data_table.loc[self.data_table['patient_id'].isin(test_patients['patient_id'])]
        self.train_dataset = Dataset(data=train_table.to_dict(orient='records'), transform=self.config['train_transforms'])
        self.val_dataset = Dataset(data=val_table.to_dict(orient='records'), transform=self.config['test_transforms'])
        self.test_dataset = Dataset(data=test_table.to_dict(orient='records'), transform=self.config['test_transforms'])
        assert len(set(test_table['patient_id']) & set((train_table['patient_id']))) == 0
        assert len(set(val_table['patient_id']) & set((train_table['patient_id']))) == 0

    @staticmethod
    def get_sampler(dataset):
        y_train = [dict['label'] for dict in dataset.data]
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = torch.from_numpy(np.array([weight[t] for t in y_train]))
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        return sampler

    def train_dataloader(self):
        train_sampler = self.get_sampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=int(self.config['batch_size']), drop_last=True, num_workers=4,
                          sampler=train_sampler, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=int(self.config['batch_size']), drop_last=False, num_workers=4,
                          persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=int(self.config['batch_size']), drop_last=False, num_workers=4,
                          persistent_workers=True, pin_memory=True)
