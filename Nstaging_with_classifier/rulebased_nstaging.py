from preprocessing.create_patient_tab_for_UKSH import patient_id_to_tabstyle
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
import ast
import numpy as np
import pandas as pd
import os


def calc_sens_spec_multiclass(pred_nstage_list: list, gt_nstage_list: list):
    sensitivity_list = []
    specificity_list = []
    for l in [0, 1, 2, 3]:
        gt = [1 if entry == l else 0 for entry in gt_nstage_list]
        pred = [1 if entry == l else 0 for entry in pred_nstage_list]
        cm = confusion_matrix(y_true=gt, y_pred=pred)
        if len(cm) == 1:
            sens = 1
            spec = 1
        else:
            TN, FP, FN, TP = cm.ravel()
            sens = TP / (TP + FN)
            spec = TN / (TN + FP)
        if not np.isnan(sens):
            sensitivity_list.append(sens)
        if not np.isnan(spec):
            specificity_list.append(spec)
    return np.mean(sensitivity_list), np.mean(specificity_list)

def calc_sens_spec_binclass(pred: list, gt: list):
    cm = confusion_matrix(y_true=gt, y_pred=pred)
    if len(cm) == 1 or len(cm) == 0:
        sens = 1
        spec = 1
    else:
        TN, FP, FN, TP = cm.ravel()
        if TP == 0 and FN == 0:
            sens = 1
        else:
            sens = TP / (TP + FN)
        spec = TN / (TN + FP)
    return sens, spec


def from_station_to_N(lymphnodes: list, pt_location: str):
    mapping_TL = {3: ['1R', '2R', '4R', '8R', '9R', '10R', '11R', '12R', '13R', '14R'],
                  2: ['1L', '2L', '3A', '3P', '4L', '5', '5L', '6', '6L', '7', '8', '8L', '9', '9L'],
                  1: ['10L', '11L']} #'12L', '13L', '14L']}
    mapping_TR = {3: ['1L', '2L', '4L', '8L', '9L', '10L', '11L', '12L', '13L', '14L'],
                  2: ['1R', '2R', '3A', '3P', '4R', '5', '5R', '6', '6R','7', '8', '8R', '9', '9R'],
                  1: ['10R', '11R']} #'12R', '13R', '14R']}
    if lymphnodes == []:
        return 0
    else:
        if pt_location == 'R':
            for stage, stations in mapping_TR.items():
                for lymphnode in lymphnodes:
                    for station in stations:
                        if lymphnode == station:
                            return stage
        elif pt_location == 'L':
            for stage, stations in mapping_TL.items():
                for lymphnode in lymphnodes:
                    for station in stations:
                        if lymphnode == station:
                            return stage

if __name__ == '__main__':
    mod1 = 'cN'
    mod2 = 'cN'
    dataset = 'UKSH-N3'
    results_path = '/workspace/KIMedLung/results'
    tab1 = pd.read_csv(os.path.join(results_path, f'ds={dataset}_level=pat_label={mod1}.csv'))
    tab2 = pd.read_csv(os.path.join(results_path, f'ds={dataset}_level=pat_label={mod2}.csv'))
    tab = pd.read_excel('/workspace/KIMedLung/data/tab_data_corrected.xlsx')
    patids = []
    gt_nstage_list = []
    pred_nstage_list = []
    for row in tab1.index:
        patient_id = tab1.loc[row, 'patient_id']
        pred_nstation = ast.literal_eval(tab1.loc[row, 'Nstation'])
        pt_location = tab1.loc[row, 'PT_Location']
        # # --- Choose:
        # pred_nstage = tab1.loc[row, 'N']
        # or ...
        pred_nstage = from_station_to_N(pred_nstation, pt_location)
        # ---
        tab2_patient_info = tab2[tab2['patient_id'] == patient_id]
        if len(tab2_patient_info) == 0:
            continue
        gt_nstage = tab2_patient_info['N'].values[0]
        if pred_nstage != gt_nstage and pred_nstage != None:
            print(f'Patient ID: {patient_id}, GT: {gt_nstage}, Pred: {pred_nstage}, LN: {pred_nstation}, PT_Location: {pt_location}')
            patids.append(patient_id_to_tabstyle(patient_id.split('_')[-1].strip('pat')))
        if pred_nstage != None:
            gt_nstage_list.append(gt_nstage)
            pred_nstage_list.append(pred_nstage)
    wrong_label = tab[tab['ID'].astype('str').isin(patids)]
    #wrong_label.to_csv(f'/share/data_rosita1/engelson/Projects/KIMedLung/results/to_check/ds={dataset}_sanity_check_{mod1}_to_{mod2}.csv', index=False)
    print(f'Acc.: {accuracy_score(gt_nstage_list, pred_nstage_list)}')
    print(f'Bal. Acc.: {balanced_accuracy_score(gt_nstage_list, pred_nstage_list)}')
    if dataset == 'UKSH':
        sens, spec = calc_sens_spec_multiclass(pred_nstage_list, gt_nstage_list)
        print(f'Sensitivity: {sens}')
        print(f'Specificity: {spec}\n')