import ast
import glob
import os
from pathlib import Path
import pandas as pd
import json
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from Nstaging_with_classifier.loggingmodule import create_confusion_matrix
from Nstaging_with_classifier.rulebased_nstaging import calc_sens_spec_binclass, calc_sens_spec_multiclass
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


path = '/workspace/KIMedLung/scripts/Nstaging_with_classifier/checkpoints'
results_path = '/workspace/KIMedLung/results'

def flatten_lists(list_of_lists: list):
    result = []
    for l in list_of_lists:
        result.extend(ast.literal_eval(l))
    return result

def extract_text(txt: str, entry_start: str, entry_stop: str):
    if entry_stop == '':
        return txt[txt.find(entry_start) + len(entry_start): len(txt)]
    else:
        return txt[txt.find(entry_start) + len(entry_start): txt.find(entry_stop)]

def extract_info(path: str):
    modality = Path(path).parent.parent.name
    task = Path(path).parent.name
    val_acc = float(extract_text(Path(path).name, 'val_bal_acc_lnlevel=', '.ckpt'))
    fold = int(extract_text(Path(path).name, 'fold=', '_epoch'))
    epoch = int(extract_text(Path(path).name, 'epoch=', '_val'))
    return modality, task, val_acc, fold, epoch

def plot_roccurve(final_result_tab: pd.DataFrame):
    for modality in final_result_tab['modality'].drop_duplicates():
        subset = final_result_tab[final_result_tab['modality'] == modality]
        plt.title(f'ROC-AUC Curve for mod={modality}')
        for idx in subset.index:
            task = subset.loc[idx, 'task']
            if task == 'CV_sampl=mask_size=64_enclns=False_maskprior=False_encptsuv=False_GINIPA=False':
                task_name = 'mask'
            elif task == 'CV_sampl=weight_map_size=64_enclns=False_maskprior=False_encptsuv=False_GINIPA=False':
                task_name = 'wm'
            elif task == 'CV_sampl=weight_map_size=64_enclns=True_maskprior=False_encptsuv=False_GINIPA=False':
                task_name = 'wm_enclns'
            elif task == 'CV_sampl=weight_map_size=64_enclns=True_maskprior=True_encptsuv=False_GINIPA=False':
                task_name = 'wm_enclns_maskprior'
            elif task == 'CV_sampl=weight_map_size=64_enclns=False_maskprior=False_encptsuv=False_GINIPA=False':
                task_name = 'wm_GINIPA'
            else:
                task_name = ''
            plt.plot(subset.loc[idx, 'rocc_results'][0], subset.loc[idx, 'rocc_results'][1], label=f'{task_name} AUC = %0.2f' % subset.loc[idx, 'auc_lnlevel'])
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

def perform_stat_test(results_tab: pd.DataFrame):
    tasks = list(results_tab['task'].drop_duplicates())
    for modality in results_tab['modality'].drop_duplicates():
        subset = results_tab[results_tab['modality'] == modality].reset_index()
        for task_idx, task in enumerate(tasks[:-1]):
            cl1 = subset[subset['task'] == task].reset_index()
            cl2 = subset[subset['task'] == tasks[task_idx + 1]].reset_index()
            cl1['misclassified'] = [1 if cl1.loc[idx, 'pred_lnlevel'] != cl1.loc[idx, 'label_lnlevel'] else 0 for idx in cl1.index]
            cl2['misclassified'] = [1 if cl2.loc[idx, 'pred_lnlevel'] != cl2.loc[idx, 'label_lnlevel'] else 0 for idx in cl2.index]
            n_00 = (np.array(cl1['misclassified'] == 1) & np.array(cl2['misclassified'] == 1)).sum()
            n_01 = (np.array(cl1['misclassified'] == 1) & np.array(cl2['misclassified'] == 0)).sum()
            n_10 = (np.array(cl1['misclassified'] == 0) & np.array(cl2['misclassified'] == 1)).sum()
            n_11 = (np.array(cl1['misclassified'] == 0) & np.array(cl2['misclassified'] == 0)).sum()
            test = mcnemar([[n_00, n_01], [n_10, n_11]], exact=False, correction=True)
            alpha=0.05
            print(f' \n mod = {modality}')
            print(f'CL1: task = {task}')
            print(f'CL2: task = {tasks[task_idx + 1]}')
            if test.pvalue < alpha:
                print("Reject Null hypotesis")
            else:
                print("Fail to reject Null hypotesis")

if __name__ == '__main__':
    # # --- results_old.json
    # result_files = sorted(glob.glob(os.path.join(path, '*', '*', 'results.json')))
    # all_results = pd.DataFrame({})
    # for idx, file in enumerate(result_files):
    #     task_result = {}
    #     with open(file, 'r') as openfile:
    #         result = json.load(openfile)
    #     for key, value in result.items():
    #         if type(value) == list:
    #             result[key] = value[0]
    #     results_df = pd.DataFrame(result).T
    #     results_df['modality'] = [Path(file).parent.parent.name] * 5
    #     results_df['task'] = [Path(file).parent.name] * 5
    #     results_df['fold'] = list(range(0, 5))
    #     all_results = pd.concat((all_results, results_df))
    # del all_results['ckpt']
    # del all_results['fold']
    # test = all_results.groupby(['modality', 'task']).mean()
    # test2 = all_results.groupby(['modality', 'task']).std()

    # --- From tables for each fold to one table
    ckpt_files = sorted(glob.glob(os.path.join(path, '*', '*', '*.csv')))
    results_tab = pd.DataFrame({})
    for file in ckpt_files:
        modality = Path(file).parent.parent.name
        task = Path(file).parent.name
        fold = int(extract_text(Path(file).name, 'fold=', '.csv'))
        tab = pd.read_csv(file)
        if 'label_lnlevel' in tab.columns:
            tab = tab.rename(columns={'label_lnlevel': 'gt_lnlevel', 'label_nstage': 'gt_nstage'})
            tab.to_csv(file)
        tab['modality'] = [modality] * len(tab)
        tab['task'] = [task] * len(tab)
        tab['fold'] = fold
        if len(results_tab) == 0:
            results_tab = tab
        else:
            results_tab = pd.concat((results_tab, tab))
    # --- Get results
    final_result_tab = pd.DataFrame({})
    for modality in results_tab['modality'].drop_duplicates():
        for task in results_tab['task'].drop_duplicates():
            subset = results_tab[(results_tab['modality'] == modality) & (results_tab['task'] == task)]
            if len(subset) > 0:
                if 'serieslevel' in modality:
                    gt_lnlevel_all = flatten_lists(subset['gt_lnlevel'].values)
                    pred_lnlevel_all = flatten_lists(subset['pred_lnlevel'].values)
                    prob_lnlevel_all = flatten_lists(subset['prob_lnlevel'].values)
                    gt_lnlevel = []
                    pred_lnlevel = []
                    prob_lnlevel = []
                    for idx, value in enumerate(gt_lnlevel_all):
                        if value != 99:
                            gt_lnlevel.append(value)
                            pred_lnlevel.append(pred_lnlevel_all[idx])
                            prob_lnlevel.append(prob_lnlevel_all[idx])
                else:
                    gt_lnlevel_all = list(subset['gt_lnlevel'])
                    pred_lnlevel_all = list(subset['pred_lnlevel'])
                    prob_lnlevel_all = list(subset['prob_lnlevel'])
                    gt_lnlevel = []
                    pred_lnlevel = []
                    prob_lnlevel = []
                    for idx, value in enumerate(gt_lnlevel_all):
                        if value != 99:
                            gt_lnlevel.append(value)
                            pred_lnlevel.append(pred_lnlevel_all[idx])
                            prob_lnlevel.append(prob_lnlevel_all[idx])
                acc_lnlevel = accuracy_score(gt_lnlevel, pred_lnlevel)
                bal_acc_lnlevel = balanced_accuracy_score(gt_lnlevel, pred_lnlevel)
                sens_lnlevel, spec_lnlevel = calc_sens_spec_binclass(pred_lnlevel, gt_lnlevel)
                rocc_results = roc_curve(gt_lnlevel, prob_lnlevel)
                auc_lnlevel = roc_auc_score(gt_lnlevel, pred_lnlevel, average='weighted')
                fig = create_confusion_matrix(gt_lnlevel, pred_lnlevel, ['true label', 'predicted label'],
                                              f'LNlevel')
                fig.savefig(os.path.join(results_path, f'LNlevel_mod={modality}_task={task}.png'))
                subset_nstage = subset.dropna(subset=['gt_nstage', 'pred_nstage'])
                subset_nstage = subset_nstage[['patient_id', 'study', 'series', 'gt_nstage', 'pred_nstage']]
                subset_nstage = subset_nstage.drop_duplicates()
                acc_nstage = accuracy_score(subset_nstage['gt_nstage'], subset_nstage['pred_nstage'])
                bal_acc_nstage = balanced_accuracy_score(subset_nstage['gt_nstage'], subset_nstage['pred_nstage'])
                sens_nstage, spec_nstage = calc_sens_spec_multiclass(subset_nstage['pred_nstage'], subset_nstage['gt_nstage'])
                fig = create_confusion_matrix(subset_nstage['gt_nstage'], subset_nstage['pred_nstage'], ['true label', 'predicted label'],
                                              f'Nstage')
                fig.savefig(os.path.join(results_path, f'Nstage_mod={modality}_task={task}.png'))
                final_result = {'modality': [modality], 'task': [task], 'acc_lnlevel': [acc_lnlevel], 'bal_acc_lnlevel': [bal_acc_lnlevel],
                                'sens_lnlevel': [sens_lnlevel], 'spec_lnlevel': [spec_lnlevel], 'auc_lnlevel': [auc_lnlevel],
                                'rocc_results': [rocc_results], 'acc_nstage': [acc_nstage], 'bal_acc_nstage': [bal_acc_nstage],
                                'sens_nstage': [sens_nstage], 'spec_nstage': [spec_nstage], 'folds': [np.unique(subset['fold'])]}
                final_result_tab = pd.concat([final_result_tab, pd.DataFrame.from_dict(final_result)])
    final_result_tab = final_result_tab.reset_index()
    #plot_roccurve(final_result_tab)
    #perform_stat_test(results_tab)
    final_result_tab = final_result_tab.drop(columns='rocc_results')
    final_result_tab.to_csv(os.path.join(results_path, 'final_results.csv'), index=False)
