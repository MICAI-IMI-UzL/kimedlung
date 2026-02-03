import time
import torch
import os
import pandas as pd
import numpy as np
from pathlib import Path
from monai.transforms import Transform, RandWeightedCropd, LoadImage, ScaleIntensityRange, BorderPadd, Compose
from monai.data import MetaTensor
from typing import Any
import SimpleITK as sitk
import ast
import h5py
import glob
import random
from mappings import trans_dict_sarrut


class ConcatChannels(Transform):

    def __init__(self, config: dict):
        self.config = config

    def __call__(self, data: dict) -> dict:
        if 'pet' in data.keys():
            data['img'] = torch.concat((data['img'], data['pet']), dim=0)
            del data['pet']
        if self.config['mask_as_prior'] and 'mask' in data.keys():
            data['img'] = torch.concat((data['img'], data['mask']), dim=0)
            del data['mask']
        return data


class CreateWeightMap(Transform):

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def create_distance_map(image, ref_point):
        coordinate_map = sitk.PhysicalPointSource(sitk.sitkVectorFloat32, size=image.GetSize(),
                                                  origin=image.GetOrigin(),
                                                  spacing=image.GetSpacing(),
                                                  direction=image.GetDirection())
        imgx = sitk.VectorIndexSelectionCast(coordinate_map, 0) - ref_point[0]
        imgy = sitk.VectorIndexSelectionCast(coordinate_map, 1) - ref_point[1]
        imgz = sitk.VectorIndexSelectionCast(coordinate_map, 2) - ref_point[2]
        coordinate_map = sitk.Sqrt(imgx**2 + imgy**2 + imgz**2)
        return coordinate_map

    def create_weight_map(self, mask: torch.Tensor, lymphnode_name: str) -> np.ndarray:
        mask_arr = mask.squeeze().numpy().astype('uint8')
        mask_itk = sitk.GetImageFromArray(mask_arr)
        ln_labelnum = trans_dict_sarrut[lymphnode_name]
        stats = sitk.LabelIntensityStatisticsImageFilter()
        stats.Execute(mask_itk, mask_itk)
        if ln_labelnum in stats.GetLabels():
            center = stats.GetCentroid(ln_labelnum)
            distance_map = self.create_distance_map(mask_itk, center)
            distance_map_arr = sitk.GetArrayFromImage(distance_map)
            weight_map = distance_map_arr.astype('uint8')
            weight_map[mask_arr != ln_labelnum] = 0
        else:
            weight_map = np.zeros_like(mask_arr)
        return weight_map

    @staticmethod
    def post_process_weight_map(weight_map: np.array) -> torch.Tensor:
        weight_map_new = np.max(weight_map) - weight_map
        weight_map_new[weight_map==0] = 0
        weight_map_new = (weight_map_new / np.max(weight_map_new))
        return torch.from_numpy(np.expand_dims(weight_map_new, axis=0))

    def __call__(self, data: dict, ln_name: str = None) -> dict:
        if ln_name is None:
            ln_name = data['lymph_node']
        prefix = os.path.join(self.config['resultspath'], 'sampl_weight_maps',
                              f"{Path(data['mask'].meta['filename_or_obj']).parent.name}")
        Path(prefix).mkdir(exist_ok=True, parents=True)
        if type(data["series_names"]) == tuple:
            ser_name = ast.literal_eval(data["series_names"])[0]
        else:
            ser_name = data["series_names"]
        weight_map_filename = Path(data['mask'].meta['filename_or_obj']).name.replace('crop_all_lymphnodes',
                                                                   f'sampl_weight_map_ser={ser_name}_ln={ln_name}')
        weight_map_filename  = weight_map_filename.replace('.nii.gz', '.npy')
        if not os.path.exists(os.path.join(prefix, weight_map_filename)):
            weight_map = self.create_weight_map(data['mask'], ln_name)
            np.save(os.path.join(prefix, weight_map_filename), weight_map)
        else:
            weight_map = np.load(os.path.join(prefix, weight_map_filename))
        data['weight_map'] = self.post_process_weight_map(weight_map)
        return data


class SamplingLymphNodes(Transform):

    def __init__(self, config: dict):
        self.config = config
        self.sampling_function = RandWeightedCropd(keys=['img', 'pet', 'mask'], w_key='weight_map',
                                                   spatial_size=self.config['spatial_size_extract'],
                                                   num_samples=config['sample_size'], allow_missing_keys=True)
        self.pad_function = BorderPadd(keys=['img', 'pet', 'mask', 'weight_map'], spatial_border=config['spatial_size'][0],
                                       allow_missing_keys=True)
        self.create_weight_map = CreateWeightMap(config)

    def create_series_file(self, prefix: str, filename_part1: str):
        for idx in range(self.config['sample_size']):
            images = sorted(glob.glob(os.path.join(prefix, f'{filename_part1}lnidx=*_ln_name=*_{idx:03d}.h5')))
            img_list = []
            pet_list = []
            mask_list = []
            for image in images:
                with h5py.File(image, 'r') as f:
                    img_list.append(f['img'][()])
                    mask_list.append(f['mask'][()])
                    if self.config['modality'] == 'PET':
                        pet_list.append(f['pet'][()])
            assert len(img_list) == len(self.config['lymphnode_names'])
            img = np.stack(img_list,axis=1)[0]
            mask = np.stack(mask_list, axis=1)[0]
            if self.config['modality'] == 'PET':
                pet = np.stack(pet_list,axis=1)[0]
            if not os.path.exists(os.path.join(prefix, f'{filename_part1}seriesfile_{idx:03d}.h5')):
                try:
                    with h5py.File(os.path.join(prefix, f'{filename_part1}seriesfile_{idx:03d}.h5'), 'w') as f:
                        f.create_dataset('img', data=img)
                        f.create_dataset('mask', data=mask)
                        if self.config['modality'] == 'PET':
                            f.create_dataset('pet', data=pet)
                except:
                    print('Could not write file.')

    def sample_lymph_nodes(self, data: Any) -> None:
        prefix = os.path.join(self.config['resultspath'],
                              f'sampled_patches_spatialsize={self.config["spatial_size_extract"]}',
                              f"{Path(data['mask'].meta['filename_or_obj']).parent.name}")
        if self.config['modality'] == 'PET':
            filename_part1 = f'sampled_patch_serimg={ast.literal_eval(data["series_names"])[0]}_' \
                             f'sersuv={ast.literal_eval(data["series_names"])[1]}_'
        elif self.config['modality'] == 'CT':
            filename_part1 = f'sampled_patch_serimg={data["series_names"]}_'
        else:
            raise Exception('Unknown modality.')
        Path(prefix).mkdir(exist_ok=True, parents=True)
        for ln_idx, ln_name in enumerate(self.config['lymphnode_names']):
            data_ln = data.copy()
            self.create_weight_map(data=data_ln, ln_name=ln_name)
            # sitk.WriteImage(sitk.GetImageFromArray(data['img'][0].numpy()), f'/workspace/{data["patient_id"]}_ct.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data['pet'][0].numpy()), f'/workspace/{data["patient_id"]}_pet.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data['mask'][0].numpy()), f'/workspace/{data["patient_id"]}_mask.nii.gz')
            data_ln = self.pad_function(data_ln)
            data_ln = self.sampling_function(data_ln)
            # sitk.WriteImage(sitk.GetImageFromArray(data_ln[0]['weight_map'][0].numpy()),
            #                 f'/workspace/{data["patient_id"]}_weightmap_ln_name={ln_name}.nii.gz')
            if data_ln[0]['weight_map'].max().item() == 0:
                for entry in data_ln:
                    entry['img'] = torch.zeros_like(entry['img'])
                    if 'pet' in entry.keys():
                        entry['pet'] = torch.zeros_like(entry['pet'])
            for idx, entry in enumerate(data_ln):
                # sitk.WriteImage(sitk.GetImageFromArray(data_ln[idx]['img'][0].numpy()),
                #                 f'/workspace/{data["patient_id"]}_ct_ln_name={ln_name}_{idx:03d}.nii.gz')
                # sitk.WriteImage(sitk.GetImageFromArray(data_ln[idx]['pet'][0].numpy()),
                #                 f'/workspace/{data["patient_id"]}_pet_ln_name={ln_name}_{idx:03d}.nii.gz')
                # sitk.WriteImage(sitk.GetImageFromArray(data_ln[idx]['mask'][0].numpy()),
                #                 f'/workspace/{data["patient_id"]}_mask_ln_name={ln_name}_{idx:03d}.nii.gz')
                if not os.path.exists(
                        os.path.join(prefix, f'{filename_part1}lnidx={ln_idx:02d}_ln_name={ln_name}_{idx:03d}.h5')):
                    try:
                        with h5py.File(
                                os.path.join(prefix, f'{filename_part1}lnidx={ln_idx:02d}_ln_name={ln_name}_{idx:03d}.h5'),
                                'a') as f:
                            f.create_dataset('img', data=entry['img'])
                            f.create_dataset('mask', data=entry['mask'])
                            if self.config['modality'] == 'PET':
                                f.create_dataset('pet', data=entry['pet'])
                    except:
                        print('Could not write file.')
        self.create_series_file(prefix, filename_part1)

    def __call__(self, data: Any) -> Any:
        data2 = data.copy()
        sample_idx = random.randint(0, self.config['sample_size'] - 1)
        prefix = os.path.join(self.config['resultspath'],
                              f'sampled_patches_spatialsize={self.config["spatial_size_extract"]}')
        if self.config['modality'] == 'PET':
            filename_part1 = f'sampled_patch_serimg={ast.literal_eval(data["series_names"])[0]}_' \
                             f'sersuv={ast.literal_eval(data["series_names"])[1]}_'
        elif self.config['modality'] == 'CT':
            filename_part1 = f'sampled_patch_serimg={data["series_names"]}_'
        else:
            raise Exception('Unknown modality.')
        if self.config['level'] == 'series':
            h5d = os.path.join(prefix, Path(data2['img']).parent.name, f'{filename_part1}seriesfile_{sample_idx:03d}.h5')
        elif self.config['level'] == 'ln':
            h5d = os.path.join(prefix, Path(data2['img']).parent.name,
                               f'{filename_part1}lnidx={self.config["lymphnode_names"].index(data2["lymph_node"]):02d}'
                               f'_ln_name={data2["lymph_node"]}_{sample_idx:03d}.h5')
        else:
            raise Exception('Unknown level.')
        if not os.path.exists(h5d):
            data2 = self.config['save_patches_transforms'](data2)
            # sitk.WriteImage(sitk.GetImageFromArray(data2['img'][0].numpy()), f'/workspace/{data2["patient_id"]}_ct.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data2['pet'][0].numpy()), f'/workspace/{data2["patient_id"]}_pet.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data2['mask'][0].numpy()), f'/workspace/{data2["patient_id"]}_mask.nii.gz')
            self.sample_lymph_nodes(data2)
            while not os.path.exists(h5d):
                time.sleep(1)
            # sitk.WriteImage(sitk.GetImageFromArray(data2['img'][0].numpy()), f'/workspace/{data2["patient_id"]}_ct_ln={data2["lymph_node"]}.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data2['pet'][0].numpy()), f'/workspace/{data2["patient_id"]}_pet_ln={data2["lymph_node"]}.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data2['mask'][0].numpy()), f'/workspace/{data2["patient_id"]}_mask_ln={data2["lymph_node"]}.nii.gz')
        with h5py.File(h5d, 'r') as f:
            data2['img'] = torch.from_numpy(f['img'][()])
            if 'pet' in data2.keys():
                data2['pet'] = torch.from_numpy(f['pet'][()])
            if 'mask' in data2.keys():
                data2['mask'] = torch.from_numpy(f['mask'][()])
            # lymph_node = 1
            # sitk.WriteImage(sitk.GetImageFromArray(data2['img'][lymph_node].numpy()), f'/workspace/{data2["patient_id"]}_ct_ln={self.config["lymphnode_names"][lymph_node]}.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data2['pet'][lymph_node].numpy()), f'/workspace/{data2["patient_id"]}_pet_ln={self.config["lymphnode_names"][lymph_node]}.nii.gz')
            # sitk.WriteImage(sitk.GetImageFromArray(data2['mask'][lymph_node].numpy()), f'/workspace/{data2["patient_id"]}_mask_ln={self.config["lymphnode_names"][lymph_node]}.nii.gz')
        return data2

class EncodeTab(Transform):

    def __init__(self, config: dict):
        self.config = config
        self.scale_pt_suv = ScaleIntensityRange(a_min=0, a_max=4, b_min=0, b_max=1)

    def __call__(self, data: dict) -> dict:
        if self.config['level'] == 'ln':
            data['ln_idx'] = MetaTensor(torch.Tensor([self.config['lymphnode_names'].index(data['lymph_node'])]))
        elif self.config['level'] == 'series':
            data['Nstation'] = str(data['Nstation'])
            data['label_lnlevel'] = MetaTensor(torch.Tensor(ast.literal_eval(data['label_lnlevel'])))
            if data['PT_Location'] == 'R':
                encoded_pt_location = 0
            elif data['PT_Location'] == 'L':
                encoded_pt_location = 1
            else:
                encoded_pt_location = None
            data['PT_Location'] = MetaTensor(torch.Tensor([encoded_pt_location]))
        else:
            raise Exception('Unknown level.')
        data['pt_suv'] = self.scale_pt_suv(MetaTensor(torch.Tensor([data['PT_SUV']])))
        return data

class ReadVOI(Transform):

    def __init__(self, config: dict):
        self.config = config
        self.load_image = LoadImage()
    def __call__(self, data: dict) -> dict:
        data2 = data.copy()
        if self.config['modality'] == 'PET':
            ln_path = ast.literal_eval(data2['ln_path'])
            if ln_path[0] == '' or ln_path[1] == '':
                data2['img'] = MetaTensor(torch.from_numpy(np.zeros([1, 64,64,64])).float())
                data2['pet'] = MetaTensor(torch.from_numpy(np.zeros([1, 64,64,64])).float())
            else:
                data2['img'] = self.load_image(ln_path[0]).unsqueeze(0)
                data2['pet'] = self.load_image(ln_path[1]).unsqueeze(0)
        elif self.config['modality'] == 'CT':
            ln_path = data2['ln_path']
            if pd.isnull(ln_path):
                data2['img'] = MetaTensor(torch.from_numpy(np.zeros([1, 64,64,64])).float())
            else:
                data2['img'] = self.load_image(ln_path).unsqueeze(0)
            del data2['ln_path']
        else:
            raise Exception('Unknown modality.')
        del data2['mask']
        return data2

class ReadSeriesVOIs(Transform):
    def __init__(self, config: dict, lnlevel_transforms: list):
        self.config = config
        self.lnlevel_transforms = Compose(lnlevel_transforms)

    def __call__(self, data: dict) -> dict:
        data1 = data.copy()
        list_ct = []
        list_pet = []
        for ln in self.config['lymphnode_names']:
            data2 = data1.copy()
            ln_path = ast.literal_eval(data2['ln_path'])[ln]
            data2['ln_path'] = str(ln_path)
            data2 = self.lnlevel_transforms(data2)
            list_ct.append(data2['img'])
            if self.config['modality'] == 'PET':
                list_pet.append(data2['pet'])
        data1['img'] = torch.concat(list_ct, dim=0)
        if self.config['modality'] == 'PET':
            data1['pet'] = torch.concat(list_pet, dim=0)
        del data1['mask']
        return data1
