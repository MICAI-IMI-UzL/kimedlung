# Deep Learning-based Mediastinal Lymph Node Assessment on PET/CT Images without Pixel-Level Annotations

This repository contains the code for the paper published in the Journal of Medical Imaging with the title above.

Our approach consists of a deep learning-based pipeline that performs binary classification of mediastinal lymph node 
stations (LNS) and N-stage prediction from PET/CT using only image-level labels for training. We aim to mimic the 
radiologists' approach for N-staging and in this way guide the learning process by utilizing prior knowledge. We 
compared three training strategies, which mainly differ in the way the subtask of N-staging is solved:
1. Training an AI-based LNS classifier followed by rule-based N-staging (TS 1),
2. training both the LNS classifier and N-staging simultaneously (TS 2)
3. freezing the pre-trained LNS classifier and refining the last network layers for N-staging (TS 3).
Prior knowledge is incorporated through two mechanisms. First, LNS are localized using atlas-to-patient registration, 
which can be used for local patch sampling. Second, patient-level knowledge is transferred to the LNS level (TS 2), 
permitting weak supervision.

## Train instructions

### Pre-processing:

- For all PET/CTs, run the atlas-to-patient registration as proposed in: https://github.com/MICAI-IMI-UzL/LNQ2023 for 
all atlases published by Lynch et al. (https://www.creatis.insa-lyon.fr/lymph-stations-atlas/).
- Combine the resulting LNS masks using majority voting.
- Create a table with information about the patient collective, which includes the following:
  - patient id
  - ground-truth N-stage
  - ground-truth list of pathological LNS
  - primary tumor location (R/L)
  - study name
  - series name
  - path to CT
  - path to PET
  - path to LNS masks

### Training:
- *train_lnlevel.py*: Training script for binary classification of LNS (TS 1).
- *train_serieslevel.py*: Training script AI-based N-staging (TS 2 and 3).

## Reference and Citation
Please refer to our work:

```
Sofija Engelson, Yannic Elser, Malte Maria Sieren, Jan Ehrhardt, Julia Andresen, Stefanie Schierholz, Tobias Keck, 
Daniel Drömann, Jörg Barkhausen, Heinz Handels, "Deep learning-based mediastinal lymph node assessment on PET/CT images 
without pixel-level annotations," J. Med. Imag. 13(1) 014507 (18 February 2026) 
https://doi.org/10.1117/1.JMI.13.1.014507
```

BibTex citation:
```
@article{10.1117/1.JMI.13.1.014507,
    author = {Sofija Engelson and Yannic Elser and Malte Maria Sieren and Jan Ehrhardt and Julia Andresen and Stefanie Schierholz and Tobias Keck and Daniel Dr{\"o}mann and J{\"o}rg Barkhausen and Heinz Handels},
    title = {{Deep learning-based mediastinal lymph node assessment on PET/CT images without pixel-level annotations}},
    volume = {13},
    journal = {Journal of Medical Imaging},
    number = {1},
    publisher = {SPIE},
    pages = {014507},
    keywords = {weakly supervised learning, mediastinal lymph nodes, N-staging, deep learning, image-level labels, priors, Lymph nodes, Education and training, Image segmentation, Tumors, Diagnostics, Image registration, Artificial intelligence, Deep learning, Machine learning, Prior knowledge},
    year = {2026},
    doi = {10.1117/1.JMI.13.1.014507},
    URL = {https://doi.org/10.1117/1.JMI.13.1.014507}
}
```

## License
See the LICENSE.txt file for license rights and limitations (CC BY-NC-ND 4.0).