# Automatic-rsfMRI-noise-detection
The official implementation of *"Keun-Soo Heo, Dong-Hee Shin, Sheng-Che Hung, Weili Lin, Han Zhang, Dinggang Shen, Tae-Eui Kam, Deep Attentive Spatio-Temporal Feature Learning for Automatic Resting-State fMRI Denoising", NeuroImage, 2022.* [paper (IN PRESS)](https://doi.org/10.1016/j.neuroimage.2022.119127)

- Contact : KeunsooHeo (gjrmstn1440@korea.ac.kr)

## Prerequisites
- Pytorch 1.7++
- Python 3.7++

## Overview
<p align="center"><img src="https://user-images.githubusercontent.com/11483057/160057719-dc43941d-5350-4d75-a7b2-dadfb1b89c1a.png" width="80%" /></p>

## Usage

### 1. Prepare data
Dataset Human Connectome Project (HCP dataset) (Smith et al., 2013) and Whitehall II imaging study (WHII-MB6 and WHII-STD datasets) available in [FSL FIX](https://www.fmrib.ox.ac.uk/datasets/FIX-training) (Salimi-Khorshidi et al., 2014).

Download the dataset and put the data into "./data" directory. Then, adjust the directory name in Dataloader contained in "util.py".

### 2. Train & Test
```
Solver.py --dataset={DATASET}
```
Adjust hyperparmeter in config.py

### Acknowledgements
This work was supported by NIH grants (MH116225, MH100217, MH104324). It utilizes approaches developed by an NIH grant (1U01MH110274) and the ef- forts of the UNC/UMN Baby Connectome Project (BCP) Consortium. It was also supported by Institute of Information & communications Technology Plan- ning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No. 2019-0-00079, Artificial Intelligence Graduate School Program(Korea Univer- sity)), and the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (No. 2020R1C1C1013830, No. 2020R1A4A1018309).
