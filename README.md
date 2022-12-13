## MKCL
 MKCL: Medical Knowledge Enhanced Radiology Report Generation with Contrastive Learning

## Overview
This repository contains code necessary to run MKCL model. 

## Requirements
- `torch:1.11.0+cu111`
- `python==3.7`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`
## Datasets
We use public IU X-Ray datasets in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

| Dataset | TRAIN | VAL | TEST |
| :------ | --------: | --------: | -----: |
| IMAGE# | 5,226 | 748 | 1,496 |
| REPORT# | 2,770 | 395 | 790 |
| PATIENT# | 2,770 | 395 | 790 |
| AVG.LEN | 37.56 | 36.78 | 33.62 |

## codes
models.py:This file contains the overall network architecture of MKCL.

utils:This file contains some defined functions.

main_train.py:This file trains the MKCL model.

main_test.py:This file tests the MKCL model.


## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

## Acknowledgment
This work is supported by grant from the Natural Science Foundation of China (No. 62072070)
