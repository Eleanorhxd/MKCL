# MKCL
This is the implementation of [Medical Knowledge Enhanced Radiology Report Generation with Contrastive Learning]
## Requirements

- `python==3.7`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`
# Datasets
We use public IU X-Ray datasets in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.
## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.
