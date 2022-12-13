# MKCL
 MKCL: Medical Knowledge Enhanced Radiology Report Generation with Contrastive Learning

# Contribution
- `This paper proposes a model MKCL, which uses IU Medical Knowledge Graph (IU-MKG) to mine the relationship among medical findings and supply medical information for the accuracy of the radiologic report. We propose Knowledge Enhanced Attention mechanism (KEA) to combine IU-MKG and extracted CT visual features, which can alleviate textual data bias and generate high-quality reports.`
- `This paper employs supervised contrastive learning to significantly capture abnormal areas in radiologic images and ease unlabeled medical images. Furthermore, we introduce a supervised contrastive loss to tackle a relatively narrow text distribution with the reports and alleviate normal phenomena that are frequently described in medical reports.`
- `We have conducted extensive comprehensive experiments on the publicly available IU X-Ray dataset. The experimental results prove the proposed modelMKCL has a better performance of multiple state-of-the-art methods in image captioning and radiology report generation.`
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
