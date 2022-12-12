import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from modules import utils


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_id = os.path.join(self.image_dir, image_path[0])
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class BaseDatasetProgressive(Dataset):
    def __init__(self, args, tokenizer, split, transform=None, limit_length= None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.vocab_path = args.vocab_path

        self.transform = transform
        #image-2-text
        self.max_seq_length = args.max_seq_length
        #text_2_text
        self.src_max_seq_length = args.src_max_seq_length
        self.tgt_max_seq_length = args.tgt_max_seq_length
        self.split = split
        self.tokenizer = tokenizer

        self.clean_report = utils.clean_report_mimic_cxr

        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]

        if isinstance(split, list):
            self.examples = self.get_folds()
        else:
            self.examples = self.ann[self.split]

        if args.normal_abnormal is not None:
            self.examples = [example for example in self.examples if example['abnormal'] == args.normal_abnormal]

        if limit_length is not None:
            self.examples = self.examples[:limit_length]
        self.report_mode = args.report_mode

        for i in range(len(self.examples)):
            # image-2-text
            self.examples[i]['ids'] = tokenizer(self.examples[i][self.report_mode])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

            '''
            input_batch = ["<s>It <mask> retriever. My <mask> cute </s>", ... ]
            decoder_input_batch = ["</s><s>My dog is cute. It is a golden retriever", ...]
            labels_batch = ["<s>My dog is cute. It is a golden retriever</s>", ...]
            '''
            #text-2-text (bart)
            # input= f"<s>{utils.clean_report_mimic_cxr(self.examples[i][self.report_mode])}</s>"
            input = f"{self.clean_report(self.examples[i][self.report_mode])}"
            decoder_input = f"</s><s>{self.clean_report(self.examples[i]['report'])}"
            label = f"<s>{self.clean_report(self.examples[i]['report'])}</s>"

            self.examples[i]['input_bart'] = input
            self.examples[i]['decoder_input'] = decoder_input
            self.examples[i]['label'] = label

    def __len__(self):
        return len(self.examples)

    def get_folds(self):
        examples = []
        for x in self.ann:
            if str(x['fold']) in self.split:
                examples.append(x)
        return examples

class IuxrayDatasetProgressive(BaseDatasetProgressive):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')

        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)

        # text-2-text (bart)
        input_bart = example['input_bart']
        decoder_input_bart = example['decoder_input']
        label_bart = example['label']
        sample = (int(float(image_id)), image, report_ids, report_masks, seq_length, input_bart, decoder_input_bart, label_bart)
        return sample