import logging
import re

import numpy as np
import cv2
import torch
import sklearn.metrics as sk

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


NUMBER_TOKEN = '<NUM>'
ALPHANUM_PATTERN = re.compile('^[a-zA-Z\\\.]+$')
NUM_PATTERN = re.compile('^[0-9]*$')

def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x

  # 37-74新加
NUMBER_TOKEN = '<NUM>'
ALPHANUM_PATTERN = re.compile('^[a-zA-Z\\\.]+$')
NUM_PATTERN = re.compile('^[0-9]*$')

def do_filter(tokens):
    new_tokens = []
    for token in tokens:
        if token != '.':
            if ALPHANUM_PATTERN.search(token) is not None:
                new_tokens.append(token)
            elif NUM_PATTERN.search(token) is not None:
                new_tokens.append(NUMBER_TOKEN)
    return new_tokens

def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x

# 新加
def clean_report_iu_xray(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                    replace('\\', '').replace("'", '').strip().lower())

    tokens = [sent_cleaner(" ".join(do_filter(sent.split()))) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def clean_report_mimic_cxr(report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(" ".join(do_filter(sent.split()))) for sent in report_cleaner(report) if
                  sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

def generate_heatmap(image, weights):
    image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

class Metrics(object):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.cond_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                            'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                            'Pleural Other', 'Fracture', 'Support Devices']
        for i in range(14):
            self.y_true.append([])
            self.y_pred.append([])

    def update(self, y_pred, y_true):
        # print(y_true.size())
        # print(y_pred.size())
        y_pred = torch.argmax(y_pred, dim=2)
        # print(y_pred.size())

        for i in range(len(self.cond_names)):
            self.y_true[i].append(y_true[:, i])
            self.y_pred[i].append(y_pred[:, i])

    def update_discrete(self, y_pred, y_true):
        # print(y_true.size())
        # print(y_pred.size())
        # print(y_pred.size())

        for i in range(len(self.cond_names)):
            self.y_true[i].append(y_true[:, i])
            self.y_pred[i].append(y_pred[:, i])


    def calculate_metrics(self):
        metrics = {}

        # Compute metrics for each condition
        for i in range(len(self.cond_names)):
            y_true = torch.cat(self.y_true[i])
            y_pred = torch.cat(self.y_pred[i])

            metrics['Positive Precision ' + self.cond_names[i]] = list(sk.precision_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]
            metrics['Positive Recall ' + self.cond_names[i]] = list(sk.recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]
            metrics['Positive F1 ' + self.cond_names[i]] = list(sk.f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0))[0]

            metrics['Uncertain Precision ' + self.cond_names[i]] = \
            list(sk.precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[0]
            metrics['Uncertain Recall ' + self.cond_names[i]] = \
            list(sk.recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[0]
            metrics['Uncertain F1 ' + self.cond_names[i]] = list(sk.f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0))[
                0]


        # Compute global metrics
        master_y_true = torch.cat([inner for outer in self.y_true for inner in outer])
        master_y_pred = torch.cat([inner for outer in self.y_pred for inner in outer])

        metrics['Micro Positive Precision'] = list(sk.precision_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]
        metrics['Micro Positive Recall'] = list(sk.recall_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]
        metrics['Micro Positive F1'] = list(sk.f1_score(master_y_true, master_y_pred, labels=[1], average=None, zero_division=0))[0]

        metrics['Micro Uncertain Precision'] = \
        list(sk.precision_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        metrics['Micro Uncertain Recall'] = \
        list(sk.recall_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]
        metrics['Micro Uncertain F1'] = list(sk.f1_score(master_y_true, master_y_pred, labels=[2], average=None, zero_division=0))[0]

        return metrics

class MetricsROC():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.y_true = []
        self.y_pred = []
        self.cond_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion', 'Lung Opacity',
                           'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                           'Pleural Other', 'Fracture', 'Support Devices']
        for i in range(14):
            self.y_true.append([])
            self.y_pred.append([])

    def update(self, y_pred, y_true):
        # print(y_true.size())
        # print(y_pred.size())
        print(y_pred[:, 0, :].size())
        # print(y_pred.size())

        for i in range(len(self.cond_names)):
            self.y_true[i].append(y_true[:, i])
            self.y_pred[i].append(y_pred[:, i, :])

    def calculate_metrics(self):
        metrics = {}

        # Compute global metrics
        master_y_true = torch.cat([inner for outer in self.y_true for inner in outer])
        master_y_pred = torch.cat([inner for outer in self.y_pred for inner in outer])

        metrics['Micro AUCROC'] = \
        list(sk.roc_auc_score(master_y_true, master_y_pred, labels=[1], average=None))[0]

        return metrics


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)