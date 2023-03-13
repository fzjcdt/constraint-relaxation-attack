import argparse
import gc
import time

import torch
import torchvision.transforms as transforms
from robustbench import load_model
from torch.utils.data import DataLoader

from attacks import CRAttack
from utils import Logger, get_model_ids

# argparse
parser = argparse.ArgumentParser(description='Evaluate models on various datasets.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'], help='dataset to evaluate')
parser.add_argument('--model-id', type=str, default=None, help='model id to evaluate')
parser.add_argument('--l-norm', type=str, default='Linf', choices=['Linf'], help='lp norm')
parser.add_argument('--batch-size', type=int, default=50, help='batch size')
parser.add_argument('--eps', type=str, default='8./255', help='epsilon')
parser.add_argument('--max-iter', type=int, default=150, help='max iteration')
parser.add_argument('--decay-steps', type=int, default=30, help='decay steps')
parser.add_argument('--target-numbers', type=int, default=3, help='target numbers')
parser.add_argument('--restart', type=int, default=5, help='restart numbers')
parser.add_argument('--seed', type=int, default=0, help='random seed')

argparse = parser.parse_args()
argparse.eps = eval(argparse.eps)

# logger
log_file = './log/' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.log'
logger = Logger(log_file)

if argparse.model_id is None:
    model_ids = get_model_ids(argparse.dataset, argparse.l_norm)
else:
    model_ids = [argparse.model_id]

# dataset
if argparse.dataset == 'cifar10':
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(
        root='./data/cifar10',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()])
    )
    data_loader = DataLoader(dataset, batch_size=argparse.batch_size, shuffle=False, num_workers=4)
elif argparse.dataset == 'cifar100':
    from torchvision.datasets import CIFAR100

    dataset = CIFAR100(
        root='./data/cifar100',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()])
    )
    data_loader = DataLoader(dataset, batch_size=argparse.batch_size, shuffle=False, num_workers=4)
elif argparse.dataset == 'imagenet':
    from robustbench.data import get_preprocessing, load_clean_dataset
    from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel


def margin(x, y):
    logits = model(x)
    u = torch.arange(x.shape[0])
    y_corr = logits[u, y].clone()
    logits[u, y] = -float('inf')
    y_others = logits.max(dim=-1)[0]

    return y_corr - y_others


device = 'cuda' if torch.cuda.is_available() else 'cpu'

for model_id in model_ids:
    logger.log('model: ' + model_id)
    model = load_model(model_name=model_id, dataset=argparse.dataset, threat_model=argparse.l_norm).to(device)
    model = model.eval()
    attacker = CRAttack(model, eps=argparse.eps, max_iter=argparse.max_iter, decay_steps=argparse.decay_steps,
                        target_numbers=argparse.target_numbers, restart=argparse.restart, seed=argparse.seed)
    success, total = 0, 0
    total_f, total_b = 0, 0
    if argparse.dataset in ['cifar10', 'cifar100']:
        for images, labels in data_loader:
            start = time.time()
            images, labels = images.to(device), labels.to(device)
            adv_img, f_num, b_num = attacker(images, labels)

            margin_min = margin(adv_img, labels)
            for m, fn, bn in zip(margin_min, f_num, b_num):
                total += 1
                if m < 0:
                    success += 1
                total_f += fn.item()
                total_b += bn.item()
                logger.log(
                    'success: {}, total: {}, margin: {}, forward: {}, backward: {}, total_f: {}, total_b: {}, time: {}'
                    .format(success, total, m.item(), fn.item(), bn.item(), total_f, total_b, time.time() - start))
        logger.log('attack success rate: {}'.format(success / total))
        logger.new_line()
    else:
        prepr = get_preprocessing(BenchmarkDataset.imagenet, ThreatModel.Linf, model_id, None)
        # Maybe you need to change the path of the dataset
        clean_x_test, clean_y_test = load_clean_dataset(BenchmarkDataset.imagenet, 5000, '/', prepr)

        for j in range(0, 5000, argparse.batch_size):
            start = time.time()
            images = clean_x_test[j:j + argparse.batch_size].to(device)
            labels = clean_y_test[j:j + argparse.batch_size].to(device)
            adv_img, f_num, b_num = attacker(images, labels)

            margin_min = margin(adv_img, labels)
            for m, fn, bn in zip(margin_min, f_num, b_num):
                total += 1
                if m < 0:
                    success += 1
                total_f += fn.item()
                total_b += bn.item()
                logger.log(
                    'success: {}, total: {}, margin: {}, forward: {}, backward: {}, total_f: {}, total_b: {}, time: {}'
                    .format(success, total, m.item(), fn.item(), bn.item(), total_f, total_b, time.time() - start))
        logger.log('attack success rate: {}'.format(success / total))

        del clean_x_test, clean_y_test
        gc.collect()
        torch.cuda.empty_cache()
