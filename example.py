import torch
import torchvision.transforms as transforms
from robustbench import load_model
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from attacks import CRAttack

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

test_loader = DataLoader(CIFAR10('./data/cifar10', train=False, transform=transforms.ToTensor()),
                         batch_size=10000, shuffle=False, num_workers=0)

X, y = [], []
for i, (x, y_) in enumerate(test_loader):
    X = x.to(device)
    y = y_.to(device)

model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf').to(device)
model = model.eval()
attacker = CRAttack(model, eps=8.0 / 255, log_path='cr_test.log')
attacker.run_standard_evaluation(X, y, bs=200)
