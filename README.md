# Constraint-Relaxation-Attack

Official implementation for the paper: **Efficient Robustness Evaluation via Constraint Relaxation**

## Overview

This repository contains the implementation of the Constraint Relaxation Attack (CRAttack), a novel approach for efficiently evaluating the adversarial robustness of deep neural networks. The attack relaxes constraints during the optimization process to find more effective adversarial examples.

## Environment Setup

- **OS**: Ubuntu 20.04.3
- **GPU**: NVIDIA Tesla V100
- **CUDA**: 11.4
- **Python**: 3.8.10
- **PyTorch**: 1.10.1
- **Torchvision**: 0.11.2

## Installation

```bash
git clone https://github.com/fzjcdt/constraint-relaxation-attack.git
cd constraint-relaxation-attack
pip install -r requirements.txt
```

## Quick Start Example

The following example demonstrates how to use CRAttack on a pre-trained model from RobustBench against the CIFAR-10 dataset:

```python
import torch
import torchvision.transforms as transforms
from robustbench import load_model
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from attacks import CRAttack

# Set device for computation
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load CIFAR-10 test dataset
test_loader = DataLoader(CIFAR10('./data/cifar10', train=False, transform=transforms.ToTensor()),
                         batch_size=10000, shuffle=False, num_workers=0)

# Extract data from loader
X, y = [], []
for i, (x, y_) in enumerate(test_loader):
    X = x.to(device)
    y = y_.to(device)

# Load pre-trained robust model from RobustBench
model = load_model(model_name='Wang2023Better_WRN-28-10', dataset='cifar10', threat_model='Linf').to(device)
model = model.eval()

# Initialize the CRAttack with epsilon=8/255 and specify log file
attacker = CRAttack(model, eps=8.0 / 255, log_path='cr_test.log')

# Run evaluation with batch size of 200
attacker.run_standard_evaluation(X, y, bs=200)
```

This example:
1. Sets up the computation device (GPU if available)
2. Loads the CIFAR-10 test dataset
3. Loads a pre-trained WideResNet model from RobustBench
4. Creates a CRAttack instance with perturbation bound ε=8/255
5. Runs the attack evaluation and logs results to [cr_test.log](./cr_test.log)

## Attack Commands

You can run attacks on various datasets using our provided scripts. The default setting evaluates all models specified in the `./model_ids/` directory for the given dataset.

```bash
# Attack models on CIFAR-10 with default epsilon (8/255)
python main.py --dataset 'cifar10'

# Attack models on CIFAR-100 with default epsilon (8/255)
python main.py --dataset 'cifar100'

# Attack models on ImageNet with epsilon=4/255
python main.py --dataset 'imagenet' --eps '4/255'
```

### Custom Model Evaluation

If you want to evaluate just one specific model, use the `--model_id` parameter:

```bash
python main.py --dataset 'cifar10' --model_id 'Wang2023Better_WRN-28-10'
```


## Full results
### CIFAR-10

#### Linf, eps=8/255
| <sub>#</sub> |                                          <sub>Model ID (Paper)</sub>                                           | <sub>Architecture</sub> | <sub>Best known robust accuracy</sub> | <sub> AutoAttack robust accuracy</sub> | <sub> AutoAttack forward number </sub> | <sub> AutoAttack backward number </sub> |<sub> CR attack robust accuracy</sub> | <sub> CR attack forward number </sub> | <sub> CR attack backward number </sub> |
|:---:|:--------------------------------------------------------------------------------------------------------------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  <sub>**1**</sub> |       <sub><sup>**[Bartoldson2024Adversarial_WRN-94-16](https://arxiv.org/abs/2404.09349)**</sup></sup>        | <sub>WideResNet-94-16</sub> |<sub>73.71% </sub> | <sub>73.71% </sub> | <sub>5810</sub> | <sub>1443</sub> | <sub>**73.59%**</sub> | <sub>108 (53.8×)</sub> | <sub> 65 (22.0×)</sub> | 
|  <sub>**2**</sub> |               <sub><sup>**[Amini2024MeanSparse](https://arxiv.org/abs/2406.05927)**</sup></sup>                | <sub>MeanSparse RaWideResNet-70-16</sub> |<sub>72.08% </sub> | <sub>72.08% </sub> | <sub>5674 </sub> | <sub>1399 </sub> | <sub>**71.85%**</sub> | <sub>104 (54.6×) </sub> | <sub>62 (22.6×)</sub> | 
|  <sub>**3**</sub> |        <sub><sup>**[Bartoldson2024Adversarial_WRN-82-8](https://arxiv.org/abs/2404.09349)**</sup></sup>        | <sub>WideResNet-82-8</sub> |<sub>71.59% </sub> | <sub>71.59% </sub> | <sub>5647</sub> | <sub>1393</sub> | <sub>**71.42%**</sub> | <sub>107 (52.8×)</sub> | <sub> 64 (21.8×)</sub> | 
|  <sub>**4**</sub> |                  <sub><sup>**[Peng2023Robust](https://arxiv.org/abs/2308.16258)**</sup></sup>                  | <sub>RaWideResNet-70-16</sub> |<sub>71.07% </sub> | <sub>71.07% </sub> | <sub>5637 </sub> | <sub>1390 </sub> | <sub>**70.99%**</sub> | <sub>104 (54.2×) </sub> | <sub>63 (22.1×) </sub> | 
|  <sub>**5**</sub> |             <sub><sup>**[Wang2023Better_WRN-70-16](https://arxiv.org/abs/2302.04638)**</sup></sup>             | <sub>WideResNet-70-16</sub> |<sub>70.69% </sub> | <sub>70.69% </sub> | <sub>5541 </sub> | <sub>1370 </sub> | <sub>**70.56%**</sub> | <sub>119 (46.6×) </sub> | <sub>73 (18.8×) </sub> | 
|  <sub>**6**</sub> |            <sub><sup>**[Cui2023Decoupled_WRN-28-10](https://arxiv.org/abs/2305.13948)**</sup></sup>            | <sub>WideResNet-28-10	</sub> |<sub>67.73% </sub> | <sub>67.73% </sub> | <sub>5344 </sub> | <sub>1322 </sub> | <sub>**67.55%**</sub> | <sub>104 (51.4×) </sub> | <sub>62 (21.3×)</sub> | 
|  <sub>**7**</sub> |               <sub><sup>**[Bai2023Improving_edm](https://arxiv.org/abs/2301.12554)**</sup></sup>               | <sub>ResNet-152 + WideResNet-70-16 + mixing network</sub> |<sub>68.06% </sub> | <sub>68.06% </sub> | <sub>5459 </sub> | <sub>1354 </sub> | <sub>**67.35%**</sub> | <sub>145 (37.7×) </sub> | <sub>92 (14.7×) </sub> | 
|  <sub>**8**</sub> |            <sub><sup>**[Wang2023Better_WRN-28-10](https://arxiv.org/abs/2302.046388)**</sup></sup>             | <sub>WideResNet-28-10</sub> |<sub>67.31% </sub> | <sub>67.31% </sub> | <sub>5338 </sub> | <sub>1322 </sub> | <sub>**67.21%**</sub> | <sub>118 (45.2×) </sub> | <sub>66 (18.4×) </sub> | 
|  <sub>**9**</sub> |       <sub><sup>**[Rebuffi2021Fixing_70_16_cutmix_extra](https://arxiv.org/abs/2103.01946)**</sup></sup>       | <sub>WideResNet-70-16</sub> |<sub>66.56% </sub> | <sub>66.58% </sub> | <sub>4998 </sub> | <sub>1243 </sub> | <sub>**66.51%**</sub> | <sub>111 (45.0×) </sub> | <sub>67 (18.6×) </sub> | 
|  <sub>**10**</sub> |        <sub><sup>**[Gowal2021Improving_70_16_ddpm_100m](https://arxiv.org/abs/2110.09468)**</sup></sup>        | <sub>WideResNet-70-16</sub> |<sub>66.10% </sub> | <sub>66.11% </sub> | <sub>5148 </sub> | <sub>1275 </sub> | <sub>**66.08%**</sub> | <sub>109 (47.2×) </sub> | <sub>66 (19.3×) </sub> | 
|  <sub>**11**</sub> |         <sub><sup>**[Gowal2020Uncovering_70_16_extra](https://arxiv.org/abs/2010.03593)**</sup></sup>          | <sub>WideResNet-70-16</sub> |<sub>65.87% </sub> | <sub>65.88% </sub> | <sub>5055 </sub> | <sub>1253 </sub> | <sub>**65.74%**</sub> | <sub>117 (43.2×) </sub> | <sub>71 (17.6×) </sub> | 
|  <sub>**12**</sub> |            <sub><sup>**[Huang2022Revisiting_WRN-A4](https://arxiv.org/abs/2212.11005)**</sup></sup>            | <sub>WideResNet-A4</sub> |<sub>65.79% </sub> | <sub>65.79% </sub> | <sub>5210 </sub> | <sub>1289 </sub> | <sub>**65.71%**</sub> | <sub>113 (46.1×) </sub> | <sub>69 (18.7×) </sub> | 
|  <sub>**13**</sub> |       <sub><sup>**[Rebuffi2021Fixing_106_16_cutmix_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup>       | <sub>WideResNet-106-16</sub> |<sub>64.58% </sub> | <sub>64.64% </sub> | <sub>4977 </sub> | <sub>1234 </sub> | <sub>**64.47%**</sub> | <sub>116 (42.9×) </sub> | <sub>71 (17.4×) </sub> | 
|  <sub>**14**</sub> |       <sub><sup>**[Rebuffi2021Fixing_70_16_cutmix_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup>        | <sub>WideResNet-70-16</sub> |<sub>64.20% </sub> | <sub>64.25% </sub> | <sub>4915 </sub> | <sub>1220 </sub> | <sub>**64.18%**</sub> | <sub>116 (42.4×) </sub> | <sub>71 (17.2×) </sub> | 
|  <sub>**15**</sub> |        <sub><sup>**[Gowal2021Improving_28_10_ddpm_100m](https://arxiv.org/abs/2110.09468)**</sup></sup>        | <sub>WideResNet-28-10</sub> |<sub>63.38% </sub> | <sub>63.44% </sub> | <sub>4928 </sub> | <sub>1221 </sub> | <sub>**63.36%**</sub> | <sub>108 (45.6×) </sub> | <sub>65 (18.8×) </sub> | 
|  <sub>**16**</sub> |         <sub><sup>**[Pang2022Robustness_WRN70_16](https://arxiv.org/pdf/2202.10103.pdf)**</sup></sup>          | <sub>WideResNet-70-16</sub> |<sub>63.35% </sub> | <sub>63.35% </sub> | <sub>4868 </sub> | <sub>1209 </sub> | <sub>**63.28%**</sub> | <sub>115 (42.3×) </sub> | <sub>70 (17.3×) </sub> | 
|  <sub>**17**</sub> |         <sub><sup>**[Rade2021Helper_extra](https://openreview.net/forum?id=BuD2LmNaU3a)**</sup></sup>          | <sub>WideResNet-34-10</sub> |<sub>62.83% </sub> | <sub>62.83% </sub> | <sub>4793 </sub> | <sub>1193 </sub> | <sub>**62.66%**</sub> | <sub>111 (43.2×) </sub> | <sub>68 (17.5×) </sub> | 
|  <sub>**18**</sub> |            <sub><sup>**[Sehwag2021Proxy_ResNest152](https://arxiv.org/abs/2104.09425)**</sup></sup>            | <sub>ResNest152</sub> |<sub>62.79% </sub> | <sub>62.79% </sub> | <sub>4758 </sub> | <sub>1181 </sub> | <sub>**62.53%**</sub> | <sub>110 (43.3×) </sub> | <sub>66 (17.9×) </sub> | 
|  <sub>**19**</sub> |         <sub><sup>**[Gowal2020Uncovering_28_10_extra](https://arxiv.org/abs/2010.03593)**</sup></sup>          | <sub>WideResNet-28-10</sub> |<sub>62.76% </sub> | <sub>62.80% </sub> | <sub>4729 </sub> | <sub>1176 </sub> | <sub>**62.71%**</sub> | <sub>121 (39.1×) </sub> | <sub>74 (15.9×) </sub> | 
|  <sub>**20**</sub> |              <sub><sup>**[Huang2021Exploring_ema](https://arxiv.org/abs/2110.03825)**</sup></sup>              | <sub>WideResNet-34-R</sub> |<sub>62.50% </sub> | <sub>62.54% </sub> | <sub>4824 </sub> | <sub>1200 </sub> | <sub>**62.49%**</sub> | <sub>106 (45.5×) </sub> | <sub>64 (18.8×) </sub> | 
|  <sub>**21**</sub> |                <sub><sup>**[Huang2021Exploring](https://arxiv.org/abs/2110.03825)**</sup></sup>                | <sub>WideResNet-34-R</sub> |<sub>**61.56%**</sub> | <sub>**61.56%**</sub> | <sub>4665 </sub> | <sub>1160 </sub> | <sub>61.59%</sub> | <sub>102 (45.7×) </sub> | <sub>61 (19.0×) </sub> | 
|  <sub>**22**</sub> |              <sub><sup>**[Dai2021Parameterizing](https://arxiv.org/abs/2110.05626)**</sup></sup>               | <sub>WideResNet-28-10-PSSiLU</sub> |<sub>61.55% </sub> | <sub>61.55% </sub> | <sub>4776 </sub> | <sub>1188 </sub> | <sub>**61.45%**</sub> | <sub>108 (44.2×) </sub> | <sub>66 (18.0×) </sub> | 
|  <sub>**23**</sub> |         <sub><sup>**[Pang2022Robustness_WRN28_10](https://arxiv.org/pdf/2202.10103.pdf)**</sup></sup>          | <sub>WideResNet-28-10</sub> |<sub>61.04% </sub> | <sub>61.04% </sub> | <sub>4774 </sub> | <sub>1186 </sub> | <sub>**60.91%**</sub> | <sub>118 (40.5×) </sub> | <sub>72 (16.5×) </sub> | 
|  <sub>**24**</sub> |          <sub><sup>**[Rade2021Helper_ddpm](https://openreview.net/forum?id=BuD2LmNaU3a)**</sup></sup>          | <sub>WideResNet-28-10</sub> |<sub>60.97% </sub> | <sub>60.97% </sub> | <sub>4727 </sub> | <sub>1174 </sub> | <sub>**60.80%**</sub> | <sub>109 (43.4×) </sub> | <sub>66 (17.8×) </sub> | 
|  <sub>**25**</sub> |       <sub><sup>**[Rebuffi2021Fixing_28_10_cutmix_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup>        | <sub>WideResNet-28-10</sub> |<sub>60.73% </sub> | <sub>60.75% </sub> | <sub>4728 </sub> | <sub>1175 </sub> | <sub>**60.68%**</sub> | <sub>117 (40.4×) </sub> | <sub>72 (16.3×) </sub> | 
|  <sub>**26**</sub> |             <sub><sup>**[Sridhar2021Robust_34_15](https://arxiv.org/abs/2106.02078)**</sup></sup>              | <sub>WideResNet-34-15</sub> |<sub>60.41% </sub> | <sub>60.41% </sub> | <sub>4664 </sub> | <sub>1159 </sub> | <sub>**60.32%**</sub> | <sub>132 (35.3×) </sub> | <sub>82 (14.1×) </sub> | 
|  <sub>**27**</sub> |                 <sub><sup>**[Sehwag2021Proxy](https://arxiv.org/abs/2104.09425)**</sup></sup>                  | <sub>WideResNet-34-10</sub> |<sub>60.27% </sub> | <sub>60.27% </sub> | <sub>4648 </sub> | <sub>1155 </sub> | <sub>**60.23%**</sub> | <sub>104 (44.7×) </sub> | <sub>62 (18.6×) </sub> | 
|  <sub>**28**</sub> |             <sub><sup>**[Wu2020Adversarial_extra](https://arxiv.org/abs/2004.05884)**</sup></sup>              | <sub>WideResNet-28-10</sub> |<sub>60.04% </sub> | <sub>60.04% </sub> | <sub>4667 </sub> | <sub>1162 </sub> | <sub>**59.98%**</sub> | <sub>120 (38.9×) </sub> | <sub>74 (15.7×) </sub> | 
|  <sub>**29**</sub> |                <sub><sup>**[Sridhar2021Robust](https://arxiv.org/abs/2106.02078)**</sup></sup>                 | <sub>WideResNet-28-10</sub> |<sub>59.66% </sub> | <sub>59.66% </sub> | <sub>4668 </sub> | <sub>1163 </sub> | <sub>**59.57%**</sub> | <sub>109 (42.8×) </sub> | <sub>66 (17.6×) </sub> | 
|  <sub>**30**</sub> |                <sub><sup>**[Zhang2020Geometry](https://arxiv.org/abs/2010.01736)**</sup></sup>                 | <sub>WideResNet-28-10</sub> |<sub>59.64% </sub> | <sub>59.64% </sub> | <sub>4641 </sub> | <sub>1160 </sub> | <sub>**59.12%**</sub> | <sub>158 (29.4×) </sub> | <sub>100 (11.6×) </sub> | 
|  <sub>**31**</sub> |               <sub><sup>**[Carmon2019Unlabeled](https://arxiv.org/abs/1905.13736)**</sup></sup>                | <sub>WideResNet-28-10</sub> |<sub>59.53% </sub> | <sub>59.53% </sub> | <sub>4558 </sub> | <sub>1137 </sub> | <sub>**59.46%**</sub> | <sub>109 (41.8×) </sub> | <sub>66 (17.2×) </sub> | 
|  <sub>**32**</sub> |         <sub><sup>**[Gowal2021Improving_R18_ddpm_100m](https://arxiv.org/abs/2110.09468)**</sup></sup>         | <sub>PreActResNet-18</sub> |<sub>58.5% </sub> | <sub>58.63% </sub> | <sub>4557 </sub> | <sub>1136 </sub> | <sub>**58.60%**</sub> | <sub>101 (45.1×) </sub> | <sub>61 (18.6×) </sub> | 
|  <sub>**33**</sub> |      <sub><sup>**[Addepalli2021Towards_WRN34](https://openreview.net/forum?id=SHB_znlW5G7)**</sup></sup>       | <sub>WideResNet-34-10</sub> |<sub>58.04% </sub> | <sub>58.04% </sub> | <sub>4452 </sub> | <sub>1113 </sub> | <sub>**58.00%**</sub> | <sub>135 (33.0×) </sub> | <sub>85 (13.1×) </sub> | 
|  <sub>**34**</sub> | <sub><sup>**[Addepalli2022Efficient_WRN_34_10](https://artofrobust.github.io/short_paper/31.pdf)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>57.81% </sub> | <sub>57.81% </sub> | <sub>4558 </sub> | <sub>1136 </sub> | <sub>**57.72%**</sub> | <sub>119 (38.3×) </sub> | <sub>74 (15.4×) </sub> | 
|  <sub>**35**</sub> |               <sub><sup>**[Chen2021LTD_WRN34_20](https://arxiv.org/abs/2111.02331)**</sup></sup>               | <sub>WideResNet-34-20</sub> |<sub>57.71% </sub> | <sub>57.71% </sub> | <sub>4572 </sub> | <sub>1139 </sub> | <sub>**57.68%**</sub> | <sub>132 (34.6×) </sub> | <sub>83 (13.7×) </sub> | 
|  <sub>**36**</sub> |       <sub><sup>**[Rade2021Helper_R18_extra](https://openreview.net/forum?id=BuD2LmNaU3a)**</sup></sup>        | <sub>PreActResNet-18</sub> |<sub>57.67% </sub> | <sub>57.67% </sub> | <sub>4573 </sub> | <sub>1140 </sub> | <sub>**57.49%**</sub> | <sub>113 (40.5×) </sub> | <sub>69 (16.5×) </sub> | 
|  <sub>**37**</sub> |               <sub><sup>**[Jia2022LAS-AT_70_16](https://arxiv.org/abs/2203.06616)**</sup></sup>                | <sub>WideResNet-70-16</sub> |<sub>57.61% </sub> | <sub>57.61% </sub> | <sub>4461 </sub> | <sub>1110 </sub> | <sub>**57.56%**</sub> | <sub>114 (39.1×) </sub> | <sub>70 (15.9×) </sub> | 
|  <sub>**38**</sub> |          <sub><sup>**[Debenedetti2022Light_XCiT-L12](https://arxiv.org/abs/2209.07399)**</sup></sup>           | <sub>XCiT-L12</sub> |<sub>**57.58%**</sub> | <sub>**57.58%**</sub> | <sub>4408 </sub> | <sub>1105 </sub> | <sub>57.63%</sub> | <sub>94 (46.9×) </sub> | <sub>56 (19.7×) </sub> | 
|  <sub>**39**</sub> |          <sub><sup>**[Debenedetti2022Light_XCiT-M12](https://arxiv.org/abs/2209.07399)**</sup></sup>           | <sub>XCiT-M12</sub> |<sub>**57.27%**</sub> | <sub>**57.27%**</sub> | <sub>4531 </sub> | <sub>1133 </sub> | <sub>57.28%</sub> | <sub>100 (45.3×) </sub> | <sub>60 (18.9×) </sub> | 
|  <sub>**40**</sub> |                 <sub><sup>**[Sehwag2020Hydra](https://arxiv.org/abs/2002.10509)**</sup></sup>                  | <sub>WideResNet-28-10</sub> |<sub>57.14% </sub> | <sub>57.14% </sub> | <sub>4495 </sub> | <sub>1120 </sub> | <sub>**57.12%**</sub> | <sub>106 (42.4×) </sub> | <sub>64 (17.5×) </sub> | 
|  <sub>**41**</sub> |            <sub><sup>**[Gowal2020Uncovering_70_16](https://arxiv.org/abs/2010.03593)**</sup></sup>             | <sub>WideResNet-70-16</sub> |<sub>57.14% </sub> | <sub>57.2% </sub> | <sub>4431 </sub> | <sub>1105 </sub> | <sub>**57.10%**</sub> | <sub>106 (41.8×) </sub> | <sub>64 (17.3×) </sub> | 
|  <sub>**42**</sub> |        <sub><sup>**[Rade2021Helper_R18_ddpm](https://openreview.net/forum?id=BuD2LmNaU3a)**</sup></sup>        | <sub>PreActResNet-18</sub> |<sub>57.09% </sub> | <sub>57.09% </sub> | <sub>4431 </sub> | <sub>1104 </sub> | <sub>**57.01%**</sub> | <sub>111 (39.9×) </sub> | <sub>68 (16.2×) </sub> | 
|  <sub>**43**</sub> |               <sub><sup>**[Chen2021LTD_WRN34_10](https://arxiv.org/abs/2111.02331)**</sup></sup>               | <sub>WideResNet-34-10</sub> |<sub>56.94% </sub> | <sub>56.94% </sub> | <sub>4386 </sub> | <sub>1095 </sub> | <sub>**56.87%**</sub> | <sub>125 (35.1×) </sub> | <sub>78 (14.0×) </sub> | 
|  <sub>**44**</sub> |            <sub><sup>**[Gowal2020Uncovering_34_20](https://arxiv.org/abs/2010.03593)**</sup></sup>             | <sub>WideResNet-34-20</sub> |<sub>56.82% </sub> | <sub>56.86% </sub> | <sub>4246 </sub> | <sub>1062 </sub> | <sub>**56.74%**</sub> | <sub>112 (37.9×) </sub> | <sub>68 (15.6×) </sub> | 
|  <sub>**45**</sub> |            <sub><sup>**[Rebuffi2021Fixing_R18_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup>            | <sub>PreActResNet-18</sub> |<sub>56.66% </sub> | <sub>56.66% </sub> | <sub>4272 </sub> | <sub>1064 </sub> | <sub>**56.57%**</sub> | <sub>113 (37.8×) </sub> | <sub>69 (15.4×) </sub> | 
|  <sub>**46**</sub> |           <sub><sup>**[Wang2020Improving](https://openreview.net/forum?id=rklOg6EFwS)**</sup></sup>            | <sub>WideResNet-28-10</sub> |<sub>**56.29%**</sub> | <sub>**56.29%**</sub> | <sub>4482 </sub> | <sub>1119 </sub> | <sub>56.43%</sub> | <sub>119 (37.7×) </sub> | <sub>73 (15.3×) </sub> | 
|  <sub>**47**</sub> |               <sub><sup>**[Jia2022LAS-AT_34_10](https://arxiv.org/abs/2203.06616)**</sup></sup>                | <sub>WideResNet-34-10</sub> |<sub>56.26% </sub> | <sub>56.26% </sub> | <sub>4335 </sub> | <sub>1080 </sub> | <sub>**56.19%**</sub> | <sub>116 (37.4×) </sub> | <sub>71 (15.2×) </sub> | 
|  <sub>**48**</sub> |                <sub><sup>**[Wu2020Adversarial](https://arxiv.org/abs/2004.05884)**</sup></sup>                 | <sub>WideResNet-34-10</sub> |<sub>56.17% </sub> | <sub>56.17% </sub> | <sub>4336 </sub> | <sub>1080 </sub> | <sub>**56.07%**</sub> | <sub>109 (39.8×) </sub> | <sub>66 (16.4×) </sub> | 
|  <sub>**49**</sub> |          <sub><sup>**[Debenedetti2022Light_XCiT-S12](https://arxiv.org/abs/2209.07399)**</sup></sup>           | <sub>XCiT-S12</sub> |<sub>56.14% </sub> | <sub>56.14% </sub> | <sub>4296 </sub> | <sub>1076 </sub> | <sub>**55.98%**</sub> | <sub>108 (39.8×) </sub> | <sub>65 (16.6×) </sub> | 
|  <sub>**50**</sub> |               <sub><sup>**[Sehwag2021Proxy_R18](https://arxiv.org/abs/2104.09425)**</sup></sup>                | <sub>ResNet-18</sub> |<sub>**55.54%**</sub> | <sub>**55.54%**</sub> | <sub>4244 </sub> | <sub>1060 </sub> | <sub>55.66%</sub> | <sub>104 (40.8×) </sub> | <sub>63 (16.8×) </sub> | 
|  <sub>**51**</sub> |                <sub><sup>**[Hendrycks2019Using](https://arxiv.org/abs/1901.09960)**</sup></sup>                | <sub>WideResNet-28-10</sub> |<sub>54.92% </sub> | <sub>54.92% </sub> | <sub>4323 </sub> | <sub>1079 </sub> | <sub>**54.88%**</sub> | <sub>104 (41.6×) </sub> | <sub>63 (17.1×) </sub> | 
|  <sub>**52**</sub> |                 <sub><sup>**[Pang2020Boosting](https://arxiv.org/abs/2002.08619)**</sup></sup>                 | <sub>WideResNet-34-20</sub> |<sub>53.74% </sub> | <sub>53.74% </sub> | <sub>4200 </sub> | <sub>1062 </sub> | <sub>**53.72%**</sub> | <sub>301 (14.0×) </sub> | <sub>203 (5.2×) </sub> | 
|  <sub>**53**</sub> |              <sub><sup>**[Cui2020Learnable_34_20](https://arxiv.org/abs/2011.11164)**</sup></sup>              | <sub>WideResNet-34-20</sub> |<sub>53.57% </sub> | <sub>53.57% </sub> | <sub>4141 </sub> | <sub>1039 </sub> | <sub>**53.11%**</sub> | <sub>90 (46.0×) </sub> | <sub>53 (19.6×) </sub> | 
|  <sub>**54**</sub> |                 <sub><sup>**[Zhang2020Attacks](https://arxiv.org/abs/2002.11242)**</sup></sup>                 | <sub>WideResNet-34-10</sub> |<sub>53.51% </sub> | <sub>53.51% </sub> | <sub>4136 </sub> | <sub>1035 </sub> | <sub>**53.44%**</sub> | <sub>118 (35.1×) </sub> | <sub>73 (14.2×) </sub> | 
|  <sub>**55**</sub> |               <sub><sup>**[Rice2020Overfitting](https://arxiv.org/abs/2002.11569)**</sup></sup>                | <sub>WideResNet-34-20</sub> |<sub>**53.42%**</sub> | <sub>**53.42%**</sub> | <sub>4138 </sub> | <sub>1037 </sub> | <sub>53.43%</sub> | <sub>106 (39.0×) </sub> | <sub>64 (16.2×) </sub> | 
|  <sub>**56**</sub> |                  <sub><sup>**[Huang2020Self](https://arxiv.org/abs/2002.10319)**</sup></sup>                   | <sub>WideResNet-34-10</sub> |<sub>53.34% </sub> | <sub>53.34% </sub> | <sub>4072 </sub> | <sub>1018 </sub> | <sub>**52.83%**</sub> | <sub>111 (36.7×) </sub> | <sub>68 (15.0×) </sub> | 
|  <sub>**57**</sub> |              <sub><sup>**[Zhang2019Theoretically](https://arxiv.org/abs/1901.08573)**</sup></sup>              | <sub>WideResNet-34-10</sub> |<sub>53.08% </sub> | <sub>53.08% </sub> | <sub>4089 </sub> | <sub>1024 </sub> | <sub>**52.45%**</sub> | <sub>97 (42.2×) </sub> | <sub>58 (17.7×) </sub> | 
|  <sub>**58**</sub> |              <sub><sup>**[Cui2020Learnable_34_10](https://arxiv.org/abs/2011.11164)**</sup></sup>              | <sub>WideResNet-34-10</sub> |<sub>52.86% </sub> | <sub>52.86% </sub> | <sub>4014 </sub> | <sub>1008 </sub> | <sub>**52.33%**</sub> | <sub>84 (47.8×) </sub> | <sub>49 (20.6×) </sub> | 
|  <sub>**59**</sub> |   <sub><sup>**[Addepalli2022Efficient_RN18](https://artofrobust.github.io/short_paper/31.pdf)**</sup></sup>    | <sub>ResNet-18</sub> |<sub>52.48% </sub> | <sub>52.48% </sub> | <sub>3968 </sub> | <sub>997 </sub> | <sub>**52.45%**</sub> | <sub>125 (31.7×) </sub> | <sub>78 (12.8×) </sub> | 
|  <sub>**60**</sub> |               <sub><sup>**[Chen2020Adversarial](https://arxiv.org/abs/2003.12862)**</sup></sup>                | <sub>ResNet-50 <br/> (3x ensemble)</sub> |<sub>51.56% </sub> | <sub>51.56% </sub> | <sub>4090 </sub> | <sub>1025 </sub> | <sub>**51.50%**</sub> | <sub>188 (21.8×) </sub> | <sub>123 (8.3×) </sub> | 
<!-- 
|  <sub>**55**</sub> | <sub><sup>**[Chen2020Efficient](https://arxiv.org/abs/2010.01278)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>51.12% </sub> | <sub>51.12% </sub> | <sub>3999 </sub> | <sub>1005 </sub> | <sub>**51.03%**</sub> | <sub>100 (40.0×) </sub> | <sub>60 (16.8×) </sub> | 
|  <sub>**56**</sub> | <sub><sup>**[Addepalli2021Towards_RN18](https://openreview.net/forum?id=SHB_znlW5G7)**</sup></sup> | <sub>ResNet-18</sub> |<sub>51.06% </sub> | <sub>51.06% </sub> | <sub>3901 </sub> | <sub>978 </sub> | <sub>**51.05%**</sub> | <sub>127 (30.7×) </sub> | <sub>79 (12.4×) </sub> | 
|  <sub>**57**</sub> | <sub><sup>**[Sitawarin2020Improving](https://arxiv.org/abs/2003.09347)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>50.72% </sub> | <sub>50.72% </sub> | <sub>3981 </sub> | <sub>999 </sub> | <sub>**50.69%**</sub> | <sub>89 (44.7×) </sub> | <sub>53 (18.8×) </sub> | 
|  <sub>**58**</sub> | <sub><sup>**[Engstrom2019Robustness](https://github.com/MadryLab/robustness)**</sup></sup> | <sub>ResNet-50</sub> |<sub>**49.25%**</sub> | <sub>**49.25%**</sub> | <sub>4016 </sub> | <sub>1010 </sub> | <sub>49.44%</sub> | <sub>90 (44.6×) </sub> | <sub>53 (19.1×) </sub> | 
|  <sub>**59**</sub> | <sub><sup>**[Zhang2019You](https://arxiv.org/abs/1905.00877)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>**44.83%**</sub> | <sub>**44.83%**</sub> | <sub>3410 </sub> | <sub>867 </sub> | <sub>44.88%</sub> | <sub>80 (42.6×) </sub> | <sub>46 (18.8×) </sub> | 
|  <sub>**60**</sub> | <sub><sup>**[Andriushchenko2020Understanding](https://arxiv.org/abs/2007.02617)**</sup></sup> | <sub>PreActResNet-18</sub> |<sub>**43.93%**</sub> | <sub>**43.93%**</sub> | <sub>3541 </sub> | <sub>891 </sub> | <sub>44.03%</sub> | <sub>93 (38.1×) </sub> | <sub>55 (16.2×) </sub> | 
|  <sub>**61**</sub> | <sub><sup>**[Wong2020Fast](https://arxiv.org/abs/2001.03994)**</sup></sup> | <sub>PreActResNet-18</sub> |<sub>43.21% </sub> | <sub>43.21% </sub> | <sub>3361 </sub> | <sub>854 </sub> | <sub>**43.20%**</sub> | <sub>88 (38.2×) </sub> | <sub>52 (16.4×) </sub> | 
-->



### CIFAR-100

#### Linf, eps=8/255
|   <sub>#</sub>   | <sub>Model ID (Paper)</sub> | <sub>Architecture</sub> | <sub>Best known robust accuracy</sub> | <sub> AutoAttack robust accuracy</sub> | <sub> AutoAttack forward number </sub> | <sub> AutoAttack backward number </sub> | <sub> CR attack robust accuracy</sub> | <sub> CR attack forward number </sub> | <sub> CR attack backward number </sub> |
|:----------------:|:--:|:---:|:---:|:---:|:---:|:---:|:-------------------------------------:|:-------------------------------------:|:--------------------------------------:|
|  <sub>**1**</sub> | <sub><sup>**[Wang2023Better_WRN-70-16](https://arxiv.org/abs/2302.04638)**</sup></sup> | <sub>WideResNet-70-16</sub> |<sub>42.67% </sub> | <sub>42.67% </sub> | <sub>3351 </sub> | <sub>844 </sub> |         <sub>**42.57%**</sub>         |        <sub>119 (28.2×) </sub>        |         <sub>71 (11.9×) </sub>         | 
| <sub>**2**</sub> | <sub><sup>**[ Cui2023Decoupled_WRN-28-10 ]( https://arxiv.org/abs/2305.13948 )**</sup></sup> | <sub> WideResNet-28-10 </sub> | <sub> 39.18 %</sub> | <sub> 39.18 %</sub> | <sub> 3009 </sub> | <sub> 764 </sub> |         <sub>**39.13%**</sub>         |       <sub> 96 (31.3 ×) </sub>       |        <sub> 54 (14.1×) </sub>         |
|  <sub>**3**</sub> | <sub><sup>**[Wang2023Better_WRN-28-10](https://arxiv.org/abs/2302.046388)**</sup></sup> | <sub>WideResNet-28-10</sub> |<sub>38.83% </sub> | <sub>38.83% </sub> | <sub>2959 </sub> | <sub>749 </sub> |         <sub>**38.67%**</sub>         |        <sub>116 (25.5×) </sub>        |         <sub>69 (10.9×) </sub>         | 
| <sub>**4**</sub> | <sub><sup>**[ Bai2023Improving_edm ]( https://arxiv.org/abs/2301.12554 )**</sup></sup> | <sub> ResNet-152+WideResNet-70-16+mixing-network </sub> | <sub> 38.72 %</sub> | <sub> 38.72 %</sub> | <sub> 3082 </sub> | <sub> 789 </sub> |         <sub>**38.63%**</sub>         |        <sub>94 (32.8×) </sub>         |        <sub> 54 (14.6×) </sub>         |
|  <sub>**5**</sub> | <sub><sup>**[Gowal2020Uncovering_extra](https://arxiv.org/abs/2010.03593)**</sup></sup> | <sub>WideResNet-70-16</sub> |<sub>36.88% </sub> | <sub>36.88% </sub> | <sub>2695 </sub> | <sub>686 </sub> |         <sub>**36.87%**</sub>         |        <sub>95 (28.4×) </sub>         |         <sub>54 (12.7×) </sub>         | 
|  <sub>**6**</sub> | <sub><sup>**[Debenedetti2022Light_XCiT-L12](https://arxiv.org/abs/2209.07399)**</sup></sup> | <sub>XCiT-L12</sub> |<sub>35.08% </sub> | <sub>35.08% </sub> | <sub>2665 </sub> | <sub>680 </sub> |         <sub>**34.96%**</sub>         |        <sub>99 (26.9×) </sub>         |         <sub>57 (11.9×) </sub>         | 
| <sub>**7**</sub> | <sub><sup>**[ Bai2023Improving_trades ]( https://arxiv.org/abs/2301.12554 )**</sup></sup> | <sub> ResNet-152+WideResNet-70-16+mixing-network </sub> | <sub> 35.15 %</sub> | <sub> 35.15 %</sub> | <sub> 2679 </sub> | <sub> 693 </sub> |         <sub>**34.80%**</sub>         |        <sub>85 (31.5×) </sub>         |        <sub> 47 (14.7×) </sub>         |
|  <sub>**8**</sub> | <sub><sup>**[Rebuffi2021Fixing_70_16_cutmix_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup> | <sub>WideResNet-70-16</sub> |<sub>34.64% </sub> | <sub>34.64% </sub> | <sub>2594 </sub> | <sub>658 </sub> |         <sub>**34.56%**</sub>         |        <sub>99 (26.2×) </sub>         |         <sub>57 (11.5×) </sub>         | 
|  <sub>**9**</sub> | <sub><sup>**[Debenedetti2022Light_XCiT-M12](https://arxiv.org/abs/2209.07399)**</sup></sup> | <sub>XCiT-M12</sub> |<sub>34.21% </sub> | <sub>34.21% </sub> | <sub>2724 </sub> | <sub>692 </sub> |         <sub>**34.09%**</sub>         |        <sub>103 (26.4×) </sub>        |         <sub>61 (11.3×) </sub>         | 
|  <sub>**10**</sub> | <sub><sup>**[Pang2022Robustness_WRN70_16](https://arxiv.org/pdf/2202.10103.pdf)**</sup></sup> | <sub>WideResNet-70-16</sub> |<sub>33.05% </sub> | <sub>33.05% </sub> | <sub>2595 </sub> | <sub>659 </sub> |         <sub>**32.97%**</sub>         |        <sub>96 (27.0×) </sub>         |         <sub>55 (12.0×) </sub>         | 
| <sub>**11**</sub> | <sub><sup>**[ Cui2023Decoupled_WRN-34-10_autoaug ]( https://arxiv.org/abs/2305.13948 )**</sup></sup> | <sub> WideResNet-34-10 </sub> | <sub> 32.52 %</sub> | <sub> 32.52 %</sub> | <sub> 2597 </sub> | <sub> 661 </sub> |        <sub>**32.46%**</sub>        |        <sub> 81 (32.1×) </sub>        |        <sub> 45 (14.7×) </sub>         |
|  <sub>**12**</sub> | <sub><sup>**[Debenedetti2022Light_XCiT-S12](https://arxiv.org/abs/2209.07399)**</sup></sup> | <sub>XCiT-S12</sub> |<sub>32.19% </sub> | <sub>32.19% </sub> | <sub>2522 </sub> | <sub>645 </sub> |         <sub>**32.10%**</sub>         |        <sub>104 (24.2×) </sub>        |         <sub>61 (10.6×) </sub>         | 
|  <sub>**13**</sub> | <sub><sup>**[Rebuffi2021Fixing_28_10_cutmix_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup> | <sub>WideResNet-28-10</sub> |<sub>32.06% </sub> | <sub>32.06% </sub> | <sub>2470 </sub> | <sub>628 </sub> |         <sub>**31.95%**</sub>         |        <sub>95 (26.0×) </sub>         |         <sub>54 (11.6×) </sub>         | 
|  <sub>**14**</sub> | <sub><sup>**[Jia2022LAS-AT_34_20](https://arxiv.org/abs/2203.06616)**</sup></sup> | <sub>WideResNet-34-20</sub> |<sub>**31.91%**</sub> | <sub>**31.91%**</sub> | <sub>2554 </sub> | <sub>653 </sub> |           <sub>31.92%</sub>           |        <sub>88 (29.0×) </sub>         |         <sub>49 (13.3×) </sub>         | 
|  <sub>**15**</sub> | <sub><sup>**[Addepalli2022Efficient_WRN_34_10](https://artofrobust.github.io/short_paper/31.pdf)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>31.85% </sub> | <sub>31.85% </sub> | <sub>2363 </sub> | <sub>604 </sub> |         <sub>**31.81%**</sub>         |        <sub>98 (24.1×) </sub>         |         <sub>56 (10.8×) </sub>         | 
| <sub>**16**</sub> | <sub><sup>**[ Cui2023Decoupled_WRN-34-10 ]( https://arxiv.org/abs/2305.13948 )**</sup></sup> | <sub> WideResNet-34-10 </sub> | <sub> 31.65 %</sub> | <sub> 31.65 %</sub> | <sub> 2518 </sub> | <sub> 640 </sub> |        <sub>**31.62%**</sub>        |        <sub> 82 (30.7×) </sub>        |        <sub> 45 (14.2×) </sub>         |
|  <sub>**17**</sub> | <sub><sup>**[Sehwag2021Proxy](https://arxiv.org/abs/2104.09425)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>31.15% </sub> | <sub>31.15% </sub> | <sub>2506 </sub> | <sub>641 </sub> |         <sub>**31.14%**</sub>         |        <sub>96 (26.1×) </sub>         |         <sub>55 (11.7×) </sub>         | 
| <sub>**18**</sub> | <sub><sup>**[ Chen2024Data_WRN_34_10 ]( https://doi.org/10.1016/j.patcog.2024.110394 )**</sup></sup> | <sub> WideResNet-34-10 </sub> | <sub> 31.13 %</sub> | <sub> 31.13 %</sub> | <sub> 2456 </sub> | <sub> 626 </sub> |        <sub>**31.12%**</sub>        |        <sub>85 (28.9×) </sub>         |        <sub> 47 (13.3×) </sub>         |
|  <sub>**19**</sub> | <sub><sup>**[Cui2020Learnable_34_10_LBGAT9_eps_8_255](https://arxiv.org/abs/2011.11164)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>31.20% </sub> | <sub>31.20% </sub> | <sub>2604 </sub> | <sub>667 </sub> |         <sub>**31.10%**</sub>         |        <sub>113 (23.0×) </sub>        |         <sub>67 (10.0×) </sub>         | 
|  <sub>**20**</sub> | <sub><sup>**[Pang2022Robustness_WRN28_10](https://arxiv.org/pdf/2202.10103.pdf)**</sup></sup> | <sub>WideResNet-28-10</sub> |<sub>31.08% </sub> | <sub>31.08% </sub> | <sub>2518 </sub> | <sub>640 </sub> |         <sub>**31.03%**</sub>         |        <sub>93 (27.1×) </sub>         |         <sub>53 (12.1×) </sub>         | 
|  <sub>**21**</sub> | <sub><sup>**[Jia2022LAS-AT_34_10](https://arxiv.org/abs/2203.06616)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>**30.77%**</sub> | <sub>**30.77%**</sub> | <sub>2469 </sub> | <sub>627 </sub> |         <sub>**30.77%**</sub>         |        <sub>86 (28.7×) </sub>         |         <sub>48 (13.1×) </sub>         | 
|  <sub>**22**</sub> | <sub><sup>**[Chen2021LTD_WRN34_10](https://arxiv.org/abs/2111.02331)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>30.59% </sub> | <sub>30.59% </sub> | <sub>2333 </sub> | <sub>598 </sub> |         <sub>**30.58%**</sub>         |        <sub>96 (24.3×) </sub>         |         <sub>55 (10.9×) </sub>         | 
|  <sub>**23**</sub> | <sub><sup>**[Addepalli2021Towards_WRN34](https://openreview.net/forum?id=SHB_znlW5G7)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>30.35% </sub> | <sub>30.35% </sub> | <sub>2566 </sub> | <sub>653 </sub> |         <sub>**30.23%**</sub>         |        <sub>94 (27.3×) </sub>         |         <sub>54 (12.1×) </sub>         | 
|  <sub>**24**</sub> | <sub><sup>**[Cui2020Learnable_34_20_LBGAT6](https://arxiv.org/abs/2011.11164)**</sup></sup> | <sub>WideResNet-34-20</sub> |<sub>30.20% </sub> | <sub>30.20% </sub> | <sub>2379 </sub> | <sub>609 </sub> |         <sub>**29.87%**</sub>         |        <sub>93 (25.6×) </sub>         |         <sub>53 (11.5×) </sub>         | 
|  <sub>**25**</sub> | <sub><sup>**[Gowal2020Uncovering](https://arxiv.org/abs/2010.03593)**</sup></sup> | <sub>WideResNet-70-16</sub> |<sub>30.03% </sub> | <sub>30.03% </sub> | <sub>2389 </sub> | <sub>607 </sub> |         <sub>**29.99%**</sub>         |        <sub>85 (28.1×) </sub>         |         <sub>48 (12.6×) </sub>         | 
|  <sub>**26**</sub> | <sub><sup>**[Cui2020Learnable_34_10_LBGAT6](https://arxiv.org/abs/2011.11164)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>29.33% </sub> | <sub>29.33% </sub> | <sub>2377 </sub> | <sub>607 </sub> |         <sub>**28.87%**</sub>         |        <sub>102 (23.3×) </sub>        |         <sub>60 (10.1×) </sub>         | 
|  <sub>**27**</sub> | <sub><sup>**[Rade2021Helper_R18_ddpm](https://openreview.net/forum?id=BuD2LmNaU3a)**</sup></sup> | <sub>PreActResNet-18</sub> |<sub>28.88% </sub> | <sub>28.88% </sub> | <sub>2235 </sub> | <sub>571 </sub> |         <sub>**28.80%**</sub>         |        <sub>85 (26.3×) </sub>         |         <sub>48 (11.9×) </sub>         | 
|  <sub>**28**</sub> | <sub><sup>**[Wu2020Adversarial](https://arxiv.org/abs/2004.05884)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>28.86% </sub> | <sub>28.86% </sub> | <sub>2345 </sub> | <sub>598 </sub> |         <sub>**28.85%**</sub>         |        <sub>84 (27.9×) </sub>         |         <sub>47 (12.7×) </sub>         | 
|  <sub>**29**</sub> | <sub><sup>**[Rebuffi2021Fixing_R18_ddpm](https://arxiv.org/abs/2103.01946)**</sup></sup> | <sub>PreActResNet-18</sub> |<sub>28.50% </sub> | <sub>28.50% </sub> | <sub>2259 </sub> | <sub>572 </sub> |         <sub>**28.37%**</sub>         |        <sub>89 (25.4×) </sub>         |         <sub>50 (11.4×) </sub>         | 
|  <sub>**30**</sub> | <sub><sup>**[Hendrycks2019Using](https://arxiv.org/abs/1901.09960)**</sup></sup> | <sub>WideResNet-28-10</sub> |<sub>**28.42%**</sub> | <sub>**28.42%**</sub> | <sub>2256 </sub> | <sub>580 </sub> |           <sub>28.47%</sub>           |        <sub>91 (24.8×) </sub>         |         <sub>52 (11.2×) </sub>         | 
<!--
|  <sub>**25**</sub> | <sub><sup>**[Addepalli2022Efficient_RN18](https://artofrobust.github.io/short_paper/31.pdf)**</sup></sup> | <sub>ResNet-18</sub> |<sub>**27.67%**</sub> | <sub>**27.67%**</sub> | <sub>2335 </sub> | <sub>600 </sub> |         <sub>**27.67%**</sub>         |        <sub>95 (24.6×) </sub>         |         <sub>55 (10.9×) </sub>         | 
|  <sub>**26**</sub> | <sub><sup>**[Cui2020Learnable_34_10_LBGAT0](https://arxiv.org/abs/2011.11164)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>27.16% </sub> | <sub>27.16% </sub> | <sub>2231 </sub> | <sub>580 </sub> |         <sub>**26.61%**</sub>         |        <sub>68 (32.8×) </sub>         |         <sub>36 (16.1×) </sub>         | 
|  <sub>**27**</sub> | <sub><sup>**[Addepalli2021Towards_PARN18](https://openreview.net/forum?id=SHB_znlW5G7)**</sup></sup> | <sub>PreActResNet-18</sub> |<sub>27.14% </sub> | <sub>27.14% </sub> | <sub>2099 </sub> | <sub>542 </sub> |         <sub>**27.12%**</sub>         |        <sub>91 (23.1×) </sub>         |         <sub>52 (10.4×) </sub>         | 
|  <sub>**28**</sub> | <sub><sup>**[Chen2020Efficient](https://arxiv.org/abs/2010.01278)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>**26.94%**</sub> | <sub>**26.94%**</sub> | <sub>2142 </sub> | <sub>550 </sub> |           <sub>27.03%</sub>           |        <sub>81 (26.4×) </sub>         |         <sub>45 (12.2×) </sub>         | 
|  <sub>**29**</sub> | <sub><sup>**[Sitawarin2020Improving](https://arxiv.org/abs/2003.09347)**</sup></sup> | <sub>WideResNet-34-10</sub> |<sub>24.57% </sub> | <sub>24.57% </sub> | <sub>1942 </sub> | <sub>504 </sub> |         <sub>**24.54%**</sub>         |        <sub>66 (29.4×) </sub>         |         <sub>35 (14.4×) </sub>         | 
|  <sub>**30**</sub> | <sub><sup>**[Rice2020Overfitting](https://arxiv.org/abs/2002.11569)**</sup></sup> | <sub>PreActResNet-18</sub> |<sub>18.95% </sub> | <sub>18.95% </sub> | <sub>1623 </sub> | <sub>423 </sub> |         <sub>**18.90%**</sub>         |        <sub>58 (28.0×) </sub>         |         <sub>29 (14.6×) </sub>         | 
-->



### ImageNet

#### Linf, eps=4/255
|   <sub>#</sub>    | <sub>Model ID (Paper)</sub> | <sub>Architecture</sub> | <sub>Best known robust accuracy</sub> | <sub> AutoAttack robust accuracy</sub> | <sub> AutoAttack forward number </sub> | <sub> AutoAttack backward number </sub> | <sub> CR attack robust accuracy</sub> | <sub> CR attack forward number </sub> | <sub> CR attack backward number </sub> |
|:-----------------:|:--:|:---:|:-------------------------------------:|:--------------------------------------:|:--------------------------------------:|:---:|:-------------------------------------:|:---:|:---:|
| <sub>**1**</sub>  | <sub><sup>**[Amini2024MeanSparse](https://arxiv.org/abs/2406.05927)**</sup></sup> | <sub>MeanSparse ConvNeXt-L</sub> |        <sub>**59.64%**</sub>         |         <sub>**59.64%**</sub>          |         <sub>5075</sub>          | <sub>1248</sub> |            <sub>59.70</sub>            | <sub>165(30.8×) </sub> | <sub>99(12.6×) </sub> |
| <sub>**2**</sub>  | <sub><sup>**[Liu2023Comprehensive_Swin-L](https://arxiv.org/abs/2302.14301)**</sup></sup> | <sub>Swin-L</sub> |           <sub>59.56%</sub>           |           <sub>59.56%</sub>            |            <sub>4918</sub>             | <sub>1212</sub> |         <sub>**59.46%**</sub>         | <sub>165(29.8×) </sub> | <sub>99(12.2×) </sub> |
| <sub>**3**</sub>  | <sub><sup>**[Liu2023Comprehensive_ConvNeXt-L](https://arxiv.org/abs/2302.14301)**</sup></sup> | <sub>ConvNeXt-L</sub> |         <sub>**58.48%**</sub>         |         <sub>**58.48%**</sub>          |            <sub>5013</sub>             | <sub>1235</sub> |           <sub>58.50%</sub>           | <sub>161(31.1×) </sub> | <sub>96(12.9×) </sub> |
| <sub>**4**</sub>  | <sub><sup>**[Singh2023Revisiting_ConvNeXt-L-ConvStem](https://arxiv.org/abs/2303.01870)**</sup></sup> | <sub>ConvNeXt-L+ConvStem</sub> |           <sub>57.7%</sub>            |            <sub>57.7%</sub>            |            <sub>5106</sub>             | <sub>1257</sub> |         <sub>**57.62%**</sub>         | <sub>158(32.3×) </sub> | <sub>94(13.4×) </sub> |
| <sub>**5**</sub>  | <sub><sup>**[Liu2023Comprehensive_Swin-B](https://arxiv.org/abs/2302.14301)**</sup></sup> | <sub>Swin-B</sub> |           <sub>56.16%</sub>           |           <sub>56.16%</sub>            |            <sub>4795</sub>             | <sub>1183</sub> |         <sub>**56.1%**</sub>          | <sub>156(30.7×) </sub> | <sub>93(12.7×) </sub> |
| <sub>**6**</sub>  | <sub><sup>**[Singh2023Revisiting_ConvNeXt-B-ConvStem](https://arxiv.org/abs/2303.01870)**</sup></sup> | <sub>ConvNeXt-B+ConvStem</sub> |           <sub>56.14%</sub>           |           <sub>56.14%</sub>            |            <sub>4946</sub>             | <sub>1215</sub> |         <sub>**56.04%**</sub>         | <sub>157(31.5×) </sub> | <sub>93(13.1×) </sub> |
| <sub>**7**</sub>  | <sub><sup>**[Liu2023Comprehensive_ConvNeXt-B](https://arxiv.org/abs/2302.14301)**</sup></sup> | <sub>ConvNeXt-B</sub> |           <sub>55.82%</sub>           |           <sub>55.82%</sub>            |            <sub>4699</sub>             | <sub>1159</sub> |         <sub>**55.8%**</sub>          | <sub>153(30.7×) </sub> | <sub>91(12.7×) </sub> |
| <sub>**8**</sub>  | <sub><sup>**[Singh2023Revisiting_ViT-B-ConvStem](https://arxiv.org/abs/2303.01870)**</sup></sup> | <sub>ViT-B+ConvStem</sub> |           <sub>54.66%</sub>           |           <sub>54.66%</sub>            |            <sub>4638</sub>             | <sub>1145</sub> |         <sub>**54.6%**</sub>          | <sub>154(30.1×) </sub> | <sub>92(12.4×) </sub> |
| <sub>**9**</sub>  | <sub><sup>**[Singh2023Revisiting_ConvNeXt-S-ConvStem](https://arxiv.org/abs/2303.01870)**</sup></sup> | <sub>ConvNeXt-S+ConvStem</sub> |           <sub>52.42%</sub>           |           <sub>52.42%</sub>            |            <sub>4514</sub>             | <sub>1116</sub> |         <sub>**52.28%**</sub>         | <sub>148(30.5×) </sub> | <sub>88(12.7×) </sub> |
| <sub>**10**</sub> | <sub><sup>**[Singh2023Revisiting_ConvNeXt-T-ConvStem](https://arxiv.org/abs/2303.01870)**</sup></sup> | <sub>ConvNeXt-T+ConvStem</sub> |           <sub>49.46%</sub>           |           <sub>49.46%</sub>            |            <sub>4416</sub>             | <sub>1093</sub> |         <sub>**49.46%**</sub>         | <sub>143(30.9×) </sub> | <sub>84(13.0×) </sub> |
| <sub>**11**</sub> | <sub><sup>**[Peng2023Robust](https://arxiv.org/abs/2308.16258)**</sup></sup> | <sub>RaWideResNet-101-2</sub> |           <sub>48.94%</sub>           |           <sub>48.94%</sub>            |            <sub>4140</sub>             | <sub>1028</sub> |         <sub>**48.84%**</sub>         | <sub>140(29.6×) </sub> | <sub>82(12.5×) </sub> |
| <sub>**12**</sub> | <sub><sup>**[Singh2023Revisiting_ViT-S-ConvStem](https://arxiv.org/abs/2303.01870)**</sup></sup> | <sub>ViT-S+ConvStem</sub> |           <sub>48.08%</sub>           |           <sub>48.08%</sub>            |            <sub>4198</sub>             | <sub>1038</sub> |         <sub>**48.04%**</sub>         | <sub>142(29.6×) </sub> | <sub>84(12.4×) </sub> |
| <sub>**13**</sub> | <sub><sup>**[Debenedetti2022Light_XCiT-L12](https://arxiv.org/abs/2209.07399)**</sup></sup> | <sub>XCiT-L12</sub> |          <sub>47.60% </sub>           |           <sub>47.60% </sub>           |            <sub>3863 </sub>            | <sub>964 </sub> |         <sub>**47.52%**</sub>         | <sub>110 (35.1×) </sub> | <sub>64 (15.1×) </sub> | 
| <sub>**14**</sub> | <sub><sup>**[Debenedetti2022Light_XCiT-M12](https://arxiv.org/abs/2209.07399)**</sup></sup> | <sub>XCiT-M12</sub> |          <sub>45.24% </sub>           |           <sub>45.24% </sub>           |            <sub>3751 </sub>            | <sub>935 </sub> |         <sub>**45.20%**</sub>         | <sub>110 (34.1×) </sub> | <sub>65 (14.4×) </sub> | 
| <sub>**15**</sub> | <sub><sup>**[Debenedetti2022Light_XCiT-S12](https://arxiv.org/abs/2209.07399)**</sup></sup> | <sub>XCiT-S12</sub> |          <sub>41.78% </sub>           |           <sub>41.78% </sub>           |            <sub>3464 </sub>            | <sub>874 </sub> |         <sub>**41.64%**</sub>         | <sub>105 (33.0×) </sub> | <sub>61 (14.3×) </sub> | 
<!--
|  <sub>**4**</sub> | <sub><sup>**[Salman2020Do_50_2](https://arxiv.org/abs/2007.08489)**</sup></sup> | <sub>WideResNet-50-2</sub> |<sub>**38.14%**</sub> | <sub>**38.14%**</sub> | <sub>3336 </sub> | <sub>841 </sub> | <sub>**38.14%**</sub> | <sub>95 (35.1×) </sub> | <sub>54 (15.6×) </sub> | 
|  <sub>**5**</sub> | <sub><sup>**[Salman2020Do_R50](https://arxiv.org/abs/2007.08489)**</sup></sup> | <sub>ResNet-50</sub> |<sub>34.96% </sub> | <sub>34.96% </sub> | <sub>2926 </sub> | <sub>739 </sub> | <sub>**34.60%**</sub> | <sub>92 (31.8×) </sub> | <sub>53 (13.9×) </sub> | 
|  <sub>**6**</sub> | <sub><sup>**[Engstrom2019Robustness](https://github.com/MadryLab/robustness)**</sup></sup> | <sub>ResNet-50</sub> |<sub>29.22% </sub> | <sub>29.22% </sub> | <sub>2473 </sub> | <sub>631 </sub> | <sub>**29.13%**</sub> | <sub>79 (31.3×) </sub> | <sub>44 (14.3×) </sub> | 
|  <sub>**7**</sub> | <sub><sup>**[Wong2020Fast](https://arxiv.org/abs/2001.03994)**</sup></sup> | <sub>ResNet-50</sub> |<sub>**26.24%**</sub> | <sub>**26.24%**</sub> | <sub>2327 </sub> | <sub>592 </sub> | <sub>26.30%</sub> | <sub>78 (29.8×) </sub> | <sub>43 (13.8×) </sub> | 
|  <sub>**8**</sub> | <sub><sup>**[Salman2020Do_R18](https://arxiv.org/abs/2007.08489)**</sup></sup> | <sub>ResNet-18</sub> |<sub>25.32% </sub> | <sub>25.32% </sub> | <sub>2016 </sub> | <sub>519 </sub> | <sub>**25.22%**</sub> | <sub>77 (26.2×) </sub> | <sub>42 (12.4×) </sub> | 
-->

