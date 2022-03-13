# Angular-AT

This is the code for the paper under review.

**Pre-requisite**: Python(3.6.4), Pytorch(0.4.1), CUDA, and numpy

**Use the following command to run the code:**

**Wideresnet**: python train_wideresnet.py

**Pre-trained Model**: https://drive.google.com/file/d/1yF8Mvta0qzbEqCDxbybM_B20QH0ByfEq/view?usp=sharing

WideResNet-34-10: 
|Method      | PGD-20 (%)    | PGD-500 (%)   | Auto-Attack (%)    | Natural (%)    |
|------------|---------------|---------------|-----------------|------------|
|Proposed    | 63.10         |  62.30        | 53.19           | 85.33      |


**Baselines**

AT-HE:  https://github.com/ShawnXYang/AT_HE/tree/master/CIFAR-10

TRADES: https://github.com/yaodongyu/TRADES

