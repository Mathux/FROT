# FROT

Pytorch/numpy implementation of "Feature Robust Optimal Transport for High-dimensional Data" paper: &nbsp; [Preprint](https://arxiv.org/abs/2005.12123)


## Installation :construction_worker:
### Python requirements
This code was tested on Python 3.7 and Python 3.8, and need the following packages:

* click
* numpy
* torch
* pot
* scipy
* sklearn

You can install them with pip.

## How to use :rocket:
Our Frot model has a `sklearn-like` interface. 

### Toy exemple
Run inside a Python Interpreter:

```python3
import numpy as np
from src.models.frot import Frot

# Define some Toy data
X = np.array([[0, 0], [1, 1], [2, 2]])
Y = np.array([[0, 1], [0, 1], [2, -2]])
group = [[0], [1]]

model = Frot(eps=0.05, eta=1.0, method='sinkhorn', pFRWD=1, pnorm=2)
model.fit(X, Y, group)

PI = model.PI_
alpha = model.alpha_
dist = model.FRWD_
```

* `PI` is the optimal transport plan
* `alpha` is the vector which describe group importance.
* `dist` is the FRWD distance

### Implementation
We implemented the FROT computation with 3 methods:

* `lp`: construct and solve a linear program
* `sinkhorn`: use frank-wolfe with sinkhorn as a inner solver
* `emd`: use frank-wolfe with emd as a inner solver


## How to cite? :clipboard:

If you find this work useful in your research, please consider citing:

```
@misc{petrovich2020feature,
    title={Feature Robust Optimal Transport for High-dimensional Data},
    author={Mathis Petrovich and Chao Liang and Yanbin Liu and Yao-Hung Hubert Tsai and Linchao Zhu and Yi Yang and Ruslan Salakhutdinov and Makoto Yamada},
    year={2020},
    eprint={2005.12123},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
```
