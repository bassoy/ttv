## Installation

Clone the repo
```bash
git clone git@github.com:bassoy/ttv.git
cd ttv
```

Install OpenBLAS (or MKL) [optional]
```bash
sudo apt install libopenblas-base libopenblas-dev libomp-dev
```

Navigate to the python wrapper folder
```bash
cd ttvpy
```

Install the pacakge in editable mode
```bash
pip install -e .
# python3 setup.py build_ext -i
```

Test the package
```bash
python3 -m unittest discover -v
```


## Python Interface

```python
C = ttv(cmode, A, b)
```
or
```python
C = ttv(cmode, A, b, version)
```
* `cmode` is the contraction mode with `1 <= cmode <= len(A.shape)`
* `A` is an input `numpy.ndarray` with typically `len(A.shape)>=2`
* `b` is an input `numpy.ndarray` with `len(b.shape)=1`
* `C` is the output `numpy.ndarray` with `len(A.shape)-1`
* `version` corresponds to the optimization level: `6` highest (OpenBLAS or MKL with OpenMP) and `1` lowest (sequential)


## Python Example
```python
import numpy as np
import ttvpy as tp

A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
b = np.arange(3, dtype=np.float64)

C = ttv(1,A,b)
```
