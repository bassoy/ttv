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

### q-mode Tensor-Vector Product
```python
C = ttv(q, A, b)
```
* `q` is the contraction mode with `1<=q<=A.ndim`
* `A` is an input `numpy.ndarray` with typically `A.ndim>=2`
* `b` is an input `numpy.ndarray` with `b.ndim=1` and `b.shape[0]=A.shape[q-1]`
* `C` is the output `numpy.ndarray` with `A.ndim-1`

### {1,...,q-1,q+1,...p}-mode Tensor-Vector Products
```python
c = ttvs(q, A, B, order)
```
* `q` is the non-contraction mode with `1<=q<=A.ndim`. Every other mode of `A` is contracted with all vectors in `B`
* `A` is an input `numpy.ndarray` with typically `A.ndim>=2`
* `B` is an input list of `numpy.ndarray`s with `B[r].ndim=1` for `1<=r<p` and `B[r].shape[0]=A.shape[r]` for `1<=r<q` and `B[r].shape[0]=A.shape[r+1]` for `q<=r<p`
* `c` is the output `numpy.ndarray` with `c.ndim=1` and `c.shape[0]=A.shape[q-1]`
* `order` corresponds to the multiplication order of the vectors (`optimal` is the default value) : 
  * `forward` multiplies vectors according to `B[1]`,...,`B[p-1]`
  * `backward` multiplies vectors according to `B[p-1]`,...,`B[1]`
  * `optimal` multiplies vectors according to their dimensionality starting with `B[r]` with `B[r].shape >= B[q].shape` for all `1 <= q < p`.


## Python Example

### 1-mode Tensor-Vector Product
```python
import numpy as np
import ttvpy as tp

A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
b = np.arange(3, dtype=np.float64)
C = ttv(1,A,b)
# C = [[40,43,46,49,],[52,55,58,61]]

```

### {1,3}-mode Tensor-Vector Products
```python
import numpy as np
import ttvpy as tp

A = np.arange(3*2*4, dtype=np.float64).reshape(3,2,4)
B = [np.arange(3, dtype=np.float64), np.arange(4, dtype=np.float64)]
c = ttvs(2,A,b)
# c = [38,86,134]

```
