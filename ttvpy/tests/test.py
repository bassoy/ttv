import unittest
import numpy as np
import ttvpy as tp

class TestTTV(unittest.TestCase):
  
  def test_ttv_mode1(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
    B = np.arange(3, dtype=np.float64)
    D = np.einsum("ijk,i->jk", A, B)
    for version in range(1,6) :
      C = tp.ttv(1, A, B, version)
      self.assertTrue(np.all(C==D))
 
  def test_ttv_mode2(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
    B = np.arange(2, dtype=np.float64)
    D = np.einsum("ijk,j->ik", A, B)
    for version in range(1,6) :
      C = tp.ttv(2, A, B, version)
      self.assertTrue(np.all(C==D))


  def test_ttv_mode3(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
    B = np.arange(4, dtype=np.float64)
    D = np.einsum("ijk,k->ij", A, B)
    for version in range(1,6) :
      C = tp.ttv(3, A, B, version)
      self.assertTrue(np.all(C==D))

if __name__ == '__main__':
    unittest.main()
    
    



