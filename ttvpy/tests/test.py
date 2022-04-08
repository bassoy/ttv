import unittest
import numpy as np
import ttvpy as tp

class TestTTV(unittest.TestCase):
  
  def test_ttv_mode1(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
    b = np.arange(3, dtype=np.float64)
    D = np.einsum("ijk,i->jk", A, b)
    C = tp.ttv(1, A, b)
    self.assertTrue(np.all(C==D))
 
  def test_ttv_mode2(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
    b = np.arange(2, dtype=np.float64)
    D = np.einsum("ijk,j->ik", A, b)
    C = tp.ttv(2, A, b)
    self.assertTrue(np.all(C==D))


  def test_ttv_mode3(self):
    A = np.arange(3*2*4, dtype=np.float64).reshape(3, 2, 4)
    b = np.arange(4, dtype=np.float64)
    D = np.einsum("ijk,k->ij", A, b)
    C = tp.ttv(3, A, b)
    self.assertTrue(np.all(C==D))


class TestqTTV(unittest.TestCase): 
  
  def test_ttvs_mode1(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(2, dtype=np.float64), np.arange(4, dtype=np.float64), np.arange(5, dtype=np.float64)]
    D  = np.einsum("ijkl,j->ikl", A, B[0])
    D  = np.einsum("ikl,k->il"  , D, B[1])
    D  = np.einsum("il,l->i"    , D, B[2])
    for order in ["forward", "backward"] :
      C = tp.ttvs(1, A, B, order)
      self.assertTrue(np.all(C==D))    
 
  def test_ttvs_mode2(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(3, dtype=np.float64), np.arange(4, dtype=np.float64), np.arange(5, dtype=np.float64)]
    D  = np.einsum("ijkl,i->jkl", A, B[0])
    D  = np.einsum("jkl,k->jl"  , D, B[1])
    D  = np.einsum("jl,l->j"    , D, B[2])
    for order in ["forward", "backward"] :
      C = tp.ttvs(2, A, B, order)
      self.assertTrue(np.all(C==D))  

  def test_ttvs_mode3(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(3, dtype=np.float64), np.arange(2, dtype=np.float64), np.arange(5, dtype=np.float64)]
    D  = np.einsum("ijkl,i->jkl", A, B[0])
    D  = np.einsum("jkl,j->kl"  , D, B[1])
    D  = np.einsum("kl,l->k"    , D, B[2])
    for order in ["forward", "backward"] :
      C = tp.ttvs(3, A, B, order)
      self.assertTrue(np.all(C==D))  

    
  def test_ttvs_mode4(self):
    A  = np.arange(3*2*4*5, dtype=np.float64).reshape(3, 2, 4, 5)
    B = [np.arange(3, dtype=np.float64), np.arange(2, dtype=np.float64), np.arange(4, dtype=np.float64)]
    D  = np.einsum("ijkl,i->jkl", A, B[0])
    D  = np.einsum("jkl,j->kl"  , D, B[1])
    D  = np.einsum("kl,k->l"    , D, B[2])
    for order in ["forward", "backward"] :
      C = tp.ttvs(4, A, B, order)
      self.assertTrue(np.all(C==D))     
    
    
if __name__ == '__main__':
    unittest.main()
    
    



