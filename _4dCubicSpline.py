# -*- coding: utf-8 -*-
"""
4D hermite cubic spline interpolation
based on: https://gist.github.com/chausies/c453d561310317e7eda598e229aea537

@author: hhuang91
"""

import torch as T

def h_poly_helper(tt):
  A = T.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype)
  return [
              sum( A[i, j]*tt[j] for j in range(4) )
          for i in range(4) ]

def h_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = T.ones_like(t)
  for i in range(1, 4):
    tt[i] = tt[i-1]*t
  return h_poly_helper(tt)


def interp_func(x, F, dim):
  "Returns interpolating function"
  """
  x, 1D, float Tensor, sampling location along dim
  F, 4D, float Tensor, sampled value
  dim, single number, int, sampling dimension along which x is located
  """
  assert len(x)>1, 'Not implemented for signle number interpolation'
  dF = T.narrow(F,dim,1,len(x)-1) - T.narrow(F,dim,0,len(x)-1)
  dx = x[1:] - x[:-1]
  newShape = [1,1,1,1]
  newShape[dim] = len(dx)
  newShape = tuple(newShape)
  dx = dx.view(newShape)
  m =  dF/dx
  m0 = T.narrow(m,dim,0,1)
  mMiddle = T.narrow(m,dim,1,len(x)-1-1) + T.narrow(m,dim,0,len(x)-1-1)
  mEnd = T.narrow(m,dim,-1,1)
  m = T.cat((m0,mMiddle/2,mEnd),dim=dim) # with shape similar as F but +1 len at dim
  F = T.cat((F,T.narrow(F,dim,-1,1)),dim=dim)  # pad F to avoid selection out of range
  def func(xs):
    I = T.searchsorted(x[1:], xs)
    dxi = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dxi)
    newShape = [1,1,1,1]
    newShape[dim] = len(I)
    newShape = tuple(newShape)
    hh0 = hh[0].view(newShape) * T.index_select(F, dim, I)
    hh1 = hh[1].view(newShape) * T.index_select(m, dim, I) * dxi.view(newShape)
    hh2 = hh[2].view(newShape) * T.index_select(F, dim, I+1) #<-- reason for padding F, I+1 could be out of range
    hh3 = hh[3].view(newShape) * T.index_select(m, dim, I+1) * dxi.view(newShape)
    return hh0 + hh1 + hh2 + hh3
  return func

def interp(x,y,z,t, Fxyzt, xs,ys,zs,ts):
  """
  x,y,z,t: 1-D Tensors, specifying sampling grid
  xs,ys,zs,ts: 1-D Tensors, specifying interpolating grid
  """
  Fyzt = interp_func(x, Fxyzt, 0)(xs)
  Fzt = interp_func(y, Fyzt, 1)(ys)
  Ft = interp_func(z, Fzt, 2)(zs)
  F = interp_func(t, Ft, 3)(ts)
  return F
