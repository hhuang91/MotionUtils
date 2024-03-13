# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:23:49 2024

FBP filter implementation in PyTorch for faster computation and gradient propagation

@author: H. Huang

"""

import numpy as np
import torch as tr
from tqdm import tqdm
from .funcIOdecor import tensorIn_TensorOut


class torchFilter:
    def __init__(self,device):
        self.device = device
        self.parkerOn = 0
        self.extrapOn = 0
        self.norm = []
        self.g = None
    
    def configFilter(self, g, fact, hamming, **kwargs):
        self.g = g
        self.fact = fact
        self.hamming = hamming
        if 'parker' in kwargs:
            self.parkerOn = kwargs['parker']
        if 'extrap' in kwargs:
            self.extrapOn = kwargs['extrap']
        if 'norm' in kwargs:
            self.norm = kwargs['norm']
    
    def genFilter(self,yShape):
        (nu, nv, nb) = yShape
        
        if self.g.SDD.size == 1:
            DS = np.tile(self.g.SDD, nb)
            u0 = np.tile(self.g.u0, nb)
            v0 = np.tile(self.g.v0, nb)
        else:
            DS = self.g.SDD
            u0 = self.g.u0
            v0 = self.g.v0
        
        # Pixel sizes in mm
        uPixelSize = self.g.PixSize[0]
        vPixelSize = self.g.PixSize[1]
    
        # Filter specification in u
        NU = np.minimum(int(2**np.ceil(np.log2(nu))),int(1.1*nu))
        
        ''' Faster Implementation --> padded NU'''
        filter = np.zeros((NU,1))
        arg1 = np.pi * self.fact * np.arange(0,int(nu/2)) # s coordinates where the filter is calculated
        h1 = self.hamming * self.vfunc(arg1) + 0.5 * (1-self.hamming) * (self.vfunc(np.pi + arg1) + self.vfunc(np.pi - arg1))
        arg2 = np.pi * self.fact * np.arange(-int(nu/2)+1, 0)
        h2 = self.hamming * self.vfunc(arg2) + 0.5 * (1-self.hamming) * (self.vfunc(np.pi + arg2) + self.vfunc(np.pi - arg2))
        filter[0:int(nu/2)] = np.expand_dims(h1, axis=1)
        filter[(NU-int(nu/2)+1):NU] = np.expand_dims(h2, axis=1)
        filter = 2 * (self.fact**2) / (2*uPixelSize*2) * filter
        self.filter = (tr.fft.rfft(tr.tensor(filter).float().to(self.device),axis=0)).view(int(NU/2)+1,1,1)
        
        ''' Original Implementation --> padded 2*NU'''
        # filter = np.zeros((2*NU,1))
        # arg1 = np.pi * self.fact * np.arange(0,nu) # s coordinates where the filter is calculated
        # h1 = self.hamming * self.vfunc(arg1) + 0.5 * (1-self.hamming) * (self.vfunc(np.pi + arg1) + self.vfunc(np.pi - arg1))
        # arg2 = np.pi * self.fact * np.arange(-nu+1, 0)
        # h2 = self.hamming * self.vfunc(arg2) + 0.5 * (1-self.hamming) * (self.vfunc(np.pi + arg2) + self.vfunc(np.pi - arg2))
        # filter[0:nu] = np.expand_dims(h1, axis=1)
        # filter[(2*NU-nu+1):(2*NU)] = np.expand_dims(h2, axis=1)
        # filter = 2 * (self.fact**2) / (2*uPixelSize*2) * filter
        # self.filter = tr.fft.rfft(tr.tensor(filter).float().to(self.device),axis=0).view(NU+1,1,1)
    
        if self.parkerOn:
            wPker = self.applyParker(nu, nv, nb, u0, DS, uPixelSize, self.g.angle)
            self.wPker = tr.tensor(wPker).float().to(self.device).view(nu,1,nb)
        
        self.p_mat = tr.zeros(nu,nv,nb).to(self.device)
        self.eta_mat = tr.zeros(nu,nv,nb).to(self.device)
        for b in tqdm(np.arange(0, nb),desc = 'pre-computing filter'):
            p_mat = uPixelSize * np.matmul(np.ones((nu,1),'float32'), (np.expand_dims(np.arange(0,nv), axis=0) - v0[b] - nv/2))
            eta_mat = vPixelSize * np.matmul((np.expand_dims(np.arange(0,nu), axis=1) - u0[b] - nu/2), np.ones((1,nv),'float32'))
            self.p_mat[:,:,b] = tr.tensor(p_mat).float().to(self.device)
            self.eta_mat[:,:,b] = tr.tensor(eta_mat).float().to(self.device)
            
    @tensorIn_TensorOut
    def torchFilterRaisedCosine(self,y,forceFilter=False):
        if self.pyyf(y) == 'prjF' and not forceFilter:
            print('projection is already filtered')
            return y
        (nu, nv, nb) = y.shape
        if self.filter is None:
            self.genFilter(y.shape)
        if self.g.SDD.size == 1:
            DS = np.tile(self.g.SDD, nb)
            A = np.tile(self.g.SAD, nb)
            u0 = np.tile(self.g.u0, nb)
            v0 = np.tile(self.g.v0, nb)
        else:
            DS = self.g.SDD
            A = self.g.SAD
            u0 = self.g.u0
            v0 = self.g.v0
        # Pixel sizes in mm
        DS = tr.tensor(DS).float().to(self.device).view(1,1,nb)
        A = tr.tensor(A).float().to(self.device).view(1,1,nb)
        u0 = tr.tensor(u0).float().to(self.device).view(1,1,nb)
        v0 = tr.tensor(v0).float().to(self.device).view(1,1,nb)
        
        NU = np.minimum(int(2**np.ceil(np.log2(nu))),int(1.1*nu))
        
        if not self.norm:
                lt = y
        elif len(self.norm) == 1:
            lt = -tr.log(tr.maximum(y/self.norm,1e-10))
        else:
            norm = tr.tensor(self.norm).to(self.device).view(1,1,nb)
            lt = -tr.log(tr.maximum(y/norm,1e-10))
            
        lt = lt*DS / tr.sqrt(DS**2 + self.p_mat**2 + self.eta_mat**2) * (A/DS)
        
        if self.parkerOn:
            lt = lt * self.wPker
        
        ''' Original Implementation --> padded 2*NU'''
        # pad_r = tr.zeros((int(NU-nu/2),lt.shape[1],lt.shape[2])).to(self.device)
        # pad_l = tr.zeros((int(NU-nu/2),lt.shape[1],lt.shape[2])).to(self.device)
        # if self.extrapOn:
        #     for j in range(lt.shape[1]):
        #         pad_r[:,j,:] = 2*lt[-1,j,:] - tr.flip(lt[int(-1-NU+nu/2):-1,j,:],[0])
        #         pad_l[:,j,:] = 2*lt[1,j,:] - tr.flip(lt[0:int(NU-nu/2),j,:],[0])
        #     pad_r[pad_r<0] = 0
        #     pad_l[pad_l<0] = 0
        
        ''' Faster Implementation --> padded NU'''
        pad_r = tr.zeros((int(NU/2-nu/2),lt.shape[1],lt.shape[2])).to(self.device)
        pad_l = tr.zeros((int(NU/2-nu/2),lt.shape[1],lt.shape[2])).to(self.device)
        if self.extrapOn:
            for j in range(lt.shape[1]):
                pad_r[:,j,:] = 2*lt[-1,j,:] - tr.flip(lt[int(-1-NU/2+nu/2):-1,j,:],[0])
                pad_l[:,j,:] = 2*lt[1,j,:] - tr.flip(lt[0:int(NU/2-nu/2),j,:],[0])
            pad_r[pad_r<0] = 0
            pad_l[pad_l<0] = 0
            
        tmp = tr.cat((pad_l,lt, pad_r), axis=0)
        pad_l_shape = pad_l.shape
        del pad_l,pad_r
        tr.cuda.empty_cache()
        tmp = tr.fft.rfft(tmp, axis=0)
        tmp = self.filter * tmp
        tmp = tr.fft.irfft(tmp,axis=0)
        y_filt = tmp[pad_l_shape[0]:(pad_l_shape[0] + nu),:,:] # un-pad 
        dAng = np.abs(np.median(self.g.angle[2:]-self.g.angle[1:-1]))
        y_filt = y_filt * dAng/2.0 * (np.pi/180.0)
    
        return y_filt
    
    # parker weights
    def applyParker(self,nu, nv, nb, u0, DS, uPixelSize, ang_vec):
        
        parkIndx = np.arange(-nu/2, nu/2, dtype='float32')
        parkIndx = np.expand_dims(parkIndx, axis=1)
        u0 = np.expand_dims(u0, axis=0)
        DS = np.expand_dims(DS, axis=0)
        gamma = -np.arctan((np.tile(parkIndx,(1,nb)) - np.tile(u0,(nu,1))) * uPixelSize / np.tile(DS,(nu,1)))
        dAng = np.median(ang_vec[1:] - ang_vec[0:(len(ang_vec)-1)])
        pkerAngle = np.zeros((1,len(ang_vec)),'float32')
    
        # create half the parker weights
        if (dAng > 0):
            pkerAngle[0,:] = ang_vec - ang_vec[0]
        else:
            pkerAngle[0,:] = ang_vec[0] - ang_vec
            gamma = -gamma
        
        overscanAngle = ((np.abs(pkerAngle[0,-1]-pkerAngle[0,0]) * np.pi/180 - np.pi)/2)
        pkerAngle = np.tile(pkerAngle, (nu,1)) * np.pi/180
    
        weightPker = np.zeros(gamma.shape,'float32')
        w_map = (pkerAngle < (2 * overscanAngle - 2 * gamma))
        weightPker[w_map] = np.sin((np.pi / 4) * pkerAngle[w_map] / (overscanAngle - gamma[w_map]))**2
    
        w_map = ((pkerAngle >= (2 * overscanAngle - 2 * gamma)) & (pkerAngle < (np.pi - 2 * gamma)))
        weightPker[w_map] = 1
    
        w_map = (pkerAngle >= (np.pi - 2 * gamma)) & (pkerAngle <= (np.pi + 2 * overscanAngle))
        weightPker[w_map] = np.sin((np.pi / 4) * (np.pi + 2 * overscanAngle - pkerAngle[w_map]) / (gamma[w_map] + overscanAngle))**2
    
        weightPker = weightPker * 2.0
    
        return weightPker
    
    # see if a projection is projection, line integral or filtered line integral
    def pyyf(self,p):
        if p.max() > 1:
            return 'lineInt';
        elif p.min() > 0:
            return 'prj';
        else:
            return 'prjF';
    # aux filter function
    def vfunc(self,s):
        i = np.argwhere(np.abs(s) > 0.0001)
        v = np.zeros(s.shape)
        ss = s[i]
        v[i] = np.sin(ss)/ss + (np.cos(ss)-1) / (ss**2)
        v[np.abs(s) <= 0.0001] = 0.5
        return v