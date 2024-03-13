# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:33:51 2023

@author: hhuang91
"""
import torch
from ._1dCubicSpline import interp

#%% 4D affine
class affineTransformer4D:
    def __init__(self,device,nBs,nPar,grad = True,deTrend = True):
        self.device = device
        self.deTrend = deTrend
        self.nBs = nBs
        self.nPar = nPar
        self.t = torch.linspace(0, 1, nBs).to(self.device)#.requires_grad_(grad)
        self.ts = torch.linspace(0, 1, nPar).to(self.device)
        self.Tx = torch.zeros(nBs).to(self.device).requires_grad_(grad)
        self.Ty = torch.zeros(nBs).to(self.device).requires_grad_(grad)
        self.Tz = torch.zeros(nBs).to(self.device).requires_grad_(grad)
        self.theta = torch.zeros(nBs).to(self.device).requires_grad_(grad)
        self.phi = torch.zeros(nBs).to(self.device).requires_grad_(grad)
        self.psi= torch.zeros(nBs).to(self.device).requires_grad_(grad)
    def __call__(self):
        Tx,Ty,Tz,theta,phi,psi = self.getInterp()
        TMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        thetaMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        phiMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        psiMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        # T
        TMat[:,0,3] = Tx
        TMat[:,1,3] = Ty
        TMat[:,2,3] = Tz
        # Theta
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        thetaMat[:,1,1] = cosTheta
        thetaMat[:,1,2] = sinTheta
        thetaMat[:,2,1] = -sinTheta
        thetaMat[:,2,2] = cosTheta
        # Phi
        cosPhi = torch.cos(phi)
        sinPhi = torch.sin(phi)
        phiMat[:,0,0] = cosPhi
        phiMat[:,0,2] = -sinPhi
        phiMat[:,2,0] = sinPhi
        phiMat[:,2,2] = cosPhi
        # Psi
        cosPsi = torch.cos(psi)
        sinPsi = torch.sin(psi)
        psiMat[:,0,0] = cosPsi
        psiMat[:,0,1] = -sinPsi
        psiMat[:,1,0] = sinPsi
        psiMat[:,1,1] = cosPsi
        # get affine Matrix
        affineMat = torch.bmm(psiMat,
                              torch.bmm(phiMat,
                                         torch.bmm(thetaMat, TMat)
                                       ) 
                              )
        return affineMat[:,0:3,:]
    def getInterp(self):
        t = torch.clamp(self.t,0,1)
        t[0] = 0.
        t[-1] = 1.
        Tx = interp(t,self.Tx,self.ts)
        Ty = interp(t,self.Ty,self.ts)
        Tz = interp(t,self.Tz,self.ts)
        theta = interp(t,self.theta,self.ts)
        phi = interp(t,self.phi,self.ts)
        psi = interp(t,self.psi,self.ts)
        if self.deTrend:
            return Tx-Tx.mean(),Ty-Ty.mean(),Tz-Tz.mean(),theta-theta.mean(),phi-phi.mean(),psi-psi.mean()
        else:
            return Tx,Ty,Tz,theta,phi,psi
    def parameters(self):
        return [self.t,self.Tx,self.Ty,self.Tz,self.theta,self.phi,self.psi]
    def reset(self,nPar = None, grad = True):
        if nPar is not None:
            self.nPar = nPar
        self.t = torch.linspace(0, 1, self.nBs).to(self.device)
        self.ts = torch.linspace(0, 1, self.nPar).to(self.device)
        self.Tx = torch.zeros(self.nBs).to(self.device).requires_grad_(grad)
        self.Ty = torch.zeros(self.nBs).to(self.device).requires_grad_(grad)
        self.Tz = torch.zeros(self.nBs).to(self.device).requires_grad_(grad)
        self.theta = torch.zeros(self.nBs).to(self.device).requires_grad_(grad)
        self.phi = torch.zeros(self.nBs).to(self.device).requires_grad_(grad)
        self.psi= torch.zeros(self.nBs).to(self.device).requires_grad_(grad)
    
    @torch.no_grad()
    def detach(self):
        Tx,Ty,Tz,theta,phi,psi = self.getInterp()
        res = torch.stack([Tx.squeeze(),Ty.squeeze(),Tz.squeeze(),
                           theta.squeeze(),phi.squeeze(),psi.squeeze()],1)
        return res
    
    def state_dict(self):
        stateDict = {}
        stateDict['deTrend'] = self.deTrend
        stateDict['nBs'] = self.nBs
        stateDict['t'] = self.t.detach().clone()
        stateDict['Tx'] = self.Tx.detach().clone() 
        stateDict['Ty'] = self.Ty.detach().clone()
        stateDict['Tz'] = self.Tz.detach().clone()
        stateDict['theta'] = self.theta.detach().clone()
        stateDict['phi'] = self.phi.detach().clone() 
        stateDict['psi'] = self.psi.detach().clone()
        return stateDict
        
    def load_state_dict(self,stateDict):
        self.deTrend = stateDict['deTrend']
        self.nBs =  stateDict['nBs']
        self.t =  stateDict['t']
        self.Tx = stateDict['Tx']
        self.Ty = stateDict['Ty']
        self.Tz = stateDict['Tz']
        self.theta = stateDict['theta']
        self.phi = stateDict['phi']
        self.psi= stateDict['psi']
        
    def changeTs(self,nPar):
        self.nPar = nPar
        self.ts = torch.linspace(0, 1, nPar).to(self.device)

#%% 4D affine but with shared rotation
class affineTransformer4DROI(affineTransformer4D):
    def __init__(self,device,nBs,nPar,grad = True,deTrend = True):
        super().__init__(device,nBs,nPar,grad,deTrend)
        self.ROIlist = None
        self.nROI = None
        
    def setROIs(self,ROIlist):
        self.ROIlist = ROIlist
        self.nROI = len(ROIlist)
        
    def convertRotation(self,rotMat):
        rotMatNew = torch.zeros(self.nROI,*rotMat.shape,device = self.device)
        for ijk,ROI in enumerate(self.ROIlist):
            """offset indicates the transformation from cntr of ROI to cntr of volume"""
            offsetT = -torch.tensor([[*ROI,0,0,0]],device = self.device) # notice the minus sign: assuming ROI value means from center of vol to center of ROI
            offsetInverseT = torch.tensor([[*ROI,0,0,0]],device = self.device)
            offsetAffine = bmmAffine(offsetT, self.device,True)
            offsetInverseAffine = bmmAffine(offsetInverseT, self.device,True)
            rotMatNew[ijk,:,:,:] = torch.bmm(offsetAffine.repeat(self.nPar,1,1),
                                               torch.bmm(rotMat, 
                                                         offsetInverseAffine.repeat(self.nPar,1,1)
                                                         )
                                            )
        return rotMatNew
    
    def __call__(self):
        Tx,Ty,Tz,theta,phi,psi = self.getInterp()
        TMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        thetaMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        phiMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        psiMat = torch.tile(torch.eye(4,4),[self.nPar,1,1]).to(self.device)
        # T
        TMat[:,0,3] = Tx
        TMat[:,1,3] = Ty
        TMat[:,2,3] = Tz
        # Theta
        cosTheta = torch.cos(theta)
        sinTheta = torch.sin(theta)
        thetaMat[:,1,1] = cosTheta
        thetaMat[:,1,2] = sinTheta
        thetaMat[:,2,1] = -sinTheta
        thetaMat[:,2,2] = cosTheta
        # Phi
        cosPhi = torch.cos(phi)
        sinPhi = torch.sin(phi)
        phiMat[:,0,0] = cosPhi
        phiMat[:,0,2] = -sinPhi
        phiMat[:,2,0] = sinPhi
        phiMat[:,2,2] = cosPhi
        # Psi
        cosPsi = torch.cos(psi)
        sinPsi = torch.sin(psi)
        psiMat[:,0,0] = cosPsi
        psiMat[:,0,1] = -sinPsi
        psiMat[:,1,0] = sinPsi
        psiMat[:,1,1] = cosPsi
        # get affine Matrix
        rotMat = torch.bmm(psiMat,
                           torch.bmm(phiMat,thetaMat)
                           )
        rotMatNew = self.convertRotation(rotMat)
        affineMat = torch.zeros(self.nROI,self.nPar,4,4,device = self.device)
        for ijk,rotMatROI in enumerate(rotMatNew):
            affineMat[ijk,:,:,:] = torch.bmm(rotMatROI,TMat)
        # affineMat = torch.bmm(psiMat,
        #                       torch.bmm(phiMat,
        #                                  torch.bmm(thetaMat, TMat)
        #                                ) 
        #                       )
        return affineMat[:,:,0:3,:]

#%% Useful Vtrans to affine function
def bmmAffine(vTrans,device,return4by4=False):
    """
    Parameters
    ----------
    vTrans : torch.Tensor
        6 DoF affine transformation in batch.
        dimension: (nBatch, 6)
        !!!Note!!! translation range is [-1,1](similar to grid sample)
    device : str, torch.device
        which device to use for computation.

    Returns
    -------
    torch.Tensor
    nBatchx4x3 transformation for input into affine_grid

    """
    Tx,Ty,Tz,theta,phi,psi = vTrans.transpose(0,1)
    nPar = Tx.shape[0]
    TMat = torch.tile(torch.eye(4,4),[nPar,1,1]).to(device)
    thetaMat = torch.tile(torch.eye(4,4),[nPar,1,1]).to(device)
    phiMat = torch.tile(torch.eye(4,4),[nPar,1,1]).to(device)
    psiMat = torch.tile(torch.eye(4,4),[nPar,1,1]).to(device)
    # T
    TMat[:,0,3] = Tx
    TMat[:,1,3] = Ty
    TMat[:,2,3] = Tz
    # Theta
    cosTheta = torch.cos(theta)
    sinTheta = torch.sin(theta)
    thetaMat[:,1,1] = cosTheta
    thetaMat[:,1,2] = sinTheta
    thetaMat[:,2,1] = -sinTheta
    thetaMat[:,2,2] = cosTheta
    # Phi
    cosPhi = torch.cos(phi)
    sinPhi = torch.sin(phi)
    phiMat[:,0,0] = cosPhi
    phiMat[:,0,2] = -sinPhi
    phiMat[:,2,0] = sinPhi
    phiMat[:,2,2] = cosPhi
    # Psi
    cosPsi = torch.cos(psi)
    sinPsi = torch.sin(psi)
    psiMat[:,0,0] = cosPsi
    psiMat[:,0,1] = -sinPsi
    psiMat[:,1,0] = sinPsi
    psiMat[:,1,1] = cosPsi
    # get affine Matrix
    affineMat = torch.bmm(psiMat,
                          torch.bmm(phiMat,
                                     torch.bmm(thetaMat, TMat)
                                   ) 
                          )
    return affineMat if return4by4 else affineMat[:,0:3,:]

# class affineTransformer:
#     def __init__(self,device,grad = True):
#         self.device = device
#         self.Tx = torch.tensor(0.).to(self.device).requires_grad_(grad)
#         self.Ty = torch.tensor(0.).to(self.device).requires_grad_(grad)
#         self.Tz = torch.tensor(0.).to(self.device).requires_grad_(grad)
#         self.theta = torch.tensor(0.).to(self.device).requires_grad_(grad)
#         self.phi = torch.tensor(0.).to(self.device).requires_grad_(grad)
#         self.psi= torch.tensor(0.).to(self.device).requires_grad_(grad)
#         return
#     def __call__(self):
#         TMat = torch.eye(4,4).to(self.device)
#         thetaMat = torch.eye(4,4).to(self.device)
#         phiMat = torch.eye(4,4).to(self.device)
#         psiMat = torch.eye(4,4).to(self.device)
#         # T
#         TMat[0,3] = self.Tx
#         TMat[1,3] = self.Ty
#         TMat[2,3] = self.Tz
#         # Theta
#         cosTheta = torch.cos(self.theta)
#         sinTheta = torch.sin(self.theta)
#         thetaMat[1,1] = cosTheta
#         thetaMat[1,2] = sinTheta
#         thetaMat[2,1] = -sinTheta
#         thetaMat[2,2] = cosTheta
#         # Phi
#         cosPhi = torch.cos(self.phi)
#         sinPhi = torch.sin(self.phi)
#         phiMat[0,0] = cosPhi
#         phiMat[0,2] = -sinPhi
#         phiMat[2,0] = sinPhi
#         phiMat[2,2] = cosPhi
#         # Psi
#         cosPsi = torch.cos(self.psi)
#         sinPsi = torch.sin(self.psi)
#         psiMat[0,0] = cosPsi
#         psiMat[0,1] = -sinPsi
#         psiMat[1,0] = sinPsi
#         psiMat[1,1] = cosPsi
#         affineMat = torch.linalg.multi_dot([thetaMat,phiMat,psiMat,TMat])
#         return affineMat[0:3,:]