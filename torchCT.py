# -*- coding: utf-8 -*-
"""
A warpper funtion that makes CudaTool compatible with pytorch
allowing gradients to flow between volumes and projections
@author: Heyuan Huang
"""
import torch
import numpy as np
import CudaTools
import copy


class BackProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ctr, pF):
        """
        Parameters
        ----------
        ctx : 
            Pytorch ConTeXt Manager.
        ctr : CudaTools.Reconstruction
            CudaTools reconstruction instance.
        pF : torch.Tensor (GPU)
            (Filter) projections data.

        Returns
        -------
        vol : torch.Tensor (GPU)
            backprojected volume.

        """
        ctr.SetImage('prjTorch',pF.detach().cpu().numpy())
        ctr.ProjectorLinearBack('prjTorch','volTorch')
        vol = torch.tensor(ctr.GetImage('volTorch').values).float().to(pF.device)
        ctx.ctr = ctr
        ctx.device = vol.device
        ctx.pm = ctr.GetGeometry()
        return vol
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Parameters
        ----------
        ctx : 
            Pytorch ConTeXt Manager..
        grad_output : torch.Tenor
            upstream gradient.

        Returns
        -------
        gradProj : torch.Tensor
            gradient to propagate downstream.

        """
        
        ctr = ctx.ctr
        device = ctx.device
        pm = ctx.pm
        ctr.SetGeometry(pm)
        ctr.SetImage('volTorch',grad_output.cpu().numpy())
        ctr.ProjectorSFForward('volTorch','prjTorch')
        gradProj = torch.tensor(ctr.GetImage('prjTorch').values).float().to(device)
        return None, gradProj
    
    
class ForwardProjectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ctr, vol):
        """
        Parameters
        ----------
        ctx : 
            Pytorch ConTeXt Manager.
        ctr : CudaTools.Reconstruction
            CudaTools reconstruction instance.
        vol : torch.Tensor (GPU)
            Volume to forward project.

        Returns
        -------
        proj : torch.Tensor (GPU)
            forward projections.

        """
        ctr.SetImage('volTorch',vol.detach().cpu().numpy())
        ctr.ProjectorSFForward('volTorch','prjTorch')
        proj = torch.tensor(ctr.GetImage('prjTorch').values).float().to(vol.device)
        ctx.ctr = ctr
        ctx.device = vol.device
        ctx.pm = ctr.GetGeometry()
        return proj
        
    @staticmethod
    def backward(ctx, grad_output):
        """
        Parameters
        ----------
        ctx : 
            Pytorch ConTeXt Manager..
        grad_output : torch.Tenor
            upstream gradient.

        Returns
        -------
        gradProj : torch.Tensor
            gradient to propagate downstream.

        """
        ctr = ctx.ctr
        device = ctx.device
        pm = ctx.pm
        ctr.SetGeometry(pm)
        ctr.SetImage('prjTorch',grad_output.cpu().numpy())
        ctr.ProjectorLinearBack('prjTorch','volTorch')
        gradVol = torch.tensor(ctr.GetImage('volTorch').values).float().to(device)
        return None, gradVol


class torchCT(torch.nn.Module):
    """
    The torch wrapper for CudaTools
    typical usage:
        tCT = torchCT([GPUid])
        tCT.SetGeometry(g, prec)
        tCT.allocateVolume()
        proj = tCT.fwdProject(vol)
        recon = tCT.backProject(proj)
    """
    
    """Initialization"""
    def __init__(self, GPUid):
        super(torchCT, self).__init__()
        self.ctr = CudaTools.Reconstruction(GPUid)
        self.g = None
    def SetGeometry(self,g,prec):
        self.g = g
        self.prec = prec
        self.ctr.SetGeometry(np.transpose(self.g.pm,[2,0,1]))
    def allocateVolume(self):
        self.ctr.SetImage('volTorch',
                          np.zeros(self.prec.XYZdim.astype(int),dtype=np.single),
                          np.array(self.prec.VoxSize)
                          )
        self.ctr.SetImage('prjTorch',
                          np.zeros([self.g.angleNum,*self.g.UVdim[-1::-1].astype(int)],dtype=np.single),
                          np.array([self.g.PixSize[0], self.g.PixSize[1], 1.0])
                          )
        
    """Baisc projeciton operation"""
    def forward(self,vol):
        self.fwdProject(vol)
    def fwdProject(self,vol):
        proj = ForwardProjectorFunction.apply(self.ctr,vol)
        return proj
    def backProject(self,pF):
        vol = BackProjectorFunction.apply(self.ctr,pF)
        return vol
    
    """per-angle projector"""
    def fwdProjectPA(self,vol):
        """
        PA => per angle
        Special case of foward projection
        where each projection is generated using different volume
        typically used for applying motion to volume during projection
        """
        self.ctr.SetImage('prjTorch',
                          np.zeros([1,*self.g.UVdim[-1::-1].astype(int)],dtype=np.single),
                          np.array([self.g.PixSize[0], self.g.PixSize[1], 1.0])
                          )
        proj = torch.zeros([self.g.angleNum,*self.g.UVdim[-1::-1].astype(int)],device = vol.device)
        for nAng in range(self.g.angleNum):
            self.ctr.SetGeometry(np.transpose(self.g.pm,[2,0,1])[nAng][None,...])
            proj[nAng,:,:] = self.fwdProject(vol[nAng])
        return proj
    def backProjectPA(self,prj):
        """
        PA => per angle
        Special case of foward projection
        where back projection generates one separate volume
        typically used for PAR
        """
        self.ctr.SetImage('prjTorch',
                          np.zeros([1,*self.g.UVdim[-1::-1].astype(int)],dtype=np.single),
                          np.array([self.g.PixSize[0], self.g.PixSize[1], 1.0])
                          )
        vol = torch.zeros([self.g.angleNum,*self.prec.XYZdim.astype(int)],device = prj.device)
        for nAng in range(self.g.angleNum):
            self.ctr.SetGeometry(np.transpose(self.g.pm,[2,0,1])[nAng][None,...])
            vol[nAng,:,:,:] = self.backProject(prj[nAng].unsqueeze(0))
        return vol
    
    """motion projectors"""
    def fwdProjectMotion(self,vol,affine):
        """
        Parameters
        ----------
        vol : torch.Tensor
            volume to forward project.
        affine : torch.Tensor
            nAnglex4x4 affine transformation

        Returns
        -------
        prj : torch.Tensor
            forward projection
        """
        g = copy.deepcopy(self.g)
        pmTensor = torch.tensor(g.pm,device=affine.device)
        newPm = torch.bmm(
                            torch.permute(pmTensor,(2,0,1)),
                            affine
                            ).detach().cpu().numpy()
        self.ctr.SetGeometry(newPm)
        prj = self.fwdProject(vol)
        return prj
    def backProjectMotion(self,prj,affine):
        """
        Parameters
        ----------
        vol : torch.Tensor
            volume to forward project.
        affine : torch.Tensor
            nAnglex4x4 affine transformation

        Returns
        -------
        prj : torch.Tensor
            forward projection
        """
        g = copy.deepcopy(self.g)
        pmTensor = torch.tensor(g.pm,device=affine.device)
        newPm = torch.bmm(
                            torch.permute(pmTensor,(2,0,1)),
                            affine
                            ).detach().cpu().numpy()
        self.ctr.SetGeometry(newPm)
        vol = self.backProject(prj)
        return vol
    def backProjectMotionPA(self,prj,affine):
        """
        Parameters
        ----------
        vol : torch.Tensor
            volume to forward project.
        affine : torch.Tensor
            nAnglex4x4 affine transformation

        Returns
        -------
        prj : torch.Tensor
            forward projection
        """
        g = copy.deepcopy(self.g)
        pmTensor = torch.tensor(self.g.pm,device=affine.device)
        newPm = torch.bmm(
                            torch.permute(pmTensor,(2,0,1)),
                            affine
                            ).detach().cpu().numpy().transpose(1,2,0)
        # self.ctr.SetGeometry(newPm)
        self.g.pm = newPm
        vol = self.backProjectPA(prj)
        self.g.pm = g.pm
        return vol
    def __del__(self):
        del self.ctr