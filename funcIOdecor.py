# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 22:02:54 2023

@author: hhuang91
"""

import torch
import numpy as np
def tensorIn_TensorOut(inputFunc):
    """
    decorator for functions that operate on tensors and are expected to return tensors
    """
    def outputFunc(self,*args,**kwargs):
        newArgs = [x if torch.is_tensor(x) else torch.tensor(x) for x in args]
        finalArgs = [x if x.device == torch.device(self.device) else x.to(self.device) for x in newArgs]
        newKwArgs = {key:(x if torch.is_tensor(x) else torch.tensor(x)) for (key,x) in kwargs.items()}
        finalKwArgs = {key:(x if x.device == torch.device(self.device) else x.to(self.device)) for (key,x) in newKwArgs.items()}
        output = inputFunc(self,*finalArgs,**finalKwArgs)
        if isinstance(output, tuple):
            newOutput = [x if torch.is_tensor(x) else torch.tensor(x) for x in output]
            finalOutput = tuple(x if x.device == torch.device(self.device) else x.to(self.device) for x in newOutput)
        else:
            newOutput = output if torch.is_tensor(output) else torch.tensor(output)
            finalOutput = newOutput if newOutput.device == torch.device(self.device) else newOutput.to(self.device)
        return finalOutput
    return outputFunc

def numpyIn_TensorOut(inputFunc):
    """
    decorator for functions that operate on numpy array/float values but are expected to return tensors
    """
    def outputFunc(self,*args,**kwargs):
        newArgs = [x.cpu().numpy() if torch.is_tensor(x) else x for x in args]
        newKwArgs = {key:(x.cpu().numpy() if torch.is_tensor(x) else x) for (key,x) in kwargs.items()}
        output = inputFunc(self,*newArgs,**newKwArgs)
        if isinstance(output, tuple):
            newOutput = [x if torch.is_tensor(x) else torch.tensor(x) for x in output]
            finalOutput = tuple(x if x.device == torch.device(self.device) else x.to(self.device) for x in newOutput)
        else:
            newOutput = output if torch.is_tensor(output) else torch.tensor(output)
            finalOutput = newOutput if newOutput.device == torch.device(self.device) else newOutput.to(self.device)
        return finalOutput
    return outputFunc

def numpyIn(inputFunc):
    """
    decorator for functions that operate on numpy array/float values and are expected to return whatever was returned by function
    """
    def outputFunc(self,*args,**kwargs):
        newArgs = [x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in args]
        newKwArgs = {key:(x.detach().cpu().numpy() if torch.is_tensor(x) else x) for (key,x) in kwargs.items()}
        output = inputFunc(self,*newArgs,**newKwArgs)
        return output
    return outputFunc

def noIn_TensorOut(inputFunc):
    """
    decorator for functions that have no arguments beside self but are expected to return tensors
    """
    def outputFunc(self):
        output = inputFunc(self)
        if isinstance(output, tuple):
            newOutput = [x if torch.is_tensor(x) else torch.tensor(x) for x in output]
            finalOutput = tuple(x if x.device == torch.device(self.device) else x.to(self.device) for x in newOutput)
        else:
            newOutput = output if torch.is_tensor(output) else torch.tensor(output)
            finalOutput = newOutput if newOutput.device == torch.device(self.device) else newOutput.to(self.device)
        return finalOutput
    return outputFunc