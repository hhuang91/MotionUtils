# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:53:18 2023

@author: hhuang91
"""
import numpy as np

from .funcIOdecor import numpyIn

class lossCollector:
    def __init__(self) -> None:
        self.loss = []
        self.bestLoss = np.inf
        self.prevX = None
        self.stopLoss = None
        self.stopLossChg = None
        self.stopLossChgNum = None
        self.stopLossChgCounter = 0
        self.stopXChg = None
        self.stopXChgNum = None
        self.stopXChgCounter = 0
    
    @numpyIn
    def setConvergeRule(self,
                        stopLoss = 0,
                        stopLossChg = 1e-10,
                        stopLossChgNum = 50,
                        stopXChg = 1e-10,
                        stopXChgNum = 50):
        if stopLoss is None:
            stopLoss = -np.inf
        self.stopLoss = stopLoss
        self.stopLossChg = stopLossChg
        self.stopLossChgNum = stopLossChgNum
        self.stopXChg = stopXChg
        self.stopXChgNum = stopXChgNum
        return
    
    def reset(self):
        self.loss = []
        self.bestLoss = np.inf
        self.prevX = None
        self.stopLossChgCounter = 0
        self.stopXChgCounter = 0
    
    @numpyIn
    def record(self,loss,x):
        ## processing X first
        # step 1: see if current x is the best solution, and store it if so
        if loss < self.bestLoss:
            self.bestLoss = loss.squeeze().item()
            isBest = True
        else:
            isBest = False
        # step 2: compute change in solution x if we have prevX, else, asign Xchg inf
        if self.prevX is not None:
            xChg = np.abs(x - self.prevX).mean()
        else:
            xChg = np.abs(x).mean()
        # step 3: previous x has no use, overwrite with current x
        self.prevX = x
        # step 4: see if change in x is smaller than defined stopXChg
        # and if so, add 1 to counter
        # if not, reset counter
        if xChg < self.stopXChg:
            self.stopXChgCounter += 1
        else:
            self.stopXChgCounter = 0
        ## Processing Loss
        # step 1: add current loss to list
        self.loss += [loss.squeeze().item()]
        # step 2: compute Loss change if we have more than two losses, else assign loss change to be inf
        if len(self.loss) > 1:
            lossChg = np.abs(self.loss[-2] - self.loss[-1])
        else:
            lossChg = np.inf
        # step 3: see if change in loss is smaller than defined stopXChg
        # and if so, add 1 to counter
        # if not, reset counter
        if lossChg < self.stopLossChg:
            self.stopLossChgCounter += 1
        else:
            self.stopLossChgCounter = 0    
        # Return if current solution is the best
        return isBest
    
    def ifConverge(self):
        if self.bestLoss <= self.stopLoss:
            return True,'Stop due to arriving at stop loss'
        elif self.stopXChgCounter >= self.stopXChgNum:
            return True,'Stop due to convergence in solution'
        elif self.stopLossChgCounter >= self.stopLossChgNum:
            return True,'Stop due to convergence in loss'
        else:
            return False, None
    
    def getTqdmMsg(self):
        msg = 'Loss:{loss:.4e}, Best Loss:{bestLoss:.4e} , lossChgCounter:{lossChgCounter:.0f}/{lossChgNum:.0f}, xChgCounter:{xChgCounter:.0f}/{xChgNum:.0f}'
        tqdmMsg = msg.format(loss = self.loss[-1],
                             bestLoss = self.bestLoss,
                             lossChgCounter = self.stopLossChgCounter,
                             lossChgNum = self.stopLossChgNum,
                             xChgCounter = self.stopXChgCounter,
                             xChgNum = self.stopXChgNum)
        return tqdmMsg