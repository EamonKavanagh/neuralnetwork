import numpy as np

from utils import sigmoid


def forwardProp(Network, dataBlocks):
    Network.hidden = sigmoid(np.dot(dataBlocks, Network.inputWeight))
    Network.hidden = np.append(np.ones((Network.hidden.shape[0],1)), Network.hidden, 1)
    Network.output = sigmoid(np.dot(Network.hidden, Network.hiddenWeight))

def backProp(Network, dataBlocks=None, dataInd=None):
    forwardProp(Network, dataBlocks)
    if dataInd is None:
        outErr = Network.output - Network.goal
    else:
        outErr = Network.output - Network.goal[dataInd]
        
    transpose = np.transpose(Network.hiddenWeight)
    hiddenErr = np.dot(outErr, transpose)*Network.hidden*(1-Network.hidden)
    hiddenErr = hiddenErr[:,1:]
    return hiddenErr, outErr
    
def calcDeriv(Network, dataBlocks=None, dataInd=None):
    if dataBlocks is None:
        dataBlocks = Network.train
    elif len(dataBlocks.shape) == 1:
        dataBlocks = dataBlocks[np.newaxis]
        
    hiddenErr, outErr = backProp(Network, dataBlocks, dataInd)
    hiddenTranspose = np.transpose(Network.hidden)
    trainTranspose = np.transpose(dataBlocks)
    Network.hiddenDeriv = np.dot(hiddenTranspose, outErr)/Network.m
    Network.inputDeriv = np.dot(trainTranspose, hiddenErr)/Network.m