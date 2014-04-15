from random import shuffle

import numpy as np

import propagation as prop
from utils import calcAccuracy
from utils import stepSize

def gradDescent(Network, numIter, alpha):
    j = 0
    while calcAccuracy(Network) < .95 and j < numIter:
        j += 1
        prop.calcDeriv(Network)
        Network.inputWeight -= alpha*Network.inputDeriv
        Network.hiddenWeight -= alpha*Network.hiddenDeriv
        print "Training set accuracy: %.3f" %calcAccuracy(Network)
    
def incrDescent(Network, numIter, alpha):
    j = 0
    while calcAccuracy(Network) < .95 and j < numIter:
        j += 1
        randvec = range(Network.m)
        shuffle(randvec)
        for i in randvec:
            prop.calcDeriv(Network, Network.train[i], i)
            Network.inputWeight -= alpha*Network.inputDeriv
            Network.hiddenWeight -= alpha*Network.hiddenDeriv
        print "Training set accuracy: %.3f" %calcAccuracy(Network)
    
def hybrDescent(Network, numIter, gamma, beta, delta, eta, eps, nhat, batchSize):
    j = 0
    mu = 0
    lastUpdate = 0
    numBatches = Network.m/batchSize
    alpha = stepSize(mu, gamma, eta)
    while calcAccuracy(Network) < .95 and j < numIter:
        j += 1
        randvec = range(Network.m)
        shuffle(randvec)
        oldInputWeight = Network.inputWeight.copy()
        oldHiddenWeight = Network.hiddenWeight.copy()
        gInput = np.zeros(Network.inputWeight.shape)
        hInput = np.zeros(Network.inputWeight.shape)
        gHidden = np.zeros(Network.hiddenWeight.shape)
        hHidden = np.zeros(Network.hiddenWeight.shape)
        
        for i in range(numBatches-1):
            obs = randvec[i*batchSize:(i+1)*batchSize]
            prop.calcDeriv(Network, Network.train[obs], obs)
            xi = 1/sum(np.power(mu,range(numBatches-i)))
            gInput += xi*Network.inputDeriv
            gHidden += xi*Network.hiddenDeriv
            hInput = mu*hInput + gInput
            hHidden = mu*hHidden + gHidden
            Network.inputWeight = oldInputWeight - alpha*hInput
            Network.hiddenWeight = oldHiddenWeight - alpha*hHidden
        
        lastUpdate += 1
        if lastUpdate > nhat or np.linalg.norm(Network.inputWeight-oldInputWeight) < eps:
            mu = beta*mu + delta
            alpha = stepSize(mu, gamma, eta)
            lastUpdate = 0
        print "Training set accuracy: %.3f" %calcAccuracy(Network)