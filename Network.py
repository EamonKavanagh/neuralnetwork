import numpy as np

import utils
import descent
import propagation


class Network(object):

    def __init__(self, train, digits, numHidden, test=None, testDigits=None):
        self.m = train.shape[0]
        self.train = np.append(np.ones((self.m,1)), train, 1)
        self.digits = digits
        oneVSall = np.identity(10)
        whichDigit = np.asarray([oneVSall[int(x)] for x in digits])
        self.goal = whichDigit
        numOut = self.goal.shape[1]
        self.output = None
        self.hidden = None
        self.inputWeight = (np.random.rand(train.shape[1]+1, numHidden)*2*.12)-.12
        self.hiddenWeight = (np.random.rand(numHidden+1, numOut)*2*.12)-.12
        self.inputDeriv = None
        self.hiddenDeriv = None
        self.test = test
        self.testDigits = testDigits
        
    def predict(self, newData=None):
        if newData is None:
            newData = self.train
        else:
            newData = np.append(np.ones((newData.shape[0],1)), newData, 1)
            
        z = utils.sigmoid(np.dot(newData, self.inputWeight))
        z = np.append(np.ones((z.shape[0],1)), z, 1)
        digitProb = utils.sigmoid(np.dot(z, self.hiddenWeight))
        return np.argmax(digitProb,1)
        
    def trainNetwork(self, method, *params):
        if method == 'GD':
            method = descent.gradDescent
        elif method == 'ID':
            method = descent.incrDescent
        else:
            method = descent.hybrDescent
            
        method(self, *params)
        if self.test is not None:
            print "Testing set accuracy"
            print utils.calcAccuracy(self, self.test, self.testDigits)  
        
    def cost(self, theta1, theta2):
        z1 = np.dot(self.train, theta1)
        a2 = utils.sigmoid(z1)
        a2 = np.append(np.ones((a2.shape[0],1)), a2, 1)
        z2 = np.dot(a2, theta2)
        h = utils.sigmoid(z2)
        return -sum(sum(self.goal*np.log(h) + (1-self.goal)*np.log(1-h)))/self.m

    def checkDeriv(self):
        propagation.calcDeriv(self)
        grad = np.concatenate((np.reshape(self.inputDeriv, -1), 
                                np.reshape(self.hiddenDeriv[:], -1)))
        eps = 1e-5
        theta1 = np.copy(self.inputWeight)
        theta2 = np.copy(self.hiddenWeight)
        perturb1 = np.zeros(theta1.shape)
        perturb2 = np.zeros(theta2.shape)
        numgrad = []
        for i in range(theta1.shape[0]):
            for j in range(theta1.shape[1]):
                perturb1[i,j] = eps
                l1 = self.cost(theta1-perturb1, theta2)
                l2 = self.cost(theta1+perturb1, theta2)
                numgrad.append((l2-l1)/(2*eps))
                perturb1[i,j] = 0
        for i in range(theta2.shape[0]):
            for j in range(theta2.shape[1]):
                perturb2[i,j] = eps
                l1 = self.cost(theta1, theta2-perturb2)
                l2 = self.cost(theta1, theta2+perturb2)
                numgrad.append((l2-l1)/(2*eps))
                perturb2[i,j] = 0
        numgrad = np.asarray(numgrad)
        return np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)