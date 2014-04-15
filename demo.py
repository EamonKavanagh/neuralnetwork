import csv
import sys

import numpy as np

from Network import Network

method = str(sys.argv[1])

print "Reading data"
data = np.loadtxt("train.csv", delimiter=",")
print "Data loaded"
numData = 10000
digits = np.asarray([x[0] for x in data[:numData]])
pixels = np.asarray([x[1:] for x in data[:numData]])
testDigits = np.asarray([x[0] for x in data[numData:numData+int(.2*numData)]])
testPixels = np.asarray([x[1:] for x in data[numData:numData+int(.2*numData)]])

numHidden = 100
n = Network(pixels, digits, numHidden, testPixels, testDigits)
numIter = 30
alpha = .5
gamma = .5
beta = 1.1
delta = .01
eta = 2
eps = .001
nhat = 10
batchSize = 50
if method == 'GD':
    n.trainNetwork('GD', numIter, alpha)
elif method =='ID':
    n.trainNetwork('ID', numIter, alpha)
else:
    n.trainNetwork('HD', numIter, gamma, beta, delta, eta, eps, nhat, batchSize)