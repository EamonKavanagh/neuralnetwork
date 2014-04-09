import numpy as np

def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))
        
def stepSize(mu,gamma,eta):
    phi = eta*(1-mu)
    if mu > 1:
        return gamma
    else:
        return (1+phi)*gamma
    
def calcAccuracy(Network, newData=None, digits=None):
    prediction = Network.predict(newData)
    if digits is None:
        digits = Network.digits
    correct = prediction == digits
    return float(sum(correct))/len(digits)