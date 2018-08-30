#neural network
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import math


def produceWeightsMatrix(breadthLayers):
	weightsMatrix = []
	for i in range(len(breadthLayers)):
		weightsMatrix.append(np.array([1]*breadthLayers[i]))

	return weightsMatrix

def sigmoidFunction(x, reCalib = False):
	if reCalib == True:
		return x/(1-x)
	return 1/(1+np.exp(-x))


def computeError(prediction, y):
	return np.sum((prediction-y)*(prediction-y)*.5)/len(y)

def computeFinalDelta(y, p, hf):
	return -1*(y-p.T)*p.T*(1-p.T)*hf.T

def makePredictions(p, y):
	predictions = []
	for i in range(len(y)):
		if p[i][1]>p[i][2]:
			predictions.append(1)
		else:
			predictions.append(0)
	return predictions
# def computeHiddenDeltas(p, ho, hi):
# 	return -1*(p-ho)*hi*(1-hi)*hf.T


# def gradientDescent(predictedWeights, actualWeights):
X, y = make_moons(n_samples = 200, noise=0.20)
print y.shape
n_dim = 2
n_hidden_dimensions = 3
n_hidden_layers = 2
n_output_dimensions = 1

w1 = np.random.random((n_dim, n_hidden_dimensions))
b1 = np.zeros((1, n_hidden_dimensions))

w2 = np.random.random((n_hidden_dimensions, n_hidden_dimensions))
b2 = np.zeros((1, n_hidden_dimensions))

w3 = np.random.random((n_hidden_dimensions, n_output_dimensions))
b3 = np.zeros((1, n_output_dimensions))
print X.shape
#layer 0 = x, layer 1 = hidden, layer 2 = hidden, layer 3 = hidden, layer 4 = output
learning_rate = 0.5
for i in range(20000):

	#forward propogation
	z1 = X.dot(w1)+b1
	a1 = sigmoidFunction(z1)

	z2 = a1.dot(w2)+b2
	a2 = sigmoidFunction(z2)

	z3 = a2.dot(w3)+b3
	p = sigmoidFunction(z3)

	#back propogation
	for i in range(n_hidden_dimensions):
			w3[i]-=learning_rate*np.sum(computeFinalDelta(y,p,a2.T[i]))/len(y)

	for i in range(n_hidden_dimensions):
		for j in range(n_hidden_dimensions):
			w2[i][j]-=learning_rate*np.sum(computeFinalDelta(p,a2.T[i],a1.T[j]))/len(y)

	for i in range(n_hidden_dimensions):
		for j in range(n_dim):
			w1.T[i][j]-=learning_rate*np.sum(computeFinalDelta(a2.T[i],a1.T[i],X.T[j]))/len(y)

print roc_auc_score(y,p)

auc_scores = []
x_range = []
for i in range(100):
	x_range.append(i*1.0/100.0)
	auc_scores.append(roc_auc_score(y,(p > (i*1.0/100.0))*1))
print x_range
print auc_scores

plt.plot(x_range, auc_scores)
plt.show()