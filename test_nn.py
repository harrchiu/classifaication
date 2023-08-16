# one hidden layer NN
# https://medium.com/p/68998a08e4f6

import numpy as np
import copy 
import csv
import matplotlib.pyplot as plt


# apply on entire array
def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

def render_arr(arr):
    vals = np.reshape(arr, (28,28))
    plt.imshow(np.array(vals, dtype=float))
    plt.show()

class NeuralNetwork:
    def __init__(self):
        self.w1 = np.load('w1.npy')
        self.w2 = np.load('w2.npy')
    
    # pass through input of X
    def feed_fwd(self, X):
        self.x_l1 = X.dot(self.w1)
        self.x_sigmoid = sigmoid(self.x_l1)
        self.x_l2 = self.x_sigmoid.dot(self.w2)
        # self.out = softmax(self.x_l2)

        self.out = copy.deepcopy(self.x_l2)
        for ind,x in enumerate(self.x_l2):
            self.out[ind] = softmax(x)
        # print(self.sigmoid_out, self.output_res)
        return self.out
    

TRAINING_DATA_LEN = 10000
csvfile = open('mnist_test.csv')
inputN = 28*28
hiddenN = 128
outputN = 10
# csvfile = open('dummy_train.csv')
# TRAINING_DATA_LEN = 9
# batch_size = 3
# inputN = 12
# hiddenN = 128
# outputN = 4
data = np.loadtxt(csvfile, delimiter=',')
nn = NeuralNetwork()

np.random.shuffle(data)

X = data[:,1:]
outputs = data[:,0]
Y = np.zeros((len(outputs), outputN), dtype=float)
for ind, ans in enumerate(outputs):
    Y[ind][int(ans)] = 1.0

print(X, X.shape, Y, Y.shape)
nn.feed_fwd(X)

correct = 0
for ind,res in enumerate(nn.out):
    if np.argmax(res) == int(outputs[ind]):
        correct += 1
    else:
        print('correct: ',int(outputs[ind]), 'output: ',np.argmax(res))
        # render_arr(X[ind])

print('accuracy:', correct/len(X))
# print("Input : \n" + str(X[0]))
print("Actual Output: \n" + str(Y[0]))
print("Predicted Output: \n" + str(nn.feed_fwd(X)[0]))
print(
    "Loss: " + str(np.mean(np.square(Y - nn.feed_fwd(X))))
)  # mean sum squared loss

