# one hidden layer NN
# https://medium.com/p/68998a08e4f6

import numpy as np
import csv
import matplotlib.pyplot as plt


# apply on entire array
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_drtv(x):
    # return x * (1 - x)
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    exponents = np.exp(x)
    return exponents/np.sum(exponents)

def render_arr(arr):
    vals = np.reshape(arr, (28,28))
    plt.imshow(np.array(vals, dtype=float))
    plt.show()

class NeuralNetwork:
    def __init__(self, inputN, hiddenN, outputN, loadData):
        # layers/nodes
        self.hiddenL = None
        self.outputL = np.zeros(outputN)

        # weights/edges
        # inputL to hiddenL; hiddenL to outputL
        if loadData:
            self.w1 = np.load('w1.npy')
            self.w2 = np.load('w2.npy')
        else:
            self.w1 = np.random.rand(inputN, hiddenN)
            self.w2 = np.random.rand(hiddenN, outputN)

    # pass through input of X
    def feed_fwd(self, X):
        self.hiddenL = sigmoid(np.dot(X, self.w1))
        self.output_res = softmax(sigmoid(np.dot(self.hiddenL, self.w2)))
        return self.output_res

    # use results Y to adjust weights
    def back_propogate(self, X, Y):
        loss_calc = 2 * (Y - self.outputL) * sigmoid_drtv(self.outputL)
        w2_delta = np.dot(self.hiddenL.T, loss_calc)
        w1_delta = np.dot(
            X.T, np.dot(loss_calc, self.w2.T) * sigmoid_drtv(self.hiddenL)
        )

        self.w2 += w2_delta
        self.w1 += w1_delta

    def train(self, X, Y):
        self.outputL = self.feed_fwd(X)
        self.back_propogate(X, Y)

    def saveWeights(self):
        np.save('w1.npy', self.w1)
        np.save('w2.npy', self.w2)

csvfile = open('mnist_train.csv')
data = np.loadtxt(csvfile, delimiter=',')
TRAINING_DATA_LEN = 60000
loadData = False # set false to start training from scratch
batch_size = 40

# load set
with open('sample_num.txt','r') as file:
    sample_num = int(file.read())
    if loadData == False or sample_num + batch_size >= TRAINING_DATA_LEN:
        sample_num = 0

inputN = 28*28
hiddenN = 2
outputN = 10
nn = NeuralNetwork(inputN, hiddenN, outputN, loadData)

for epoch in range(100):

    # get current slice of training data and call function
    while sample_num < TRAINING_DATA_LEN:
        # print('loading samples:', sample_num,'to', sample_num+batch_size-1)
        with open('sample_num.txt','w') as file:
            file.write(str(sample_num+batch_size))

        training_data = data[sample_num:sample_num+batch_size]
        X = training_data[:,1:]
        outputs = training_data[:,0]
        Y = np.zeros((len(outputs),10), dtype=float)
        for ind, ans in enumerate(outputs):
            Y[ind][int(ans)] = 1.0

        nn.train(X, Y)
        sample_num += batch_size

    print("--- epoch #", epoch)
    # print("Input : \n" + str(X))
    print("Actual Output: \n" + str(Y[0]))
    print("Predicted Output: \n" + str(nn.feed_fwd(X)[0]))
    print(
        "Loss: " + str(np.mean(np.square(Y - nn.feed_fwd(X))))
    )  # mean sum squared loss
    sample_num = 0
    nn.saveWeights()

# test new values
if False and q % 10 == 0:
    next_ind = np.random.randint(TRAINING_DATA_LEN)
    sample_input = data[next_ind][1:]
    sample_output = data[next_ind][0]

    print("\n\n---- Final Testing ---- ")
    # print("Input : \n" + str(X))
    print("Actual Output: \n" + str(sample_output))
    print("Predicted Output: \n" + str(nn.feed_fwd(sample_input)))
    print("Loss: " + str(np.mean(np.square([[sample_output]] - nn.feed_fwd(sample_input)))))
    render_arr(sample_input)
