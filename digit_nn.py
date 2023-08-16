# one hidden layer NN
# https://medium.com/p/68998a08e4f6

import numpy as np
import csv
import matplotlib.pyplot as plt


# apply on entire array
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_drtv(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)
    # return x * (1 - x)
    # return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

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
            # self.w1 = np.random.rand(inputN, hiddenN)
            # self.w2 = np.random.rand(hiddenN, outputN)
            self.w1 = np.random.uniform(-1.,1.,size=(inputN, hiddenN))/np.sqrt(inputN*hiddenN).astype(np.float32)
            self.w2 = np.random.uniform(-1.,1.,size=(hiddenN, outputN))/np.sqrt(hiddenN*outputN).astype(np.float32)
    
    # pass through input of X
    def feed_fwd(self, X):
        # self.hiddenL = sigmoid(np.dot(X, self.w1))
        # self.output_res = softmax(sigmoid(np.dot(self.hiddenL, self.w2)))
        # # self.output_res = sigmoid(np.dot(self.hiddenL, self.w2))
        # return self.output_res

        self.x_l1 = np.dot(X, self.w1)
        self.x_sigmoid = sigmoid(self.x_l1)
        self.x_l2 = self.x_sigmoid.dot(self.w2)
        # self.out = softmax(self.x_l2)

        self.out = np.copy(self.x_l2)
        for ind,x in enumerate(self.out):
            self.out[ind] = softmax(x)
        # print(self.sigmoid_out, self.output_res)
        return self.out
    
    # use results Y to adjust weights
    def back_propogate(self, X, Y):
        # loss_calc = 2 * (Y - self.outputL) * sigmoid_drtv(self.outputL)
        # loss_calc = 2 * (Y - self.outputL)/10 * d_softmax(self.sigmoid_out)
        # w2_delta = np.dot(self.hiddenL.T, loss_calc)
        # w1_delta = np.dot(
        #     X.T, np.dot(loss_calc, self.w2.T) * sigmoid_drtv(self.hiddenL)
        # )

        # loss_calc = 2 * (Y - self.outputL)/10 * d_softmax(self.sigmoid_out)
        # w2_delta = np.dot(self.hiddenL.T, loss_calc)
        # w1_delta = np.dot(
        #     X.T, np.dot(self.w2, loss_calc.T).T * sigmoid_drtv(self.x1_p)
        # )

        error=2*(Y - self.outputL)/self.out.shape[0]*d_softmax(self.x_l2)
        w2_delta=self.x_sigmoid.T@error
        
        
        error=((self.w2).dot(error.T)).T*sigmoid_drtv(self.x_l1)
        w1_delta=X.T@error

        # loss_calc_2=((l2).dot(error.T)).T*d_sigmoid(x_l1p)

        self.w2 -= w2_delta*0.001
        self.w1 -= w1_delta*0.001

    def train(self, X, Y):
        self.outputL = self.feed_fwd(X)
        self.back_propogate(X, Y)

    def saveWeights(self):
        np.save('w1.npy', self.w1)
        np.save('w2.npy', self.w2)

TRAINING_DATA_LEN = 60000
# csvfile = open('mnist_train.csv')
# batch_size = 256
# inputN = 28*28
# hiddenN = 200
# outputN = 10
csvfile = open('dummy_train.csv')
TRAINING_DATA_LEN = 1
batch_size = 8
inputN = 3
hiddenN = 128
outputN = 2
data = np.loadtxt(csvfile, delimiter=',')
loadData = False # set false to start training from scratch

# load set
with open('sample_num.txt','r') as file:
    sample_num = int(file.read())
    if loadData == False or sample_num + batch_size >= TRAINING_DATA_LEN:
        sample_num = 0

nn = NeuralNetwork(inputN, hiddenN, outputN, loadData)

accuracy = 100
for epoch in range(20000):

    # get current slice of training data and call function
    while sample_num < TRAINING_DATA_LEN:
        # print('loading samples:', sample_num,'to', sample_num+batch_size-1)
        with open('sample_num.txt','w') as file:
            file.write(str(sample_num+batch_size))

        training_data = data[sample_num:sample_num+batch_size]
        X = training_data[:,1:]
        outputs = training_data[:,0]
        Y = np.zeros((len(outputs), outputN), dtype=float)
        for ind, ans in enumerate(outputs):
            Y[ind][int(ans)] = 1.0
            # Y[ind][0] = outputs[ind]
        nn.train(X, Y)
        sample_num += batch_size
    sample_num = 0

    if epoch % 1000 == 0:
        print("--- epoch #", epoch)
        # print("Input : \n" + str(X))
        print("Actual Output: \n" + str(Y))
        print("Predicted Output: \n" + str(nn.feed_fwd(X)))
        print(
            "Loss: " + str(np.mean(np.square(Y - nn.feed_fwd(X))))
        )  # mean sum squared loss

    if (epoch % 10 == 0):
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
