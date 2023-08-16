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

def sigmoid_drtv(x):
    x = np.clip(x, -20, 20)
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
        # self.hiddenL = None
        # self.outputL = np.zeros(outputN)

        # weights/edges
        # inputL to hiddenL; hiddenL to outputL
        if loadData:
            self.w1 = np.load('w1.npy')
            self.w2 = np.load('w2.npy')
        else:
            self.w1 = (np.random.uniform(-1.,1.,size=(inputN, hiddenN))/np.sqrt(inputN*hiddenN)).astype(np.float32)
            self.w2 = (np.random.uniform(-1.,1.,size=(hiddenN, outputN))/np.sqrt(hiddenN*outputN)).astype(np.float32)
    
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
    
    # use results Y to adjust weights
    def back_propogate(self, X, Y):
        error=2*(self.out - Y)/self.out.shape[0]*d_softmax(self.x_l2)
        w2_delta=self.x_sigmoid.T@error
        
        
        error=((self.w2).dot(error.T)).T*sigmoid_drtv(self.x_l1)
        w1_delta=X.T@error

        # loss_calc_2=((l2).dot(error.T)).T*d_sigmoid(x_l1p)

        # self.w2 -= w2_delta*0.001
        # self.w1 -= w1_delta*0.001
        # print('training', X, Y, self.out)
        # print(w2_delta, w1_delta)

    # forward and backward pass
    def train(self, X, Y):
        self.out = self.feed_fwd(X)
        self.back_propogate(X, Y)

    def saveWeights(self):
        np.save('w1.npy', self.w1)
        np.save('w2.npy', self.w2)

TRAINING_DATA_LEN = 60000
csvfile = open('mnist_train.csv')
batch_size = 128
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
loadData = True # set false to start training from scratch

# load set
with open('sample_num.txt','r') as file:
    try:
        sample_num = int(file.read())
    except:
        sample_num = 0
    if loadData == False or sample_num + batch_size >= TRAINING_DATA_LEN:
        sample_num = 0

nn = NeuralNetwork(inputN, hiddenN, outputN, loadData)

accuracies = []
for epoch in range(10000):

    # pre shuffle acc max: 0.3671875
    np.random.shuffle(data)
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
        nn.train(X, Y)
        correct = 0
        for ind,res in enumerate(nn.out):
            if np.argmax(res) == int(outputs[ind]):
                correct += 1
        accuracies.append(correct/len(X))

        sample_num += batch_size
    sample_num = 0

    if epoch % 10 == 0:
        print('accuracy:', np.average(accuracies[-10:-1]))
        print("--- epoch #", epoch)
        # print("Input : \n" + str(X[0]))
        print("Actual Output: \n" + str(Y[0]))
        print("Predicted Output: \n" + str(nn.feed_fwd(X)[0]))
        print(
            "Loss: " + str(np.mean(np.square(Y - nn.feed_fwd(X))))
        )  # mean sum squared loss

    if (epoch % 50 == 0):
        nn.saveWeights()

# test new values
if True:
    next_ind = np.random.randint(TRAINING_DATA_LEN)
    sample_input = np.array([data[next_ind][1:]])
    
    activated_index_output = int(data[next_ind][0])
    sample_output = np.zeros(outputN)
    sample_output[activated_index_output] = 2.0

    print(sample_input, sample_output)

    print("\n\n---- Final Testing ---- ")
    # print("Input : \n" + str(X))
    print("Actual Output: \n" + str(sample_output))
    print("Predicted Output: \n" + str(nn.feed_fwd(sample_input)))
    print("Loss: " + str(np.mean(np.square(sample_output - nn.feed_fwd(sample_input)))))
    # render_arr(sample_input)
