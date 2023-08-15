# https://medium.com/p/68998a08e4f6

import numpy as np

# one hidden layer NN


# apply on entire array
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_drtv(x):
    return x * (1 - x)
    # return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self, X, Y):
        self.y = Y

        # layers/nodes
        self.inputL = X  # np.zeros_like(X)
        self.hiddenL = None
        self.outputL = np.zeros_like(Y)
        inputN = len(X[0])
        hiddenN = 4
        outputN = len(Y[0])

        # weights/edges
        # inputL to hiddenL; hiddenL to outputL
        self.w1 = np.random.rand(inputN, hiddenN)
        self.w2 = np.random.rand(hiddenN, outputN)

    def feed_fwd(self, inputL=None):
        if inputL is None:
            inputL = self.inputL
        self.hiddenL = sigmoid(np.dot(inputL, self.w1))
        self.output_res = sigmoid(np.dot(self.hiddenL, self.w2))
        return self.output_res

    def back_propogate(self):
        loss_calc = 2 * (self.y - self.outputL) * sigmoid_drtv(self.outputL)
        w2_delta = np.dot(self.hiddenL.T, loss_calc)
        w1_delta = np.dot(
            self.inputL.T, np.dot(loss_calc, self.w2.T) * sigmoid_drtv(self.hiddenL)
        )

        self.w2 += w2_delta
        self.w1 += w1_delta

    def train(self):
        self.outputL = self.feed_fwd()
        self.back_propogate()


# initialize weights (inputs x hidden, hidden x out)
# each weight column belongs to one output row
X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
Y = np.array(([0], [1], [1], [0]), dtype=float)
nn = NeuralNetwork(X, Y)

for i in range(1500):
    nn.train()
    if i % 100 == 0:
        print("for iteration # " + str(i))
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(Y))
        print("Predicted Output: \n" + str(nn.feed_fwd()))
        print(
            "Loss: " + str(np.mean(np.square(Y - nn.feed_fwd())))
        )  # mean sum squared loss

# test new values
if False:
    print("\n\n---- Final Testing ---- ")
    print("Input : \n" + str(X))
    print("Actual Output: \n" + str(Y))
    print("Predicted Output: \n" + str(nn.feed_fwd([0, 0, 1])))
    print("Loss: " + str(np.mean(np.square([[0]] - nn.feed_fwd([0, 0, 1])))))
