nt("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print(
            "Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))
        )  # mean sum squared loss
        pri