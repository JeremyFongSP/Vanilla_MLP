# import math
import csv
import numpy as np
import pandas as pd
import pdb
import cProfile, pstats, io
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# np.seterr(divide='ignore', invalid='ignore')


class Nnet:


    def __init__(self, inputsize, hiddensize, hiddennum, labels, lmbda, alpha,
                 iternum):
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.hiddennum = hiddennum
        self.labels = labels
        self.lmbda = lmbda
        self.alpha = alpha
        self.iternum = iternum
        self.thetanum = hiddennum + 1
        self.thetas = []
        self.hist_cost = np.zeros(iternum)
        self.hist_acc = np.zeros(iternum)
        self.cost = 0
        self.p = 0
        self.sizesarray = np.array([inputsize, labels])
        for i in range(hiddennum):
            self.sizesarray = np.insert(self.sizesarray, 1, self.hiddensize)

    def getthetas(self, df):
        tsizetotal = 0
        for idx in range(self.thetanum):
            tsizetotal += (self.sizesarray[idx]+1) * self.sizesarray[idx + 1]

        # Starting point at thetas in 'Results.csv'
        start = df.iloc[-tsizetotal:]
        thetas = []

        # From start to length of theta, place in theta variable then
        # move start to new location
        for i in range(self.thetanum):
            tsize = (self.sizesarray[i] + 1) * self.sizesarray[i + 1]
            thetas.append(np.array(start.iloc[0:int(tsize)]))
            start = start.iloc[int(tsize):]
            thetas[i] = thetas[i].reshape(self.sizesarray[i] + 1,
                                          self.sizesarray[i + 1])
        self.thetas = thetas

    def computecostfunction(self, X, y, numexamples):
        labels = np.arange(self.labels)
        y1 = np.equal(y, labels).astype(int)
        a = [X]                 # a = X
        z = []

        np.set_printoptions(threshold=np.inf)
        # pdb.set_trace()
        # Find all a and z for all layers
        # pdb.set_trace()
        for idx in range(self.thetanum):
            a[idx] = np.array(np.c_[np.ones((numexamples, 1)), a[idx]])
            # Add 1s
            z.append(np.matmul(a[idx], self.thetas[idx]))
            # Compute z
            a.append(sigmoid(z[idx]))
            # Sigmoid z

        # Gamma
        # gamma = 1

        # Cost Function Computation
        # J = (1/numexamples)*np.sum((np.sum((
        #     np.multiply(-y1, (1-a[-1])**gamma*np.log(a[-1])) - np.multiply((1-y1), (1-a[-1])**gamma*np.log(1-a[-1]))), axis=0)), axis=0)
        # J = (1/numexamples)*np.sum((np.sum((
            # np.multiply(-y1, np.log(a[-1])) - np.multiply((1-y1), np.log(1-a[-1]))), axis=0)), axis=0)
        # The np.where(...) prevents the program from computing np.log(0) and outputting nan to history
        J = (1/numexamples)*np.sum((np.sum((
            np.multiply(-y1, np.log(a[-1])) - np.multiply((1-y1), np.log(1 - np.where(a[-1] > 0.99999999999, 0.99999999999, a[-1])))), axis=0)), axis=0)
        # J = (1/numexamples)*np.sum((np.sum((
        #     np.multiply(-y1, np.where(np.absolute(a[-1]) > 0.000001, np.log(a[-1]), 10**-5)) - np.multiply((1-y1), np.where(np.absolute(1-a[-1]) > 0.000001, np.log(1-a[-1]), 10**-5))), axis=0)), axis=0)

        # pdb.set_trace()
        # Regularization
        regsum = 0
        for i in range(self.thetanum):
            regsum += sum(sum(np.power(self.thetas[i][:, 1::], 2)))
    #        print(regsum)
        reg = self.lmbda/(2*numexamples) * regsum

        return J + reg

    # @profile    # This function takes 0.339 seconds to run
    def backpropagation(self, X, y, numexamples):
        Del = []
        DelT = []
        for i in range(self.thetanum):
            Del.append(np.transpose(np.zeros(np.shape(self.thetas[i]))))
            DelT.append(np.transpose(np.zeros(np.shape(self.thetas[i]))))

        labels = np.arange(self.labels)
        y1 = np.equal(y, labels).astype(int)

        for i in range(numexamples):
    #        a = [X[i]]
            a = [object]*(self.thetanum+1)
            a[0] = X[i]
            z = [object]*(self.thetanum)

            for idx in range(self.thetanum):
    #            a[idx] = np.array(np.append(1, a[idx]))  # Add 1s
                a[idx] = np.concatenate(([1], a[idx]), axis=None)
                # Faster than np.append
    #            a[idx] = np.insert(a[idx], 0, 1)      # This takes much more time to run
                z[idx] = (np.array(np.array(np.matmul(a[idx],
                                            self.thetas[idx]))))
                # Compute z
    #            z.append(np.array(np.array(np.matmul(a[idx], thetas[idx]))))
    # Compute z
    #            a.append([sigmoid(z[idx])])  # Sigmoid z
                a[idx+1] = [sigmoid(z[idx])]
    #       a0 = 1x5        inc. bias
    #       a1 = 1x6        inc. bias
    #       a2 = 1x6        inc. bias
    #       a3 = 1x3        no bias

    #       z0 = 1x5        a0 x theta[0] => 1x5 x 5x5 = 1x5
    #       z1 = 1x5        a1 x theta[1] => 1x6 x 6x5 = 1x5
    #       z2 = 1x3        a2 x theta[2] => 1x6 x 6x3 = 1x3

    #       theta0 = 5x5
    #       theta1 = 6x5
    #       theta2 = 6x3
            d = [np.transpose(a[-1] - y1[i])]

            ui = 0
            for idx in range(self.thetanum-1, 0, -1):
                d.append(np.multiply(np.matmul(self.thetas[idx][1:, :], d[ui]), np.transpose([(sigmoidgradient(z[idx-1]))])))   # 3x3 x 3x1 = 3x1 .x 3x1
                ui += 1
            d.reverse()

            for idx in range(self.thetanum):
    #            Del[idx] = Del[idx] + np.matmul(d[idx], [a[idx]])
                Del[idx] = np.matmul(d[idx], [a[idx]])
                DelT[idx] = np.transpose(Del[idx])
    #            print("d[idx]: ", d[idx], " a[idx]: ", a[idx])
    #            thetas[idx] = thetas[idx] - alpha*DelT[idx]
    #         print("thetas")
    #         print(thetas)
    #         print("DelT")
    #         print(DelT)
    #         print("alpha * DelT")
    #         print(np.multiply(alpha, DelT))
    #         exit(1)
            self.thetas = np.subtract(self.thetas, np.multiply(self.alpha, DelT))
    #    exit(1)
    #    print("Del: ")
    #    print(Del)
    #   When using stochastic gradient descent, instead of returning Del,
    #   return thetas. Perform gradient descent after each example
    #    print("New thetas:")
    #    print(thetas)
        #self.thetas = thetas
    #    return Del

    def trainneuralnet(self, X, y, numexamples):

        # Initialize Theta and x to random values
        # thetas = []
        for i in range(self.thetanum):
            self.thetas.append(np.array(np.random.rand(self.sizesarray[i] + 1,
                                                  self.sizesarray[i + 1])
                                   * 0.01))

        for idx, val in enumerate(self.thetas):
            print("Initial Theta%i [%i X %i]:" % (idx, self.sizesarray[idx] +
                                                  1, self.sizesarray[idx + 1]))
            print(val)

        try:
            print("Opening File...")
            fw = open("Results.csv", "w+")
            fw2 = open("History.csv", "w+")
            print("Overwriting Results.csv & History.csv")
        except PermissionError:
            print("File is currently opened. Close and try again.")
            exit()

        J = np.array(np.zeros((self.iternum, 1)))  # TO-DO: Use run until error is small
        try:
            for i in tqdm(range(self.iternum)):
                J[i] = self.computecostfunction(X, y, numexamples)
                #    print("cost: ")
                #    print(J)
    #            Del = backpropagation(X, y, lmbda, thetas, numexamples)
                self.backpropagation(X, y, numexamples)
                #TO-DO: Add regularization
                #    print("Del: ")
                #    print(Del)

                # Regularization (0s appended for bias term)
    #            reg = []
    #            grad = []

    #            for idx in range(THET_NUM):
    #                reg.append(
    #                    (lmbda / numexamples) * np.append(np.zeros((1, thetas[idx].shape[1])), thetas[idx][1::, :], axis=0))
    #                grad.append(np.add((1 / numexamples) * Del[idx], np.transpose(reg[idx])))
    #                thetas[idx] = thetas[idx] - np.transpose(alpha * grad[idx])  # Thetas permanently change
                #    print("grad: ")
                #    print(grad)
                #    print("thetas: ")

                # Export Costs to Results.csv
                fw.write(str(J[i][0]) + "\n")

                # Prepare Cost and Accuracy History
                self.hist_cost[i] = J[i][0]
                self.predict(X)
                h0 = np.array(self.p)
                self.hist_acc[i] = np.mean(np.array(np.equal(h0[1], np.transpose(y))).astype(int))*100

                # Print Progress Bar & Iteration Number
                # print(progbar(i, self.iternum),
                #       "- {iter}/{tot_iter} completed".format(iter=i,
                #                         tot_iter=self.iternum), end='\r')
                # if i % 25 == 0
                #    print("\r%s iterations completed..." % (str(i)), end='\r')

            # # Create Plots
            # epochs = range(numiteration)
            #
            # # Cost Function
            # plt.plot(epochs, hist_cost, 'r', label='Cost')
            # plt.title('Training Cost vs Epochs')
            # plt.figure()
            #
            # # Accuracy
            # plt.plot(epochs, hist_acc, 'b', label='Acc')
            # plt.title('Accuracy vs Epochs')

            # Flatten Thetas for export
            thetasflattened = self.thetas.copy()
            for idx in range(self.thetanum):
                thetasflattened[idx] = thetasflattened[idx].flatten().tolist()
            thetasflattened = sum(thetasflattened, [])
            #        print(thetasflattened)
            #        print(thetas)

            for i in range(np.size(thetasflattened)):
                fw.write(str(thetasflattened[i]) + "\n")
            fw.close()

            for n in range(self.iternum):
                fw2.write(str(self.hist_cost[n]) + "," + str(self.hist_acc[n])
                          + "\n")
            fw2.close()

            print("Data written to Results.txt successfully")
        except AssertionError:
            print("An error has occurred when looping or writing data")
            exit(1)

        # self.thetas = thetas

    def predict(self, X):
        m = np.size(X, axis=0)
        p = []
    #    p = list([np.zeros((m, 1)), np.zeros((m, 1))])
        h = [X]
        for idx in range(self.thetanum):
            h.append(sigmoid(np.matmul(np.array(np.c_[np.ones((m, 1)),
                                                h[idx]]), self.thetas[idx])))
    #    print(h[-1])
        p.append(np.array(np.amax(h[-1], axis=1)))      # percent
        p.append(np.array(np.argmax(h[-1], axis=1)))    # label
    #    p[0] = np.array(np.amax(h2, axis=1))        # percent
    #    p[1] = np.array(np.argmax(h2, axis=1))      # label
        np.set_printoptions(threshold=np.inf)
        #    print("p")
    #    print(p)
        self.p = p

    def printpredict(self, h0, y):
        if self.labels == 1:
            print(str(np.mean(np.array(np.equal(np.round(h0[0]), np.transpose(y))).astype(int))*100) + " %")
        else:
            print(str(np.mean(np.array(np.equal(h0[1], np.transpose(y))).astype(int))*100) + " %")
        return


def profile(fnc):

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoidgradient(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z))

'''
def computenumericalgradient(X, y, lmbda, thetas, numexamples):
    nn_params = np.append(thetas[0].flatten(), thetas[1].flatten(), axis=0)
#    print("nn_params: ")
#    print(nn_params)
    numgrad = np.array(np.zeros(nn_params.shape))
#    print("numgrad: ")
#    print(numgrad)
    perturb = np.array(np.zeros(nn_params.shape))
#    print("perturb: ")
#    print(perturb)

    eps = 1e-4
    for p in range(nn_params.size):
        perturb[p] = eps
        p1eps = perturb[0:thetas[0].size].reshape(thetas[0].shape)
        p2eps = perturb[thetas[0].size:thetas[0].size+thetas[1].size].reshape(thetas[1].shape)
        thetastempminus = [thetas[0] - p1eps, thetas[1] - p2eps]
        thetastempplus = [thetas[0] + p1eps, thetas[1] + p2eps]
        loss1 = computecostfunction(X, y, lmbda, thetastempminus, numexamples)
#        print("loss1: ")
#        print(loss1)
        loss2 = computecostfunction(X, y, lmbda, thetastempplus, numexamples)
#        print("loss2: ")
#        print(loss2)
        numgrad[p] = (loss2 - loss1) / (2*eps)
#        print("numgrad: ")
#        print(numgrad)
        perturb[p] = 0
    temp1 = np.array(numgrad[0:thetas[0].size].reshape(thetas[0].shape))
    temp2 = np.array(numgrad[thetas[0].size:thetas[0].size+thetas[1].size].reshape(thetas[1].shape))
    numgrad = np.array([np.transpose(temp1), np.transpose(temp2)])
    return numgrad


def gradientcheck():
    # Test data
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    lmbda = 0
    Theta1 = np.array((np.random.rand(input_layer_size+1, hidden_layer_size)*2-1)/10)    # 4x5
    Theta2 = np.array((np.random.rand(hidden_layer_size+1, num_labels)*2-1)/10)          # 6x3
    Thetas = [Theta1, Theta2]
    xtemp1 = np.array((np.random.rand(m, input_layer_size)*2-1)/10)                      # 5x3
    y = np.array([[1], [2], [0], [1], [2]])
    print("y:")
    print(y)
    Del = backpropagation(xtemp1, y, lmbda, Thetas, m)
    grad = np.array([(1/m)*Del[0], (1/m)*Del[1]])
    numgrad = computenumericalgradient(xtemp1, y, lmbda, Thetas, m)

    print("Grad:")
    print(grad)
    print("Numerical Grad:")
    print(numgrad)
    # Compare Gradient to Numerical Gradient
    comp = grad - numgrad

    print("Compare:")
    print(comp)
'''

def progbar(i, total_iter):
    prog_bar = '|          |'
    prog_list = list(prog_bar)
    frac = int(i/total_iter*10)
    for n in range(frac):
        prog_list[n+1] = '>'
    prog_bar = ''.join(prog_list)
    return prog_bar


# Notes:
# 1) Code is not optimized for performance, simply a proof of concept
# 2) In order for the program to work, we need to give it a file that has the
#    data with examples as rows and features as columns. The last column of the
#    file must be the labels y as it extracts the last column for supervised
#    learning. The result is stored in a file called 'Results.csv' along with
#    the Theta values (Weights) at the end of the files
# 3) If the model has already been trained, the theta values are taken from the
#    'Results.csv' and goes through the examples to return a prediction percent
# 4) In the case of multiple hidden layers, the algorithm currently takes a long
#    time to learn therefore it is recommended to change the value of alpha (the
#    learning rate) for something between 1 and 10 (for single hidden layer, 0.3
#    performs well). The disadvantage is that the data tends to oscillate, which
#    is often expected when alpha is high.
# 5) Stochastic gradient descent was implemented which significantly increased
#    performance and reduced training time. For some reason, the OCTAVE examples
#    does not work properly when training the neural net. Need to verify if code
#    works for bigger matrices/ number of examples
#
# Notes: Remember to change IN_SIZE if using a different data set


### ---------------------- Begin Program ------------------- ###


def main():
    # Initialization
    IN_SIZE = 400           # bias not included
    HID_SIZE = 256         # bias not included
    HID_NUMS = 1
    LABEL_NUM = 10
    # THET_NUM = HID_NUMS + 1
    # sizesarray = np.array([IN_SIZE, LABEL_NUM])
    # for h in range(HID_NUMS):
    #     sizesarray = np.insert(sizesarray, 1, HID_SIZE)
    numexamples = 0              # number of examples counted from file length
    X = []
    xtemp = []
    LMBDA = 0
    ALPHA = .3
    print("Set number of iterations: ")
    while 1:
        try:
            NUM_ITER = int(input())
            break
        except ValueError:
            print("Wrong type, try again")

    start_load = time.time()
    mynnet = Nnet(IN_SIZE, HID_SIZE, HID_NUMS, LABEL_NUM, LMBDA, ALPHA,
                  NUM_ITER)
    print('{:^24s}'.format("--Neural Network--"))
    print('{:^24s}'.format(str(mynnet.sizesarray)))

    fileno = "OCT"

    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, '../Data/%s_Made_up_data.csv' % fileno)
    # Variables for history
    # hist_cost = np.zeros(numiteration)
    # hist_acc = np.zeros(numiteration)

    # Open Data File
    with open(filename, "r") as fr:
    # with open("Sleep_data_training_set.csv", "r") as fr:
        fr.readline()  # Skip first line
        csvReader = csv.reader(fr)
        for row in csvReader:
            xtemp.append(row)
            numexamples += 1

    X = np.array(xtemp).astype(np.float)
    y = np.array(X[:, mynnet.inputsize:])
    X = X[:, :mynnet.inputsize]

    # Check if gradient works, disable gradientcheck when actually training NN
    # gradientcheck()
    fr.close()
    print("Load time:", time.time() - start_load, "\n")
    # Prompt
    print('{:^24s}'.format("Use existing data? (Y/N)"))
    ans = input().upper()
    while ans != 'E' and ans != 'Y' and ans != 'N':
        print("Invalid input, try again:")
        ans = input().upper()

    # Use Thetas at end of file to make predictions, skip to end of program
    if ans == 'E':
        exit(0)
    elif ans == 'Y':
        start_time = time.time()
        try:
            filename = 'Results.csv'
            df = pd.read_csv(filename, header=None)
            mynnet.getthetas(df)
            thetas = mynnet.thetas
            # thetas = getthetas(df, sizesarray, THET_NUM)
        except FileNotFoundError:
            print("No existing data, train model before using this option")
            print("Program has ended")
            exit(0)

        for idx, val in enumerate(thetas):
            print("Using Theta%i as:" % idx)
            print(val)
        fr.close()
    # Make new file and start training model
    # Make new file and start training model
    else:
        start_time = time.time()
        mynnet.trainneuralnet(X, y, numexamples)
        thetas = mynnet.thetas
        # thetas = trainneuralnet(X, y, lmbda, sizesarray, numexamples, THET_NUM, numiteration)

    print("Trained in %s seconds" % (time.time() - start_time))
    mynnet.predict(X)
    # h0 = np.array(predict(thetas, X))
    h0 = np.array(mynnet.p)
    print("\n*********************")
    print("Training Set Accuracy")
    # print(y)
    # print(str(np.mean(np.array(np.equal(h0[1], np.transpose(y))).astype(int))*100) + " %")
    mynnet.printpredict(h0, y)
    # printpredict(h0, y, LABEL_NUM)
    print("*********************")

    while 1:
        print("----------------------------------")
        print("1- Type DATA for number prediction")
        print("2- Type UPLOAD to select file")
        print("3- Type PLOT to graph")
        print("4- Type EXIT to end")
        print("5- Type SHOW [number] to select image")
        print("----------------------------------")
        print("Command: ")
        ans = input()
        input_list = ans.split(' ')
        if str(input_list[0]).upper() == 'EXIT':
            break
        elif str(input_list[0]).upper() == 'DATA':
            xs = np.zeros((IN_SIZE, 1))
            for k in range(IN_SIZE):
                while 1:
                    print("data " + str(k + 1) + ":")
                    ans = input()
                    try:
                        xs[k] = float(ans)
                        break
                    except ValueError:
                        print("Not a number, try again:")
                        continue
            mynnet.predict(np.transpose(xs))
            pred = mynnet.p
            # pred = predict(thetas, np.transpose(xs))
            print("\nThere is " + str(pred[0] * 100) +
                  "% chance that the answer is " + str(pred[1]))
        elif str(input_list[0]).upper() == 'UPLOAD':
            pred_m = 0
            xtemp = []
            print("Enter file name with extension:")
            while 1:
                try:
                    file_name = input()
                    with open(file_name, "r") as fpr:
                        fpr.readline()  # Skip first line
                        csvReader = csv.reader(fpr)
                        for row in csvReader:
                            xtemp.append(row)
                            pred_m += 1
                    pred_X = np.array(xtemp).astype(np.float)
                    pred_y = np.array(pred_X[:, IN_SIZE:])
                    pred_X = pred_X[:, :IN_SIZE]

                    mynnet.predict(pred_X)
                    ext_h0 = np.array(mynnet.p)
                    # ext_h0 = np.array(predict(thetas, pred_X))
                    print("Prediction array:")
                    print(ext_h0[1])
                    print("**********************")
                    print("External File Accuracy")
                    print(str(np.mean(np.array(np.equal(ext_h0[1], np.transpose(pred_y))).astype(int)) * 100) + " %")
                    print("**********************")
                    fpr.close()
                    break
                except FileNotFoundError:
                    print("File Not Found. Please try again:")
                except ValueError:
                    print("An Error Occurred when converting file numbers to float. Check file and try again.")
                    exit(1)
        elif str(input_list[0]).upper() == 'PLOT':

            # Open History.csv File
            hist_temp = []
            with open('History.csv', "r") as fpr2:
                # fpr2.readline()
                csvReader2 = csv.reader(fpr2)
                for row in csvReader2:
                    hist_temp.append(row)

            # Create Plots from History.csv
            epochs = range(NUM_ITER)
            hist_cost = np.array([i[0] for i in hist_temp]).astype(np.float)
            hist_acc = np.array([i[1] for i in hist_temp]).astype(np.float)

            # Cost Function
            plt.plot(epochs, hist_cost, 'y', label='Cost')
            plt.title('Training Cost vs Epochs')
            plt.figure()

            # Accuracy
            plt.plot(epochs, hist_acc, 'r', label='Acc')
            plt.title('Accuracy vs Epochs')

            plt.show()

        elif str(input_list[0]).upper() == 'SHOW':
            try:
                im_nums = [int(x) for x in input_list[1:]]
                for im in im_nums:
                    plt.imshow(np.reshape(pred_X[im], (20, 20)))
                    plt.show()
            except FileNotFoundError:
                print("Please UPLOAD a file first")
            except NameError:
                print("Please UPLOAD a file first")
            except ValueError:
                print("Wrong value in arguments")
            except TypeError:
                print("Wrong value type in arguments")

        else:
            print("Command not found. Please try again: ")
            continue

    print("Program has ended.")
    exit(0)

### ------------------------- END ----------------------------- ###


if __name__ == '__main__':
    main()
