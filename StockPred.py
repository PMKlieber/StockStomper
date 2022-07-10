import numpy as np
import scipy as sp
import itertools
import keras
from keras.layers import Dense,Input,Dropout,Reshape,Conv1D,MaxPooling1D,UpSampling1D,Add,Concatenate
from keras import Model,Sequential
import keras.backend as K


ops = ['nadam', 'adagrad', 'adadelta', 'rmsprop', 'adamax', 'adam']
acs = ['elu', 'hard_sigmoid', 'linear', 'relu', 'tanh', 'softsign', 'softmax', 'softplus', 'sigmoid', 'selu']
losses = ['hinge', 'logcosh', 'mae', 'mape', 'mse', 'msle', 'poisson', 'squared_hinge', 'binary_crossentropy']

from StockData import StockDataHandler


def ploss(Truth, Pred):
    eps = 10 ** (-6)
    buyFac = (Pred + eps) / (1 - Pred + eps)
    sellFac = (1 - Pred + eps) / (Pred + eps)
    return (buyFac * -Truth) + (sellFac * Truth)


class StockPred:
    sdh = None
    model = None

    def loadNpyData(self, npyfile='smallstox2.npy',verbose=False):
        """
        Have the Stock Data Handler load and prepare stock data from a Numpy dump
        :param npyfile: Filename of Numpy dump
        """
        self.sdh = StockDataHandler(None);
        self.sdh.loadFromNpy(npyfile)
        self.sdh.fillmissingdates(verbose=1)
        self.sdh.calcLogDif()

    def __init__(self):
        self.loadNpyData()

    def buildmod(self, lays, dropouts, ac, ac2):
        """
        Creates a Keras Deep Learning model from a list of layer sizes, a list of dropout rates, and two activation functions, and returns a Keras model
        with the specified layers, dropouts, and activations

        :param lays: a list of integers, each integer is the number of neurons in a layer
        :param dropouts: list of dropout values for each layer
        :param ac: Activation function for the hidden layers
        :param ac2: The activation function for the output layer
        :return: A model with the input layer, hidden layers and output layer.
        """
        inp = Input(shape=(lays[0],))
        dbgstr = "#0 ({},{}) ".format(lays[0], ac)
        lay = Dense(lays[1], activation=ac)(inp)
        if len(dropouts) > 0:
            lay = Dropout(dropouts[0])(lay)
            dbgstr = dbgstr + "Dropout({})".format(dropouts[0])
        print(dbgstr)
        for layn in range(2, len(lays) - 1):
            dbgstr = "#{} ".format(layn)
            lay = Dense(lays[layn], activation=ac)(lay)
            dbgstr = dbgstr + "({},{}) ".format(lays[layn], ac)
            if len(dropouts) >= layn:
                lay = Dropout(dropouts[layn - 2])(lay)
                dbgstr = dbgstr + "Dropout({})".format(dropouts[layn - 1])
            print(dbgstr)
        otp = Dense(lays[-1], activation=ac2)(lay)
        self.model = Model(inp, otp)
        return self.model

    def buildDataSet(self, inHistLength=100, inDataCols="OLHC", outDataCols="D", verbose=False):
        """
        It takes the data from the data handler, and creates a data set of x,y values for training a ML model

        :param inHistLength: The number of days of historical data to use as input, defaults to 100 (optional)
        :param inDataCols: The columns of the input data to use, defaults to OLHC (optional)
        :param outDataCols: The column(s) of the output data, defaults to D (optional)
        :param verbose: 0=no output, 1=output per stock, 2=output per row, defaults to False (optional)
        """
        x = []
        y = []
        ret = []
        slc = lambda i: sp.log(i[1:] / i[0])
        inData = self.sdh.getDataArray(inDataCols,verbose=verbose)
        outData = self.sdh.getDataArray(outDataCols,verbose=verbose)
        for stockInd in range(0, len(self.sdh.symi)):
            for predDate in range(inHistLength, len(self.sdh.vdates) - 1):
                if outData[stockInd, predDate] != 0:
                    x.append(np.vstack(inData[stockInd, predDate - inHistLength:predDate]).flatten())
                    y.append(outData[stockInd, predDate])
                    if verbose > 1: print(x[-1], y[-1])
            if verbose > 0:
                print("Added rows for {} ({}/{}), total={}".format(self.sdh.symi[stockInd], stockInd,len(self.sdh.symi),len(x)))
        return (sp.array(x), sp.array(y))


    def trainAndTestData(self, testSetPct=.05, testSetRandom=False):
        """
        Create a default dataset, split it into train and test portions,
        create a model, and progressively train it with intermittent tests
        to show accuracy and progress
        :param testSetPct: The percent of the dataset to set aside as testing data
        :param testSetRandom: Whether to pick test cases from the end or randomly
        """
        x, y = self.buildDataSet(inDataCols="D",outDataCols="D")
        if testSetRandom == False:
            self.trainX = x[:-int(len(x) * (testSetPct))]
            self.trainY = y[:-int(len(x) * (testSetPct))]
            self.testX = x[-int(len(x) * (testSetPct)):]
            self.testY = y[-int(len(x) * (testSetPct)):]
        else:
            randor = sp.arange(0, len(x))
            np.random.shuffle(randor)
            self.trainX = x[randor[:-int(len(x) * (testSetPct))]]
            self.trainY = y[randor[:-int(len(x) * (testSetPct))]]
            self.testX = x[randor[-int(len(x) * (testSetPct)):]]
            self.testY = y[randor[-int(len(x) * (testSetPct)):]]
        model = spd.buildmod([100,50,25,20,15,10,1], [], 'tanh', 'sigmoid')
        for i in range(1, 50):
            op = keras.optimizers.Adadelta(.1**(2+i/10))
            model.compile(op, ploss)
            model.fit(self.trainX,self.trainY, epochs=i, shuffle=1, validation_split=.05)
            print(model.predict(x[-10:]))
            print(sp.sum(model.predict(self.testX) * self.testY))



for i in range(1, 50):
    op = keras.optimizers.Adadelta(.1**(2+i/10))
    model.compile(op, ploss)
    model.fit(x[:-25000],y[:-25000], epochs=i, shuffle=1, validation_split=.05)
    print(model.predict(x[-10:]))
    print(sp.sum(model.predict(x[-25000:]) * y[-25000:]))
