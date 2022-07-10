import csv
import zipfile

import scipy as sp
from dateutil import parser

from StockData import StockDataHandler

class StockPred:
    sdh=None

    def loadNpyData(self,npyfile='smallstox2.npy'):
        self.sdh = StockDataHandler(None);
        self.sdh.loadFromNpy(npyfile)
        self.sdh.fillmissingdates();
        self.sdh.calcLogDif()

    def __init__(self):
        self.loadNpyData()


    def buildmod(lays,dropouts,ac,ac2):
        """
        It takes a list of layer sizes, a list of dropout rates, and two activation functions, and returns a Keras model
        with the specified layers, dropouts, and activations

        :param lays: a list of integers, each integer is the number of neurons in a layer
        :param dropouts: list of dropout values for each layer
        :param ac: Activation function for the hidden layers
        :param ac2: The activation function for the output layer
        :return: A model with the input layer, hidden layers and output layer.
        """
        inp=Input(shape=(lays[0],))
        dbgstr="#0 ({},{}) ".format(lays[0],ac)
        lay=Dense(lays[1],activation=ac)(inp)
        if len(dropouts)>0:
            lay=Dropout(dropouts[0])(lay)
            dbgstr=dbgstr+"Dropout({})".format(dropouts[0])
        print (dbgstr)
        for layn in range(2,len(lays)-1):
            dbgstr="#{} ".format(layn)
            lay=Dense(lays[layn],activation=ac)(lay)
            dbgstr=dbgstr+"({},{}) ".format(lays[layn],ac)
            if len(dropouts)>=layn:
                lay=Dropout(dropouts[layn-2])(lay)
                dbgstr=dbgstr+"Dropout({})".format(dropouts[layn-1])
            print(dbgstr)
        otp=Dense(lays[-1],activation=ac2)(lay)
        return Model(inp,otp)

