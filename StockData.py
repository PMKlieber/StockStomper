import csv
import zipfile

import scipy as sp
from dateutil import parser


class StockDataHandler:
    zf = ""
    sym = {}
    symi = []
    vdates = []
    symnames = {}

    def __init__(self, zfloc="~/Quotesz.zip", exchanges=['NYSE', 'NASDAQ', 'AMEX']):
        if zfloc is None:
            return
        zf = zipfile.ZipFile(zfloc)
        for ex in exchanges:
            fns = [i.filename for i in zf.filelist if i.filename[:len(ex)] == ex and i.filename[-3:] == 'csv']
            fns.sort()
            for i in fns:
                csv1 = [k for k in csv.reader([j.decode('utf-8') for j in zf.open(i).readlines()])]
                for j in csv1:
                    if j[0] not in self.sym:
                        self.sym[j[0]] = {}
                    dts = parser.parse(j[1]).strftime('%Y%m%d')
                self.sym[j[0]][dts] = sp.array([float(l) for l in j[2:6]])
                if dts not in self.vdates:
                    self.vdates.append(dts)
                dmax = max([len(self.sym[i]) for i in self.sym])
            print("{} -> {}".format(i, dts))

    # Load stock histories from Numpy dump file
    def loadFromNpy(self, npyloc="bigsym.npy"):
        self.sym = sp.load(npyloc, allow_pickle=1).flatten()[0]
        self.symi = list(self.sym)

    # Refresh list of dates
    def refreshDates(self):
        self.vdates = set()
        self.vdates.update(*[set([j for j in self.sym[i]]) for i in self.sym])
        self.vdates = sorted(list(self.vdates))

    # Filling in missing dates for each stock.
    def fillmissingdates(self, verbose=False):
        # Return a stock with open,low,high,close all equal to input's close value
        sic = lambda i: sp.array([i[3], i[3], i[3], i[3]])
        # Create a set of all dates included in given set of stock histories
        self.refreshDates()
        self.symi = list(self.sym)
        # Find stock histories that are missing dates
        for j in self.sym:
            if verbose:
                print("Checking {}  ({}/{})".format(j, self.symi.index(j), len(self.sym)))
            for k in self.vdates:
                if k not in self.sym[j]:
                    # Set missing date's history's (Open,Low,High,Close) all to latest existing date before missing date
                    self.sym[j][k] = sic(
                        self.sym[j][([list(self.sym[j])[0]] + sorted([i for i in self.sym[j] if int(i) < int(k)]))[-1]])
                    if verbose:
                        print("{} missing {}, filling with {}".format(j, k, self.sym[j][k]))
                    self.sym[j] = dict(sorted(self.sym[j].items()))

    # Merge histories from usym into self
    def mergesym(self, usym, overwrite=False, newsym=False):
        vdas = set()
        vdas.update(*[set(usym[j].keys()) for j in usym])
        for k in vdas.difference(self.vdates):
            self.vdates.append(k)
        self.vdates = sorted(set(self.vdates))
        for i in [j for j in usym if (j in self.sym or newsym == True)]:
            print(i)
            if i not in sym:
                self.sym[i] = {}
                print("ADDING {}".format(i))
            for j in [j for j in usym[i] if (overwrite == True or (j not in self.sym[i]))]:
                self.sym[i][j] = usym[i][j]
        self.sym[i] = dict(sorted(self.sym[i].items()))
        self.symi = [i for i in self.sym]
        for i in self.symi:
            if len(self.sym[i]) == 0:
                self.sym.pop(i)
            else:
                (i, len(self.sym[i]))
                self.sym[i] = dict(sorted(self.sym[i].items()))
                lastl = [self.sym[i][list(self.sym[i])[-1]][3]] * 4
                for j in self.vdates[::-1]:
                    if j not in self.sym[i]:
                        self.sym[i][j] = lastl
                    else:
                        (i, len(self.sym[i]))
                self.sym[i] = dict(sorted(self.sym[i].items()))

    def calcLogDif(self, returnLogDiff=False, eps=.01):
        """
        Calculates the log of the ratio between the current stock's price and its previous price

        :param returnLogDiff: if True returns the log difference, otherwise only sets it internally
        :param eps: This is a small number that is added to the price to avoid division by zero
        :return: The log difference of the prices.
        """
        oc = []
        self.refreshDates()
        self.symi = list(self.sym)
        for i in self.symi:
            p1 = sp.array([(self.sym[i][j][3]) for j in self.vdates])
            p1 = p1 + eps
            lp1 = sp.log(p1[1:] / p1[:-1])
            oc.append(lp1)
        self.oc = sp.array(oc)
        if returnLogDiff:
            return self.oc

    def pruneBriefStocks(self, minDatesReq=0.5):
        """
        Removes stocks that only have data for a small number of dates

        :param minDatesReq: The minimum number of dates, either as an integer number, or a fraction of total dates
        """
        if minDatesReq < 1:
            minDatesReq = int(len(self.vdates) * minDatesReq)
        newsymi = [self.symi[j[0]] for j in sp.argwhere(sp.sum(self.oc != 0, 1) >= minDatesReq)]
        newsym = dict([(i, self.sym[i]) for i in newsymi])
        self.sym = newsym
        self.symi = list(self.sym)

    def pruneSparseDates(self, minStocksReq=0.5):
        """
        Removes dates that only have data for a small number of stocks

        :param minDatesReq: The minimum number of stocks, either as an integer number, or a fraction of total
        """
        if minStocksReq < 1:
            minStocksReq = int(len(self.symi) * minStocksReq)
        newdates = [self.vdates[j[0]] for j in sp.argwhere(sp.sum(self.oc != 0, 0) >= minStocksReq)]
        for j in self.symi:
            for k in [i for i in self.sym[j] if i not in newdates]:
                self.sym[j].pop(k)
        self.vdates = newdates

    def loadSymNames(self,symnamedir='/root/symnames/', exs=['AMEX', 'NYSE', 'NASDAQ']):
        """
        Load names associated with stock symbols from file
        :param symnamedir:  Directory containing symbol name files
        :param exs: Exchanges to load names from
        """
        for sm in exs:
            of = open("{}/{}.txt".format(symnamedir, sm)).readlines()[1:]
            for j in of:
                j = j.split("\t")
                self.symnames[j[0]] = j[1][:-1]

    def getDataArray(self,cols='OLHCD',verbose=False):
        #refresh dates and symbol list
        self.refreshDates()
        self.symi=list(self.sym)
        self.calcLogDif()
        ret=[]
        for stockind,stockname in enumerate(self.symi):
            col=[]
            for stockdateind,stockdate in enumerate(self.vdates[:-1]):
                row=[]
                for k in cols:
                    if k=="O": #Open
                        row.append(self.sym[stockname][stockdate][0])
                    if k=="L": #Low
                        row.append(self.sym[stockname][stockdate][1])
                    if k=="H": #High
                        row.append(self.sym[stockname][stockdate][2])
                    if k=="C": #Close
                        row.append(self.sym[stockname][stockdate][3])
                    if k=="D": #Difference Log
                        row.append(self.oc[stockind][stockdateind])
                col.append(row)
            if verbose: print("Added {}, rows={}".format(stockname,len(ret)))
            ret.append(col)
        return sp.array(ret)
