#==============================================================================
#     Neural Netowrk Python Library
#     Aleks Mercik
#     May 2016
#==============================================================================

import os
import pickle
import numpy
import math
import random

class Network:
    def __init__(self, ID="DEFAULT", shape=[1,1,1], bias = 1):
        self.nodeNet = []
        self.pathNet = []
        self.ID = ID
        self.shape = shape
        self.biasValue = bias
        if self.ID == "DEFAULT":
            print ("Creating temporary network.")
            self.makeNet()
        else:
            if os.path.isfile("data/"+self.ID+".netdata"):
                try:
                    self.load()
                    print ("Network \""+self.ID+"\" loaded successfully.")
                except:
                    print ("Loading failed. Creating temporary network.")
                    self.makeNet()
            else:
                try:
                    self.makeNet()
                    self.save()
                    print ("New network \""+self.ID+"\" saved successfully.")
                except: 
                    print ("New network creation failed. Creating temporary network.")
                    self.makeNet()
            
    def save(self):
        try:
            if not os.path.isdir("data"):
                os.makedirs("data")
            dataWrite = open("data/" + self.ID + ".netdata", "wb")
            pickle.dump([self.nodeNet, self.pathNet, self.ID, self.shape, self.biasValue], dataWrite, -1)
            dataWrite.close()
            dataWrite = open("data/" + self.ID +"_BACKUP.netdata", "wb")
            pickle.dump([self.nodeNet, self.pathNet, self.ID, self.shape, self.biasValue], dataWrite, -1)
            dataWrite.close()
        except:
            raise

    def load(self):
        try:
            if not os.path.exists("data/" + self.ID + ".netdata"):
                raise
                return
            try:
                dataRead = open("data/" + self.ID + ".netdata", "rb")
                inData = pickle.load(dataRead)
                dataRead.close()
            except:
                print ("Loading from save failed. Loading from backup.")
                dataRead = open("data/" + self.ID + "_BACKUP.netdata", "rb")
                inData = pickle.load(dataRead)
                dataRead.close()
            self.nodeNet = inData[0]
            self.pathNet = inData[1]
            self.ID = inData[2]
            self.shape = inData[3]
            self.biasValue = inData[4]
        except:
            raise
        
    def forward(self, inValues):
        self.clear()
        for i in range(len(self.nodeNet[0][0:-1])):
            self.nodeNet[0][i].add(inValues[i])
        for i in self.pathNet:
            for j in i:
                for k in j:
                    self.nodeNet[k.endIndex[0]][k.endIndex[1]].add(self.nodeNet[k.startIndex[0]][k.startIndex[1]].output()[1]*k.weight)   
        return [i.output()[1] for i in (self.nodeNet[-1])]
    
    def addPath(self, sIndex, eIndex):
        self.pathNet[sIndex[0]][sIndex[1]].append(Path(sIndex, eIndex))

    def makeNet(self):
        for i in self.shape:
            tempLayer = [Node() for j in range(i+1)]
            tempLayer[-1].isBias = True
            tempLayer[-1].sum = [self.biasValue, self.biasValue]
            self.nodeNet.append(tempLayer)
        
        for i in range(len(self.nodeNet)-1):
            self.pathNet.append([])
            for j in range(len(self.nodeNet[i])):
                self.pathNet[i].append([])
                for k in range(len(self.nodeNet[i+1])-1): # i = Path Layer, j = Node Number, k = Path
                    self.pathNet[i][j].append(Path([i,j],[i+1,k]))
        self.nodeNet[-1].pop()
        
    def clear(self):
        for i in self.nodeNet:
            for j in i:
                if not j.isBias:
                    j.sum = [0, 0]
                    j.sumSigged = False
                else:
                    j.sum = [self.biasValue, self.biasValue]
        for i in self.pathNet:
            for j in i:
                for k in j:
                    k.delta = 0

    def backProp(self, expectedInput, expectedOutput, A=1):
        self.forward(expectedInput)

        for paths in self.pathNet[-1]:
            for i in paths:
                endNodeIndex = i.endIndex
                startNodeIndex = i.startIndex
                endNodeSum = self.nodeNet[endNodeIndex[0]][endNodeIndex[1]].sum
                startNodeSum = self.nodeNet[startNodeIndex[0]][startNodeIndex[1]].sum
                i.delta = -(expectedOutput[endNodeIndex[1]] - endNodeSum[1])*endNodeSum[1]*(1-endNodeSum[1])
                i.weight -= A * i.delta * startNodeSum[1]
        
        for x in self.pathNet[0:-1]:
            for y in x:
                for i in y:
                    endNodeIndex = i.endIndex
                    startNodeIndex = i.startIndex
                    endNodeSum = self.nodeNet[endNodeIndex[0]][endNodeIndex[1]].sum
                    startNodeSum = self.nodeNet[startNodeIndex[0]][startNodeIndex[1]].sum
                    deltaSum = 0
                    for path in self.pathNet[endNodeIndex[0]][endNodeIndex[1]]:
                        deltaSum += path.delta * path.weight
                    i.delta = deltaSum*endNodeSum[1]*(1-endNodeSum[1])
                    i.weight -= A * i.delta * startNodeSum[1]
        

    def printNet(self):
        for i in self.nodeNet:
            for j in i:
                print ([self.nodeNet.index(i),i.index(j)],j.sum)
        for i in self.pathNet:
            for j in i:
                for k in j:
                    print (k.pathInfo())
        

class Path:
    def __init__(self, startIndex, endIndex):
        self.weight = ((numpy.random.random()*2)-1)
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.delta = 0
    def pathInfo(self):
        return str(self.startIndex) + " -- " + str(self.weight) + " --> " + str(self.endIndex)

class Node:
    def __init__(self, isBias = False):
        self.sumSigged = False
        self.sum = [0, 0]       
        self.isBias = isBias

    def sigF(self):
        try:
            self.sum[1] = 1.0 / (1.0 + math.exp(-self.sum[0]))
        except:
            pass
        
    def output(self):
        if not self.sumSigged and not self.isBias:
            self.sigF()
            self.sumSigged = True
        return self.sum
        
    def add(self, a):
        self.sum[0]+=a
        if self.sumSigged:
            self.sumSigged = False

##Example Of Use---------------------------------------------------------------
#,
#Net = Network(ID = "TEST",shape=[1, 2, 1])
#
#print Net.forward([1])
#print Net.forward([0]),"\n"
#    
#Net.printNet()
#print ""
#
#LC = 5
#
#for i in range(100000):
#    Net.backProp([1],[0],LC)
#    Net.backProp([0],[1],LC)
#    if i%1000 == 0:
#        LC+=1
#        print Net.forward([1]), Net.forward([0])
#
#    Net.save()
#
#Net.printNet()
#
#Net = Network(shape=[1,2, 1])
#
#for i in range(10000):
#    netInput = random.randint(0,1)
#    
#    netOutput = int(not bool(netInput))
#        
#    netReturn = Net.forward([netInput]) 
#        
#    Net.backProp([netInput], [netOutput], 2)
#    
#for i in range(100):
#    print Net.forward([0])
#    print Net.forward([1]),"\n"
#
#for j in range(1, 4):
#    for k in range(1,5):
#        Net = Network(shape=[5] + [k]*j + [1])
#        K = 2
#        for i in range(10000):
#            _in = [random.randint(0, 1),random.randint(0, 1),random.randint(0, 1),random.randint(0, 1),random.randint(0, 1)]
#            _out = [0]
#            
#            if sum(_in) >= 3:
#                _out = [1]
#            Net.backProp(_in, _out, K)
#            if not i % 2000:
#                #K *= 0.999
#                print j, k, _in, _out,Net.forward(_in), Net.shape
                
