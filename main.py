import numpy as np
from NNetwork3 import Network
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import random

## Custom Functions : rangeMap
def rangeMap(value, valLow, valHigh, numLow, numHigh, constrain = False):
    valDiff = abs(valHigh-valLow)
    numDiff = abs(numHigh-numLow)
    temp = value
    if temp > valHigh and constrain: temp = valHigh
    elif temp < valLow and constrain: temp = valLow 
    temp =  (((temp - valLow) / float(valDiff)) * numDiff)+numLow
    return temp
##

size = 30

Net = Network(shape = [size, size, size])

#Net.printNet()

fig = plt.figure()
ax = fig.gca(projection='3d')

temp = []

plotSin = [(math.sin(rangeMap(i, 0, size, 0, math.pi*2))+1)/2 for i in range(size)]
plotCos = [(math.cos(rangeMap(i, 0, size, 0, math.pi*2))+1)/2 for i in range(size)]
plotSin_2 = [(math.sin(rangeMap(i, 0, size, 0, math.pi*2))**2+1)/2 for i in range(size)]

learnCounts = 300
for c in range(learnCounts):
    print("Processing:",100*c/learnCounts,"%")
    
    Net.backProp(plotSin_2, plotSin_2)
    Net.backProp(plotCos, plotCos)
    Net.backProp(plotSin, plotSin)
    
    temp.append(Net.forward(plotSin))

print("Done...")

X = range(size)
Y = range(learnCounts)
X, Y = np.meshgrid(X, Y)

surface = ax.plot_surface(X, Y, temp , cmap=cm.coolwarm,linewidth=0, antialiased=False)

Net.save()

print("3D Sine...")
plt.show()

print("All functions...")
plt.plot(plotSin); plt.plot(Net.forward(plotSin))
plt.plot(plotCos); plt.plot(Net.forward(plotCos))
plt.plot(plotSin_2); plt.plot(Net.forward(plotSin_2))

plt.show()
