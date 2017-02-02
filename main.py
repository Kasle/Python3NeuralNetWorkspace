from NNetwork3 import Network

Net = Network(shape = [1, 2, 3])

print (Net.forward([1]))
print (Net.forward([0]))
print (Net.forward([-1]))

Net.printNet()
