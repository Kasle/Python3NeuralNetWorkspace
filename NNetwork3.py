# ==============================================================================
#     Neural Netowrk Python Library
#     Aleks Mercik
#     March 2017
# ==============================================================================

import math
import numpy as np

sig_m = np.vectorize(lambda x : 1 / ( 1 + math.exp(-x)))

class Network:
    def __init__(self, ID="DEFAULT", shape=(1,1,1), bias = 1):
        self.node_net = [np.zeros(i) for i in shape]
        self.path_net_forward = [[np.random.rand(shape[i+1])*2 - 1 for j in range(shape_val)] for i, shape_val in enumerate(shape[:-1])]
        self.path_net_recurrent = [[np.random.rand(i)*2 - 1 for j in range(i)] for i in shape[1:-1]]

        self.ID = ID
        self.shape = shape
        self.bias = bias

    def forward(self, forward_input):
        recurrent_last = [np.zeros(i) for i in self.shape[1:-1]]
        forward_output = []
        np_forward_input = np.array(forward_input)

        for input_index, input_segment in enumerate(np_forward_input):
            self.node_net = [np.zeros(i) for i in self.shape]
            self.node_net[0] += sig_m(input_segment)
            for layer_index in range(len(self.node_net)-1):
                for node_index, node_value in enumerate(self.node_net[layer_index]):
                    self.node_net[layer_index+1] += node_value * self.path_net_forward[layer_index][node_index]
                    if (layer_index < len(self.node_net)-2):
                        self.node_net[layer_index+1] += recurrent_last[node_index] * self.path_net_recurrent[layer_index][node_index]

                self.node_net[layer_index+1] = sig_m(self.node_net[layer_index+1])
            recurrent_last = self.node_net[1:-1]
            forward_output.append(self.node_net[-1])

        return forward_output



        

if __name__ == "__main__":
    a = Network()
    print(a.forward([[1],[1],[0],[1],[1]]))
