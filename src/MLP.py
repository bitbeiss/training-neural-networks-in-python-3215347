import numpy as np
import math

class Perceptron:
    """A single neuron with the sigmoid activation function.
       Attributes:
          inputs: The number of inputs in the perceptron, not counting the bias.
          bias:   The bias term. By default it's 1.0."""

    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs (+1 for the bias).""" 
        self.weights = [1 for i in range(inputs + 1)]
        print (self.weights)
        self.bias = bias

    def run(self, x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        return self.sigmoid(x_sum)

    def set_weights(self, w_init):
        self.weights = np.array(w_init)

    def sigmoid(self, x):
        return 1/(1 + pow(math.e,-x))
    

class MultilayerPerceptron:
    
    def __init__(self, layers, bias = 1.0):
        self.layers = np.array(layers, dtype=object)                    # network layers
        self.bias = bias
        self.network = []
        self.values = []

        for i in range(len(layers)):                                    #interate each layer
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]       #values for each neuron in the layer
            if i > 0:                                                   #only inputs are not neurons
                for j in range(self.layers[i]):                         #all neurons in one layer
                    self.network[i].append(Perceptron(inputs = self.layers[i-1], bias = self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)

    def set_weights(self, w_init):
        for layerx in range(len(w_init)):
            for neuronx in range(len(w_init[layerx])):
                neuron = self.network[layerx + 1][neuronx]
                neuron.set_weights(w_init[layerx][neuronx])

    def printWeights(self):
        print()
        for i in range(1, len(self.network)):                                  #all network layers
            for j in range(self.layers[i]):                                    #every neuron in layer i
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
        print()

    def run(self, x):
        x = np.array(x,dtype=object)                                            #transform x to numpy array
        self.values[0] = x
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        #note: the last item in the value array is a single-entry list
        return self.values[-1]                                                  #give back the last item in the list
#---------------

neuron = Perceptron(inputs=2)

neuron.set_weights([10,10,-15])
print("AND Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0,0])))
print("0 0 = {0:.10f}".format(neuron.run([0,1])))
print("0 0 = {0:.10f}".format(neuron.run([1,0])))
print("0 0 = {0:.10f}".format(neuron.run([1,1])))
#---

neuron.set_weights([100,100,-15])

print("OR Gate:")
print("0 0 = {0:.10f}".format(neuron.run([0,0])))
print("0 0 = {0:.10f}".format(neuron.run([0,1])))
print("0 0 = {0:.10f}".format(neuron.run([1,0])))
print("0 0 = {0:.10f}".format(neuron.run([1,1])))
#---------------

print()
print("Multi Layer XOR Neural Network")
#setup the ml network (XOR)
mlNetwork = MultilayerPerceptron(layers=[2,2,1])             #three layers (in/hidden/out), bias 1.0 (default)
neuronWeights = [[[-10,-10,15],[15,15,-10]],[[10,10,-15]]]  #define weights for single neurons
mlNetwork.set_weights(neuronWeights)
mlNetwork.printWeights()

#run the network
inputs = [[0,0], [0,1], [1,0],[1,1]]
for inputPair in inputs:
    output = mlNetwork.run(inputPair)[0]
    print(inputPair,"0 0 = {0:.10f}".format(output))
