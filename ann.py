#! /usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from ann.maths import *
from ann.neuron import *
from ann.layer import *
from ann.network import *
from ann.tests import *

# --------------------------------------------------
if __name__ == '__main__':
  if len(sys.argv) > 1:
    if sys.argv[1] == 'test':
      Tests.call_tests(__name__)
  # create a layer to take a word in input
  # input_layer = Layer(Function.tanh, Function.mse, 3, 2)
  # layer1 = Layer(Function.tanh, Function.mse, 3, 3)

  # output_layer = Layer(Function.tanh, Function.mse, 1, 3)

  # # train network to learn XOR
  # network = Network(input_layer, layer1, output_layer)
  # network.update_info()
  # inputs = [[0, 1],  [1, 0], [0, 0], [1, 1]]
  # targets = [[1], [1], [0], [0]]
  # network.dojo(inputs, targets, 0.0001)

  # make a neural network of 1 input layer with 2 inputs (sigmoid, mse), one hidden layer with 2 neurons (sigmoid, mse) and one output layer with 2 outputs (sigmoid, mse)
  input_layer = Layer(Function.sigmoid, Function.mse, 2, 2)
  layer1 = Layer(Function.sigmoid, Function.mse, 2, 2)
  output_layer = Layer(Function.sigmoid, Function.mse, 1, 2)
  network = Network(input_layer, layer1, output_layer)
  network.update_info



