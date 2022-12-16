from .neuron import *

class Layer(object):

  def __init__(self, activation, loss, number_of_neurons, number_of_inputs, bias_value=None, weights_value=None, name=None):
    self.name = name if name is not None else str(id(self))
    self.neurons = [Neuron(activation, loss, number_of_inputs, bias_value, weights_value, self.name + "_" + str(i)) for i in range(number_of_neurons)]
    self.number_of_neurons = number_of_neurons
    self.number_of_inputs = number_of_inputs
    self.inputs = [0 for i in range(number_of_inputs)]

  def predict(self, inputs):
    outputs = []
    for neuron in self.neurons:
      outputs.append(neuron.predict(inputs))
    return outputs

  def forward(self, inputs, targets=None):
    outputs = []
    if targets is None:
      for neuron in self.neurons:
        outputs.append(neuron.forward(inputs, None))
    else:
      self.targets = targets
      for i in range(len(self.neurons)):
        outputs.append(self.neurons[i].forward(inputs, targets[i]))
    return outputs

  def backward(self, learning_rate, backward=False):
    for neuron in self.neurons:
      neuron.backward(learning_rate, backward)

  def reset(self, bias_value=None, weights_value=None):
    for neuron in self.neurons:
      neuron.reset(bias_value, weights_value)


