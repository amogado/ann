
class Bias(object):
  def __init__(self, value=None):
    if value is None:
      self.value = MathTool.random_float(-1, 1)
    else:
      self.value = value
    self.gradient = 0

# --------------------------------------------------
class Weight(object):
  def __init__(self, value=None):
    if value is None:
      self.value = MathTool.random_float(-1, 1)
    else:
      self.value = value
    self.gradient = 0

# --------------------------------------------------

class Activation(object):
  def __init__(self, function):
    self.function = function
    if function == Function.sigmoid:
      self.derivative = Derivative.sigmoid
    elif function == Function.tanh:
      self.derivative = Derivative.tanh
    elif function == Function.relu:
      self.derivative = Derivative.relu
    elif function == Function.sigmoid2:
      self.derivative = Derivative.sigmoid2
    else:
      raise Exception('Unknown activation function: ' + function)

# --------------------------------------------------

class Loss(object):
  def __init__(self, function):
    self.function = function
    if function == Function.mse:
      self.derivative = Derivative.mse
    else:
      raise Exception('Unknown loss function: ' + function)
from .maths import *

class Neuron(object):
  def __init__(self, activation, loss, number_of_inputs, bias_value=None, weights_value=None, name=None):
    self.activation = Activation(activation)
    self.loss = Loss(loss)
    self.bias = Bias(bias_value)
    self.weights = [Weight(weights_value) for i in range(number_of_inputs)]
    self.inputs = [0 for i in range(number_of_inputs)]
    self.output = 0
    self.error = 0
    self.name = name if name is not None else self
    self.number_of_inputs = number_of_inputs

  @staticmethod
  def linear_combination(neuron, inputs):
    weighted_sum = 0
    for i in range(len(inputs)):
      weighted_sum += neuron.weights[i].value * inputs[i]
    return weighted_sum + neuron.bias.value

  def predict(self, inputs):
    output = self.activation.function(Neuron.linear_combination(self, inputs))
    return output

  def forward(self, inputs, target=None):
    self.inputs = inputs
    self.weighted_sum = Neuron.linear_combination(self, inputs)
    self.output = self.activation.function(self.weighted_sum)
    if target is not None:
      self.target = target
    else:
      self.target = None
    return self.output

  def calculate_gradient(self, error=None, backward=False):
    if not backward:
      self.error = self.loss.derivative(self.target, self.output) if error is None else error
    # print(" target: " + str(self.target))
    # print(" output: " + str(self.output))
    # print(" error: " + str(self.error))
    self.activation.gradient = self.activation.derivative(self.weighted_sum)
    # print(" activation gradient: " + str(self.activation.gradient))
    self.bias.gradient = self.error * self.activation.gradient
    # print(" bias gradient: " + str(self.bias.gradient))
    for i in range(len(self.inputs)):
      self.weights[i].gradient = self.inputs[i] * self.bias.gradient
      # print(" weight gradient: " + str(self.weights[i].gradient))

  def update_superparameters(self, learning_rate):
    self.bias.value -= learning_rate * self.bias.gradient
    for i in range(len(self.inputs)):
      self.weights[i].value -= learning_rate * self.weights[i].gradient

  def backward(self, learning_rate, backward=False):
    self.calculate_gradient(backward=backward)
    self.update_superparameters(learning_rate)

  def get_info(self):
    info = {}
    info['name'] = self.name
    info['bias'] = self.bias.value
    info['activation'] = self.activation.function.__name__
    info['loss'] = self.loss.function.__name__
    info['weights'] = [weight.value for weight in self.weights]
    info['output'] = self.output
    info['error'] = self.error
    info['target'] = self.target
    return info

  def reset(self, bias_value=None, weights_value=None):
    self.bias = Bias(bias_value)
    self.weights = [Weight(weights_value) for i in range(self.number_of_inputs)]
    self.inputs = [0 for i in range(self.number_of_inputs)]
    self.output = 0
    self.error = 0
    self.target = None
