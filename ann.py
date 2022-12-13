#! /usr/bin/env python

import sys
import os
import unittest
import math
import json

# --------------------------------------------------
class Tests(unittest.TestCase):
  # def test_class_MathTool(self):
  #   self.assertIsInstance(MathTool, object)

  # def test_method_MathTool_e(self):
  #   self.assertAlmostEqual(MathTool.e, 2.718281828459045)

  # def test_method_MathTool_exp(self):
  #   self.assertAlmostEqual(MathTool.exp(1), 2.718281828459045)
  #   self.assertAlmostEqual(MathTool.exp(2), 7.3890560989306495)
  #   self.assertAlmostEqual(MathTool.exp(3), 20.085536923187664)
  #   self.assertAlmostEqual(MathTool.exp(4), 54.59815003314423)
  #   self.assertAlmostEqual(MathTool.exp(5), 148.41315910257657)

  # def test_method_MathTool_log(self):
  #   self.assertAlmostEqual(MathTool.log(1), 0.0)
  #   self.assertAlmostEqual(MathTool.log(2), 0.6931471805599453)
  #   self.assertAlmostEqual(MathTool.log(3), 1.0986122886681098)
  #   self.assertAlmostEqual(MathTool.log(4), 1.3862943611198906)
  #   self.assertAlmostEqual(MathTool.log(5), 1.6094379124341003)

  # def test_class_Function(self):
  #   self.assertIsInstance(Function, object)

  # def test_method_Function_sigmoid(self):
  #   self.assertAlmostEqual(Function.sigmoid(0), 0.5)
  #   self.assertAlmostEqual(Function.sigmoid(1), 0.7310585786300049)
  #   self.assertAlmostEqual(Function.sigmoid(2), 0.8807970779778823)
  #   self.assertAlmostEqual(Function.sigmoid(3), 0.9525741268224334)
  #   self.assertAlmostEqual(Function.sigmoid(4), 0.9820137900379085)

  # def test_method_Function_tanh(self):
  #   self.assertAlmostEqual(Function.tanh(0), 0.0)
  #   self.assertAlmostEqual(Function.tanh(1), 0.7615941559557649)
  #   self.assertAlmostEqual(Function.tanh(2), 0.9640275800758169)
  #   self.assertAlmostEqual(Function.tanh(3), 0.9950547536867305)
  #   self.assertAlmostEqual(Function.tanh(4), 0.999329299739067)

  # def test_method_Function_relu(self):
  #   self.assertAlmostEqual(Function.relu(0), 0.0)
  #   self.assertAlmostEqual(Function.relu(1), 1.0)
  #   self.assertAlmostEqual(Function.relu(2), 2.0)
  #   self.assertAlmostEqual(Function.relu(3), 3.0)
  #   self.assertAlmostEqual(Function.relu(4), 4.0)

  # def test_method_Function_mse(self):
  #   self.assertAlmostEqual(Function.mse(1, 1), 0.0)
  #   self.assertAlmostEqual(Function.mse(1, 2), 1.0)
  #   self.assertAlmostEqual(Function.mse(2, 1), 1.0)
  #   self.assertAlmostEqual(Function.mse(2, 2), 0.0)

  # def test_method_MathTool_max(self):
  #   self.assertEqual(MathTool.max(1, 2), 2)
  #   self.assertEqual(MathTool.max(2, 1), 2)
  #   self.assertEqual(MathTool.max(1, 1), 1)

  # def test_class_Derivative(self):
  #   self.assertIsInstance(Derivative, object)

  # def test_method_Derivative_sigmoid(self):
  #   self.assertAlmostEqual(Derivative.sigmoid(0), 0.25)
  #   self.assertAlmostEqual(Derivative.sigmoid(1), 0.19661193324148185)
  #   self.assertAlmostEqual(Derivative.sigmoid(2), 0.10499358540350662)
  #   self.assertAlmostEqual(Derivative.sigmoid(3), 0.04517665973091267)
  #   self.assertAlmostEqual(Derivative.sigmoid(4), 0.01766270621332736)

  # def test_method_Derivative_tanh(self):
  #   self.assertAlmostEqual(Derivative.tanh(0), 1.0)
  #   self.assertAlmostEqual(Derivative.tanh(1), 0.41997434161402614)
  #   self.assertAlmostEqual(Derivative.tanh(2), 0.07065082485316466)
  #   self.assertAlmostEqual(Derivative.tanh(3), 0.009866037165440922)
  #   self.assertAlmostEqual(Derivative.tanh(4), 0.0013409507920792178)

  # def test_method_Derivative_relu(self):
  #   self.assertAlmostEqual(Derivative.relu(0), 0.0)
  #   self.assertAlmostEqual(Derivative.relu(1), 1.0)
  #   self.assertAlmostEqual(Derivative.relu(2), 1.0)
  #   self.assertAlmostEqual(Derivative.relu(3), 1.0)
  #   self.assertAlmostEqual(Derivative.relu(4), 1.0)

  # def test_method_Derivative_mse(self):
  #   self.assertAlmostEqual(Derivative.mse(1, 1), 0.0)
  #   self.assertAlmostEqual(Derivative.mse(1, 2), 2.0)
  #   self.assertAlmostEqual(Derivative.mse(2, 1), -2.0)
  #   self.assertAlmostEqual(Derivative.mse(2, 2), 0.0)

  # def test_method_MathTool_random_float(self):
  #   self.assertGreaterEqual(MathTool.random_float(0, 1), 0.0)
  #   self.assertLessEqual(MathTool.random_float(0, 100), 100.0)
  #   self.assertGreaterEqual(MathTool.random_float(0, 100), 0.0)
  #   self.assertNotEqual(MathTool.random_float(0, 100), MathTool.random_float(0, 100))

  # def test_class_Bias(self):
  #   self.assertIsInstance(Bias, object)

  # def test_method_Bias_init(self):
  #   bias = Bias(1.0)
  #   self.assertEqual(bias.value, 1.0)
  #   bias2 = Bias()
  #   self.assertGreaterEqual(bias2.value, -1.0)
  #   self.assertLessEqual(bias2.value, 1.0)

  # def test_class_Weight(self):
  #   self.assertIsInstance(Weight, object)

  # def test_method_Weight_init(self):
  #   weight = Weight(1.0)
  #   self.assertEqual(weight.value, 1.0)
  #   weight2 = Weight()
  #   self.assertGreaterEqual(weight2.value, -1.0)
  #   self.assertLessEqual(weight2.value, 1.0)

  # def test_class_Activation(self):
  #   self.assertIsInstance(Activation, object)

  # def test_method_Activation_init(self):
  #   activation = Activation(Function.sigmoid)
  #   self.assertEqual(activation.function, Function.sigmoid)
  #   self.assertEqual(activation.derivative, Derivative.sigmoid)
  #   activation2 = Activation(Function.tanh)
  #   self.assertEqual(activation2.function, Function.tanh)
  #   self.assertEqual(activation2.derivative, Derivative.tanh)
  #   activation3 = Activation(Function.relu)
  #   self.assertEqual(activation3.function, Function.relu)
  #   self.assertEqual(activation3.derivative, Derivative.relu)

  # def test_class_Loss(self):
  #   self.assertIsInstance(Loss, object)

  # def test_method_Loss_init(self):
  #   loss = Loss(Function.mse)
  #   self.assertEqual(loss.function, Function.mse)
  #   self.assertEqual(loss.derivative, Derivative.mse)

  # def test_class_Neuron(self):
  #   self.assertIsInstance(Neuron, object)

  # def test_method_Neuron_init(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   self.assertEqual(neuron.activation.function, Function.sigmoid)
  #   self.assertEqual(neuron.activation.derivative, Derivative.sigmoid)
  #   self.assertEqual(neuron.loss.function, Function.mse)
  #   self.assertEqual(neuron.loss.derivative, Derivative.mse)
  #   self.assertGreaterEqual(neuron.bias.value, -1.0)
  #   self.assertLessEqual(neuron.bias.value, 1.0)
  #   self.assertEqual(len(neuron.weights), 1)
  #   self.assertGreaterEqual(neuron.weights[0].value, -1.0)
  #   self.assertLessEqual(neuron.weights[0].value, 1.0)

  # def test_method_Neuron_linear_combination(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   neuron.weights[0].value = 1.0
  #   neuron.bias.value = 1.0
  #   self.assertEqual(Neuron.linear_combination(neuron, [1]), 2.0)

  # def test_method_Neuron_predict(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   neuron.weights[0].value = 1.0
  #   neuron.bias.value = 1.0
  #   self.assertAlmostEqual(neuron.predict([1]), 0.8807970779778823)

  # def test_method_Neuron_forward(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   neuron.weights[0].value = 1.0
  #   neuron.bias.value = 1.0
  #   self.assertAlmostEqual(neuron.forward([1], 1), 0.8807970779778823)
  #   self.assertEqual(neuron.target, 1)

  # def test_method_Neuron_calculate_gradient(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   neuron.weights[0].value = 1.0
  #   neuron.bias.value = 1.0
  #   neuron.forward([1], 1)
  #   neuron.calculate_gradient()
  #   self.assertAlmostEqual(neuron.bias.gradient, -0.025031084347353506)
  #   self.assertAlmostEqual(neuron.weights[0].gradient, -0.025031084347353506)

  # def test_method_Neuron_update_superparameters(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   neuron.weights[0].value = 1.0
  #   neuron.bias.value = 1.0
  #   neuron.forward([1], 1)
  #   neuron.calculate_gradient()
  #   neuron.update_superparameters(0.1)
  #   self.assertAlmostEqual(neuron.bias.value, 1.0025031084347354)
  #   self.assertAlmostEqual(neuron.weights[0].value, 1.0025031084347354)
  #   self.assertNotEqual(neuron.bias.value, 1.0)
  #   self.assertNotEqual(neuron.weights[0].value, 1.0)

  # def test_method_Neuron_backward(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1)
  #   neuron.weights[0].value = 1.0
  #   neuron.bias.value = 1.0
  #   predict = neuron.predict([1])
  #   neuron.forward([1], 0)
  #   neuron.backward(0.1)
  #   self.assertAlmostEqual(neuron.bias.value, 0.981504391354034)
  #   self.assertAlmostEqual(neuron.weights[0].value, 0.981504391354034)
  #   self.assertNotEqual(neuron.bias.value, 1.0)
  #   self.assertNotEqual(neuron.weights[0].value, 1.0)
  #   new_predict = neuron.predict([1])
  #   self.assertNotEqual(predict, new_predict)
  #   self.assertLess(new_predict, predict)

  # def test_class_Layer(self):
  #   self.assertIsInstance(Layer, object)

  # def test_method_Layer_init(self):
  #   number_of_neurons = 4
  #   number_of_inputs = 2
  #   bias_value = 0.5
  #   weights_value = 0.5
  #   layer = Layer(Function.sigmoid, Function.mse, number_of_neurons, number_of_inputs, bias_value, weights_value)
  #   self.assertEqual(len(layer.neurons), number_of_neurons)
  #   self.assertEqual(len(layer.neurons[0].weights), number_of_inputs)
  #   self.assertEqual(len(layer.neurons[1].weights), number_of_inputs)
  #   self.assertEqual(len(layer.neurons[2].weights), number_of_inputs)
  #   self.assertEqual(len(layer.neurons[3].weights), number_of_inputs)
  #   self.assertEqual(layer.number_of_neurons, number_of_neurons)
  #   self.assertEqual(layer.number_of_inputs, number_of_inputs)
  #   for neuron in layer.neurons:
  #     self.assertEqual(neuron.activation.function, Function.sigmoid)
  #     self.assertEqual(neuron.activation.derivative, Derivative.sigmoid)
  #     self.assertEqual(neuron.loss.function, Function.mse)
  #     self.assertEqual(neuron.loss.derivative, Derivative.mse)
  #     self.assertGreaterEqual(neuron.bias.value, -1.0)
  #     self.assertLessEqual(neuron.bias.value, 1.0)
  #     self.assertEqual(len(neuron.weights), number_of_inputs)
  #     for weight in neuron.weights:
  #       self.assertGreaterEqual(weight.value, -1.0)
  #       self.assertLessEqual(weight.value, 1.0)
    
  # def test_method_Layer_predict(self):
  #   layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   self.assertAlmostEqual(layer.predict([1, 1])[0], 0.9525741268224334)
  #   self.assertAlmostEqual(layer.predict([1, 1])[1], 0.9525741268224334)
  #   self.assertAlmostEqual(layer.predict([1, 1])[2], 0.9525741268224334)
  #   self.assertAlmostEqual(layer.predict([1, 1])[3], 0.9525741268224334)

  # def test_method_Layer_forward(self):
  #   layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[0], 0.9525741268224334)
  #   self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[1], 0.9525741268224334)
  #   self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[2], 0.9525741268224334)
  #   self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[3], 0.9525741268224334)
  #   self.assertEqual(layer.targets, [1, 1, 1, 1])

  # def test_method_Layer_backward(self):
  #   layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   predict = layer.predict([1, 1])
  #   layer.forward([1, 1], [1, 1, 1, 1])
  #   layer.backward(0.1)
  #   new_predict = layer.predict([1, 1])
  #   self.assertNotEqual(predict, new_predict)

  # def test_class_Network(self):
  #   self.assertIsInstance(Network, object)

  # def test_method_Network_init(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   self.assertEqual(network.layers[0], input_layer)
  #   self.assertEqual(network.layers[1], layer2)
  #   self.assertEqual(network.layers[2], layer3)
  #   self.assertEqual(network.layers[3], output_layer)

  # def test_method_Network_update_info(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   network.update_info()
  #   self.assertEqual(network.number_of_layers, 4)
  #   self.assertEqual(network.number_of_inputs, 2)
  #   self.assertEqual(network.number_of_outputs, 1)
  #   self.assertEqual(network.number_of_neurons, 13)
  #   self.assertEqual(network.number_of_weights, 44)
  #   self.assertEqual(network.number_of_biases, 13)
  #   self.assertEqual(network.number_of_parameters, 57)
  #   self.assertEqual(network.number_of_hidden_layers, 2)

  # def test_method_Network_get_layer(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   self.assertEqual(network.get_layer(0), input_layer)
  #   self.assertEqual(network.get_layer(1), layer2)
  #   self.assertEqual(network.get_layer(2), layer3)
  #   self.assertEqual(network.get_layer(3), output_layer)

  # def test_method_Network_get_next_layer(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   self.assertEqual(network.get_next_layer(0), layer2)
  #   self.assertEqual(network.get_next_layer(1), layer3)
  #   self.assertEqual(network.get_next_layer(2), output_layer)
  #   self.assertEqual(network.get_next_layer(3), None)

  # def test_method_Network_get_previous_layer(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   self.assertEqual(network.get_previous_layer(0), None)
  #   self.assertEqual(network.get_previous_layer(1), input_layer)
  #   self.assertEqual(network.get_previous_layer(2), layer2)
  #   self.assertEqual(network.get_previous_layer(3), layer3)

  # def test_method_Network_get_layer_index(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   self.assertEqual(network.get_layer_index(input_layer), 0)
  #   self.assertEqual(network.get_layer_index(layer2), 1)
  #   self.assertEqual(network.get_layer_index(layer3), 2)
  #   self.assertEqual(network.get_layer_index(output_layer), 3)

  # def test_method_Network_predict(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   self.assertEqual(network.predict([1, 1]), [0.9931208367972153])

  # def test_method_Network_forward(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
  #   network = Network(input_layer, layer2, layer3, output_layer)
  #   network.forward([1, 1], [0])
  #   self.assertEqual(network.layers[0].neurons[0].output, 0.9525741268224334)
  #   self.assertEqual(network.layers[0].neurons[1].output, 0.9525741268224334)
  #   self.assertEqual(network.layers[0].neurons[2].output, 0.9525741268224334)
  #   self.assertEqual(network.layers[0].neurons[3].output, 0.9525741268224334)
  #   self.assertEqual(network.layers[1].neurons[0].output, 0.9919203680349579)
  #   self.assertEqual(network.layers[1].neurons[1].output, 0.9919203680349579)
  #   self.assertEqual(network.layers[1].neurons[2].output, 0.9919203680349579)
  #   self.assertEqual(network.layers[1].neurons[3].output, 0.9919203680349579)
  #   self.assertEqual(network.layers[2].neurons[0].output, 0.9930888320533605)
  #   self.assertEqual(network.layers[2].neurons[1].output, 0.9930888320533605)
  #   self.assertEqual(network.layers[2].neurons[2].output, 0.9930888320533605)
  #   self.assertEqual(network.layers[2].neurons[3].output, 0.9930888320533605)
  #   self.assertEqual(network.layers[3].neurons[0].output, 0.9931208367972153)

  # def test_method_Layer_forward_no_target(self):
  #   layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 0)
  #   layer.forward([1, 1])
  #   self.assertEqual(layer.neurons[0].output, 0.5)
  #   self.assertEqual(layer.neurons[1].output, 0.5)
  #   self.assertEqual(layer.neurons[2].output, 0.5)
  #   self.assertEqual(layer.neurons[3].output, 0.5)

  # def test_method_Neuron_get_info(self):
  #   neuron = Neuron(Function.sigmoid, Function.mse, 1, 0, 0, name="test")
  #   neuron.forward([1], [1])
  #   self.assertEqual(neuron.get_info(), {'name': 'test', 'bias': 0, 'activation': 'sigmoid', 'loss': 'mse', 'weights': [0], 'output': 0.5, 'error': 0, 'target': [1]})

  # def test_method_Network_calculate_output_error(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
  #   layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
  #   network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
  #   network.forward([1, 1], [1])
  #   network.calculate_output_error()
  #   self.assertEqual(network.layers[-1].neurons[0].error, -1)

  # def test_method_Network_calculate_previous_layers_errors(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
  #   layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
  #   network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
  #   network.forward([1, 1], [1])
  #   network.calculate_output_error()
  #   network.calculate_previous_layers_errors()
  #   self.assertEqual(network.layers[-2].neurons[0].error, -1)
  #   self.assertEqual(network.layers[-2].neurons[1].error, -1)
  #   self.assertEqual(network.layers[-2].neurons[2].error, -1)
  #   self.assertEqual(network.layers[-2].neurons[3].error, -1)

  # def test_method_Network_update_superparameters(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
  #   layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
  #   network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
  #   network.forward([1, 1], [1])
  #   network.calculate_output_error()
  #   network.calculate_previous_layers_errors()
  #   network.update_superparameters(0.1)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[0].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[1].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[2].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[3].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].bias.value, 0.025)

  # def test_method_Network_backward(self):
  #   input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
  #   layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
  #   layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
  #   layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
  #   output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
  #   network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
  #   network.forward([1, 1], [1])
  #   learning_rate = 0.1
  #   network.backward(learning_rate)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[0].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[1].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[2].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].weights[3].value, 0.0125)
  #   self.assertEqual(network.layers[-2].neurons[0].bias.value, 0.025)

  # def test_method_Network_train_AND(self):
  #   input_layer = Layer(Function.tanh, Function.mse, 4, 2, 0, 1, 'input')
  #   layer2 = Layer(Function.tanh, Function.mse, 4, 4, 0, 0, '1')
  #   layer3 = Layer(Function.tanh, Function.mse, 4, 4, 0, 0, '2')
  #   layer4 = Layer(Function.tanh, Function.mse, 4, 4, 0, 0, '3')
  #   output_layer = Layer(Function.tanh, Function.mse, 1, 4, 0, 0, 'output')
  #   network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
  #   inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
  #   targets = [[1], [0], [0], [0]]
  #   network.train(inputs, targets, 0.05, 1000)

  def test_method_Network_train_XOR(self):
    input_layer = Layer(Function.tanh, Function.mse, 5, 2, None, None, 'input')
    layer2 = Layer(Function.tanh, Function.mse, 5, 5, None, None, '1')
    layer3 = Layer(Function.tanh, Function.mse, 5, 5, None, None, '2')
    layer4 = Layer(Function.tanh, Function.mse, 5, 5, None, None, '3')
    output_layer = Layer(Function.tanh, Function.mse, 1, 5, None, None, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    targets = [[0], [1], [1], [0]]
    network.train(inputs, targets, 0.015, 10000)
    self.assertAlmostEqual(round(network.predict([1, 1])[0]), 0.0, places=2)
    self.assertAlmostEqual(round(network.predict([1, 0])[0]), 1.0, places=2)
    self.assertAlmostEqual(round(network.predict([0, 1])[0]), 1.0, places=2)
    self.assertAlmostEqual(round(network.predict([0, 0])[0]), 0.0, places=2)

  def test_method_Network_export_to_json(self):
    input_layer = Layer(Function.tanh, Function.mse, 5, 2, None, None, 'input')
    layer2 = Layer(Function.tanh, Function.mse, 5, 5, None, None, '1')
    layer3 = Layer(Function.tanh, Function.mse, 5, 5, None, None, '2')
    layer4 = Layer(Function.tanh, Function.mse, 5, 5, None, None, '3')
    output_layer = Layer(Function.tanh, Function.mse, 1, 5, None, None, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    targets = [[0], [1], [1], [0]]
    network.train(inputs, targets, 0.125, 10000)
    network.export_to_json('test.json')
    self.assertTrue(os.path.exists('test.json'))
    os.remove('test.json')







# --------------------------------------------------
class MathTool(object):

  e = 2.718281828459045

  @staticmethod
  def exp(x):
    return math.exp(x)

  @staticmethod
  def log(x):
    return math.log(x)

  @staticmethod
  def max(x, y):
    max = x
    if y > max:
      max = y
    return max

  @staticmethod
  def random_float(min, max):
    rand = os.urandom(8) # 64-bit
    random_float = (float(int.from_bytes(rand, byteorder='big')) / (1 << 64))
    return min + (max - min) * random_float


# --------------------------------------------------
class Function(object):
  @staticmethod
  def sigmoid(x):
    return 1 / (1 + MathTool.exp(-x))

  @staticmethod
  def sigmoid2(x):
    return 1 / (1 + MathTool.exp(-x))


  @staticmethod
  def tanh(x):
    return (MathTool.exp(x) - MathTool.exp(-x)) / (MathTool.exp(x) + MathTool.exp(-x))

  @staticmethod
  def relu(x):
    return MathTool.max(0, x)
  
  @staticmethod
  def mse(y, y_hat):
    return (y - y_hat) ** 2


# --------------------------------------------------
class Derivative(object):
  @staticmethod
  def sigmoid(x):
    return Function.sigmoid(x) * (1 - Function.sigmoid(x))

  @staticmethod
  def sigmoid2(x):
    return (MathTool.exp(-x)) / ((1+MathTool.exp(-x))**2)

  @staticmethod
  def tanh(x):
    return 1 - Function.tanh(x) ** 2

  @staticmethod
  def relu(x):
    if x > 0:
      return 1
    else:
      return 0

  @staticmethod
  def mse(y, y_hat):
    return -2 * (y - y_hat)



# --------------------------------------------------
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

# --------------------------------------------------

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

# --------------------------------------------------

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

# --------------------------------------------------

class Network(object):
  def __init__(self, *layers):
    self.layers = []
    for i in range(len(layers)):
      self.layers.append(layers[i])
    self.number_of_layers = len(self.layers)

  def update_info(self):
    self.number_of_layers = len(self.layers)
    self.number_of_inputs = self.layers[0].number_of_inputs
    self.number_of_outputs = self.layers[-1].number_of_neurons
    self.number_of_hidden_layers = self.number_of_layers - 2
    self.number_of_neurons = 0
    for layer in self.layers:
      self.number_of_neurons += layer.number_of_neurons
    self.number_of_weights = sum([layer.number_of_neurons * layer.number_of_inputs for layer in self.layers])
    self.number_of_biases = sum([layer.number_of_neurons for layer in self.layers])
    self.number_of_parameters = self.number_of_weights + self.number_of_biases

  def get_layer(self, index):
    return self.layers[index]

  def get_next_layer(self, index):
    next_layer = None
    if index < self.number_of_layers - 1:
      next_layer = self.layers[index + 1]
    return next_layer

  def get_previous_layer(self, index):
    previous_layer = None
    if index > 0:
      previous_layer = self.layers[index - 1]
    return previous_layer

  def get_layer_index(self, layer):
    return self.layers.index(layer)

  def predict(self, inputs):
    input_layer = self.layers[0]
    input_layer.inputs = inputs
    for layer in self.layers:
      output = layer.predict(layer.inputs)
      current_layer_index = self.get_layer_index(layer)
      next_layer = self.get_next_layer(current_layer_index)
      if next_layer is not None:
        next_layer.inputs = output
    return output

  def forward(self, inputs, targets):
    self.update_info()
    input_layer = self.layers[0]
    input_layer.inputs = inputs
    for layer in self.layers:
      target = None if layer is not self.layers[-1] else targets
      output = layer.forward(layer.inputs, target)
      current_layer_index = self.get_layer_index(layer)
      next_layer = self.get_next_layer(current_layer_index)
      if next_layer is not None:
        next_layer.inputs = output
    return output

  def calculate_output_error(self):
    # print("Network : " + str(self))
    output_layer = self.layers[-1]
    # print("Output Layer index : " + str(self.get_layer_index(output_layer)))
    for i in range(output_layer.number_of_neurons):
      # print("Neuron " + str(i) + " : " + str(output_layer.neurons[i].error))
      # print("Neuron Name : " + str(output_layer.neurons[i].name))
      # print("Neuron Target : " + str(output_layer.neurons[i].target))
      # print("Neuron Error : " + str(output_layer.neurons[i].error))
      output_layer.neurons[i].calculate_gradient()
      # print("Neuron Error (after the gradient process) : " + str(output_layer.neurons[i].error))

    self.output_error = sum([neuron.error for neuron in output_layer.neurons])

  def calculate_previous_layers_errors(self):
    output_error = self.output_error
    for i in range(self.number_of_layers - 1, 0, -1):
      if i == self.number_of_layers - 1:
        for neuron in self.layers[i].neurons:
          neuron.calculate_gradient(output_error)
      else:
        next_layer_error = sum([neuron.error for neuron in self.layers[i + 1].neurons])
        for neuron in self.layers[i].neurons:
          neuron.calculate_gradient(next_layer_error)

  def update_superparameters(self, learning_rate):
    for layer in self.layers:
      layer.backward(learning_rate, backward=True)


  def backward(self, learning_rate):
    self.calculate_output_error()
    self.calculate_previous_layers_errors()
    self.update_superparameters(learning_rate)

  def train(self, inputs, targets, learning_rate, epochs, verbose=False):
    error = 1
    for epoch in range(epochs):
      # if epoch == 0:
      #   last_error = 5
      # if epoch != 0:
      #   print(abs(last_error), abs(self.output_error))
      #   if abs(last_error) <= abs(self.output_error):
      #     learning_rate = float(learning_rate / 10)
      #   last_error = self.output_error
      # print(epoch)
      for i in range(len(inputs)):
        self.forward(inputs[i], targets[i])
        self.backward(learning_rate)
        error = self.output_error
      if verbose:
        print("a = " + str(learning_rate) + "; Epoch : " + str(epoch) + " | Error : " + str(self.output_error))
    return error


  def export_to_json(self, filename):
    # store the model in a json variable
    json_string = json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)
    # write the model in a json file
    with open(filename, 'w') as f:
      f.write(json_string)

  @staticmethod
  def import_from_json(filename):
    json_string = ""
    with open(filename, 'r') as f:
      json_string = f.read()
    return json.parse(json_string)



  def dojo(self, inputs, targets, learning_rate):
    # train the network
    error = 1
    last_error = 2
    flap = 0
    stale = 0
    step = 0.1
    grow = 0
    self.update_info()
    i = 0
    running = True
    while abs( error ) > 0.001 and running == True:
      error = self.train(inputs, targets, learning_rate+0.00000000001, 100, verbose=False)
      self.print_outputs()
      # self.reset_parameters()
      for inps in inputs:
        print(self.predict(inps))
      if i != 0:
        if abs(last_error) + abs(error) != abs(last_error + error):
          flap += 1
        if last_error == error:
          stale += 1
        if 0 > last_error > error or error > last_error > 0:
          print('GROW')
          grow += 1
        if (flap + 1) % 100 == 0:
          learning_rate -= (step * learning_rate)
          flap = 0
          running = False
        if (stale + 1) % 200 == 0:
          learning_rate += (1.1 * learning_rate)
          stale = 0
          running = False
        if (grow + 1) % 50 == 0:
          learning_rate -= (step * 10 * learning_rate)
          grow = 0
          running = False
      print("a: " + str(learning_rate))
      print("Error: " + str(error))
      print("flap: " + str(flap) + " grow: " + str(grow) + " stale: " + str(stale))
      last_error = error
      i += 1

  def get_all_neurons(self):
    neurons = []
    for i in range(len(self.layers)):
      for j in range(len(self.layers[i].neurons)):
        neurons.append(self.layers[i].neurons[j])
    return neurons

  def print_outputs(self):
    neurons = self.get_all_neurons()
    outputs = []
    for i in range(len(neurons)):
      outputs.append(neurons[i].output)
    print(outputs)

  def reset_parameters(self, bias_value=None, weights_value=None):
    neurons = self.get_all_neurons()
    for i in range(len(neurons)):
      if neurons[i].output == 1 or neurons[i].output == 0:
        neurons[i].bias = Bias(bias_value)
        neurons[i].weights = [Weight(weights_value) for i in range(neurons[i].number_of_inputs)]

        




# --------------------------------------------------
if __name__ == '__main__':
  # unittest.main()
  print('ok')
  # create a layer to take a word in input
  # input_layer = Layer(Function.tanh, Function.mse, 3, 2)
  # layer1 = Layer(Function.tanh, Function.mse, 3, 3)

  # output_layer = Layer(Function.tanh, Function.mse, 1, 3)

  # train network to learn XOR
  # network = Network(input_layer, layer1, output_layer)
  # network.update_info()
  # inputs = [[0, 1],  [1, 0], [0, 0], [1, 1]]
  # targets = [[1], [1], [0], [0]]
  # network.dojo(inputs, targets, 0.0001)




