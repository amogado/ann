import unittest
from .maths import *
from .neuron import *
from .layer import *
from .network import *

# --------------------------------------------------------------------
class Tests(unittest.TestCase):

  @staticmethod
  def call_tests(name):
    __name__ = name
    if __name__ == '__main__':
      unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
      exit("Not in the main")

  def test_class_MathTool(self):
    self.assertIsInstance(MathTool, object)

  def test_method_MathTool_e(self):
    self.assertAlmostEqual(MathTool.e, 2.718281828459045)

  def test_method_MathTool_exp(self):
    self.assertAlmostEqual(MathTool.exp(1), 2.718281828459045)
    self.assertAlmostEqual(MathTool.exp(2), 7.3890560989306495)
    self.assertAlmostEqual(MathTool.exp(3), 20.085536923187664)
    self.assertAlmostEqual(MathTool.exp(4), 54.59815003314423)
    self.assertAlmostEqual(MathTool.exp(5), 148.41315910257657)

  def test_method_MathTool_log(self):
    self.assertAlmostEqual(MathTool.log(1), 0.0)
    self.assertAlmostEqual(MathTool.log(2), 0.6931471805599453)
    self.assertAlmostEqual(MathTool.log(3), 1.0986122886681098)
    self.assertAlmostEqual(MathTool.log(4), 1.3862943611198906)
    self.assertAlmostEqual(MathTool.log(5), 1.6094379124341003)

  def test_class_Function(self):
    self.assertIsInstance(Function, object)

  def test_method_Function_sigmoid(self):
    self.assertAlmostEqual(Function.sigmoid(0), 0.5)
    self.assertAlmostEqual(Function.sigmoid(1), 0.7310585786300049)
    self.assertAlmostEqual(Function.sigmoid(2), 0.8807970779778823)
    self.assertAlmostEqual(Function.sigmoid(3), 0.9525741268224334)
    self.assertAlmostEqual(Function.sigmoid(4), 0.9820137900379085)

  def test_method_Function_tanh(self):
    self.assertAlmostEqual(Function.tanh(0), 0.0)
    self.assertAlmostEqual(Function.tanh(1), 0.7615941559557649)
    self.assertAlmostEqual(Function.tanh(2), 0.9640275800758169)
    self.assertAlmostEqual(Function.tanh(3), 0.9950547536867305)
    self.assertAlmostEqual(Function.tanh(4), 0.999329299739067)

  def test_method_Function_relu(self):
    self.assertAlmostEqual(Function.relu(0), 0.0)
    self.assertAlmostEqual(Function.relu(1), 1.0)
    self.assertAlmostEqual(Function.relu(2), 2.0)
    self.assertAlmostEqual(Function.relu(3), 3.0)
    self.assertAlmostEqual(Function.relu(4), 4.0)

  def test_method_Function_mse(self):
    self.assertAlmostEqual(Function.mse(1, 1), 0.0)
    self.assertAlmostEqual(Function.mse(1, 2), 1.0)
    self.assertAlmostEqual(Function.mse(2, 1), 1.0)
    self.assertAlmostEqual(Function.mse(2, 2), 0.0)

  def test_method_MathTool_max(self):
    self.assertEqual(MathTool.max(1, 2), 2)
    self.assertEqual(MathTool.max(2, 1), 2)
    self.assertEqual(MathTool.max(1, 1), 1)

  def test_class_Derivative(self):
    self.assertIsInstance(Derivative, object)

  def test_method_Derivative_sigmoid(self):
    self.assertAlmostEqual(Derivative.sigmoid(0), 0.25)
    self.assertAlmostEqual(Derivative.sigmoid(1), 0.19661193324148185)
    self.assertAlmostEqual(Derivative.sigmoid(2), 0.10499358540350662)
    self.assertAlmostEqual(Derivative.sigmoid(3), 0.04517665973091267)
    self.assertAlmostEqual(Derivative.sigmoid(4), 0.01766270621332736)

  def test_method_Derivative_tanh(self):
    self.assertAlmostEqual(Derivative.tanh(0), 1.0)
    self.assertAlmostEqual(Derivative.tanh(1), 0.41997434161402614)
    self.assertAlmostEqual(Derivative.tanh(2), 0.07065082485316466)
    self.assertAlmostEqual(Derivative.tanh(3), 0.009866037165440922)
    self.assertAlmostEqual(Derivative.tanh(4), 0.0013409507920792178)

  def test_method_Derivative_relu(self):
    self.assertAlmostEqual(Derivative.relu(0), 0.0)
    self.assertAlmostEqual(Derivative.relu(1), 1.0)
    self.assertAlmostEqual(Derivative.relu(2), 1.0)
    self.assertAlmostEqual(Derivative.relu(3), 1.0)
    self.assertAlmostEqual(Derivative.relu(4), 1.0)

  def test_method_Derivative_mse(self):
    self.assertAlmostEqual(Derivative.mse(1, 1), 0.0)
    self.assertAlmostEqual(Derivative.mse(1, 2), 2.0)
    self.assertAlmostEqual(Derivative.mse(2, 1), -2.0)
    self.assertAlmostEqual(Derivative.mse(2, 2), 0.0)

  def test_method_MathTool_random_float(self):
    self.assertGreaterEqual(MathTool.random_float(0, 1), 0.0)
    self.assertLessEqual(MathTool.random_float(0, 100), 100.0)
    self.assertGreaterEqual(MathTool.random_float(0, 100), 0.0)
    self.assertNotEqual(MathTool.random_float(0, 100), MathTool.random_float(0, 100))

  def test_class_Bias(self):
    self.assertIsInstance(Bias, object)

  def test_method_Bias_init(self):
    bias = Bias(1.0)
    self.assertEqual(bias.value, 1.0)
    bias2 = Bias()
    self.assertGreaterEqual(bias2.value, -1.0)
    self.assertLessEqual(bias2.value, 1.0)

  def test_class_Weight(self):
    self.assertIsInstance(Weight, object)

  def test_method_Weight_init(self):
    weight = Weight(1.0)
    self.assertEqual(weight.value, 1.0)
    weight2 = Weight()
    self.assertGreaterEqual(weight2.value, -1.0)
    self.assertLessEqual(weight2.value, 1.0)

  def test_class_Activation(self):
    self.assertIsInstance(Activation, object)

  def test_method_Activation_init(self):
    activation = Activation(Function.sigmoid)
    self.assertEqual(activation.function, Function.sigmoid)
    self.assertEqual(activation.derivative, Derivative.sigmoid)
    activation2 = Activation(Function.tanh)
    self.assertEqual(activation2.function, Function.tanh)
    self.assertEqual(activation2.derivative, Derivative.tanh)
    activation3 = Activation(Function.relu)
    self.assertEqual(activation3.function, Function.relu)
    self.assertEqual(activation3.derivative, Derivative.relu)

  def test_class_Loss(self):
    self.assertIsInstance(Loss, object)

  def test_method_Loss_init(self):
    loss = Loss(Function.mse)
    self.assertEqual(loss.function, Function.mse)
    self.assertEqual(loss.derivative, Derivative.mse)

  def test_class_Neuron(self):
    self.assertIsInstance(Neuron, object)

  def test_method_Neuron_init(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    self.assertEqual(neuron.activation.function, Function.sigmoid)
    self.assertEqual(neuron.activation.derivative, Derivative.sigmoid)
    self.assertEqual(neuron.loss.function, Function.mse)
    self.assertEqual(neuron.loss.derivative, Derivative.mse)
    self.assertGreaterEqual(neuron.bias.value, -1.0)
    self.assertLessEqual(neuron.bias.value, 1.0)
    self.assertEqual(len(neuron.weights), 1)
    self.assertGreaterEqual(neuron.weights[0].value, -1.0)
    self.assertLessEqual(neuron.weights[0].value, 1.0)

  def test_method_Neuron_linear_combination(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    neuron.weights[0].value = 1.0
    neuron.bias.value = 1.0
    self.assertEqual(Neuron.linear_combination(neuron, [1]), 2.0)

  def test_method_Neuron_predict(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    neuron.weights[0].value = 1.0
    neuron.bias.value = 1.0
    self.assertAlmostEqual(neuron.predict([1]), 0.8807970779778823)

  def test_method_Neuron_forward(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    neuron.weights[0].value = 1.0
    neuron.bias.value = 1.0
    self.assertAlmostEqual(neuron.forward([1], 1), 0.8807970779778823)
    self.assertEqual(neuron.target, 1)

  def test_method_Neuron_calculate_gradient(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    neuron.weights[0].value = 1.0
    neuron.bias.value = 1.0
    neuron.forward([1], 1)
    neuron.calculate_gradient()
    self.assertAlmostEqual(neuron.bias.gradient, -0.025031084347353506)
    self.assertAlmostEqual(neuron.weights[0].gradient, -0.025031084347353506)

  def test_method_Neuron_update_superparameters(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    neuron.weights[0].value = 1.0
    neuron.bias.value = 1.0
    neuron.forward([1], 1)
    neuron.calculate_gradient()
    neuron.update_superparameters(0.1)
    self.assertAlmostEqual(neuron.bias.value, 1.0025031084347354)
    self.assertAlmostEqual(neuron.weights[0].value, 1.0025031084347354)
    self.assertNotEqual(neuron.bias.value, 1.0)
    self.assertNotEqual(neuron.weights[0].value, 1.0)

  def test_method_Neuron_backward(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1)
    neuron.weights[0].value = 1.0
    neuron.bias.value = 1.0
    predict = neuron.predict([1])
    neuron.forward([1], 0)
    neuron.backward(0.1)
    self.assertAlmostEqual(neuron.bias.value, 0.981504391354034)
    self.assertAlmostEqual(neuron.weights[0].value, 0.981504391354034)
    self.assertNotEqual(neuron.bias.value, 1.0)
    self.assertNotEqual(neuron.weights[0].value, 1.0)
    new_predict = neuron.predict([1])
    self.assertNotEqual(predict, new_predict)
    self.assertLess(new_predict, predict)

  def test_class_Layer(self):
    self.assertIsInstance(Layer, object)

  def test_method_Layer_init(self):
    number_of_neurons = 4
    number_of_inputs = 2
    bias_value = 0.5
    weights_value = 0.5
    layer = Layer(Function.sigmoid, Function.mse, number_of_neurons, number_of_inputs, bias_value, weights_value)
    self.assertEqual(len(layer.neurons), number_of_neurons)
    self.assertEqual(len(layer.neurons[0].weights), number_of_inputs)
    self.assertEqual(len(layer.neurons[1].weights), number_of_inputs)
    self.assertEqual(len(layer.neurons[2].weights), number_of_inputs)
    self.assertEqual(len(layer.neurons[3].weights), number_of_inputs)
    self.assertEqual(layer.number_of_neurons, number_of_neurons)
    self.assertEqual(layer.number_of_inputs, number_of_inputs)
    for neuron in layer.neurons:
      self.assertEqual(neuron.activation.function, Function.sigmoid)
      self.assertEqual(neuron.activation.derivative, Derivative.sigmoid)
      self.assertEqual(neuron.loss.function, Function.mse)
      self.assertEqual(neuron.loss.derivative, Derivative.mse)
      self.assertGreaterEqual(neuron.bias.value, -1.0)
      self.assertLessEqual(neuron.bias.value, 1.0)
      self.assertEqual(len(neuron.weights), number_of_inputs)
      for weight in neuron.weights:
        self.assertGreaterEqual(weight.value, -1.0)
        self.assertLessEqual(weight.value, 1.0)
    
  def test_method_Layer_predict(self):
    layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    self.assertAlmostEqual(layer.predict([1, 1])[0], 0.9525741268224334)
    self.assertAlmostEqual(layer.predict([1, 1])[1], 0.9525741268224334)
    self.assertAlmostEqual(layer.predict([1, 1])[2], 0.9525741268224334)
    self.assertAlmostEqual(layer.predict([1, 1])[3], 0.9525741268224334)

  def test_method_Layer_forward(self):
    layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[0], 0.9525741268224334)
    self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[1], 0.9525741268224334)
    self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[2], 0.9525741268224334)
    self.assertAlmostEqual(layer.forward([1, 1], [1, 1, 1, 1])[3], 0.9525741268224334)
    self.assertEqual(layer.targets, [1, 1, 1, 1])

  def test_method_Layer_backward(self):
    layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    predict = layer.predict([1, 1])
    layer.forward([1, 1], [1, 1, 1, 1])
    layer.backward(0.1)
    new_predict = layer.predict([1, 1])
    self.assertNotEqual(predict, new_predict)

  def test_class_Network(self):
    self.assertIsInstance(Network, object)

  def test_method_Network_init(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    self.assertEqual(network.layers[0], input_layer)
    self.assertEqual(network.layers[1], layer2)
    self.assertEqual(network.layers[2], layer3)
    self.assertEqual(network.layers[3], output_layer)

  def test_method_Network_update_info(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    network.update_info()
    self.assertEqual(network.number_of_layers, 4)
    self.assertEqual(network.number_of_inputs, 2)
    self.assertEqual(network.number_of_outputs, 1)
    self.assertEqual(network.number_of_neurons, 13)
    self.assertEqual(network.number_of_weights, 44)
    self.assertEqual(network.number_of_biases, 13)
    self.assertEqual(network.number_of_parameters, 57)
    self.assertEqual(network.number_of_hidden_layers, 2)

  def test_method_Network_get_layer(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    self.assertEqual(network.get_layer(0), input_layer)
    self.assertEqual(network.get_layer(1), layer2)
    self.assertEqual(network.get_layer(2), layer3)
    self.assertEqual(network.get_layer(3), output_layer)

  def test_method_Network_get_next_layer(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    self.assertEqual(network.get_next_layer(0), layer2)
    self.assertEqual(network.get_next_layer(1), layer3)
    self.assertEqual(network.get_next_layer(2), output_layer)
    self.assertEqual(network.get_next_layer(3), None)

  def test_method_Network_get_previous_layer(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    self.assertEqual(network.get_previous_layer(0), None)
    self.assertEqual(network.get_previous_layer(1), input_layer)
    self.assertEqual(network.get_previous_layer(2), layer2)
    self.assertEqual(network.get_previous_layer(3), layer3)

  def test_method_Network_get_layer_index(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    self.assertEqual(network.get_layer_index(input_layer), 0)
    self.assertEqual(network.get_layer_index(layer2), 1)
    self.assertEqual(network.get_layer_index(layer3), 2)
    self.assertEqual(network.get_layer_index(output_layer), 3)

  def test_method_Network_predict(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    self.assertEqual(network.predict([1, 1]), [0.9931208367972153])

  def test_method_Network_forward(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 1, 1)
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 1, 1)
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 1, 1)
    network = Network(input_layer, layer2, layer3, output_layer)
    network.forward([1, 1], [0])
    self.assertEqual(network.layers[0].neurons[0].output, 0.9525741268224334)
    self.assertEqual(network.layers[0].neurons[1].output, 0.9525741268224334)
    self.assertEqual(network.layers[0].neurons[2].output, 0.9525741268224334)
    self.assertEqual(network.layers[0].neurons[3].output, 0.9525741268224334)
    self.assertEqual(network.layers[1].neurons[0].output, 0.9919203680349579)
    self.assertEqual(network.layers[1].neurons[1].output, 0.9919203680349579)
    self.assertEqual(network.layers[1].neurons[2].output, 0.9919203680349579)
    self.assertEqual(network.layers[1].neurons[3].output, 0.9919203680349579)
    self.assertEqual(network.layers[2].neurons[0].output, 0.9930888320533605)
    self.assertEqual(network.layers[2].neurons[1].output, 0.9930888320533605)
    self.assertEqual(network.layers[2].neurons[2].output, 0.9930888320533605)
    self.assertEqual(network.layers[2].neurons[3].output, 0.9930888320533605)
    self.assertEqual(network.layers[3].neurons[0].output, 0.9931208367972153)

  def test_method_Layer_forward_no_target(self):
    layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 0)
    layer.forward([1, 1])
    self.assertEqual(layer.neurons[0].output, 0.5)
    self.assertEqual(layer.neurons[1].output, 0.5)
    self.assertEqual(layer.neurons[2].output, 0.5)
    self.assertEqual(layer.neurons[3].output, 0.5)

  def test_method_Neuron_get_info(self):
    neuron = Neuron(Function.sigmoid, Function.mse, 1, 0, 0, name="test")
    neuron.forward([1], [1])
    self.assertEqual(neuron.get_info(), {'name': 'test', 'bias': 0, 'activation': 'sigmoid', 'loss': 'mse', 'weights': [0], 'output': 0.5, 'error': 0, 'target': [1]})

  def test_method_Network_calculate_output_error(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
    layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    network.forward([1, 1], [1])
    network.calculate_output_error()
    self.assertEqual(network.layers[-1].neurons[0].error, -1)

  def test_method_Network_calculate_previous_layers_errors(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
    layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    network.forward([1, 1], [1])
    network.calculate_output_error()
    network.calculate_previous_layers_errors()
    self.assertEqual(network.layers[-2].neurons[0].error, -1)
    self.assertEqual(network.layers[-2].neurons[1].error, -1)
    self.assertEqual(network.layers[-2].neurons[2].error, -1)
    self.assertEqual(network.layers[-2].neurons[3].error, -1)

  def test_method_Network_update_superparameters(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
    layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    network.forward([1, 1], [1])
    network.calculate_output_error()
    network.calculate_previous_layers_errors()
    network.update_superparameters(0.1)
    self.assertEqual(network.layers[-2].neurons[0].weights[0].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].weights[1].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].weights[2].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].weights[3].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].bias.value, 0.025)

  def test_method_Network_backward(self):
    input_layer = Layer(Function.sigmoid, Function.mse, 4, 2, 0, 1, 'input')
    layer2 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '1')
    layer3 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '2')
    layer4 = Layer(Function.sigmoid, Function.mse, 4, 4, 0, 0, '3')
    output_layer = Layer(Function.sigmoid, Function.mse, 1, 4, 0, 0, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    network.forward([1, 1], [1])
    learning_rate = 0.1
    network.backward(learning_rate)
    self.assertEqual(network.layers[-2].neurons[0].weights[0].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].weights[1].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].weights[2].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].weights[3].value, 0.0125)
    self.assertEqual(network.layers[-2].neurons[0].bias.value, 0.025)

  def test_method_Network_train_AND(self):
    input_layer = Layer(Function.tanh, Function.mse, 4, 2, 0, 1, 'input')
    layer2 = Layer(Function.tanh, Function.mse, 4, 4, 0, 0, '1')
    layer3 = Layer(Function.tanh, Function.mse, 4, 4, 0, 0, '2')
    layer4 = Layer(Function.tanh, Function.mse, 4, 4, 0, 0, '3')
    output_layer = Layer(Function.tanh, Function.mse, 1, 4, 0, 0, 'output')
    network = Network(input_layer, layer2, layer3, layer4, output_layer)
    
    inputs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    targets = [[1], [0], [0], [0]]
    network.train(inputs, targets, 0.05, 1000)

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
