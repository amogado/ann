import pickle

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
    output_layer = self.layers[-1]
    for i in range(output_layer.number_of_neurons):
      output_layer.neurons[i].calculate_gradient()

    self.output_error = sum([neuron.error for neuron in output_layer.neurons])

  def calculate_previous_layers_errors(self):
    output_error = self.output_error
    for i in range(self.number_of_layers - 1, -1, -1):
      # print("layer: " + str(i))
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
      for i in range(len(inputs)):
        self.forward(inputs[i], targets[i])
        self.backward(learning_rate)
        error = self.output_error
      if verbose:
        print("a = " + str(learning_rate) + "; Epoch : " + str(epoch) + " | Error : " + str(self.output_error))
    return error


  def export_to_file(self, filename):
    with open(filename, 'wb+') as file:
      pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)


  @staticmethod
  def import_from_file(filename):
    with open(filename, 'rb+') as f:
      return pickle.load(f)




  def dojo(self, inputs, targets, learning_rate, max_loops=None):
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
    while abs( error ) > 0.001 and running == True or (max_loops is None or i < max_loops):
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

