import math
import os


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


