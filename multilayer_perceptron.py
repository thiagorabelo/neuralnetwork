import math
import random

import util

from typing import List, Union, Callable, TypeVar, Iterable, Tuple

from matrix import Matrix
from util import MatBaseType, MatType, MatProxyType, RowType, ColType, Number


class ActivationFunction:

    def __init__(self,
                 func: Callable[[Number], Number],
                 dfunc: Callable[[Number], Number]):
        self._func = func
        self._dfunc = dfunc

    def func(self, val: Number, *args):
        return self._func(val)

    def dfunc(self, val: Number, *args):
        return self._dfunc(val)

    def __call__(self, val: Number, *args):
        return self._func(val)


@util.clip(-700, 700)
def _sigmoid(val: Number) -> Number:
    return 1.0 / (1.0 + math.exp(-val))


def _dsigmoid(val: Number) -> Number:
    sig = _sigmoid(val)
    return sig * (1.0 - sig)


def _dtanh(val: Number) -> Number:
    tanh = math.tanh(val)
    return 1.0 - math.pow(tanh, 2.0)


def _relu(val: Number) -> Number:
    return max(0, val)


def _drelu(val: Number) -> Number:
    return 1.0 if val > 0.0 else 0.0


ACTIVATIONS_FUNCTIONS = {
    'sigmoid': ActivationFunction(
        _sigmoid,
        _dsigmoid
    ),

    'tanh': ActivationFunction(
        math.tanh,
        _dtanh
    ),

    'relu': ActivationFunction(
        _relu,
        _drelu
    ),

    'sin': ActivationFunction(
        math.sin,
        math.cos
    ),

    'linear': ActivationFunction(
        lambda val: val,
        lambda _: 1.0
    ),

    'custom': ActivationFunction(
        lambda val: 0.5 * math.pow(val, 2.0) - 18,
        lambda val: val
    )
}


class MLP:

    def __init__(self,
                 n_inputs: int,
                 layers: List[int],
                 activation_func: ActivationFunction,
                 activation_func_output: ActivationFunction = None):

        self.n_inputs = n_inputs
        self.n_outputs = layers[-1]
        self.layers_weights = []
        self.layers_bias = []

        self.activation_function = activation_func

        if activation_func_output:
            self.activation_func_output = activation_func_output
        else:
            self.activation_func_output = self.activation_function

        weights_list = [self.n_inputs] + layers

        for index, layer in enumerate(weights_list[1:], 1):
            weights = Matrix(layer, weights_list[index - 1])
            bias = Matrix(layer, 1)

            weights.randomize(lambda: random.uniform(-1.0, 1.0))
            bias.randomize(lambda: random.uniform(-1.0, 1.0))

            self.layers_weights.append(weights)
            self.layers_bias.append(bias)

    def randomise_weights(self, rand: Callable[[], Number] = None) -> None:
        if rand:
            args = (rand, )
        else:
            args = tuple()

        for weights, bias in zip(self.layers_weights, self.layers_bias):
            weights.randomize(*args)
            bias.randomize(*args)

    def linear_combination(self,
                           input_matrix: MatBaseType,
                           weights: MatBaseType,
                           bias: MatBaseType) -> MatBaseType:
        # | w11 w21 wb1 | * | i1 |
        # | w12 w22 wb2 |   | i2 |
        #                   |  1 |

        # | w11*i1 + w21*i2 + wb1*1 |
        # | w12*i1 + w22*i2 + wb2*1 |

        # Same as:

        # | w11 w21 | * | i1 | + | wb1 |
        # | w12 w22 |   | i2 |   | wb2 |

        # | w11*i1 + w21*i2 | + | wb1 | = | w11*i1 + w21*i2 + wb1 |
        # | w12*i1 + w22*i2 |   | wb2 |   | w12*i1 + w22*i2 + wb2 |
        matrix = weights @ input_matrix
        matrix += bias
        return matrix

    def apply_activation_function(self, matrix: MatBaseType, last_layer: bool) -> MatBaseType:
        if not last_layer:
            matrix.imap(self.activation_function)
        else:
            matrix.imap(self.activation_func_output)

        return matrix

    def walk_layers(self) -> Iterable[Tuple[MatBaseType, MatBaseType, int, bool]]:
        last_layer_index = len(self.layers_weights) - 1

        for index, (weights, bias) in enumerate(zip(self.layers_weights, self.layers_bias)):
            yield weights, bias, index, index == last_layer_index

    def predict(self, input_array: List[Number]) -> List[Number]:
        matrix = Matrix.from_array(self.n_inputs, 1, input_array)

        for weights, bias, index, is_last_layer in self.walk_layers():
            matrix = self.linear_combination(matrix, weights, bias)
            matrix = self.apply_activation_function(matrix, is_last_layer)

        return list(matrix)


class Supervisor:

    learning_rate: float = 0.8

    def __init__(self, mlp: MLP, learning_rate: float = None, normalize: bool = False):
        self.mlp = mlp
        self.backpropagation = BackpropagationHelper(self)
        self.learning_rate = learning_rate or Supervisor.learning_rate
        self.normalize: bool = normalize
        self.normalizator: util.Normalizator = None

    def forward_signal(self, matrix: MatBaseType) -> MatBaseType:
        self.backpropagation.phi_layers[0] = matrix

        for weights, bias, index, is_last_layer in self.mlp.walk_layers():
            matrix = self.mlp.linear_combination(matrix, weights, bias)
            self.backpropagation.linear_combinations[index] = matrix

            matrix = self.mlp.apply_activation_function(matrix.copy(), is_last_layer)
            if not is_last_layer:
                self.backpropagation.phi_layers[index + 1] = matrix

        return matrix

    def train(self, matrix: MatBaseType, target: MatBaseType) -> Number:
        # Δ/δ	Delta/delta
        # Φ/φ	Phi/phi

        # δ = local_gradient
        # Δ = delta
        # φ = phi

        matrix = self.forward_signal(matrix)

        error = target - matrix
        inst_average_error = (error @ error.t).get(0, 0) / 2.0

        self.backpropagation.backpropagate(error)
        self.backpropagation.adjust_weights()

        return inst_average_error

    def train_set(self,
                  train_set: Iterable[Union[Tuple[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]],
                  min_error: float,
                  max_epochs: int):
        average_global_error = 0.0
        train_set_size = len(train_set)

        if self.normalize:
            norm = util.Normalizator(train_set, self.mlp.n_inputs, self.mlp.n_outputs)
            self.normalizator = norm
            train_set = tuple(
                (norm.normalize_inputs(train_data[0]), norm.normalize_targets(train_data[1]))
                for train_data in train_set
            )

        train_set = tuple(
            (Matrix.from_array(self.mlp.n_inputs, 1, input_array),
             Matrix.from_array(len(target_array), 1, target_array))
            for input_array, target_array in train_set
        )

        for epoch in range(1, max_epochs+1):
            random_train_set = random.sample(train_set, len(train_set))

            for iteration, (input_array, target_array) in enumerate(random_train_set, 1):
                try:
                    inst_average_error = self.train(input_array, target_array)
                except Exception as ex:
                    print(f"ERROR ON: Epoch={epoch}, Iteration={iteration}")
                    raise ex
                average_global_error += inst_average_error

            average_global_error /= train_set_size

            print(f'AvgGlobalError={round(average_global_error, 15)} - Epoch={epoch}')

            if average_global_error <= min_error:
                break


class BackpropagationHelper:

    def __init__(self, supervisor: Supervisor):
        mlp = supervisor.mlp

        self.supervisor: Supervisor = supervisor

        # vi(n)
        self.linear_combinations: List[MatBaseType] = [None] * min(len(mlp.layers_weights),
                                                                   len(mlp.layers_bias))
        # φ(vi(n))
        self.phi_layers: List[MatBaseType] = [None] * len(self.linear_combinations)

        # δi(n)
        self.gradients: List[MatBaseType] = [None] * len(self.linear_combinations)

        # Δwi(n)
        self.deltas_w: List[MatBaseType] = [None] * len(self.linear_combinations)

        # Δbi(n)
        self.deltas_b: List[MatBaseType] = [None] * len(self.linear_combinations)

    def backpropagate(self, error: MatBaseType) -> None:
        mlp = self.supervisor.mlp
        supervisor = self.supervisor

        phi_layers = self.phi_layers
        layers_weights = mlp.layers_weights
        gradients = self.gradients
        deltas_w = self.deltas_w
        deltas_b = self.deltas_b

        reversed_linear_combinations = util.enumerate_reversed(self.linear_combinations)

        # Calculate output layer deltas
        index, output_layer_linear_combinations = next(reversed_linear_combinations)
        derivative = output_layer_linear_combinations.map(mlp.activation_func_output.dfunc)
        gradients[index] = -error * derivative

        # deltas_w[index] = -supervisor.learning_rate * (gradients[index] @ phi_layers[index].t)
        deltas_w[index] = gradients[index] @ phi_layers[index].t
        deltas_w[index] *= -supervisor.learning_rate

        deltas_b[index] = -supervisor.learning_rate * gradients[index]

        # Calculate hidden layer deltas
        for index, linear_combination in reversed_linear_combinations:
            derivative = linear_combination.map(mlp.activation_function.dfunc)
            mult_gradients_weights = layers_weights[index + 1].t @ gradients[index + 1]

            gradients[index] = derivative * mult_gradients_weights

            # deltas_w[index] = -supervisor.learning_rate * (gradients[index] @ phi_layers[index].t)
            deltas_w[index] = gradients[index] @ phi_layers[index].t
            deltas_w[index] *= -supervisor.learning_rate

            deltas_b[index] = (-supervisor.learning_rate * gradients[index])

    def adjust_weights(self) -> None:
        mlp = self.supervisor.mlp

        for index, (deltas_w, deltas_b) in enumerate(zip(self.deltas_w, self.deltas_b)):
            mlp.layers_weights[index] += deltas_w
            mlp.layers_bias[index] += deltas_b
