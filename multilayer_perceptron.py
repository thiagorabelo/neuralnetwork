import math

from typing import List, Union, Callable, TypeVar, Iterable, Tuple

from matrix import Matrix


Number = Union[int, float]
MBase = TypeVar('MBase', bound='MatrixBase')


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


def _sigmoid(val: Number) -> Number:
    return 1.0 / (1.0 + math.exp(-val))


def _dsigmoid(val: Number) -> Number:
    sig = _sigmoid(val)
    return sig * (1.0 - sig)


def _dtanh(val: Number) -> Number:
    tanh = math.tanh(val)
    return 1.0 - math.pow(tanh, 2.0)


ACTIVATIONS_FUNCTIONS = {
    'sigmoid': ActivationFunction(
        _sigmoid,
        _dsigmoid
    ),

    'tanh': ActivationFunction(
        math.tanh,
        _dtanh
    ),

    'linear': ActivationFunction(
        lambda val: val,
        lambda _: 1
    )
}


class MLP:

    activation_function: ActivationFunction = ACTIVATIONS_FUNCTIONS['sigmoid']

    def __init__(self, n_inputs: int, layers: List[int], pure_linear_output=False):
        self.n_inputs = n_inputs
        self.layers_weights = []
        self.layers_bias = []
        self.pure_linear_output = pure_linear_output

        weights_list = [self.n_inputs] + layers

        for index, layer in enumerate(weights_list[1:], 1):
            weights = Matrix(layer, weights_list[index - 1])
            bias = Matrix(layer, 1)

            weights.randomize()
            bias.randomize()

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

    def linear_combination(self, input_matrix: MBase, weights: MBase, bias) -> MBase:
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

    def apply_activation_function(self, matrix: MBase, last_layer: bool) -> MBase:
        if not last_layer or not self.pure_linear_output:
            matrix.imap(self.activation_function)
        return matrix

    def walk_layers(self) -> Iterable[Tuple[MBase, MBase, int, bool]]:
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

    def __init__(self, mlp: MLP, learning_rate: float = None):
        self.mlp = mlp
        self.backpropagation = BackpropagationHelper(self)
        self.learning_rate = learning_rate or Supervisor.learning_rate

    def train(self, input_array: List[Number], target_array: List[Number]) -> None:
        # Δ/δ	Delta/delta
        # Φ/φ	Phi/phi

        # δ = local_gradient
        # Δ = delta
        # φ = phi

        matrix = Matrix.from_array(self.mlp.n_inputs, 1, input_array)
        target = Matrix.from_array(len(target_array), 1, target_array)

        self.backpropagation.phi_layers[0] = matrix

        for weights, bias, index, is_last_layer in self.mlp.walk_layers():
            matrix = self.mlp.linear_combination(matrix, weights, bias)
            self.backpropagation.linear_combinations[index] = matrix

            matrix = self.mlp.apply_activation_function(matrix * 1, is_last_layer)
            if not is_last_layer:
                self.backpropagation.phi_layers[index + 1] = matrix

        error = target - matrix
        # inst_average_error = (error @ error.t).get(0, 0) / 2.0
        # TODO: Calc Global Average Error

        linear_combinations = reversed(self.backpropagation.linear_combinations)
        phi_layers = list(reversed(self.backpropagation.phi_layers))
        layers_weights = list(reversed(self.mlp.layers_weights))

        # Backpropagation
        for index, linear_combination in enumerate(linear_combinations):
            if not index == 0:
                derived = linear_combination.map(self.mlp.activation_function.dfunc)
                mult_gradients_weights = layers_weights[index - 1].t @ self.backpropagation.gradients[index - 1]
                self.backpropagation.gradients[index] = derived * mult_gradients_weights
                self.backpropagation.deltas_w[index] = -self.learning_rate * (self.backpropagation.gradients[index]
                                                                              @ phi_layers[index].t)
                self.backpropagation.deltas_b[index] = -self.learning_rate * self.backpropagation.gradients[index]
            else:
                derived = linear_combination.map(self.mlp.activation_function.dfunc)
                self.backpropagation.gradients[index] = -error * derived
                self.backpropagation.deltas_w[index] = -self.learning_rate * (self.backpropagation.gradients[index]
                                                                              @ phi_layers[index].t)
                self.backpropagation.deltas_b[index] = -self.learning_rate * self.backpropagation.gradients[index]

        # Weights corrections
        for index, (deltas, deltas_b) in enumerate(zip(reversed(self.backpropagation.deltas_w),
                                                       reversed(self.backpropagation.deltas_b))):
            self.mlp.layers_weights[index] += deltas
            self.mlp.layers_bias[index] += deltas_b


class BackpropagationHelper:

    def __init__(self, supervisor: Supervisor):
        self.supervisor = supervisor

        # vi(n)
        self.linear_combinations: List[MBase] = [None] * min(len(self.supervisor.mlp.layers_weights),
                                                             len(self.supervisor.mlp.layers_bias))
        # φ(vi(n))
        self.phi_layers: List[MBase] = [None] * len(self.linear_combinations)

        # δi(n)
        self.gradients: List[MBase] = [None] * len(self.linear_combinations)

        # Δwi(n)
        self.deltas_w: List[MBase] = [None] * len(self.linear_combinations)

        # Δbi(n)
        self.deltas_b: List[MBase] = [None] * len(self.linear_combinations)


if __name__ == '__main__':
    def known_test():
        mlp = MLP(2, [2, 1])
        sup = Supervisor(mlp)

        ws = [
            # L1
            0.8,   0.3,
            -0.6, -0.4,

            # b1
            0.7,
            -0.4,

            # L2
            0.7,
            -0.8,

            # b2
            -0.3,
        ]

        ws_iter = iter(ws)

        for weights, bias in zip(mlp.layers_weights, mlp.layers_bias):
            weights.randomize(lambda: next(ws_iter))
            bias.randomize(lambda: next(ws_iter))

        for weights, bias in zip(mlp.layers_weights, mlp.layers_bias):
            weights.print()
            bias.print()
            print('\n')

        sup.train([1, 1], [0])

    def xor_test():
        mlp = MLP(2, [2, 1])
        sup = Supervisor(mlp)

        train_set = (
            ([0, 0], [0]),
            ([0, 1], [1]),
            ([1, 0], [1]),
            ([1, 1], [0]),
        )

        import time
        import random

        start_time = time.time()
        for i in range(3000):
            random_train_set = random.sample(train_set, len(train_set))
            for input_array, target_array in random_train_set:
                sup.train(input_array, target_array)
                print(i)
        end_time = time.time()
        print('TEMPO GASTO = ' + str(end_time - start_time) + '\n')

        # buffer = [f'\n### {i}'] + [''] * len(train_set)
        buffer = [''] * len(train_set)
        for idx, (input_array, target_array) in enumerate(train_set, 0):
            output = mlp.predict(input_array)
            buffer[idx] = f"{input_array[0]} ^ {input_array[1]} = {output[0]} :: {target_array[0]}"
        print('\n'.join(buffer))

    xor_test()
