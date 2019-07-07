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


ACTIVATIONS_FUNCTIONS = {
    'sigmoid': ActivationFunction(
        lambda val: 1.0 / (1.0 + math.exp(val)),
        lambda val: val * (1.0 - val)
    ),

    'tanh': ActivationFunction(
        math.tanh,
        lambda val: 1 - (val * val)
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

    def compute_layer(self, input_matrix: MBase, weights: MBase, bias, last_layer: bool):
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
        matrix = weights * input_matrix
        matrix += bias

        if last_layer and not self.pure_linear_output:
            matrix.map(self.activation_function)

        return matrix

    def walk_layers(self) -> Iterable[Tuple[MBase, MBase, int, bool]]:
        last_layer_index = len(self.layers_weights) - 1

        for index, (weights, bias) in enumerate(zip(self.layers_weights, self.layers_bias)):
            yield weights, bias, index, index == last_layer_index

    def predict(self, input_array: List[Number]) -> List[Number]:
        matrix = Matrix.from_array(self.n_inputs, 1, input_array)

        for weights, bias, index, is_last_layer in self.walk_layers():
            matrix = self.compute_layer(matrix, weights, bias, is_last_layer)

        return list(matrix)
