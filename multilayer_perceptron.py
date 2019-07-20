import math
import random

from functools import wraps
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


def clip(min_val: Number, max_val: Number) -> \
        Callable[[Number, Number], Callable[[Number], Number]]:
    def decorator(func: Callable[[Number], Number]) -> Callable[[Number], Number]:
        @wraps(func)
        def wrapper(value: Number) -> Number:
            return func(min_val if value < min_val else
                        max_val if value > max_val else
                        value)
        return wrapper
    return decorator


@clip(-700, 700)
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
        if not last_layer:
            matrix.imap(self.activation_function)
        else:
            matrix.imap(self.activation_func_output)

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

    def train(self, input_array: List[Number], target_array: List[Number]) -> Number:
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

            matrix = self.mlp.apply_activation_function(matrix.copy(), is_last_layer)
            if not is_last_layer:
                self.backpropagation.phi_layers[index + 1] = matrix

        error = target - matrix
        inst_average_error = (error @ error.t).get(0, 0) / 2.0
        # TODO: Calc Global Average Error

        linear_combinations = reversed(
            self.backpropagation.linear_combinations)
        phi_layers = list(reversed(self.backpropagation.phi_layers))
        layers_weights = list(reversed(self.mlp.layers_weights))

        # Backpropagation
        for index, linear_combination in enumerate(linear_combinations):
            if not index == 0:
                derivative = linear_combination.map(
                    self.mlp.activation_function.dfunc)
                mult_gradients_weights = (layers_weights[index - 1].t
                                          @ self.backpropagation.gradients[index - 1])

                self.backpropagation.gradients[index] = derivative * \
                    mult_gradients_weights

                self.backpropagation.deltas_w[index] = (-self.learning_rate *
                                                        (self.backpropagation.gradients[index]
                                                         @ phi_layers[index].t))

                self.backpropagation.deltas_b[index] = (-self.learning_rate
                                                        * self.backpropagation.gradients[index])
            else:
                derivative = linear_combination.map(
                    self.mlp.activation_func_output.dfunc)
                self.backpropagation.gradients[index] = -error * derivative

                self.backpropagation.deltas_w[index] = (-self.learning_rate *
                                                        (self.backpropagation.gradients[index]
                                                         @ phi_layers[index].t))

                self.backpropagation.deltas_b[index] = (-self.learning_rate
                                                        * self.backpropagation.gradients[index])

        # Weights corrections
        for index, (deltas, deltas_b) in enumerate(zip(reversed(self.backpropagation.deltas_w),
                                                       reversed(self.backpropagation.deltas_b))):
            self.mlp.layers_weights[index] += deltas
            self.mlp.layers_bias[index] += deltas_b

        return inst_average_error

    def train_set(self,
                  train_set: List[Tuple[Number, Number]],
                  min_error: float,
                  max_epochs: int):
        average_global_error = 0.0
        train_set_size = len(train_set)

        for epoch in range(1, max_epochs+1):
            random_train_set = random.sample(train_set, len(train_set))

            for iteration, (input_array, target_array) in enumerate(random_train_set, 1):
                try:
                    inst_average_error = self.train(input_array, target_array)
                except:
                    print(f"ERROR ON: Epoch={epoch}, Iteration={iteration}")
                    raise
                average_global_error += inst_average_error

            average_global_error /= train_set_size

            print(
                f'AvgGlobalError={round(average_global_error, 15)} - Epoch={epoch}')

            if average_global_error <= min_error:
                break


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


def arange(start, stop=None, step=1.0):
    if not step:
        raise ValueError('step can not be None or 0')

    if stop == None and start > 0.0:
        start, stop = 0.0, start
    elif stop == None and start < 0.0 and step > 0.0:
        step *= -1
        start, stop = 0.0, start
    elif start > stop and step > 0.0:
        step *= -1

    if start < stop:
        while start < stop:
            yield start
            start += step
    else:
        while start > stop:
            yield start
            start += step


if __name__ == '__main__':
    def known_test():
        mlp = MLP(2, [2, 1])
        sup = Supervisor(mlp)

        ws = [
            # L1
            0.8, 0.3,
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

        sup.train_set(train_set, 0.05, 10000)

        buffer = [''] * len(train_set)
        for idx, (input_array, target_array) in enumerate(train_set, 0):
            output = mlp.predict(input_array)
            buffer[idx] = f"{input_array[0]} ^ {input_array[1]} = {output[0]} :: {target_array[0]}"
        print('\n'.join(buffer))

    def test_function():
        # def func(x):
        #     return math.pow(x, 2.0) - 10.0*x + 21

        # Variando do x' até x'' (3 -> 4), dividido em 100 partes
        # train_set = tuple(
        #     ([3.0 + i*(4.0/20.0)], [func(3.0 + i*(4.0/20.0))])
        #     for i in range(21)
        # )

        # https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html;jsessionid=d7ccdb5dad86ecd28c93a845c8c8
        def func(x):
            return 2*math.pow(x, 3) - math.pow(x, 2) + 10*x - 4

        train_set = tuple(
            ([i], [func(i)])
            for i in arange(-2.0, 2.0, 0.1)
        )

        import random
        from pylab import plot, show

        mlp = MLP(1, [10, 20, 1],
                  ACTIVATIONS_FUNCTIONS['sigmoid'],
                  ACTIVATIONS_FUNCTIONS['linear'])

        mlp.randomise_weights(lambda: random.uniform(-1.0, 1.0))

        sup = Supervisor(mlp, 0.01)

        sup.train_set(train_set, 0.05, 1000)

        # validation = tuple(
        #     ([x], [func(x)])
        #     for x in range(-7, 18)
        # )

        # validation = tuple(
        #     ([x], [func(x)])
        #     for x in range(-10, 10)
        # )

        plot(
            [i[0][0] for i in train_set], [i[1][0] for i in train_set], 'b',
            [i[0][0] for i in train_set], [mlp.predict(i[0]) for i in train_set], 'r'
        )
        show()

    test_function()
