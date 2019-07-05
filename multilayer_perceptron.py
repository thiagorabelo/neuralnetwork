from typing import List, Union

from matrix import Matrix


Number = Union[int, float]


class MLP:

    def __init__(self, n_inputs: int, layers: List[int]):
        self.n_inputs = n_inputs
        self.layers_weights = []
        self.layers_bias = []

        weights_list = [self.n_inputs] + layers

        for index, layer in enumerate(weights_list[1:], 1):
            weights = Matrix(layer, weights_list[index - 1])
            bias = Matrix(layer, 1)

            weights.randomize()
            bias.randomize()

            self.layers_weights.append(weights)
            self.layers_bias.append(bias)

    def predict(self, input_array: List[Number]) -> List[Number]:
        matrix = Matrix.from_array(self.n_inputs, 1, input_array)

        for weights, bias in zip(self.layers_weights, self.layers_bias):
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

            matrix = weights * matrix
            matrix += bias

        return list(matrix)
