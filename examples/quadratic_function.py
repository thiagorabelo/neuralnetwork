import util
import math

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    import random
    from pylab import plot, show

    def func(x):
        return math.pow(x, 2.0) - 10.0 * x + 21

    # Variando do x' até x'' (3 -> 7), dividido em 100 partes
    train_set = tuple(
        ([i], [func(i)])
        for i in util.divide_arange(1.0, 9.0, 40)
    )

    mlp = MLP(1, [10, 30, 1],
              ACTIVATIONS_FUNCTIONS['sigmoid'],
              ACTIVATIONS_FUNCTIONS['linear'])

    mlp.randomise_weights(lambda: random.uniform(-1.0, 1.0))

    sup = Supervisor(mlp, 0.01)

    sup.train_set(train_set, 0.005, 3000)

    validation = tuple(
        ([x], [func(x)])
        for x in util.divide_arange(0.0, 10.0, 200)
    )

    plot(
        [i[0][0] for i in validation], [i[1][0] for i in validation], 'b',
        [i[0][0] for i in validation], [mlp.predict(i[0]) for i in validation], 'r'
    )
    show()


if __name__ == '__main__':
    main()
