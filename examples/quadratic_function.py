import util
import math

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    from pylab import plot, show

    def func(x):
        return math.pow(x, 2.0) - 10.0 * x + 21

    # Variando do x' até x'' (-19 -> 29), dividido em 50 partes
    train_set = tuple(
        ([i], [func(i)])
        for i in util.divide_arange(-19.0, 29.0, 50)
    )

    scaler = util.TrainSetScaler(train_set, -1.0, 1.0)
    train_set = scaler.scale_train_set(train_set)

    mlp = MLP(1, [10, 30, 20, 10, 1],
              ACTIVATIONS_FUNCTIONS['tanh'],
              ACTIVATIONS_FUNCTIONS['custom'])

    sup = Supervisor(mlp, 0.001)

    sup.train_set(train_set, 0.0001, 5000)

    validation = tuple(
        ([x], [func(x)])
        for x in util.divide_arange(-19.0, 29.0, 200)
    )

    plot(
        [i[0][0] for i in validation], [i[1][0] for i in validation], 'b',
        [i[0][0] for i in validation], [scaler.reverse_target(mlp.predict(scaler.scale_input(i[0])))
                                        for i in validation], 'r'
    )
    show()


if __name__ == '__main__':
    main()
