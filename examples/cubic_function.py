import util
import math

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    # https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html;jsessionid=d7ccdb5dad86ecd28c93a845c8c8
    def func(x):
        return 2*math.pow(x, 3) - math.pow(x, 2) + 10*x - 4

    train_set = tuple(
        ([i], [func(i)])
        for i in util.divide_arange(-9.0, 9.0, 100)
    )

    from pylab import plot, show

    mlp = MLP(1, [10, 30, 20, 10, 1],
              ACTIVATIONS_FUNCTIONS['tanh'],
              ACTIVATIONS_FUNCTIONS['linear'])

    sup = Supervisor(mlp, 0.0003, True)

    sup.train_set(train_set, 0.0001, 5000)
    norm = sup.normalizator

    validation = tuple(
        ([x], [func(x)])
        for x in util.divide_arange(-11.0, 11.0, 50)
    )

    plot(
        [i[0][0] for i in validation], [i[1][0] for i in validation], 'b',
        [i[0][0] for i in validation], [norm.denormalize_targets(
            mlp.predict(norm.normalize_inputs(i[0]))) for i in validation], 'r'
    )
    show()


if __name__ == '__main__':
    main()
