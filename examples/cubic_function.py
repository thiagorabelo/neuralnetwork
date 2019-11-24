import util
import math

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    # https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html;jsessionid=d7ccdb5dad86ecd28c93a845c8c8
    def func(x):
        return 2*math.pow(x, 3) - math.pow(x, 2) + 10*x - 4

    train_set = tuple(
        ([i], [func(i)])
        for i in util.divide_arange(-3.0, 3.0, 40)
    )

    import random
    from pylab import plot, show

    mlp = MLP(1, [10, 30, 1],
              ACTIVATIONS_FUNCTIONS['sigmoid'],
              ACTIVATIONS_FUNCTIONS['linear'])

    mlp.randomise_weights(lambda: random.uniform(-1.0, 1.0))

    sup = Supervisor(mlp, 0.01, True)

    sup.train_set(train_set, 0.005, 3000)
    norm = sup.normalizator

    validation = tuple(
        (norm.normalize_inputs([x]), norm.normalize_targets([func(x)]))
        for x in util.divide_arange(-4.0, 4.0, 200)
    )

    plot(
        [i[0][0] for i in validation], [i[1][0] for i in validation], 'b',
        [i[0][0] for i in validation], [mlp.predict(i[0]) for i in validation], 'r'
    )
    show()


if __name__ == '__main__':
    main()
