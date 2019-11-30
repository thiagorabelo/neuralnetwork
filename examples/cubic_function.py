import util
import math

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    from pylab import plot, show

    # https://www.mathworks.com/help/deeplearning/ug/improve-neural-network-generalization-and-avoid-overfitting.html;jsessionid=d7ccdb5dad86ecd28c93a845c8c8
    def func(x):
        return 2*math.pow(x, 3) - math.pow(x, 2) + 10*x - 4

    train_set = tuple(
        ([i], [func(i)])
        for i in util.divide_arange(-9.0, 9.0, 100)
    )

    scaler = util.TrainSetScaler(train_set, -1.0, 1.0)
    train_set = scaler.scale_train_set(train_set)

    mlp = MLP(1, [10, 30, 20, 10, 1],
              ACTIVATIONS_FUNCTIONS['tanh'],
              ACTIVATIONS_FUNCTIONS['linear'])

    sup = Supervisor(mlp, 0.0003)

    sup.train_set(train_set, 0.0001, 5000)

    validation = tuple(
        ([x], [func(x)])
        for x in util.divide_arange(-12.0, 12.0, 200)
    )

    plot(
        [i[0][0] for i in validation], [i[1][0] for i in validation], 'b',
        [i[0][0] for i in validation], [scaler.reverse_target(mlp.predict(scaler.scale_input(i[0])))
                                        for i in validation], 'r'
    )
    show()


if __name__ == '__main__':
    main()
