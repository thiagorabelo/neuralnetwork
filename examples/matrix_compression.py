from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    from pylab import plot, show, figure

    matrix_data = [[0.5192, 0.7719, 0.0654, 0.4428, 0.9772],
                   [0.4677, 0.3291, 0.4459, 0.7433, 0.2053],
                   [0.1714, 0.3725, 0.9397, 0.9649, 0.2550],
                   [0.0703, 0.1238, 0.5263, 0.1601, 0.5177],
                   [0.1052, 0.8416, 0.3686, 0.4239, 0.3272]]

    train_set = tuple(
        (row, row) for row in matrix_data
    )

    mlp = MLP(5, [5, 3, 5, 5],
              ACTIVATIONS_FUNCTIONS['sigmoid'],
              ACTIVATIONS_FUNCTIONS['linear'])

    sup = Supervisor(mlp, 0.05)

    loss_list = []

    def append_loss(_, loss):
        loss_list.append(loss)

    sup.train_set(train_set, 0.01, 100000, append_loss)

    print()

    for idx, data in enumerate(matrix_data, 1):
        output = mlp.predict(data)
        data_str = f'[{", ".join("%0.4f" % d for d in data)}]'
        output_str = f'[{", ".join("%0.4f" % o for o in output)}]'
        print(f'{data_str} = {output_str}')
        plot(
            [idx] * len(data), data, 'bo',
            [idx] * len(output), output, 'ro'
        )
    figure()
    plot(list(range(len(loss_list))), loss_list, 'b')
    show()


if __name__ == '__main__':
    main()
