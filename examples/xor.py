import time

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def main():
    mlp = MLP(2, [2, 1], ACTIVATIONS_FUNCTIONS['sigmoid'])
    sup = Supervisor(mlp)

    train_set = (
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0]),
    )

    start_time = time.time()
    sup.train_set(train_set, 0.0005, 10000)
    end_time = time.time()
    print(f"Spent time {end_time-start_time}")

    buffer = [''] * len(train_set)
    for idx, (input_array, target_array) in enumerate(train_set, 0):
        output = mlp.predict(input_array)
        buffer[idx] = f"{input_array[0]} ^ {input_array[1]} = {output[0]} :: {target_array[0]}"
    print('\n'.join(buffer))


if __name__ == '__main__':
    main()
