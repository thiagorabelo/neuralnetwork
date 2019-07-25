from PIL import Image

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def load_images_data(
        images_set=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                    'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                    'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                    'y', 'z')):
    images = [
        (i_name, Image.open(f"./examples/images/alphabet/{i_name}.png"))
        for i_name in images_set
    ]

    return [(name, list(image.getdata())) for name, image in images]


def convert_data(data):
    chars = [
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0],
        [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1],
        [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 1, 0],
        [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1],
        [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0],
        [1, 1, 0, 1],

        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]

    return [(pixels, answer, name) for (name, pixels), answer in zip(data, chars)]


def main():
    data = load_images_data()
    data = convert_data(data)

    input_layer = len(data[0][0])
    output_layer = len(data[0][1])

    train_set = tuple(
        (pixels, answer) for pixels, answer, name in data
    )

    mlp = MLP(input_layer,
              [input_layer, 8, output_layer],
              ACTIVATIONS_FUNCTIONS['sigmoid'])

    sup = Supervisor(mlp)

    sup.train_set(train_set, 0.01, 100)

    for input_array, target_array, name in data:
        output = [round(result) for result in mlp.predict(input_array)]
        print(f"{name} - {target_array} :: {output}")


if __name__ == '__main__':
    main()
