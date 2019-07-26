from PIL import Image

from multilayer_perceptron import MLP, Supervisor, ACTIVATIONS_FUNCTIONS


def load_images_data():
    images_set = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                  'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                  'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
                  'y', 'z']
    images = [
        (i_name, Image.open(f"./examples/images/alphabet/{i_name}.png"))
        for i_name in images_set
    ]

    return [(name, list(image.getdata())) for name, image in images]


def convert_data(data):
    chars = [
        list(int(i) for i in s) for s in
        ['00000', '00001', '00010', '00011', '00100', '00101', '00110', '00111',
         '01000', '01001', '01010', '01011', '01100', '01101', '01110', '01111',
         '10000', '10001', '10010', '10011', '10100', '10101', '10110', '10111',
         '11000', '11001', '11010', '11011', '11100', '11101', '11110', '11111']
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

    sup.train_set(train_set, 0.001, 100)

    for input_array, target_array, name in data:
        output = [round(result) for result in mlp.predict(input_array)]
        print(f"{name} - Expected={target_array} :: Predicted={output}")


if __name__ == '__main__':
    main()
