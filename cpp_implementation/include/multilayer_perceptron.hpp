#ifndef MULTILAYER_PERCEPTRON
#define MULTILAYER_PERCEPTRON


#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <tuple>
#include <vector>

#include "matrix.hpp"


double sigmoid_(double val)
{
    return 1.0 / (1.0 + std::exp(-std::clamp(val, -40.0, 40.0)));
}


double dsigmoid_(double val)
{
    double sig = sigmoid_(val);
    return sig * (1.0 - sig);
}


double tanh_(double val)
{
    return std::tanh(val);
}


double dtanh_(double val)
{
    double tanh = std::tanh(val);
    return 1.0 - std::pow(tanh, 2.0);
}


double relu_(double val)
{
    return std::max(0.0, val);
}


double drelu_(double val)
{
    return val > 0.0 ? 1.0 : 0.0;
}


double sin_(double val)
{
    return std::sin(val);
}


double cos_(double val)
{
    return std::cos(val);
}


double linear_(double val)
{
    return val;
}


double dlinear_(double val)
{
    return 1.0;
}


struct ActivationFunction
{
    std::function<double(double)> func;
    std::function<double(double)> derived_func;
};


ActivationFunction SIGMOID{ sigmoid_, dsigmoid_ };
ActivationFunction TANH{ tanh_, dtanh_ };
ActivationFunction RELU{ relu_, drelu_ };
ActivationFunction SIN{ sin_, cos_ };
ActivationFunction LINEAR{ linear_, dlinear_ };


const double _default_initial_weights_range[2] = {-1.0, 1.0};


class MLP
{
    public:
        MLP(
            size_t n_inputs,
            std::vector<size_t> layers,
            ActivationFunction& ac,
            ActivationFunction& ac_output,
            // std::vector<double> initial_weights_range = {-1.0, 1.0}
            double initial_weights_range_begin = -1.0,
            double initial_weights_range_end = 1.0
        ) : m_n_inputs{n_inputs}, m_n_outputs{layers.back()}, m_ac{ac}, m_ac_output{ac_output}
        {
            std::vector<size_t> weight_list = {m_n_inputs};
            weight_list.insert(weight_list.end(), layers.begin(), layers.end());

            size_t index = 1;
            for (auto layer = std::next(weight_list.begin()); layer != weight_list.end(); ++layer) {
                Matrix<double> weights{*layer, weight_list[index - 1]};
                Matrix<double> bias{*layer, 1};

                std::default_random_engine gen;
                std::uniform_real_distribution<double> dist(initial_weights_range_begin, initial_weights_range_end);

                auto randomizer = [&dist, &gen]()  { return dist(gen); };

                weights.randomize(randomizer);
                bias.randomize(randomizer);

                m_layers_weights.push_back(weights);
                m_layers_bias.push_back(bias);

                index += 1;
            }
        }

        Matrix<double> linear_combination(Matrix<double>& input_matrix,
                                          Matrix<double>& weights,
                                          Matrix<double>& bias)
        {
            Matrix<double> matrix = (weights * input_matrix);
            return matrix + bias;
        }

        Matrix<double>& apply_activation_function(Matrix<double>& matrix, bool last_layer)
        {
            std::function<double(double)> af = last_layer ? m_ac.func : m_ac_output.func;

            for (size_t row = 0; row < matrix.rows(); row++) {
                for (size_t col = 0; col < matrix.cols(); col++) {
                    matrix.set(row, col) = af(matrix.get(row, col));
                }
            }

            return matrix;
        }

        struct walk_layers
        {
            size_t m_index;
            MLP& m_mlp;

            walk_layers(MLP& mlp) : m_index{0}, m_mlp{mlp}
            {
            }

            std::tuple<size_t, bool> next()
            {
                bool last_layer_index = m_mlp.m_layers_weights.size() - 1;
                size_t index = m_index;
                m_index += 1;
                return std::make_tuple(index, index == last_layer_index);
            }

            bool has_next()
            {
                return m_index < m_mlp.m_layers_weights.size();
            }
        };

        std::vector<double> predict(double input_array[], size_t length, bool take_ownership)
        {
            Matrix<double> matrix{m_n_inputs, 1, input_array, take_ownership};
            walk_layers walker{*this};

            while (walker.has_next()) {
                size_t index;
                bool is_last_layer;

                std::tie(index, is_last_layer) = walker.next();
                Matrix<double>& weight = m_layers_weights[index];
                Matrix<double>& bias = m_layers_bias[index];

                matrix = linear_combination(matrix, weight, bias);
                matrix = apply_activation_function(matrix, is_last_layer);
            }

            return {}; // TODO: Termine esta parte
        }

    private:
        size_t m_n_inputs;
        size_t m_n_outputs;

        std::vector<Matrix<double>> m_layers_weights;
        std::vector<Matrix<double>> m_layers_bias;

        ActivationFunction& m_ac;
        ActivationFunction& m_ac_output;
};


#endif /* MULTILAYER_PERCEPTRON */
