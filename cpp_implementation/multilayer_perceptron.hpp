#include <algorithm>
#include <cmath>
#include <functional>


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
