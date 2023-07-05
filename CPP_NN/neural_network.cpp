#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}


class Layer {
    public:
        vector<vector<double>> weights;
        vector<double> biases;
        vector<double> outputs;
        
        Layer(int n_inputs, int n_neurons) {
            for (int i = 0; i < n_neurons; i++) {
                vector<double> neuron_weights;
                for (int j = 0; j < n_inputs; j++) {
                    neuron_weights.push_back(0.0);
                }
                weights.push_back(neuron_weights);
                biases.push_back(0.0);
                outputs.push_back(0.0);
            }
        }

        void call(vector<vector<double>> inputs) {
            for (int i = 0; i < weights.size(); i++) {
                double output = 0.0;
                for (int j = 0; j < weights[i].size(); j++) {
                    output += inputs[0][j] * weights[i][j];
                }
                output += biases[i];
                // apply sigmoid function
                output = sigmoid(output);
                outputs[i] = output;
            }
        }
};
