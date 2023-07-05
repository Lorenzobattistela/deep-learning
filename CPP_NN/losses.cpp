#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class Loss {
public:
    vector<double> y_true;
    vector<double> y_pred;

    Loss(const vector<double>& y_true, const vector<double>& y_pred) {
        this->y_true = y_true;
        this->y_pred = y_pred;
    }
};

class BinaryCrossEntropy : public Loss {
public:
    BinaryCrossEntropy(const vector<double>& y_true, const vector<double>& y_pred)
        : Loss(y_true, y_pred) {}

    double call() {
        double loss = 0.0;
        for (int i = 0; i < y_true.size(); i++) {
            loss += y_true[i] * log(y_pred[i]) + (1 - y_true[i]) * log(1 - y_pred[i]);
        }
        return -loss;
    }
};

class MeanSquaredError : public Loss {
public:
    MeanSquaredError(const vector<double>& y_true, const vector<double>& y_pred)
        : Loss(y_true, y_pred) {}

    double call() {
        double loss = 0.0;
        for (int i = 0; i < y_true.size(); i++) {
            loss += pow(y_true[i] - y_pred[i], 2);
        }
        return loss / y_true.size();
    }
};

int main () {
    vector<double> y_true = {1, 0, 1, 0};
    vector<double> y_pred = {0.9, 0.2, 0.8, 0.1};

    BinaryCrossEntropy bce(y_true, y_pred);
    cout << bce.call() << endl;

    MeanSquaredError mse(y_true, y_pred);
    cout << mse.call() << endl;

    return 0;
}
