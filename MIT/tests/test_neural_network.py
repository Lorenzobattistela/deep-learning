from ..neural_network import Layer, sigmoid

def test_layer_innit():
    l = Layer(n_inputs=2, n_neurons=2)
    assert len(l.weights) == 2
    assert len(l.weights[0]) == 2
    assert len(l.biases) == 1
    assert len(l.biases[0]) == 2
    for bias in l.biases:
        for z in bias:
            assert z == 0.

def test_sigmoid():
    z = 0
    res = sigmoid(0)
    assert res == 0.5
        
def test_layer_calling():
    l = Layer(n_inputs=1, n_neurons=1)
    res = l.call([[1]])
    assert res == sigmoid(l.weights)
    
