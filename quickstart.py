import tensorflow as tf

print("Tensorflow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential is used for stacking layers where each layer has one input tensor
# and one output tensor. Layers are functions with a known mathematical structure
# that can be reused.

# Flatten: for ex, a layer with shape (10, 10, 10) would be falttened to (1000,)

# Dense: regular densely connected neural network layer

# Dense implements the operation: output = activation(dot(input, kernel) + bias) where activation is the element-wise activation function passed as the activation argument, kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True). These are all attributes of Dense.

# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)

predictions = model(x_train[:1]).numpy()
print("Predictions:", predictions)

# softmax function converts these logits to probabilities for each class
print(tf.nn.softmax(predictions).numpy())


# The loss function takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example. This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print("Loss function:", loss_function(y_train[:1], predictions).numpy())

model.compile(optimizer="adam", loss=loss_function, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)
