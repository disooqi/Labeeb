A machine learning engine designed and developed to be both easy to use and source code readable. It is a straightforward implementation of different algorithms and techniques of machine learning in Python. You can use it for small projects and/or educational purposes.

Backpropagation is implemented in boring detail such that derivative steps is taken carefully and without any implicit or hidden 
details

Natasy is Arabic word means the skilled scientist, or the clever doctor.


Natasy is intended to be easy to use and intuitive to build your neural network.
To build you network to classify the will known MNIST dataset:
1. Define your network
```python
my_NN = NeuralNetwork(n_features=400, n_classes=10)
```
2. Build up the layers as you want
```python
my_NN.add_layer(100, activation=Activation.leaky_relu, dropout_keep_prob=1)    
my_NN.add_layer(12, activation=Activation.softmax_stable, output_layer=True)
```
3. Finally, call the optimizer
```python
gd_optimizer = Optimizer(loss='multinomial_cross_entropy', method='adam') # gd-with-momentum gradient-descent rmsprop adam
gd_optimizer.minimize(nn01, epochs=100, mini_batch_size=5000, learning_rate=.1, regularization_parameter=0, dataset=mnist)
```

The following is the complete source code for the example. More examples can be found under Natasy/examples.


