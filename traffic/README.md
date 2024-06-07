Trial 1
2 convolution + max pool
1 input layer with 128 nodes - activation relu
dropout - 0.5
1 output layer with 43 nodes - softmax

    training - 7ms/step
    acuracy: 0.057
    loss: 3.4841

Trial 2
2 convolution + max pool
1 input layer with 128 nodes - activation relu
1 hidden layer with 128 nodes - activation relu
dropout - 0.5
1 output layer with 43 nodes - softmax

    training - 7ms/step
    acuracy: 0.9604
    loss: 0.1654

Trial 3
2 convolution + max pool
1 input layer with 128 nodes - activation relu
dropout - 0.2
1 hidden layer with 256 nodes - activation relu
dropout - 0.2
1 output layer with 43 nodes - softmax

    training - 7ms/step
    acuracy: 0.9636
    loss: 0.1482
