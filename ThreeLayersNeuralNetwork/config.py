class config:
    input_dim = 3072
    hidden_dim = 1024
    hidden_dim2 = 256
    output_dim = 10
    decay_rate = 0.95
    learning_rate = 0.005
    beta = 0.01
    batch_size = 32
    num_epochs = 10
    # relu or relu1(leaky relu) or sigmoid
    activation = 'relu' 