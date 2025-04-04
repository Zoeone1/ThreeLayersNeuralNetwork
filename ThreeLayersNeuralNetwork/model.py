#CIFAR-10 
import numpy as np
import json

#确保结果可复现
np.random.seed(0)

# 保存和读取模型参数
def save_model(model, model_name):
    for k,v in model.items():
        model[k] = v.tolist()
    with open(f'{model_name}.json', 'w') as f:
        json.dump(model, f)

def load_model(model_name):
    with open(f'{model_name}.json', 'r') as f:
        model = json.load(f)
    for k,v in model.items():
        model[k] = np.array(v)
    return model

def relu(x):
    return np.maximum(0, x)

def relu1(x):
    return np.maximum(0.2*x, x)

def sigmoid(z):
    return 1/(1+np.exp(z))

def sigmoid_derivation(a):
    return a*(1-a)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

def loss(y, y_predicted, model, beta): #beta 正则化强度
    L2_Reg = (beta/2)*(np.sum(np.square(model['w1'])) + np.sum(np.square(model['w2'])) + np.sum(np.square(model['w3'])))
    cross_loss = np.mean(-np.sum(y * np.log(y_predicted), axis=1)) + L2_Reg
    return cross_loss

def accuracy(y, y_predicted):
    return np.mean(np.argmax(y_predicted, axis=1) == np.argmax(y, axis=1))

def init_model(input_dim, hidden_dim, hidden_dim2, output_dim):  
    w1 = np.random.randn(input_dim, hidden_dim)/np.sqrt(input_dim)
    b1 = np.zeros(hidden_dim)
    w2 = np.random.randn(hidden_dim, hidden_dim2)/np.sqrt(hidden_dim)
    b2 = np.zeros(hidden_dim2)
    w3 = np.random.randn(hidden_dim2, output_dim)/np.sqrt(hidden_dim2)
    b3 = np.zeros(output_dim)
    model = {'w1':w1,'b1':b1, 'w2':w2,'b2':b2, 'w3':w3,'b3':b3}
    return model

def forward_propagation(model, X, activation = 'relu'):
    w1, b1, w2, b2, w3, b3 = model['w1'], model['b1'], model['w2'], model['b2'], model['w3'], model['b3']
    
    z1 = np.dot(X, w1) + b1
    if activation == 'relu':
        X1 = relu(z1)
    elif activation == 'relu1':
        X1 = relu1(z1)
    elif activation == 'sigmoid':
        X1 = sigmoid(z1)
    else:
        raise ValueError("activation function is unsupported")
    
    z2 = np.dot(X1, w2) + b2
    if activation == 'relu':
        X2 = relu(z2)
    elif activation == 'relu1':
        X2 = relu1(z2)
    elif activation == 'sigmoid':
        X2 = sigmoid(z2)
    else:
        raise ValueError("activation function is unsupported")
    
    z3 = np.dot(X2, w3) + b3
    y_predicted = softmax(z3)
    cache = {'X1':X1, 'X2':X2, 'z1':z1, 'z2':z2}

    return y_predicted, cache

def backward_propagation(model, cache, X, y, y_predicted, beta, activation = 'relu'):
    w1, w2, w3 = model['w1'], model['w2'], model['w3']
    X1, X2, z1, z2 = cache['X1'], cache['X2'], cache['z1'], cache['z2']

    dz3 = y_predicted - y
    dw3 = np.dot(X2.T, dz3) + beta * w3
    db3 = np.sum(dz3, axis=0)

    dX2 = np.dot(dz3, w3.T)
    if activation == 'relu':
        dz2 = np.where(X2 > 0, dX2, 0)
    elif activation == 'relu1':
        dz2 = np.where(X2 > 0, dX2, 0.2 * dX2)
    elif activation == 'sigmoid':
        dz2 = sigmoid_derivation(X2) * dX2

    dw2 = np.dot(X1.T, dz2) + beta * w2
    db2 = np.sum(dz2, axis=0)

    dX1 = np.dot(dz2, w2.T)
    if activation == 'relu':
        dz1 = np.where(X1 > 0, dX1, 0)
    elif activation == 'relu1':
        dz1 = np.where(X1 > 0, dX1, 0.2 * dX1)
    elif activation == 'sigmoid':
        dz1 = sigmoid_derivation(X1) * dX1
    
    dw1 = np.dot(X.T, dz1) + beta * w1
    db1 = np.sum(dz1, axis=0)
    grads  = {'w1':dw1, 'b1':db1, 'w2':dw2, 'b2':db2, 'w3':dw3, 'b3':db3}
    return grads

def update_model(model, grads, learning_rate, beta):
    model['w1'] = model['w1'] - learning_rate * (grads['w1'] + beta * model['w1'])
    model['w2'] = model['w2'] - learning_rate * (grads['w2'] + beta * model['w2'])
    model['w3'] = model['w3'] - learning_rate * (grads['w3'] + beta * model['w3'])
    model['b1'] = model['b1'] - learning_rate * (grads['b1'] + beta * model['b1'])
    model['b2'] = model['b2'] - learning_rate * (grads['b2'] + beta * model['b2'])
    model['b3'] = model['b3'] - learning_rate * (grads['b3'] + beta * model['b3'])
    return model

def one_hot(labels, num_classes):
    one_hot_labels = np.zeros((labels.size, num_classes))
    one_hot_labels[np.arange(labels.size), labels] = 1
    return one_hot_labels
