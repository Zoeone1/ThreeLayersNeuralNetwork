from load_data import read_images,read_labels
from model import *
from config import config
from train import train_model
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#read data
X_train = []
y_train = []

for i in range(1, 6):
    data_batch_path = r"images\cifar-10-python\cifar-10-batches-py\data_batch_{}".format(i)
    X_batch = read_images(data_batch_path)
    X_train.append(X_batch)
    y_batch = read_labels(data_batch_path)
    y_train.append(y_batch)
X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_train = one_hot(y_train.T, 10)


test_batch_path = r"images\cifar-10-python\cifar-10-batches-py\test_batch"
X_test = read_images(test_batch_path)
y_test = read_labels(test_batch_path)
y_test = one_hot(y_test.T, 10)

#read parameter
input_dim = config.input_dim
hidden_dim = config.hidden_dim
hidden_dim2 = config.hidden_dim2
output_dim = config.output_dim
learning_rate = config.learning_rate
decay_rate = config.decay_rate
beta = config.beta
batch_size = config.batch_size
num_epochs = config.num_epochs
activation = config.activation

#train
model = init_model(input_dim, hidden_dim, hidden_dim2, output_dim)
model, best_model, history_train_losses, history_train_accuracies, history_test_losses, history_test_accuracies = train_model(model, X_train, y_train, X_test, y_test, learning_rate, decay_rate, beta, num_epochs, batch_size, activation)
save_model(best_model, "best_model_"+activation)

#plot
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
ax[0].plot(history_train_losses, label='Training loss')
ax[0].plot(history_test_losses, label='Testing loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].plot(history_train_accuracies, label='Training accuracy')
ax[1].plot(history_test_accuracies, label='Testing accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()