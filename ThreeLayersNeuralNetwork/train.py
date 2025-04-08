from model import *
from tqdm import tqdm
from test import test_model

def train_model(model, X_train, y_train, X_test, y_test, learning_rate, decay_rate, beta, num_epochs, batch_size, activation='relu'):
    num_train = X_train.shape[0]
    num_batches = num_train//batch_size

    history_train_losses = []
    history_train_accuracies = []
    history_test_losses = []
    history_test_accuracies = []
    best_test_accuracy  = 0
    best_model = model

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        random_index = np.random.choice(num_train, num_train, replace=False)
        X_train_epoch = X_train[random_index]
        y_train_epoch = y_train[random_index]

        for i in range(num_batches):
            X_batch = X_train_epoch[i*batch_size: (i+1)*batch_size]
            y_batch = y_train_epoch[i*batch_size: (i+1)*batch_size]
            y_predicted, cache = forward_propagation(model, X_batch, activation)
            grads = backward_propagation(model, cache, X_batch, y_batch, y_predicted, beta, activation)
            model = update_model(model, grads, learning_rate, beta)

        y_train_predicted, _ = forward_propagation(model, X_train, activation)
        train_loss = loss(y_train, y_train_predicted, model, beta)
        train_accuracy = accuracy(y_train, y_train_predicted)
        save_model(model, "currentModel")
        test_loss, test_accuracy = test_model("currentModel", X_test, y_test, beta, activation='relu')

        if best_test_accuracy < test_accuracy:
            best_test_accuracy = test_accuracy
            best_model = model

        history_train_losses.append(train_loss)
        history_train_accuracies.append(train_accuracy)
        history_test_losses.append(test_loss)
        history_test_accuracies.append(test_accuracy)
        learning_rate *= decay_rate

        print(f'Epoch {epoch+1}/{num_epochs}: Training loss = {train_loss}, Training accuracy = {train_accuracy}, Testing loss = {test_loss}, Testing accuracy = {test_accuracy}')

    return model, best_model, history_train_losses, history_train_accuracies, history_test_losses, history_test_accuracies
