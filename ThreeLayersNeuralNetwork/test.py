from model import load_model, accuracy, forward_propagation, loss
from model import load_model, forward_propagation, accuracy

def test_model(model_name, X_test, y_test, beta, activation='relu'):

    model = load_model(model_name)
    y_test_predicted, _ = forward_propagation(model, X_test, activation)
    test_accuracy = accuracy(y_test, y_test_predicted)
    test_loss = loss(y_test, y_test_predicted, model, beta)
    
    return test_loss, test_accuracy
