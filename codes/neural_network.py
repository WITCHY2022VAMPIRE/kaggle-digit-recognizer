import numpy as np
#from sklearn.preprocessing import StandardScaler # scaling data
from sklearn.neural_network import MLPClassifier # neural network
from datetime import datetime # Timing

def rescale(data_features, max_value, min_value = 0.0):
    return (data_features - min_value) / (max_value - min_value)

def train_neural_network(train_features, train_labels, rescale_base, **MLPClassifier_args):
#    print('Rescaling training data...')
    train_features_s = rescale(train_features, rescale_base)
    
    nn = MLPClassifier(**MLPClassifier_args) # hidden_layer_sizes = layers, batch_size = len(train_labels))#, alpha = input_alpha)#, tol = 1e-5)#, activation= "logistic")
#    print('Training neural network...')
    start_time = datetime.now()
    nn.fit(train_features_s, train_labels)
    end_time = datetime.now()

    print('Loss: ', nn.loss_, ', Train score: ', nn.score(train_features_s, train_labels), ', Iteration: ', nn.n_iter_, ', Time spent: ', str(end_time - start_time))
    return nn

def test_neural_network(trained_neural_network, test_features, test_labels, rescale_base):
    test_predict = neural_network_predict(trained_neural_network, test_features, rescale_base)

    #print('Sklearn test score: ', trained_neural_network.score(test_features_s, test_labels))
    correctness = test_predict == test_labels
    n_correct = np.sum(correctness)
    n_total = len(test_labels)
    print('Correct cases / Total cases: ', n_correct, '/', n_total)
    print('--- Accuracy: ', n_correct / n_total, " ---" )
    
    wrong_features = []
    wrong_labels = []
    wrong_predict = []
    for i in range(len(test_labels)):
        if not correctness[i]:
            wrong_features.append(test_features[i])
            wrong_labels.append(test_labels[i])
            wrong_predict.append(test_predict[i])

    return n_correct, n_total, wrong_features, wrong_labels, wrong_predict

def neural_network_predict(trained_neural_network, test_features, rescale_base):
#    print('Rescaling test data...')
    test_features_s = rescale(test_features, rescale_base)
#    print('Predicting...')
    return trained_neural_network.predict(test_features_s)

