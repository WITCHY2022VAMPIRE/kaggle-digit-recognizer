import numpy as np
#from sklearn.preprocessing import StandardScaler # scaling data
from sklearn.neural_network import MLPClassifier # neural network

def rescale(data_features, max_value, min_value = 0.0):
    return (data_features - min_value) / (max_value - min_value)

def train_neural_network(train_features, train_labels, rescale_base, input_alpha):
    # Rescale data
    print('Rescaling training data...')
    train_features_s = rescale(train_features, rescale_base)
    
    nn = MLPClassifier(hidden_layer_sizes = (100, 100), max_iter = 400, alpha= input_alpha, activation= "logistic")
    print('Training neural network...')
    nn.fit(train_features_s, train_labels)
    print('Lost: ', nn.loss_)
    print('Train score: ', nn.score(train_features_s, train_labels))
    return nn

def test_neural_network(trained_neural_network, test_features, test_labels, rescale_base):
    # Rescale data
    print('Rescaling test data...')
    test_features_s = rescale(test_features, rescale_base)

    print('Sklearn test score: ', trained_neural_network.score(test_features_s, test_labels))

    test_predict = trained_neural_network.predict(test_features_s)
    n_correct = np.sum(test_predict == test_labels)
    n_total = len(test_labels)
    print('Total test number: ', n_total)
    print('Correct test cases: ', n_correct)
    print('Accuracy: ', n_correct / n_total )
    
    wrong_features= []
    wrong_labels= []
    wrong_predict= []
    for i in range(len(test_labels)):
        if test_predict[i] != test_labels[i]:
            wrong_features.append(test_features[i])
            wrong_labels.append(test_labels[i])
            wrong_predict.append(test_predict[i])

    return wrong_features, wrong_labels, wrong_predict
    
