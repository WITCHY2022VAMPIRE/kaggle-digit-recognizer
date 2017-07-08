import numpy as np
#from sklearn.preprocessing import StandardScaler # scaling data
from sklearn.neural_network import MLPClassifier # neural network
from datetime import datetime # Timing

def rescale(data_features, max_value, min_value = 0.0):
    return (data_features - min_value) / (max_value - min_value)

def train_neural_network(train_features, train_labels, rescale_base, input_alpha = 0.0001):
    # Rescale data
    print('Rescaling training data...')
    train_features_s = rescale(train_features, rescale_base)
    
    nn = MLPClassifier(hidden_layer_sizes = (400, 200))# , batch_size = len(train_labels))#, alpha = input_alpha)#, tol = 1e-5)#, activation= "logistic")
    print('Training neural network...')
    start_time = datetime.now()
    nn.fit(train_features_s, train_labels)
    end_time = datetime.now()
    print('Lost: ', nn.loss_)
    print('Train score: ', nn.score(train_features_s, train_labels))
    print('Iteration: ', nn.n_iter_)
    print('Time used: ', str(end_time - start_time))
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
    
    wrong_features = []
    wrong_labels = []
    wrong_predict = []
    for i in range(len(test_labels)):
        if test_predict[i] != test_labels[i]:
            wrong_features.append(test_features[i])
            wrong_labels.append(test_labels[i])
            wrong_predict.append(test_predict[i])

    return wrong_features, wrong_labels, wrong_predict
    
