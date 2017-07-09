import numpy as np
import read_data as rd
import plot_digit as pd
import pca_reduction as pr
import neural_network as nn
import output_results as out

length_x = 28 # width of the image
length_y = 28 # height of the image
rescale_base = 255
test_split = 3000
print_wrong_cases = False
output_predict = False

def par_tune(train_features, train_labels, reduced_features):
#    for i in range(5):
#        alpha= 0.01 * 10**(-1*i)
#        print("alpha= %f" % alpha)

    test_layers = (
        (400), (600), (800), (1000), (1200), (1400), (1600),
#        (400, 400), (800, 800), (1200, 1200), (1400, 1400), (1600, 1600),
#        (400, 400, 400), (1200, 1200, 1200), (1600, 1600, 1600),
#        (400, 400, 400, 400), (1200, 1200, 1200, 1200), (1600, 1600, 1600, 1600)
    )
    for tlayer in test_layers:
        print("********** Layers:", tlayer, "**********")
#        print("***** NONONO PCA *****")
        for i in range(5):
            NEWtrain_f, NEWtrain_l, NEWtest_f, NEWtest_l= make_train_test(train_features, train_labels, 3000)
            trained_nn = nn.train_neural_network(NEWtrain_f, NEWtrain_l, rescale_base, hidden_layer_sizes = tlayer)
            wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, NEWtest_f, NEWtest_l, rescale_base)
#        print("***** PCA *****")
#        trained_nn = nn.train_neural_network(reduced_features[:n_train], train_labels[:n_train], rescale_base)
#        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, reduced_features[n_train:n_train_data], train_labels[n_train:n_train_data], rescale_base)
        print()
    
#    for i in range(10):
#        input_tol= 10**(-1*i)
#        print("Tol:", input_tol)
#        trained_nn = nn.train_neural_network(train_features[0:n_data-test_split], train_labels[0:n_data-test_split], rescale_base, hidden_layer_sizes = (700), tol= input_tol)
#        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, train_features[n_train:n_train_data], train_labels[n_train:n_train_data], rescale_base)
#        print()

def make_train_test(train_features, train_labels, n_test):
    n_data= len(train_labels)
    combine= np.concatenate( (train_features, train_labels.reshape( (n_data, 1) )), axis=1 )
    np.random.shuffle(combine)
    return combine[0:(n_data-n_test), 0:-1], combine[0:(n_data-n_test), -1], combine[(n_data-n_test):, 0:-1], combine[(n_data-n_test):, -1]
    

#def main():
if __name__ == '__main__':
    print('=== Kaggle Digit Recognizer ===')

    ###### Reading data ######
    print('Reading data')
    (train_features, train_labels, test_features) = rd.read()

    ###### Print shapes ######
    print(train_features.shape)
    print(train_labels.shape)
    #    print(train_features[1,])
    #    print(train_labels[1])
    print(test_features.shape)
    #pd.display(train_features[1], length_x, length_y)

    ###### Perform PCA reduction ######
    reduced_features= pr.pca_reduction(train_features)
    print(reduced_features.shape)

    ###### Tune pars of NN ######
#    par_tune(train_features, train_labels, reduced_features)
    
    ###### Training NN and get results ######
    trained_nn = nn.train_neural_network(train_features, train_labels, rescale_base, hidden_layer_sizes = (1200, 1200, 1200), tol= 1e-10)

    ###### Print out wrong cases ######
    if print_wrong_cases:
        print('Wrong cases:')
        for i in range(min(10, len(wrong_l))):
            print("(i, label, predict)", i, ",", wrong_l[i], ",", wrong_p[i])
            pd.display(wrong_f[i], length_x, length_y)

    ###### Output predicts for test data ######
    if output_predict:
        print('--- Output predicts for test data ---')
        test_predict = nn.neural_network_predict(trained_nn, test_features, rescale_base)
        out.output(test_predict, 'test_predict.csv')

'''
if __name__ == '__main__':
    main()
'''
