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
print_wrong_cases = True
output_predict = True
test_layers_combination = False
use_PCA = False
tune_parameters = False
avg_run = 3

def par_tune(train_features, train_labels):
#    for i in range(avg_run):
#        alpha= 0.01 * 10**(-1*i)
#        print("alpha= %f" % alpha)
    best_accuracy = -1.0
    
    if test_layers_combination:
        test_layers = (
            (50,), (300,), (500,), (700,),
            #(400), (600), (700), (800), (1200), (1600),
            #(400, 100), (800, 200), (1200, 300), (1600, 400),
            #(400, 100, 40), (800, 400, 200), (1600, 800, 200),
            #(400, 300, 200, 100), (400, 400, 400, 400), (800, 800, 800, 800)
        )
    else:
        test_layers = ((700,),)
    
    for tlayer in test_layers:
        print("********** Layers:", tlayer, "**********")
        n_test_tot = 0
        n_test_cor = 0
        for i in range(avg_run):
            train_f_shuff, train_l_shuff, test_f_shuff, test_l_shuff= make_train_test(train_features, train_labels, test_split)
            
            trained_nn = nn.train_neural_network(train_f_shuff, train_l_shuff, rescale_base, hidden_layer_sizes = tlayer, tol = 2e-5)
            n_cor_i, n_tot_i, wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, test_f_shuff, test_l_shuff, rescale_base)

            n_test_cor += n_cor_i
            n_test_tot += n_tot_i

#        print("***** PCA *****")
#        trained_nn = nn.train_neural_network(reduced_features[:n_train], train_labels[:n_train], rescale_base)
#        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, reduced_features[n_train:n_train_data], train_labels[n_train:n_train_data], rescale_base)
        if best_accuracy < n_test_cor/n_test_tot:
            best_accuracy = n_test_cor/n_test_tot
            best_trained_nn = trained_nn
            best_trained_nn.name = tlayer
        print('--- Average accuracy: ', n_test_cor/n_test_tot, ' ---')
        print()

    
#    for i in range(10):
#        input_tol= 10**(-1*i)
#        print("Tol:", input_tol)
#        trained_nn = nn.train_neural_network(train_features[0:n_data-test_split], train_labels[0:n_data-test_split], rescale_base, hidden_layer_sizes = (700), tol= input_tol)
#        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, train_features[n_train:n_train_data], train_labels[n_train:n_train_data], rescale_base)
#        print()

    print('Best neural network is', best_trained_nn.name)
    return best_trained_nn

def make_train_test(features, labels, n_test):
    '''
    Shuffle the training data and slpit them into training set and test set.
    '''
    n_data= len(labels)
#    combine= np.concatenate( (features, labels.reshape( (n_data, 1) )), axis=1 )
#    np.random.shuffle(combine)
#    return combine[0:(n_data-n_test), 0:-1], combine[0:(n_data-n_test), -1], combine[(n_data-n_test):, 0:-1], combine[(n_data-n_test):, -1]
    shuffle_index = list(range(n_data))
    np.random.shuffle(shuffle_index)
    features_shuff = np.array([features[i] for i in shuffle_index])
    labels_shuff = np.array([labels[i] for i in shuffle_index])
    return features_shuff[n_test:], labels_shuff[n_test:], features_shuff[:n_test], labels_shuff[:n_test]

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
    if use_PCA:
        train_features, pca_model, n_rf = pr.pca_reduction(train_features)
        print(train_features.shape)
    
    ###### Tune pars of NN ######
    if tune_parameters:
        trained_nn = par_tune(train_features, train_labels)        
    else:
        ## Training NN and get results ######
        trained_nn = nn.train_neural_network(train_features, train_labels, rescale_base, hidden_layer_sizes = (700), tol= 1e-10)

    ###### Print out wrong cases ######
    if print_wrong_cases and not tune_parameters and not use_PCA:
        print('Wrong cases:')
        for i in range(min(10, len(wrong_l))):
            print("(i, label, predict)", i, ",", wrong_l[i], ",", wrong_p[i])
            pd.display(wrong_f[i], length_x, length_y)

    ###### Output predicts for test data ######
    if output_predict:
        print('--- Output predicts for test data ---')
        if use_PCA:
            test_features = pr.pca_transform(test_features, pca_model, n_rf)
    
        test_predict = nn.neural_network_predict(trained_nn, test_features, rescale_base)
        out.output(test_predict, '../data/test_predict.csv')
