import read_data as rd
import plot_digit as pd
import pca_reduction as pr
import neural_network as nn

length_x = 28
length_y = 28
rescale_base = 255
test_split = 3000
print_wrong_cases = False

#def main():
if __name__ == '__main__':
    print('Kaggle Digit Recognizer')

    ###### Reading data ######
    print('Reading data')
    (train_features, train_labels, test_features, test_labels) = rd.read()

    ###### Print shapes ######
    print(train_features.shape)
    print(train_labels.shape)
    #    print(train_features[1,])
    #    print(train_labels[1])
    print(test_features.shape)
    print(test_labels.shape)
    #pd.display(train_features[1], length_x, length_y)

    ###### Perform PCA reduction ######
#    reduced_features= pr.pca_reduction(train_features)

    ###### Training NN and get results ######
    #    for i in range(5):
    #        alpha= 0.01 * 10**(-1*i)
    #        print("alpha= %f" % alpha)
    n_train_data = len(train_labels)
    n_train = n_train_data - test_split

    test_layers=(
#        (100),
#        (100, 100),
#        (100, 100, 100),
#        (100, 100, 100, 100),
#        (400),
#        (400, 200),
#        (400, 200, 100),
#        (700),
#        (700, 700),
#        (700, 700, 700, 700)
        (100), (200), (400), (700), (800), (1200), (1600), (3200)
    )
    for tlayer in test_layers:
        print("Layers:", tlayer)
        trained_nn = nn.train_neural_network(train_features[:n_train], train_labels[:n_train], rescale_base, layers= tlayer)
        wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, train_features[n_train:n_train_data], train_labels[n_train:n_train_data], rescale_base)
        print()
#    trained_nn = nn.train_neural_network(reduced_features[:n_train], train_labels[:n_train], rescale_base)
#    wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, reduced_features[n_train:n_train_data], train_labels[n_train:n_train_data], rescale_base)

    ###### Print out wrong cases ######
#    if print_wrong_cases:
#        print('Wrong cases:')
#        for i in range(min(10, len(wrong_l))):
#            print("(i, label, predict)", i, ",", wrong_l[i], ",", wrong_p[i])
#            pd.display(wrong_f[i], length_x, length_y)

'''
if __name__ == '__main__':
    main()
'''
