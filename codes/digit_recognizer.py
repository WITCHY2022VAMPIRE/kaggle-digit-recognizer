import read_data as rd
import plot_digit as pd
import neural_network as nn

length_x = 28
length_y = 28
rescale_base = 255
test_split = 1000

def main():
    print('Kaggle Digit Recognizer')

    print('Reading data')
    (train_features, train_labels, test_features, test_labels) = rd.read()

    print(train_features.shape)
    print(train_labels.shape)
#    print(train_features[1,])
#    print(train_labels[1])
    print(test_features.shape)
    print(test_labels.shape)
    
    #pd.display(train_features[1], length_x, length_y)

#    for i in range(5):
#        alpha= 0.01 * 10**(-1*i)
#        print("alpha= %f" % alpha)
    n_train = len(train_labels)
    trained_nn = nn.train_neural_network(train_features[:(n_train - test_split)], train_labels[:(n_train - test_split)], rescale_base, 0.0001)
    wrong_f, wrong_l, wrong_p = nn.test_neural_network(trained_nn, train_features[(n_train - test_split):n_train], train_labels[(n_train - test_split):n_train], rescale_base)

    print('Wrong cases:')
    for i in range(min(10, len(wrong_l))):
        print("(i, label, predict)", i, ",", wrong_l[i], ",", wrong_p[i])
        pd.display(wrong_f[i], length_x, length_y)

if __name__ == '__main__':
    main()
