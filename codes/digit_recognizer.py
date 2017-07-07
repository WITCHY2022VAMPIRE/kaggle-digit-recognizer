import read_data as rd
import plot_digit as pd

def main():
    print('Kaggle Digit Recognizer')

    print('Reading data')
    (train_features, train_labels, test_features, test_labels) = rd.read()

    print(train_features.shape)
    print(train_labels.shape)
    print(train_features[1,])
    

if __name__ == '__main__':
    main()
