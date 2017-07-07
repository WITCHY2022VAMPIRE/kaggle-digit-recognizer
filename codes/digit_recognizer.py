import read_data as rd
import plot_digit as pd

length_x = 28
length_y = 28

def main():
    print('Kaggle Digit Recognizer')

    print('Reading data')
    (train_features, train_labels, test_features, test_labels) = rd.read()

    print(train_features.shape)
    print(train_labels.shape)
    print(train_features[1,])
    print(train_labels[1])
    
    pd.display(train_features[1], length_x, length_y)
    

if __name__ == '__main__':
    main()
