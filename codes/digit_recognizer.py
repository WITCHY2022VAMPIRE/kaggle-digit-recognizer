import read_data as rd

def main():
    print('Kaggle Digit Recognizer')

    print('Reading data')
    (train_data, test_data) = rd.read()
    print(train_data, test_data)

    

if __name__ == '__main__':
    main()
