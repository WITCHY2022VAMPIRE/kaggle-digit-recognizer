def output(test_predict, file_name):
    print('### Writing test result file to', file_name, '...')
    with open('../data/' + file_name, mode = 'w', buffering = 262144) as OFILE:
        print('ImageId,Label', file = OFILE)
        for i in range(len(test_predict)):
            print(i + 1, ',', test_predict[i], file = OFILE, sep = '')
    print("Writing result complete!")
        
    
