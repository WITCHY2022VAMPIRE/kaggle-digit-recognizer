def output(test_predict, file_name):
    print('### Writing test result file to', file_name, '...')
    with open(file_name, 'w') as OFILE:
        print('ImageId,Label', file = OFILE)
        for i in range(len(test_predict)):
            print(i + 1, ',', test_predict[i], file = OFILE, sep = '')
    print("Writing result complete!")
        
    
