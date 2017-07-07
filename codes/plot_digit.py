import matplotlib.pyplot as plt
import numpy as np

def display(single_data, length_x, length_y):
    plt.style.use('grayscale')
    plt.imshow(single_data.reshape(length_x, length_y))
    plt.show()

if __name__ == '__main__':
    display(np.random.uniform(size=(100)), 10, 10)
