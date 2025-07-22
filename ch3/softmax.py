import numpy as np
import matplotlib.pylab as plt

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y

if __name__ == '__main__':
    x = np.arange(0, 5.0, 0.1)
    y = softmax(x)
    print(np.sum(y))
    plt.plot(x, y)
    plt.show()
