import numpy as np

def shuffle_dataset(x, t):
    """对数据集进行shuffle

    Parameters
    ----------
    x : 训练数据
    t : 教师数据

    Returns
    -------
    x, t : 进行shuffle的训练数据和教师数据
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t