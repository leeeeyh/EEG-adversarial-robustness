import numpy as np
import scipy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承


def min_max(x):
    min = np.min(x)
    max = np.max(x)
    x = (x-min)/(max-min)
    return x

def bp_filter(data, fp1=4., fp2=40., f_sample=250):
    b, a = scipy.signal.butter(3, [2*fp1/f_sample,2*fp2/f_sample], 'bandpass')   
    filter_data = scipy.signal.filtfilt(b, a, data)
    filter_data = filter_data.astype('float32')
    return filter_data


def exponential_moving_standardize(
        data, factor_new=0.001, init_block_size=1000, eps=1e-4
):
    r"""Perform exponential moving standardization.
    Compute the exponental moving mean :math:`m_t` at time `t` as
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    Then, compute exponential moving variance :math:`v_t` at time `t` as
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{->v_t}, eps)`.
    Parameters
    ----------
    data: np.ndarray (n_channels, n_times)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization.
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: np.ndarray (n_channels, n_times)
        Standardized data.
    """
    data = data.T
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        i_time_axis = 0
        init_mean = np.mean(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_std = np.std(
            data[0:init_block_size], axis=i_time_axis, keepdims=True
        )
        init_block_standardized = (
            data[0:init_block_size] - init_mean) / np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized.T
    

class GetData(Dataset):   # 继承Dataset类
    def __init__(self, X, y): 
        # 把数据和标签拿出来        
        self.data = X
        self.label = y
        # 数据集的长度
        self.length = len(y)
        
    # 下面两个魔术方法比较好写，直接照着这个格式写就行了 
    def __getitem__(self, index): # 参数index必写
        return self.data[index], self.label[index]
    
    def __len__(self): 
        return self.length # 只需返回一个数据集长度即可

def shuffle_data(X,y,seed):
    indices = np.arange(X.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    return X, y

def standarize(data):
    print('Standarize...')
    data=data.transpose(0,2,1)
    a = data.shape[0]
    b = data.shape[1]
    data = data.reshape(a*b,-1)
    # mean_arr = np.mean(data, axis=0)
    # std_arr = np.std(data, axis=0, ddof=1)  #ddof=1为样本标准差
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    mean_arr =scaler.mean_
    var_arr = scaler.var_
    std_arr = np.sqrt(var_arr)
    data = data.reshape(a,b,-1)
    data = data.transpose(0,2,1)
    return data
