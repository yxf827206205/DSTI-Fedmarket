import torch, copy
import numpy as np
import torch.utils.data
from lib.utils import smooth_fill_zeros
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler

def normalize_dataset(data, normalizer, column_wise=False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data[..., 0].min(axis=0, keepdims=True)
            maximum = data[..., 0].max(axis=0, keepdims=True)
        else:
            minimum = data[..., 0].min()
            maximum = data[..., 0].max()
        scaler = MinMax01Scaler(minimum, maximum)
        data[..., 0] = scaler.transform(data[..., 0])
        print('Normalize the dataset by MinMax01 Normalization')
    elif normalizer == 'max11':
        if column_wise:
            minimum = data[..., 0].min(axis=0, keepdims=True)
            maximum = data[..., 0].max(axis=0, keepdims=True)
        else:
            minimum = data[..., 0].min()
            maximum = data[..., 0].max()
        scaler = MinMax11Scaler(minimum, maximum)
        data[..., 0] = scaler.transform(data[..., 0])
        print('Normalize the dataset by MinMax11 Normalization')
    elif normalizer == 'std':
        if column_wise:
            mean = data[..., 0].mean(axis=0, keepdims=True)
            std = data[..., 0].std(axis=0, keepdims=True)
        else:
            mean = data[..., 0].mean()
            std = data[..., 0].std()
        scaler = StandardScaler(mean, std)
        data[..., 0] = scaler.transform(data[..., 0])
        print('Normalize the dataset by Standard Normalization')
    elif normalizer == 'None':
        scaler = NScaler()
        data[..., 0] = scaler.transform(data[..., 0])
        print('Does not normalize the dataset')
    elif normalizer == 'cmax':
        #column min max, to be depressed
        #note: axis must be the spatial dimension, please check !
        scaler = ColumnMinMaxScaler(data[..., 0].min(axis=0), data[..., 0].max(axis=0))
        data[..., 0] = scaler.transform(data[..., 0])
        print('Normalize the dataset by Column Min-Max Normalization')
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    '''
    :param data: [B, *]
    :param val_days:
    :param test_days:
    :param interval: interval (15, 30, 60) minutes
    :return:
    '''
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    train_data = data[:int(data_len * (1-val_ratio-test_ratio))]
    val_data = data[int(data_len * (1-val_ratio-test_ratio)):int(data_len * (1-test_ratio))]
    test_data = data[int(data_len * (1-test_ratio)):]

    # test_data = data[-int(data_len*test_ratio):]
    # val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    # train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, node_indices, batch_size, shuffle=True, drop_last=True, device='cpu'):
    # cuda = True if 'cuda' in device else False
    # TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    TensorFloat = torch.FloatTensor
    X, Y, node_indices = TensorFloat(X), TensorFloat(Y), torch.LongTensor(node_indices)
    data = torch.utils.data.TensorDataset(X, Y, node_indices)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader

def Add_Window_Horizon(data, window=3, horizon=1, single=False):
    '''
    :param data: shape [B, ...]
    :param window:
    :param horizon:
    :return: X is [B, W, ...], Y is [B, H, ...]
    '''
    length = len(data)
    end_index = length - horizon - window + 1
    X = []      #windows
    Y = []      #horizon
    index = 0
    if single:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window+horizon-1:index+window+horizon])
            index = index + 1
    else:
        while index < end_index:
            X.append(data[index:index+window])
            Y.append(data[index+window:index+window+horizon])
            index = index + 1
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def split_data(X, Y, X_i, Y_i,val_ratio, test_ratio):
    data_len = X.shape[0]
    random_indices = np.random.permutation(data_len)
    X = X[random_indices,...]
    Y = Y[random_indices,...]
    X_i = X_i[random_indices,...]
    Y_i = Y_i[random_indices,...]

    x_tra = X_i[:int(data_len * (1-val_ratio-test_ratio))]
    x_val = X_i[int(data_len * (1-val_ratio-test_ratio)):int(data_len * (1-test_ratio))]
    x_test = X_i[int(data_len * (1-test_ratio)):]

    y_tra = Y_i[:int(data_len * (1-val_ratio-test_ratio))]
    y_val = Y_i[int(data_len * (1-val_ratio-test_ratio)):int(data_len * (1-test_ratio))]
    y_test = Y[int(data_len * (1-test_ratio)):]

    return x_tra, y_tra, x_val, y_val, x_test, y_test


def get_dataloader(args, normalizer = 'std', single=False):
    #load raw st dataset
    data = load_st_dataset(args)        # T, N, 1
    #normalize st data
    # data, scaler = normalize_dataset(data, normalizer, args.column_wise)
    
    data_i = data.copy()
    data_i[..., 0] = smooth_fill_zeros(data_i[..., 0])

    X, Y = Add_Window_Horizon(data, args.lag, args.horizon, single)
    X_i, Y_i = Add_Window_Horizon(data_i, args.lag, args.horizon, single)
    x_tra, y_tra, x_val, y_val, x_test, y_test = split_data(X, Y, X_i, Y_i, args.val_ratio, args.test_ratio)
    x_tra, scaler = normalize_dataset(x_tra, normalizer, args.column_wise)
    x_val[..., 0] = scaler.transform(x_val[..., 0])
    x_test[..., 0] = scaler.transform(x_test[..., 0])
    
    args.logger.info(f'Train: {x_tra.shape}, {y_tra.shape}')
    args.logger.info(f'Val: {x_val.shape}, {y_val.shape}')
    args.logger.info(f'Test: {x_test.shape}, {y_test.shape}')

    def reshape_data(x, y):
        B, T, N, C_prime = x.shape
        
        node_indices = np.tile(np.arange(N), B)  # (B*N,)
        node_indices = node_indices.reshape(B, N)
        return x, y, node_indices
    x_tra, y_tra, train_node_indices = reshape_data(x_tra, y_tra)
    x_val, y_val, val_node_indices = reshape_data(x_val, y_val)
    x_test, y_test, test_node_indices = reshape_data(x_test, y_test)
    ##############get dataloader######################
    train_dataloader = data_loader(x_tra, y_tra, train_node_indices, args.batch_size, shuffle=True, drop_last=True, device=args.device)
    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader(x_val, y_val, val_node_indices, args.batch_size, shuffle=False, drop_last=True, device=args.device)
    test_dataloader = data_loader(x_test, y_test, test_node_indices, args.batch_size, shuffle=False, drop_last=False, device=args.device)
    return train_dataloader, val_dataloader, test_dataloader, scaler
