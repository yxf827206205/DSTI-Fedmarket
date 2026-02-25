import os
import numpy as np
import pandas as pd
import pickle

def load_st_dataset(args):
    dataset = args.dataset
    if 'PeMSD4FLOW' in dataset:
        data = np.load('./data/PeMSD4/data.npz')
        data = data['data']
        data = data[:,:,0:3]
    elif 'PeMSD4OCCUPANCY' in dataset:
        data1 = np.load('./data/PeMSD4/pems04.npz')
        data1 = data1['data']
        data1 = data1[:,:,1:2]
        data2 = np.load('./data/PeMSD4/data.npz')
        data2 = data2['data']
        data2 = data2[:,:,1:3]
        data = np.concatenate((data1, data2), axis=-1)
    elif 'PeMSD4SPEED' in dataset:
        data1 = np.load('./data/PeMSD4/pems04.npz')
        data1 = data1['data']
        data1 = data1[:,:,2:3]
        data2 = np.load('./data/PeMSD4/data.npz')
        data2 = data2['data']
        data2 = data2[:,:,1:3]
        data = np.concatenate((data1, data2), axis=-1)
    elif 'PeMSD7' in dataset:
        # [修复逻辑] PeMSD7 原始数据只有流量，需要手动添加时间特征
        df = pd.read_csv('./data/PeMSD7/data.csv')
        data = df.drop(columns='time').to_numpy(dtype=np.float64)
        
        # 扩展维度 (T, N) -> (T, N, 1)
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
            
        T, N, _ = data.shape
        steps_per_day = getattr(args, 'steps_per_day', 288) # 默认288 (5分钟间隔)
        
        # 1. 生成 Time of Day (TOD): 归一化到 [0, 1]
        tod = (np.arange(T) % steps_per_day) / steps_per_day 
        tod = np.tile(tod.reshape(T, 1, 1), (1, N, 1))
        
        # 2. 生成 Day of Week (DOW): 0-6
        dow = (np.arange(T) // steps_per_day) % 7
        dow = np.tile(dow.reshape(T, 1, 1), (1, N, 1))
        
        # 3. 拼接: (T, N, 1) -> (T, N, 3)
        data = np.concatenate([data, tod, dow], axis=-1)
        
    elif 'PeMSD8FLOW' in dataset:
        data = np.load('./data/PeMSD8/data.npz')
        data = data['data']
        data = data[:,:,0:3]
    elif 'PeMSD8OCCUPANCY' in dataset:
        data1 = np.load('./data/PeMSD8/pems08.npz')
        data1 = data1['data']
        data1 = data1[:,:,1:2]
        data2 = np.load('./data/PeMSD8/data.npz')
        data2 = data2['data']
        data2 = data2[:,:,1:3]
        data = np.concatenate((data1, data2), axis=-1)
    elif 'PeMSD8SPEED' in dataset:
        data1 = np.load('./data/PeMSD8/pems08.npz')
        data1 = data1['data']
        data1 = data1[:,:,2:3]
        data2 = np.load('./data/PeMSD8/data.npz')
        data2 = data2['data']
        data2 = data2[:,:,1:3]
        data = np.concatenate((data1, data2), axis=-1)
    elif 'METR_LA' in dataset:
        data = np.load('./data/METR_LA/data.npz')
        data = data['data']
        data = data[:,:,0:3]
    elif 'PEMS_BAY' in dataset:
        data = np.load('./data/PEMS_BAY/data.npz')
        data = data['data']
        data = data[:,:,0:3]
    else:
        raise ValueError
        
    # 根据客户端配置切分节点 (T, N_subset, 3)
    data = data[:, args.nodes]
    
    # 再次检查维度，防止遗漏
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data