import numpy as np
import matplotlib.pyplot as plt
def kalman_filter(data):
    dt = 1 
    # 状态向量的初始化，假设起始速度为0
    X = np.array([data[0, 0], data[0, 1], 0, 0]).reshape((4, 1))
    
    # 状态转移矩阵和观测矩阵的定义
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])
    
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    # 状态协方差矩阵的初始化
    P = np.eye(4)
    
    # 测量噪声和过程噪声的协方差矩阵的定义
    R = np.eye(2) * 0.1
    Q = np.eye(4) * 0.01
    
    # 存储滤波结果的数组
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        # 预测步骤
        X_ = F @ X
        P_ = F @ P @ F.T + Q
        
        # 更新步骤
        y = data[i, :].reshape((2, 1)) - H @ X_
        S = H @ P_ @ H.T + R
        K = P_ @ H.T @ np.linalg.inv(S)
        X = X_ + K @ y
        P = (np.eye(4) - K @ H) @ P_
        
        # 将滤波结果存储到数组中
        filtered_data[i, :] = X[:2].T

    return filtered_data

data = np.load('list_point.npy')[:30,:]

# 对数据进行卡尔曼滤波
filtered_data = kalman_filter(data)

# 绘制原始数据和滤波后的数据的散点图以及按照顺序连接的折线图
plt.scatter(data[:,0], data[:,1], c='r', label='Raw Data')
plt.scatter(filtered_data[:,0], filtered_data[:,1], c='b', label='Filtered Data')
plt.plot(data[:,0], data[:,1], '--', c='r', label='Raw Data Path')
plt.plot(filtered_data[:,0], filtered_data[:,1], '--', c='b', label='Filtered Data Path')
plt.legend()
plt.show()