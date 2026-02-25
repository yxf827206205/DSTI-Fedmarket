import os
import re
input=24
output=12
dataset="PeMSD4FLOW"
num_client=5
local=2
# 日志文件基础路径
base_path = f"log/in{input}_out{output}"
folder_prefix = f"{dataset}_FED_"
log_suffix = f"/{num_client}p_metis_in{input}_out{output}_adptpolu_2th_{local}locals_fedavg.log"

# 要提取的正则表达式
pattern = r"Average Horizon, MAE: ([\d\.]+), RMSE: ([\d\.]+), MAPE: ([\d\.]+)%"

# 用于存储结果的变量
mae_list = []
rmse_list = []
mape_list = []

# 遍历地址
for i in range(1, num_client+1):
    folder_path = f"{folder_prefix}{i}"
    file_path = os.path.join(base_path, folder_path + log_suffix)
    
    try:
        # 打开并读取文件
        with open(file_path, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1].strip()
            
            # 匹配正则表达式
            match = re.search(pattern, last_line)
            if match:
                mae, rmse, mape = map(float, match.groups())
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(mape)
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"处理文件时出错 {file_path}: {e}")

# 计算平均值
average_mae = sum(mae_list) / len(mae_list) if mae_list else None
average_rmse = sum(rmse_list) / len(rmse_list) if rmse_list else None
average_mape = sum(mape_list) / len(mape_list) if mape_list else None

# 输出结果
print(f"MAE 平均值: {average_mae}")
print(f"RMSE 平均值: {average_rmse}")
print(f"MAPE 平均值: {average_mape}")