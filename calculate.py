import os
import re

# 1. 修改基础配置
num_client = 5  # 根据图片显示有5个client，如果是28个就改回28
log_dir = "logs"  # 图片中显示的文件夹名称

# 要提取的正则表达式 (保持不变)
pattern = r"Average Horizon, MAE: ([\d\.]+), RMSE: ([\d\.]+), MAPE: ([\d\.]+)%"

mae_list = []
rmse_list = []
mape_list = []

# 2. 遍历地址逻辑修改
for i in range(1, num_client + 1):
    # 直接拼接文件名，例如 logs/client_1.log
    file_name = f"client_{i}.log"
    file_path = os.path.join(log_dir, file_name)
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if not lines:
                continue
            last_line = lines[-1].strip()
            
            match = re.search(pattern, last_line)
            if match:
                mae, rmse, mape = map(float, match.groups())
                mae_list.append(mae)
                rmse_list.append(rmse)
                mape_list.append(mape)
            else:
                print(f"未能匹配数据 (请确认日志最后一行格式): {file_path}")
                
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"处理文件时出错 {file_path}: {e}")

# 3. 计算并输出结果 (保持不变)
if mae_list:
    average_mae = sum(mae_list) / len(mae_list)
    average_rmse = sum(rmse_list) / len(rmse_list)
    average_mape = sum(mape_list) / len(mape_list)

    print(f"--- 统计结果 ({len(mae_list)} 个客户端) ---")
    print(f"MAE 平均值: {average_mae:.4f}")
    print(f"RMSE 平均值: {average_rmse:.4f}")
    print(f"MAPE 平均值: {average_mape:.4f}%")
else:
    print("没有成功提取到任何数据，请检查日志格式。")