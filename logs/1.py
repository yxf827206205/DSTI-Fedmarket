import re
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_and_plot(log_file_path):
    # 用来存储解析的数据
    epochs_data = {}
    horizon_data = []

    # 正则表达式
    # 匹配 Epoch 平均 Loss (区分 Train 和 Val)
    re_loss = re.compile(r'\*{10}(Train|Val) Epoch (\d+): Average Loss: ([\d\.]+)')
    # 匹配 Epoch 的 MAE, RMSE, MAPE
    re_metrics = re.compile(r'\*{10}(Train|Val) Epoch (\d+): MAE: ([\d\.]+) RMSE: ([\d\.]+) MAPE: ([\d\.]+)')
    # 匹配 Horizon
    re_horizon = re.compile(r'Horizon (\d+), MAE: ([\d\.]+), RMSE: ([\d\.]+), MAPE: ([\d\.]+)%')

    # 读取文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析 Loss
            match_loss = re_loss.search(line)
            if match_loss:
                phase, epoch, loss = match_loss.groups()
                epoch = int(epoch)
                if epoch not in epochs_data:
                    epochs_data[epoch] = {'epoch': epoch}
                epochs_data[epoch][f'{phase.lower()}_loss'] = float(loss)
                continue

            # 解析 Metrics
            match_metrics = re_metrics.search(line)
            if match_metrics:
                phase, epoch, mae, rmse, mape = match_metrics.groups()
                epoch = int(epoch)
                if epoch not in epochs_data:
                    epochs_data[epoch] = {'epoch': epoch}
                epochs_data[epoch][f'{phase.lower()}_mae'] = float(mae)
                epochs_data[epoch][f'{phase.lower()}_rmse'] = float(rmse)
                epochs_data[epoch][f'{phase.lower()}_mape'] = float(mape)
                continue

            # 解析 Horizon
            match_horizon = re_horizon.search(line)
            if match_horizon:
                horizon, mae, rmse, mape = match_horizon.groups()
                horizon_data.append({
                    'horizon': int(horizon),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape)
                })

    # 将数据转为 DataFrame 方便绘图
    df_epochs = pd.DataFrame(list(epochs_data.values())).sort_values('epoch')
    df_horizon = pd.DataFrame(horizon_data).sort_values('horizon')

    # ==========================
    # 绘图 1: 训练过程 (Epochs)
    # ==========================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training and Validation Metrics over Epochs', fontsize=16)

    # Loss 曲线
    if 'train_loss' in df_epochs.columns and 'val_loss' in df_epochs.columns:
        axes[0, 0].plot(df_epochs['epoch'], df_epochs['train_loss'], label='Train Loss', marker='o', markersize=3)
        axes[0, 0].plot(df_epochs['epoch'], df_epochs['val_loss'], label='Val Loss', marker='x', markersize=3)
        axes[0, 0].set_title('Average Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # MAE 曲线
    if 'train_mae' in df_epochs.columns and 'val_mae' in df_epochs.columns:
        axes[0, 1].plot(df_epochs['epoch'], df_epochs['train_mae'], label='Train MAE', marker='o', markersize=3)
        axes[0, 1].plot(df_epochs['epoch'], df_epochs['val_mae'], label='Val MAE', marker='x', markersize=3)
        axes[0, 1].set_title('MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # RMSE 曲线
    if 'train_rmse' in df_epochs.columns and 'val_rmse' in df_epochs.columns:
        axes[1, 0].plot(df_epochs['epoch'], df_epochs['train_rmse'], label='Train RMSE', marker='o', markersize=3)
        axes[1, 0].plot(df_epochs['epoch'], df_epochs['val_rmse'], label='Val RMSE', marker='x', markersize=3)
        axes[1, 0].set_title('RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # MAPE 曲线
    if 'train_mape' in df_epochs.columns and 'val_mape' in df_epochs.columns:
        axes[1, 1].plot(df_epochs['epoch'], df_epochs['train_mape'], label='Train MAPE', marker='o', markersize=3)
        axes[1, 1].plot(df_epochs['epoch'], df_epochs['val_mape'], label='Val MAPE', marker='x', markersize=3)
        axes[1, 1].set_title('MAPE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAPE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300)
    plt.show()

    # ==========================
    # 绘图 2: Horizon 预测指标
    # ==========================
    if not df_horizon.empty:
        fig2, ax1 = plt.subplots(figsize=(10, 6))
        fig2.suptitle('Metrics by Prediction Horizon', fontsize=16)

        color1 = 'tab:red'
        ax1.set_xlabel('Horizon')
        ax1.set_ylabel('MAE / RMSE', color=color1)
        ax1.plot(df_horizon['horizon'], df_horizon['mae'], marker='o', color='crimson', label='MAE')
        ax1.plot(df_horizon['horizon'], df_horizon['rmse'], marker='s', color='darkorange', label='RMSE')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_xticks(df_horizon['horizon'])
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 实例化第二个共享 X 轴的 Y 轴用于绘制 MAPE (%)
        ax2 = ax1.twinx()  
        color2 = 'tab:blue'
        ax2.set_ylabel('MAPE (%)', color=color2)
        ax2.plot(df_horizon['horizon'], df_horizon['mape'], marker='^', color='royalblue', label='MAPE (%)')
        ax2.tick_params(axis='y', labelcolor=color2)

        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

        plt.tight_layout()
        plt.savefig('horizon_metrics.png', dpi=300)
        plt.show()

# 将原来的代码改成：
parse_log_and_plot('client_5.log')