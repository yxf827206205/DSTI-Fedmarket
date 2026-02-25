#!/bin/bash

# ================= 配置区域 =================
DATASET="PeMSD4FLOW"
NODE_NUM=307
NUM_CLIENTS=5
SERVER_PORT=50019   
DEVICE="cuda:0"
COMPRESSION=0.10

WINDOW=24
HORIZON=12
# ===========================================

# 0. 【重要】杀掉旧进程，防止端口 9998/9999 冲突
echo ">>> Cleaning up old processes..."
pkill -u $USER -9 -f server.py
pkill -u $USER -9 -f client.py
mkdir -p logs
rm -f logs/*.log

# 1. 启动服务器 (加上 -u)
echo ">>> Starting Server for $DATASET..."
# 注意：如果你的脚本在根目录，可能需要写 ntk/server.py，这里我按你的写法
python -u server.py \
  -n $NUM_CLIENTS \
  -p $SERVER_PORT \
  -i 127.0.0.1 \
  -N $NODE_NUM \
  -dsp 32 \
  -dsu 32 \
  --device $DEVICE > logs/server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID. Logs -> logs/server.log"
sleep 10

# 2. 启动客户端
for (( i=1; i<=NUM_CLIENTS; i++ ))
do
    echo ">>> Starting Client $i / $NUM_CLIENTS ..."
    
    # 【关键修改】加上 --local_epochs 1
    python -u client.py \
      $DATASET "FED" 0 0 0 0 0 0 0 0 0 $DEVICE \
      --cid $i \
      --fedavg \
      --num_clients $NUM_CLIENTS \
      -sip 127.0.0.1 \
      -sp $SERVER_PORT \
      -cp $((SERVER_PORT + i)) \
      --compression_rate $COMPRESSION \
      --device $DEVICE \
      --lag $WINDOW \
      --horizon $HORIZON \
      --in_steps $WINDOW \
      --out_steps $HORIZON \
      --local_epochs 4 > logs/client_$i.log 2>&1 &
    
    sleep 3
done

echo ">>> All Clients started. Now watching Server log..."
# 自动帮你查看日志，看看是不是一轮一输出
tail -f logs/server.log