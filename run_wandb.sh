#!/bin/bash
# ================= STCIM-Fed with wandb =================
# 支持多数据集运行，自动连接 wandb 记录实验
# 用法:
#   bash run_wandb.sh                    # 默认 METR_LA
#   bash run_wandb.sh METR_LA            # 指定数据集
#   bash run_wandb.sh PeMSD8FLOW         # PeMSD8 Flow
#   bash run_wandb.sh PeMSD4FLOW         # PeMSD4 Flow
#   bash run_wandb.sh PEMS_BAY           # PEMS-BAY
# ========================================================

set -e

# ================= 配置区域 (可修改) =================
DATASET="${1:-METR_LA}"        # 默认跑 METR_LA
DEVICE="${2:-cuda:1}"          # 默认 GPU (cuda:1 has more VRAM)
NUM_CLIENTS=${3:-5}           # 可通过第3个参数覆盖
SERVER_PORT=${4:-50030}       # 可通过第4个参数覆盖
COMPRESSION=0.10
LOCAL_EPOCHS=6
WANDB_PROJECT="STCIM-Fed-v3-le6"  # wandb 项目名 (local_epochs=6)
WANDB_ENTITY="yxf827206205-shandong-university"    # wandb team 名

# 根据数据集自动设置参数
case $DATASET in
    METR_LA)
        NODE_NUM=207
        DSP=2
        DSU=2
        WINDOW=12
        HORIZON=12
        ;;
    PEMS_BAY)
        NODE_NUM=325
        DSP=4
        DSU=4
        WINDOW=12
        HORIZON=12
        ;;
    PeMSD4FLOW|PeMSD4)
        NODE_NUM=307
        DSP=32
        DSU=32
        WINDOW=24
        HORIZON=12
        ;;
    PeMSD4SPEED)
        NODE_NUM=307
        DSP=32
        DSU=32
        WINDOW=12
        HORIZON=9
        NUM_CLIENTS=${3:-8}
        ;;
    PeMSD4OCCUPANCY)
        NODE_NUM=307
        DSP=32
        DSU=32
        WINDOW=6
        HORIZON=1
        ;;
    PeMSD7)
        NODE_NUM=883
        DSP=32
        DSU=32
        WINDOW=12
        HORIZON=12
        ;;
    PeMSD8FLOW)
        NODE_NUM=170
        DSP=32
        DSU=32
        WINDOW=6
        HORIZON=1
        ;;
    PeMSD8SPEED)
        NODE_NUM=170
        DSP=32
        DSU=32
        WINDOW=6
        HORIZON=1
        ;;
    PeMSD8OCCUPANCY)
        NODE_NUM=170
        DSP=32
        DSU=32
        WINDOW=6
        HORIZON=1
        ;;
    *)
        echo "[ERROR] Unknown dataset: $DATASET"
        echo "Supported: METR_LA, PEMS_BAY, PeMSD4FLOW, PeMSD4SPEED, PeMSD4OCCUPANCY, PeMSD7, PeMSD8FLOW, PeMSD8SPEED, PeMSD8OCCUPANCY"
        exit 1
        ;;
esac
# =====================================================

echo "=============================================="
echo "  Dataset:       $DATASET"
echo "  Nodes:         $NODE_NUM"
echo "  Clients:       $NUM_CLIENTS"
echo "  Window:        $WINDOW -> $HORIZON"
echo "  Device:        $DEVICE"
echo "  Compression:   $COMPRESSION"
echo "  wandb Project: $WANDB_PROJECT"
echo "  wandb Entity:  $WANDB_ENTITY"
echo "=============================================="

# 0. 清理旧进程 (跳过如果 SKIP_CLEANUP=1)
if [ "${SKIP_CLEANUP}" != "1" ]; then
    echo ">>> Cleaning up old processes..."
    pkill -u $USER -9 -f "server.py" 2>/dev/null || true
    pkill -u $USER -9 -f "client.py" 2>/dev/null || true
    sleep 2
fi

LOG_DIR="logs/${DATASET}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "  Log Dir:       $LOG_DIR"

# 1. 启动服务器
echo ">>> Starting Server for $DATASET..."
nohup python -u server.py \
  -n $NUM_CLIENTS \
  -p $SERVER_PORT \
  -i 127.0.0.1 \
  -N $NODE_NUM \
  -dsp $DSP \
  -dsu $DSU \
  --device $DEVICE > "$LOG_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID. Logs -> $LOG_DIR/server.log"
sleep 10

# 2. 启动客户端 (带 wandb)
for (( i=1; i<=NUM_CLIENTS; i++ ))
do
    echo ">>> Starting Client $i / $NUM_CLIENTS (wandb enabled)..."

    WANDB_RUN_NAME="${DATASET}_c${i}_$(date +%m%d_%H%M)"

    nohup python -u client.py \
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
      --local_epochs $LOCAL_EPOCHS \
      --use_wandb \
      --wandb_project "$WANDB_PROJECT" \
      --wandb_entity "$WANDB_ENTITY" \
      --wandb_run_name "$WANDB_RUN_NAME" \
      > "$LOG_DIR/client_$i.log" 2>&1 &

    sleep 3
done

echo ""
echo ">>> All $NUM_CLIENTS clients started with wandb logging!"
echo ">>> View wandb dashboard: https://wandb.ai"
if [ "${SKIP_TAIL}" != "1" ]; then
    echo ">>> Watching server log..."
    tail -f "$LOG_DIR/server.log"
else
    echo ">>> Logs at: $LOG_DIR/"
fi
