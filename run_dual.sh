#!/bin/bash
# ================= 同时跑两个实验 =================
# PeMSD4SPEED: 8 clients, 12→9, cuda:1, port 50040
# PeMSD4FLOW:  5 clients, 24→12, cuda:2, port 50050
# =================================================

set -e

echo "=============================================="
echo "  Dual Experiment Runner"
echo "  1) PeMSD4SPEED  8 clients  12→9   cuda:1"
echo "  2) PeMSD4FLOW   5 clients  24→12  cuda:2"
echo "=============================================="

# 0. 清理旧进程
echo ">>> Cleaning up old processes..."
pkill -u $USER -9 -f "server.py" 2>/dev/null || true
pkill -u $USER -9 -f "client.py" 2>/dev/null || true
sleep 2

# 1. 启动 PeMSD4SPEED (8 clients, cuda:1, port 50040)
echo ""
echo ">>> [Exp 1] Launching PeMSD4SPEED..."
SKIP_TAIL=1 bash run_wandb.sh PeMSD4SPEED cuda:1 8 50040 &
EXP1_PID=$!

# 等待第一个实验的 server + clients 都启动完
# server 等 10s + 8 clients × 3s = 34s，留余量
sleep 50

# 2. 启动 PeMSD4FLOW (5 clients, cuda:2, port 50050)
# SKIP_CLEANUP=1 防止杀掉第一个实验的进程
echo ""
echo ">>> [Exp 2] Launching PeMSD4FLOW..."
SKIP_CLEANUP=1 SKIP_TAIL=1 bash run_wandb.sh PeMSD4FLOW cuda:2 5 50050 &
EXP2_PID=$!

echo ""
echo "=============================================="
echo "  Both experiments launched!"
echo "  PeMSD4SPEED PID: $EXP1_PID"
echo "  PeMSD4FLOW  PID: $EXP2_PID"
echo "  wandb project: STCIM-Fed-v2"
echo "  Logs: logs/PeMSD4SPEED_*/ and logs/PeMSD4FLOW_*/"
echo "=============================================="
echo ""
echo ">>> Waiting for both experiments to finish..."
wait
echo ">>> All experiments completed!"
