#!/bin/bash
# NCCL超时问题诊断和训练启动脚本

echo "=== SOKE 训练启动脚本 (NCCL 调试版本) ==="
echo "时间: $(date)"

# 设置NCCL环境变量
export NCCL_TIMEOUT=7200  # 超时设置为2小时
export NCCL_DEBUG=INFO   # 启用详细日志
export NCCL_DEBUG_SUBSYS=ALL  # 所有子系统的调试信息
export NCCL_IB_DISABLE=0  # 启用InfiniBand（如果有）
export NCCL_P2P_DISABLE=0  # 启用P2P通信
export NCCL_BLOCKING_WAIT=1  # 阻塞等待模式
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch分布式详细日志
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 异步错误处理

# 额外的诊断设置
export CUDA_LAUNCH_BLOCKING=0  # 不阻塞CUDA启动（性能考虑）
export PYTHONUNBUFFERED=1  # 不缓冲Python输出

# 显示GPU信息
echo ""
echo "=== GPU 状态 ==="
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# 清理之前可能残留的进程
echo "=== 清理残留进程 ==="
pkill -f "train.py" 2>/dev/null || true
sleep 2

# 选择配置文件
CONFIG=${1:-"configs/soke.yaml"}
echo "=== 使用配置文件: $CONFIG ==="

# 启动训练
echo "=== 开始训练 ==="
echo "NCCL_TIMEOUT=$NCCL_TIMEOUT"
echo "NCCL_DEBUG=$NCCL_DEBUG"
echo ""

# 使用调试配置还是正常配置
if [[ "$2" == "debug" ]]; then
    echo "使用调试模式（较小配置）"
    CONFIG="configs/soke_debug.yaml"
fi

# 运行训练，将输出同时保存到文件
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

python train.py --cfg $CONFIG 2>&1 | tee $LOG_FILE

echo ""
echo "=== 训练结束 ==="
echo "日志保存在: $LOG_FILE"
