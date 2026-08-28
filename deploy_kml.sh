#!/bin/bash
# ==============================================================================
# GBDT Benchmark 部署脚本 (KML 开发机版)
#
# 用途：在 KML 开发机上一键搭建环境 + 跑 XGBoost/LightGBM/CatBoost/TabPFN 对比
#
# 用法：
#   bash deploy_kml.sh           # 默认部署 + 跑分类任务
#   RUN_REGRESSION=1 bash deploy_kml.sh   # 同时跑回归
#   ENABLE_TABPFN=1 TABPFN_API_KEY=xxx bash deploy_kml.sh  # 启用 TabPFN
#
# 持久化：所有内容装在 /home/liwangsheng/gbdt_bench，重启不丢
# ==============================================================================
set -e

WORKDIR=/home/liwangsheng/gbdt_bench
ENV_NAME=gbdt
PYTHON_VER=3.10
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "===================================================="
echo "GBDT Benchmark 部署到 KML 开发机"
echo "工作目录: $WORKDIR"
echo "Conda env: $ENV_NAME (Python $PYTHON_VER)"
echo "===================================================="

# 1) 建工作目录
mkdir -p "$WORKDIR"
cd "$WORKDIR"

# 2) 复制 benchmark 脚本到工作目录（假设同目录有 benchmark_gbdt.py）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/benchmark_gbdt.py" ]; then
    cp -f "$SCRIPT_DIR/benchmark_gbdt.py" "$WORKDIR/"
    echo "[1/4] benchmark_gbdt.py 已就位"
else
    echo "[错误] 找不到 benchmark_gbdt.py，请确保它与 deploy_kml.sh 同目录"
    exit 1
fi

# 3) 初始化 conda
# shellcheck disable=SC1091
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source /usr/local/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || true

if ! command -v conda >/dev/null 2>&1; then
    echo "[错误] 找不到 conda，请确认开发机已安装 Anaconda/Miniconda"
    exit 1
fi

# 4) 创建/复用 env
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[2/4] conda env '$ENV_NAME' 已存在，复用"
else
    echo "[2/4] 创建 conda env '$ENV_NAME' (Python $PYTHON_VER)..."
    conda create -y -n "$ENV_NAME" python="$PYTHON_VER"
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip -q

# 5) 装 GBDT 框架
echo "[3/4] 安装 GBDT 框架 (xgboost/lightgbm/catboost)..."
pip install -i "$PIP_INDEX" \
    xgboost lightgbm catboost scikit-learn pandas numpy scipy

# 6) 可选：TabPFN（需 PyTorch，体积大）
if [ "${ENABLE_TABPFN:-0}" = "1" ]; then
    echo "[3b/4] 安装 TabPFN (含 PyTorch ~500MB)..."
    pip install -i "$PIP_INDEX" tabpfn
fi

echo "[4/4] 环境就绪，开始跑 benchmark"
echo ""

# 7) 跑 benchmark
cd "$WORKDIR"
python benchmark_gbdt.py 2>&1 | tee "benchmark_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "===================================================="
echo "完成！结果已保存到 $WORKDIR/benchmark_*.log"
echo "后续重跑：conda activate $ENV_NAME && cd $WORKDIR && python benchmark_gbdt.py"
echo "===================================================="
