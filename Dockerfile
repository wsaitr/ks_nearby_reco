# ==============================================================================
# GBDT Benchmark 镜像 (KML 平台用)
#
# 构建后包含：
#   - Conda env: gbdt (Python 3.10)
#   - XGBoost / LightGBM / CatBoost / scikit-learn / pandas / numpy
#   - TabPFN (含 PyTorch 2.x，体积较大)
#   - benchmark_gbdt.py（从 git 仓库 clone 到 /code）
#
# 启动开发机后：
#   conda activate gbdt
#   cd /code
#   python benchmark_gbdt.py                       # 跑分类（默认 5万样本）
#   N_SAMPLES=200000 python benchmark_gbdt.py      # 调样本量
#   RUN_REGRESSION=1 python benchmark_gbdt.py      # 同时跑回归
#   ENABLE_TABPFN=1 TABPFN_API_KEY=xxx python benchmark_gbdt.py  # 启用 TabPFN
# ==============================================================================

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# 海外代理（装包阶段用，装完清除）
ENV http_proxy=http://oversea-squid4.sgp.txyun:11080 \
    https_proxy=http://oversea-squid4.sgp.txyun:11080

# ---------- 1) 系统依赖 ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-ca-certificates git curl vim openssh-client && \
    rm -rf /var/lib/apt/lists/*

# ---------- 2) Miniconda（走清华镜像，快） ----------
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# ---------- 3) Python 3.10 env ----------
RUN conda create -y -n gbdt python=3.10 && conda clean -afy

# ---------- 4) GBDT 三大框架 + 依赖（清华镜像） ----------
SHELL ["/bin/bash", "-lc"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate gbdt && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        xgboost lightgbm catboost scikit-learn pandas numpy scipy

# ---------- 5) TabPFN（含 PyTorch ~500MB） ----------
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate gbdt && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tabpfn

# ---------- 6) 清除构建期代理（运行期如需可重新设） ----------
ENV http_proxy="" https_proxy="" no_proxy=""

# ---------- 7) 自动激活 conda env ----------
RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate gbdt' \
        >> /etc/bash.bashrc

# ---------- 8) 代码已被 KML clone 到 /code ----------
WORKDIR /code

CMD ["bash"]
