# ==============================================================================
# GBDT Benchmark 镜像 (KML 平台用)
#
# 构建后包含：
#   - Conda env: gbdt (Python 3.10)
#   - XGBoost / LightGBM / CatBoost / scikit-learn / pandas / numpy
#   - TabPFN (含 PyTorch 2.x，体积较大)
#   - benchmark_gbdt.py（从 git 仓库 clone 到 /code）
#
# 代理策略：
#   - 国内镜像（清华 pypi/anaconda、ubuntu 阿里云源）不走代理，直连更快
#   - 海外资源（如 GitHub、PyPI 国际）走 KML 海外代理 oversea-squid4.sgp.txyun:11080
#   - 按 KML 推荐：在需要代理的 RUN 里用 `export http_proxy=... https_proxy=...`
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

# 代理地址（KML 海外代理，按需在 RUN 里 export 使用）
ARG HTTP_PROXY=http://oversea-squid4.sgp.txyun:11080
ARG HTTPS_PROXY=http://oversea-squid4.sgp.txyun:11080

# ---------- 1) 系统依赖 ----------
# 先换阿里云 apt 源（国内直连，不走代理），再装包
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
        /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.aliyun.com/ubuntu|g' \
        /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        wget bzip2 ca-certificates git curl vim openssh-client && \
    rm -rf /var/lib/apt/lists/*

# ---------- 2) Miniconda（走清华镜像，国内直连） ----------
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# 配置 conda 使用清华镜像
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --set show_channel_urls yes

# ---------- 3) Python 3.10 env ----------
RUN conda create -y -n gbdt python=3.10 && conda clean -afy

# ---------- 4) GBDT 三大框架 + 依赖（清华 pypi 镜像，国内直连） ----------
SHELL ["/bin/bash", "-lc"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate gbdt && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip && \
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        xgboost lightgbm catboost scikit-learn pandas numpy scipy

# ---------- 5) TabPFN（含 PyTorch ~500MB，清华镜像） ----------
# 若清华镜像没同步到某个包，回退到海外代理 + PyPI 国际源
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate gbdt && \
    (pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tabpfn || \
     (export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY && \
      pip install tabpfn))

# ---------- 6) 运行期不设代理（如需访问海外可在容器内再 export） ----------
ENV http_proxy="" https_proxy="" no_proxy=""

# ---------- 7) 自动激活 conda env ----------
RUN echo 'source /opt/conda/etc/profile.d/conda.sh && conda activate gbdt' \
        >> /etc/bash.bashrc

# ---------- 8) 代码已被 KML clone 到 /code ----------
WORKDIR /code

CMD ["bash"]
