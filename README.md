# GBDT 框架对比 Benchmark (KML 镜像版)

对比 **XGBoost / LightGBM / CatBoost / TabPFN** 四个表格数据建模框架，配套 KML 平台镜像构建。

## 仓库内容

| 文件 | 说明 |
|---|---|
| `benchmark_gbdt.py` | 对比脚本：同一数据集跑 4 个框架，输出精度 + 训练/推理时间 |
| `Dockerfile` | KML 「构建新镜像」用，构建后镜像含 conda env `gbdt`（Python 3.10）+ 四大框架 |
| `deploy_kml.sh` | 备用：开发机直接部署脚本（不构建镜像时用） |
| `README.md` | 本文件 |
| `LICENSE` | GPL v3 |

## 文件

| 文件 | 说明 |
|---|---|
| `benchmark_gbdt.py` | 对比脚本：同一数据集跑 4 个框架，输出精度 + 训练/推理时间 |
| `deploy_kml.sh` | KML 开发机一键部署脚本（建 conda env + 装包 + 跑） |

## 在 KML 开发机部署

```bash
# 1) 把这两个文件传到开发机 /home/liwangsheng/ 下（scp 或 web 终端粘贴）
# 2) 在 KML 终端执行：
cd /home/liwangsheng
bash deploy_kml.sh
```

**可选环境变量：**

```bash
# 同时跑回归任务
RUN_REGRESSION=1 bash deploy_kml.sh

# 启用 TabPFN（需先到 https://ux.priorlabs.ai 注册接受 license 拿 API key）
ENABLE_TABPFN=1 TABPFN_API_KEY=xxx bash deploy_kml.sh

# 调样本量（默认 50000）
N_SAMPLES=200000 bash deploy_kml.sh
```

## 环境说明

- 部署目录：`/home/liwangsheng/gbdt_bench`（重启不丢）
- Conda env：`gbdt`（Python 3.10）
- 自动检测 GPU，有 GPU 则开 `device=cuda` / `task_type=GPU`
- KML 内置 PyPI 镜像通常更快，可改 `deploy_kml.sh` 里的 `PIP_INDEX`

## 重跑

```bash
conda activate gbdt
cd /home/liwangsheng/gbdt_bench
python benchmark_gbdt.py
```

## 4 个框架一览

| 框架 | 角色 | 协议 |
|---|---|---|
| XGBoost | 经典 GBDT 鼻祖，生态最成熟 | Apache-2.0 |
| LightGBM | 速度最快，海量数据首选 | MIT |
| CatBoost | 类别特征最强，默认参数最强 | Apache-2.0 |
| TabPFN | 2025 最新表格基础模型，in-context learning | 商业 license |

## KML 开发机注意事项

- ⚠️ 开发机重启后**系统环境会重置**，但 `/home/liwangsheng/` 和 `/share/` 持久
  - 本脚本装在 `/home/liwangsheng/gbdt_bench`，conda env 在 `~/miniconda3/envs/gbdt`，都不丢
- ⚠️ 装好环境后建议**做快照**（KML 平台 - 开发机详情 - 快照标签页）
- ⚠️ 开发机不用于跑常驻服务，跑完 benchmark 就退出
