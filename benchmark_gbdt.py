"""
GBDT 框架对比 demo (KML 开发机版)
在同一份数据集上跑：
  1. XGBoost        (经典 GBDT 鼻祖)
  2. LightGBM       (速度最快)
  3. CatBoost       (类别特征最强)
  4. TabPFN         (2025 最新表格基础模型，in-context learning，需 license)

支持：
  - 自动检测 GPU，开启 gpu_hist / cuda
  - 环境变量配置样本量、是否跑回归、是否启用 TabPFN
  - 分类 + 回归双任务

环境变量:
  N_SAMPLES       默认 50000，样本量
  RUN_REGRESSION  默认 0，设为 1 同时跑回归任务
  ENABLE_TABPFN   默认 0，设为 1 启用 TabPFN（需先 export TABPFN_API_KEY）
  FORCE_CPU        默认 0，设为 1 强制 CPU 模式
"""

import os
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             mean_squared_error, r2_score)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_SAMPLES = int(os.environ.get("N_SAMPLES", "50000"))
N_FEATURES = 30
N_INFORMATIVE = 20
RUN_REGRESSION = os.environ.get("RUN_REGRESSION", "0") == "1"
ENABLE_TABPFN = os.environ.get("ENABLE_TABPFN", "0") == "1"
FORCE_CPU = os.environ.get("FORCE_CPU", "0") == "1"


def detect_gpu():
    """检测可用 GPU，返回 (has_gpu, device_str)"""
    if FORCE_CPU:
        return False, "cpu"
    # 1) nvidia-smi
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return True, "cuda"
    except Exception:
        pass
    # 2) torch
    try:
        import torch
        if torch.cuda.is_available():
            return True, "cuda"
    except Exception:
        pass
    return False, "cpu"


def make_clf_data():
    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE, n_redundant=5,
        n_classes=2, random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    rs = np.random.RandomState(RANDOM_STATE)
    df["cat1"] = rs.choice(["A", "B", "C", "D"], size=N_SAMPLES)
    df["cat2"] = rs.choice(["x", "y", "z"], size=N_SAMPLES)
    return train_test_split(df, y, test_size=0.2,
                            random_state=RANDOM_STATE, stratify=y)


def make_reg_data():
    X, y = make_regression(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE, noise=0.1,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(N_FEATURES)])
    rs = np.random.RandomState(RANDOM_STATE)
    df["cat1"] = rs.choice(["A", "B", "C", "D"], size=N_SAMPLES)
    df["cat2"] = rs.choice(["x", "y", "z"], size=N_SAMPLES)
    return train_test_split(df, y, test_size=0.2, random_state=RANDOM_STATE)


def bench(name, model, X_train, X_test, y_train, y_test, task="clf"):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_time = time.perf_counter() - t0

    if task == "clf":
        metric = accuracy_score(y_test, y_pred)
        metric_name = "accuracy"
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, proba)
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")
        return OrderedDict(name=name, accuracy=metric, auc=auc,
                           train_s=round(train_time, 3),
                           infer_s=round(infer_time, 4))
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return OrderedDict(name=name, mse=round(mse, 4), r2=round(r2, 4),
                           train_s=round(train_time, 3),
                           infer_s=round(infer_time, 4))


# ===== 分类任务 =====
def run_xgboost_clf(X_train, X_test, y_train, y_test, has_gpu):
    import xgboost as xgb
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_cols = ["cat1", "cat2"]
    Xtr = X_train.copy(); Xte = X_test.copy()
    Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols])
    Xte[cat_cols] = enc.transform(Xte[cat_cols])
    kwargs = dict(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    if has_gpu:
        kwargs["tree_method"] = "hist"
        kwargs["device"] = "cuda"
    model = xgb.XGBClassifier(**kwargs)
    return bench("XGBoost", model, Xtr, Xte, y_train, y_test, "clf")


def run_lightgbm_clf(X_train, X_test, y_train, y_test, has_gpu):
    import lightgbm as lgb
    cat_cols = ["cat1", "cat2"]
    Xtr = X_train.copy(); Xte = X_test.copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    kwargs = dict(
        n_estimators=300, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    if has_gpu:
        kwargs["device"] = "gpu"
    model = lgb.LGBMClassifier(**kwargs)
    return bench("LightGBM", model, Xtr, Xte, y_train, y_test, "clf")


def run_catboost_clf(X_train, X_test, y_train, y_test, has_gpu):
    from catboost import CatBoostClassifier
    cat_cols = ["cat1", "cat2"]
    kwargs = dict(
        iterations=300, depth=8, learning_rate=0.1,
        random_state=RANDOM_STATE, verbose=False,
        cat_features=cat_cols,
    )
    if has_gpu:
        kwargs["task_type"] = "GPU"
        kwargs["devices"] = "0"
    model = CatBoostClassifier(**kwargs)
    return bench("CatBoost", model, X_train, X_test, y_train, y_test, "clf")


def run_tabpfn_clf(X_train, X_test, y_train, y_test):
    if not ENABLE_TABPFN:
        return OrderedDict(name="TabPFN", accuracy=float("nan"), auc=float("nan"),
                           train_s=float("nan"), infer_s=float("nan"),
                           note="disabled (set ENABLE_TABPFN=1)")
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        return OrderedDict(name="TabPFN", accuracy=float("nan"), auc=float("nan"),
                           train_s=float("nan"), infer_s=float("nan"),
                           note="not installed")
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_cols = ["cat1", "cat2"]
    Xtr = X_train.copy(); Xte = X_test.copy()
    Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols])
    Xte[cat_cols] = enc.transform(Xte[cat_cols])
    # TabPFN 推荐 <= 10000 样本，超过会 warn
    try:
        model = TabPFNClassifier(ignore_pretraining_limits=True)
        return bench("TabPFN", model, Xtr, Xte, y_train, y_test, "clf")
    except Exception as e:
        return OrderedDict(name="TabPFN", accuracy=float("nan"), auc=float("nan"),
                           train_s=float("nan"), infer_s=float("nan"),
                           note=f"error: {str(e)[:60]}")


# ===== 回归任务 =====
def run_xgboost_reg(X_train, X_test, y_train, y_test, has_gpu):
    import xgboost as xgb
    from sklearn.preprocessing import OrdinalEncoder
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_cols = ["cat1", "cat2"]
    Xtr = X_train.copy(); Xte = X_test.copy()
    Xtr[cat_cols] = enc.fit_transform(Xtr[cat_cols])
    Xte[cat_cols] = enc.transform(Xte[cat_cols])
    kwargs = dict(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    if has_gpu:
        kwargs["tree_method"] = "hist"
        kwargs["device"] = "cuda"
    model = xgb.XGBRegressor(**kwargs)
    return bench("XGBoost", model, Xtr, Xte, y_train, y_test, "reg")


def run_lightgbm_reg(X_train, X_test, y_train, y_test, has_gpu):
    import lightgbm as lgb
    cat_cols = ["cat1", "cat2"]
    Xtr = X_train.copy(); Xte = X_test.copy()
    for c in cat_cols:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = Xte[c].astype("category")
    kwargs = dict(
        n_estimators=300, num_leaves=63, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=RANDOM_STATE, verbose=-1,
    )
    if has_gpu:
        kwargs["device"] = "gpu"
    model = lgb.LGBMRegressor(**kwargs)
    return bench("LightGBM", model, Xtr, Xte, y_train, y_test, "reg")


def run_catboost_reg(X_train, X_test, y_train, y_test, has_gpu):
    from catboost import CatBoostRegressor
    cat_cols = ["cat1", "cat2"]
    kwargs = dict(
        iterations=300, depth=8, learning_rate=0.1,
        random_state=RANDOM_STATE, verbose=False,
        cat_features=cat_cols,
    )
    if has_gpu:
        kwargs["task_type"] = "GPU"
        kwargs["devices"] = "0"
    model = CatBoostRegressor(**kwargs)
    return bench("CatBoost", model, X_train, X_test, y_train, y_test, "reg")


def main():
    print("=" * 80)
    print(f"GBDT Benchmark | N_SAMPLES={N_SAMPLES} | GPU detect...")
    has_gpu, device = detect_gpu()
    print(f"GPU: {'YES (' + device + ')' if has_gpu else 'NO (cpu mode)'}")
    print(f"RUN_REGRESSION={RUN_REGRESSION}  ENABLE_TABPFN={ENABLE_TABPFN}")
    print("=" * 80)

    # ===== 分类 =====
    print(f"\n[分类任务] 生成数据: {N_SAMPLES} 样本, {N_FEATURES} 数值 + 2 类别特征")
    X_train, X_test, y_train, y_test = make_clf_data()
    print(f"训练集 {X_train.shape}, 测试集 {X_test.shape}")

    results = []
    print(">>> XGBoost ...")
    results.append(run_xgboost_clf(X_train, X_test, y_train, y_test, has_gpu))
    print(">>> LightGBM ...")
    results.append(run_lightgbm_clf(X_train, X_test, y_train, y_test, has_gpu))
    print(">>> CatBoost ...")
    results.append(run_catboost_clf(X_train, X_test, y_train, y_test, has_gpu))
    print(">>> TabPFN ...")
    results.append(run_tabpfn_clf(X_train, X_test, y_train, y_test))

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("分类任务对比结果")
    print("=" * 80)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ===== 回归 =====
    if RUN_REGRESSION:
        print(f"\n[回归任务] 生成数据: {N_SAMPLES} 样本, {N_FEATURES} 数值 + 2 类别特征")
        X_train, X_test, y_train, y_test = make_reg_data()
        print(f"训练集 {X_train.shape}, 测试集 {X_test.shape}")

        results = []
        print(">>> XGBoost ...")
        results.append(run_xgboost_reg(X_train, X_test, y_train, y_test, has_gpu))
        print(">>> LightGBM ...")
        results.append(run_lightgbm_reg(X_train, X_test, y_train, y_test, has_gpu))
        print(">>> CatBoost ...")
        results.append(run_catboost_reg(X_train, X_test, y_train, y_test, has_gpu))

        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("回归任务对比结果")
        print("=" * 80)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 版本
    print("\n>>> 框架版本:")
    for pkg in ["xgboost", "lightgbm", "catboost", "tabpfn", "sklearn",
                "numpy", "pandas"]:
        try:
            mod = __import__(pkg if pkg != "sklearn" else "sklearn")
            v = getattr(mod, "__version__", "?")
            print(f"  {pkg:12s}: {v}")
        except Exception as e:
            print(f"  {pkg:12s}: <{e}>")


if __name__ == "__main__":
    main()
