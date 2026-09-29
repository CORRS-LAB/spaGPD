# spaGPD 复现流程

本文档说明如何使用仓库中提供的 `Dataset1_simulated` 示例数据复现一次 spaGPD 运行结果。

## 1. 安装环境

```bash
cd spaGPD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

如果不需要运行测试，也可以只安装主程序：

```bash
python -m pip install -e .
```

## 2. 查看示例数据

示例数据保存在：

```text
data/Dataset1_simulated/processed/
```

该目录包含以下文件：

- `sc_reference.h5ad`：scRNA-seq reference 表达矩阵和细胞类型标签
- `simulated_ST_with_spatial.h5ad`：模拟 ST 表达矩阵和空间坐标
- `cell_types.txt`：细胞类型顺序
- `spatial_coordinates.csv`：spot 空间坐标
- `true_proportions.csv`：模拟数据的真实 spot-by-cell-type proportion matrix

## 3. 运行 quick test

quick test 会减少训练轮数，用于检查环境、数据和代码能否完整跑通：

```bash
python -m spagpd \
  --quick \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated_quick \
  --method-name spaGPD
```

注意：`--quick` 模式下的数值结果只用于 smoke test，不代表论文最终参数下的完整结果。

## 4. 运行最终参数示例

使用论文最终模型参数运行：

```bash
python -m spagpd \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated \
  --method-name spaGPD
```

该命令会使用固定的最终参数训练 spaGPD，包括：

- 50 维 GCN 图结构表示
- 30 维 PCA 表达特征
- 80 维联合特征
- 基于 scRNA-seq 细胞标签构建的 pseudo-proportion supervision
- 神经网络反卷积模块
- marker gene-guided NNLS calibration，marker weight 为 0.3

## 5. 查看输出结果

运行完成后，输出目录中会生成以下文件：

- `spaGPD_prop.csv`：最终 spot-by-cell-type 预测比例矩阵
- `spaGPD_metrics.csv`：RMSE、PCC、SSIM 和 JSD 指标
- `spaGPD_run_diagnostics.csv`：运行参数和诊断信息
- `spaGPD_module_record.json`：模型模块记录和 marker gene 信息

仓库中已经提供了一次 quick test 的输出结果：

```text
results/Dataset1_simulated_quick_check/
```

其中 `spaGPD_metrics.csv` 可用于确认示例数据能够正常复现出评价指标。

## 6. 运行测试

```bash
python -m pytest spagpd/tests/test_smoke.py -v
```

测试脚本会在示例数据上以 quick mode 运行 spaGPD，并检查预测矩阵、评价指标、诊断文件和模块记录是否正常生成。
