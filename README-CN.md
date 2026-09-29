# spaGPD

spaGPD（Spatial Graph-Guided Proportion Deconvolution）是一个用于空间转录组细胞类型比例反卷积的 Python 包。模型以 scRNA-seq 参考数据和空间转录组数据为输入，输出每个空间 spot 中不同细胞类型的预测比例。

本仓库整理的是论文中最终使用的 spaGPD 版本，包含模型代码、示例数据、复现脚本和运行说明。

## 方法概览

spaGPD 的主要流程包括：

1. 对 scRNA-seq reference 和 ST 数据进行共同基因匹配与统一预处理；
2. 基于 scRNA-seq 表达相似性构建参考细胞图，基于 ST 空间坐标构建空间 spot 图；
3. 使用共享 GCN 学习 50 维图结构表示；
4. 将图表示与 30 维 PCA 表达特征拼接为 80 维联合特征；
5. 基于 reference cell label 构建 pseudo-proportion supervision；
6. 训练神经网络反卷积头预测 spot-by-cell-type proportion matrix；
7. 使用 marker gene-guided calibration 对预测结果进行校准。

默认参数为论文最终模型设置：

| 参数 | 默认值 |
|---|---:|
| GCN embedding dimension | 50 |
| PCA dimension | 30 |
| Joint feature dimension | 80 |
| Graph training epochs | 400 |
| Deconvolution epochs | 300 |
| scRNA-seq kNN neighbors | 15 |
| ST spatial kNN neighbors | 10 |
| Pseudo-label temperature | 0.1 |
| Graph loss weight beta | 0.1 |
| Marker fusion weight | 0.3 |
| Random seed | 11 |

## 文件结构

```text
spaGPD/
  spagpd/                  spaGPD Python 包
    graph/                 图构建与邻接矩阵归一化
    embeddings/            GCN 图表示学习模块
    features/              PCA 表达特征
    pseudo/                pseudo-proportion supervision
    deconv/                神经网络反卷积模块
    marker/                marker gene-guided calibration
    eval/                  RMSE、PCC、SSIM、JSD 评价指标
    pipeline/              完整运行流程
    cli/                   命令行入口
    tests/                 smoke test
  data/
    Dataset1_simulated/    可直接运行的示例模拟数据，需要先解压缩Dataset1_simulated.zip
  examples/                示例运行脚本
  results/                 运行后输出目录
  WALKTHROUGH.md           逐步复现说明
  reproduce_example.sh     一键复现实例脚本
  pyproject.toml
  requirements.txt
```

## 安装

建议使用 Python 3.10 或以上版本。

```bash
cd spaGPD
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

如果需要运行测试，可以安装测试依赖：

```bash
python -m pip install -e ".[test]"
```

如果不使用 editable install，也可以按 `requirements.txt` 安装依赖后直接在仓库根目录运行。

## 示例数据

仓库中提供了一个可直接复现的模拟数据集：

```text
# 需要先解压缩Dataset1_simulated.zip
cd data
unzip Dataset1_simulated.zip
data/Dataset1_simulated/processed/
```

该数据集是一个 mouse cortex-like 模拟空间转录组数据，具有已知的 spot 水平真实细胞类型比例，可用于检查 spaGPD 的完整运行流程和模拟数据评价指标。数据规模如下：

| 项目 | 数量 |
|---|---:|
| Spatial spots | 490 |
| Reference cells | 10,266 |
| Shared genes | 320 |
| Cell types | 7 |

该目录包含：

| 文件 | 说明 |
|---|---|
| `sc_reference.h5ad` | scRNA-seq reference count matrix 和细胞类型标签 |
| `simulated_ST_with_spatial.h5ad` | ST 表达矩阵和空间坐标 |
| `cell_types.txt` | 细胞类型顺序 |
| `spatial_coordinates.csv` | spot 空间坐标 |
| `true_proportions.csv` | 模拟数据的真实 spot-by-cell-type proportion matrix |

对于新的模拟数据集，也需要整理成相同的 `processed/` 格式。如果只进行真实数据预测而不计算模拟数据指标，`true_proportions.csv` 不是必须文件；但如果要复现 RMSE、PCC、SSIM 和 JSD，则需要提供真实比例矩阵。

## 快速测试

快速测试会减少训练 epoch，用于检查环境、数据和代码是否能跑通：

```bash
python -m spagpd \
  --quick \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated_quick \
  --method-name spaGPD
```

## 复现实例

运行论文最终参数设置：

```bash
python -m spagpd \
  --processed-dir data/Dataset1_simulated/processed \
  --output-dir results/Dataset1_simulated \
  --method-name spaGPD
```

也可以直接运行：

```bash
bash reproduce_example.sh
```

## 输出文件

运行完成后会在输出目录生成：

| 文件 | 说明 |
|---|---|
| `spaGPD_prop.csv` | 最终 spot-by-cell-type 预测比例矩阵 |
| `spaGPD_metrics.csv` | 模拟数据上的 RMSE、PCC、SSIM、JSD 指标 |
| `spaGPD_run_diagnostics.csv` | 运行参数和诊断信息 |
| `spaGPD_module_record.json` | 模型模块记录和 marker gene 信息 |

## 运行测试

```bash
python -m pytest spagpd/tests/test_smoke.py -v
```

测试会在示例数据上运行 quick mode，并检查是否生成预测矩阵和评价指标文件。

## 结果简述

spaGPD 结合了表达特征、空间图结构、reference cell label 构建的伪比例监督，以及 marker gene 引导校准。在模拟数据中，该模型用于评估 spot 水平和 cell type 水平的比例预测准确性；在真实数据中，模型预测结果可进一步结合专家注释或病理区域标注进行空间富集区域验证。

更多运行步骤见 `WALKTHROUGH.md`。
