# 使用指南

环境安装见 [项目首页](../README.md#安装)。以下命令均从仓库根目录运行，并使用已激活的 `video` 环境。

## 输入准备

以下位置相对 `fmri_video/`，输入数据和资源需在本地准备。

| 输入 | 位置或配置方式 |
|---|---|
| SPM 四 session 一阶结果 | `data/derivatives/spm_first_level_4sessions/` |
| 视频帧 | `data/stimuli/video_frames/` |
| Atlas | `resources/atlases/`，或通过 `--atlas-dir` 指定 |
| MVPA 参考资源 | `data/derivatives/mvpa_reference/` |
| 未平滑 BOLD 时间序列 | 通过 `--timeseries-root` 指定 |

SPM/MATLAB 用于预处理和一阶建模，脚本位于 [`scripts/matlab/`](scripts/matlab/)。组水平空间统计需要 FSL `randomise`；皮层表面可视化需要 FreeSurfer/pycortex；视频语义检测需要 Torch/Torchvision。按所用分析配置对应工具。

示例中的 `/path/to/fmrinii` 需替换为实际数据路径。输入保持只读；每次新分析使用新的 run ID，输出统一放在 `fmri_video/outputs/runs/<run_id>/`。

## 分析流程

### 综合分析

先核对研究设计与输入，再在新的运行中执行扩展分析：

```bash
export FMRI_VIDEO_RUN_ID=example_revision_audit
fmri-video-revision --analysis audit

export FMRI_VIDEO_RUN_ID=example_revision_expanded
fmri-video-revision --analysis expanded --n-permutations 10000
```

设计预期为 87 名被试、1,044 张 beta 图和 522 张 contrast 图。

### 功能连接

Background FC 从未平滑 BOLD 中移除任务响应，估计被试级残差连接；Group FC 执行群体统计、NBS 和拓扑检验；sensitivity FC 评估处理选择与质量控制的影响。

```bash
FMRI_VIDEO_TIMESERIES_ROOT=/path/to/fmrinii
RUN_DIR=fmri_video/outputs/runs/example_background_fc_full

fmri-video-background-fc --stage audit \
  --timeseries-root "$FMRI_VIDEO_TIMESERIES_ROOT" --output-dir "$RUN_DIR"

fmri-video-background-fc --stage all \
  --timeseries-root "$FMRI_VIDEO_TIMESERIES_ROOT" --output-dir "$RUN_DIR"

fmri-video-group-fc --stage all --run-dir "$RUN_DIR" \
  --n-permutations 10000 --n-jobs 8

fmri-video-sensitivity-fc --stage all --variants all \
  --timeseries-root "$FMRI_VIDEO_TIMESERIES_ROOT" --run-dir "$RUN_DIR" \
  --n-permutations 10000 --n-jobs 8
```

上述阶段在同一 run 目录中衔接。修正已完成的分析时，创建新的 run 并记录原运行。

### MVPA 与单项方法

MVPA 数据准备入口：

```bash
fmri-video-original-mvpa --analysis prepare \
  --output-dir fmri_video/outputs/runs/example_original_mvpa_prepare
```

单项方法可通过模块入口运行，例如 UMAP：

```bash
python -m fmri_video_analysis.methods.embedding.umap
```

正式 MVPA 推断使用保持研究设计结构的置换。简单的 subject-wise sign flip 用于描述性诊断，不作为正式 MVPA p 值。其他方法的源码位置见 [实验索引](EXPERIMENTS.md)。

## 运行辅助工具

`fmri-video` 提供以下辅助命令，适合输入排查、计算前的小样本验证和已记录运行的重执行。参数可通过相应的 `--help` 查看：

| 命令 | 用途 |
|---|---|
| `fmri-video preflight --help` | 检查环境、输入、输出目录、磁盘和计算资源 |
| `fmri-video pilot --help` | 运行流程支持的小样本试验；Background FC 使用四被试样本 |
| `fmri-video reproduce --help` | 根据执行 manifest 在新的 run ID 下重执行分析，默认检查代码漂移 |

Group FC 和 sensitivity FC 支持 preflight，不提供 pilot。执行记录保存在 run 的 `metadata/` 下；重执行仍需相应的本地输入、环境与资源。
