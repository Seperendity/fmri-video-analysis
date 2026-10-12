# fMRI Video Analysis

本项目研究**自然场景复杂度与视觉目标跟踪时脑活动及功能连接的关系**。研究比较被试在单车辆（Simple）和多车辆（Complex）航拍视频中跟踪指定车辆时的 fMRI 响应，并以动态无目标视频（Control）作为参照。

重点考察三个问题：复杂场景是否伴随更强的视觉与注意相关脑区活动；视觉活动模式能否跨被试和视频区分两种场景；功能连接是否呈现全脑整体变化，或特定脑区与网络之间的选择性调整。

## 分析内容

- 刺激刻画：视频帧的视觉特征与语义属性。
- 激活分析：条件对比、重复测量与 ROI 分析。
- MVPA：跨视频解码、三分类与分类器比较。
- RSA：交叉验证距离、crossnobis 与 searchlight。
- 低维表征：UMAP 及参数敏感性分析。
- 功能连接：Background FC、NBS、网络拓扑与稳健性分析。

## 安装

推荐使用 Python 3.11 和 Conda。先根据本机安装位置调整 [`environment.yml`](fmri_video/environment.yml) 中的 FSL、FreeSurfer 和 Conda 路径，然后在仓库根目录运行：

```bash
conda env create -f fmri_video/environment.yml
conda activate video
export PYTHONNOUSERSITE=1
export FMRI_VIDEO_HEADLESS=1
```

环境文件会安装本项目及 Python 依赖。已有 `video` 环境时，直接激活并运行 `python -m pip install -e './fmri_video[full]'` 即可。

## 使用

以被试级 Background FC 为例：

```bash
fmri-video-background-fc --stage all \
  --timeseries-root /path/to/fmrinii \
  --output-dir fmri_video/outputs/runs/example_background_fc
```

将 `/path/to/fmrinii` 替换为实际数据目录，并为每次新分析指定新的输出目录。输入准备和完整分析流程见 [使用指南](fmri_video/USAGE.md)。

## 项目结构

```text
fmri_video/
├── src/fmri_video_analysis/  # 分析方法、流程与图表生成
├── scripts/                 # Python 和 SPM/MATLAB 脚本
├── tests/                   # 回归测试
├── pyproject.toml           # Python 依赖与命令入口
├── environment.yml          # Conda 环境
├── USAGE.md                 # 详细使用指南
└── EXPERIMENTS.md            # 实验与模块索引
```

各实验与源码模块的对应关系见 [实验索引](fmri_video/EXPERIMENTS.md)。
