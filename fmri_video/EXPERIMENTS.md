# Formal Experiment Map

该目录是正式四 session 视频观看 fMRI 项目的唯一维护入口。实验代码按“共享基础设施 → 单项方法 → 多阶段流程 → 报告与验证”组织。

## Directory tree

```text
fmri_video/
├── src/fmri_video_analysis/
│   ├── core/                 # 共享设计、路径、环境、统计、校验与 provenance
│   ├── methods/
│   │   ├── stimulus/         # 视频刺激低层/语义特征表征
│   │   ├── activation/       # 激活差异、重复测量与 ROI 筛选
│   │   ├── decoding/         # 跨视频 MVPA、三分类与分类器稳健性
│   │   ├── rsa/              # crossnobis、交叉验证距离与 searchlight RSA
│   │   ├── embedding/        # UMAP 表征及参数敏感性分析
│   │   └── connectivity/     # beta-series、background FC、NBS 与图拓扑
│   ├── pipelines/            # 正式多阶段实验入口
│   ├── reporting/            # 结果契约、统计审计、表格与论文图生成源码
│   └── visualization/        # 皮层表面等通用可视化
├── scripts/
│   ├── matlab/               # SPM 预处理和四 session 一阶模型
│   ├── run_revision.py       # 修订版多方法分析入口
│   └── run_original_mvpa.py  # 原始 MVPA 的可复现入口
├── tests/                    # 设计、统计、CLI、修复契约和 RSA 回归测试
├── pyproject.toml            # Python 包、依赖与 fmri-video 命令
└── environment.yml           # video Conda 环境
```

## Experiments and modules

| 实验部分 | 科学问题 | 主要代码 |
|---|---|---|
| SPM 预处理与一阶模型 | 将 8 个采集 run 预处理，并以正式前 4 个 session 建模 Simple、Complex、Control | `scripts/matlab/precessall.m`, `scripts/matlab/firstlevel_all_4ses.m` |
| 刺激刻画 | 量化视频帧的低层视觉与语义属性，为脑-刺激关联分析提供特征 | `methods/stimulus/characterization.py` |
| 激活分析 | 检验三条件的群体激活差异、重复测量效应和候选 ROI | `methods/activation/` |
| MVPA 解码 | 检验条件信息能否跨视频泛化，并比较三分类网络和分类器 | `methods/decoding/`, `pipelines/original_mvpa.py` |
| RSA | 用交叉验证距离、crossnobis 和 searchlight 检验表征几何 | `methods/rsa/` |
| 低维表征 | 用 UMAP 描述条件/被试表征结构，并检查参数敏感性 | `methods/embedding/` |
| Beta-series 与时序拓扑 | 从一阶 beta 或时序数据估计连接及拓扑特征；主要用于与正式 FC 路线互证 | `methods/connectivity/beta_series.py`, `timeseries_topology.py` |
| 正式 background FC | 对未平滑四 run BOLD 去除任务响应后估计残差连接，生成被试级 FC | `methods/connectivity/background_fc.py`, `pipelines/background_fc.py` |
| 群体 FC 推断 | 对正式被试级 FC 进行组水平统计、NBS 和网络拓扑检验 | `methods/connectivity/group_fc.py`, `nbs.py`, `pipelines/group_fc.py` |
| FC 稳健性 | 改变提取/切块/QC 等设定，复核连接、可靠性、NBS 与拓扑结论 | `methods/connectivity/sensitivity_fc.py`, `pipelines/sensitivity_fc.py` |
| 刺激—连接关系 | 将 block 刺激特征与连接变化关联，并处理复合 atlas 映射 | `methods/connectivity/block_stimulus.py`, `composite_atlas.py` |
| 修订版综合分析 | 串联审计、扩展分析和修订所需的多方法结果 | `pipelines/revision.py`, `scripts/run_revision.py` |
| 结果审计与交付 | 校验布局、设计、运动、统计和修复契约，生成正式表格/图；生成物本身不跟踪 | `reporting/` |

## Tracking boundary

Git 跟踪上述源码、测试、`USAGE.md`、本文件及环境/打包配置。以下内容只保留在本地：`data/`、`resources/`、`outputs/`、`provenance/` 和 `docs/` 中的论文与报告。旧 AAL3 notebook 路线已迁到 `/cy/data/fmri_video/archive/legacy_aal3/`，不再参与正式实验或回归测试。
