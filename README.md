# lunwen — TSN 重构实验工作区

2026-10-22 的实验工作区快照:时序网络(TSN)重构方案的实现、配置、权重与审计记录。

## 目录结构

```
tsn_reconf_exp/                  主工程
  core_sim.py                     核心仿真环境
  tsn_env.py                      TSN 环境定义(Gym 风格)
  predictor_mapper.py             预测器 / 状态映射
  train_predictors.py             预测器训练
  train_rl_and_baselines.py       RL 与基线模型训练
  run_all_experiments.py          全量实验入口
  configs/
    experiment_config.yaml        实验参数
    global_flows.yaml             全局流量配置
  checkpoints/                    RL 权重 (*.pt)
  results/                        结果图与指标 json
  audit_20261021/                 审计:2026-10-21 快照
  audit_latest_20261021/          审计:最新快照
  audit_cause_20261022/           审计:性能落后归因(2026-10-22)
  audit_current_20261022/         审计:当前方案(2026-10-22)
```

## 方案与实验文档(仓库根目录)

| 文件 | 内容 |
|---|---|
| `当前方案与实验问题分析_20261022.md` | 当前方案与待解问题 |
| `方案性能落后原因分析_20261022.md` | 性能落后归因分析 |
| `实验结果与完整验证报告_20261021.md` | 实验结果与完整验证报告 |
| `实验参数设置(4).md` | 实验参数设置 |
| `实验分组与外部对比计划.md` | 实验分组与外部对比计划 |
| `2026.10.20(8).md` | 方案文档主稿 |
| `网络建模图(6).pptx` | 网络建模图 |

## 使用

```bash
pip install -r requirements.txt   # 若需要,按 configs 里的依赖补
python tsn_reconf_exp/run_all_experiments.py
```

`*.orig` 是编辑工具留下的备份副本,内容与对应主文件基本一致;`__pycache__` / `*.pyc` 已通过 `.gitignore` 排除。
