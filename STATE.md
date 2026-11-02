# 项目状态（2026-09-26）

## 这是什么
michaelnny/alpha_zero 的 fork。原版 AlphaZero（围棋 / 五子棋）代码在 `alpha_zero/`。
本人在 2025-02 尝试用 DQN 训练五子棋（根目录 `training_gomoku_dqn*.py`、`play.py`），训练失败。
2026-09-24 新建 `dqn_gomoku/` 包重做，已训出能稳定击败规则型对手的 9x9 模型。

## 旧 DQN 为什么失败（已诊断，旧文件保留未改）
1. 每局把黑白双方 reward 相加，非平局恒为 1.0，曲线是直线是指标问题不是学习问题。
2. 自对弈 TD 目标用 `r + γ·max Q(s')`，s' 是对手视角，应为 negamax `r − γ·max Q(s')`。
3. 输家的最后一步从未拿到 −1。
4. 目标网络的 max 没有屏蔽非法动作。
5. 3825→128→128→225 的全连接网络无空间结构。
另外 `alpha_zero/core/mcts_v2.py` 的 `uct_search` 被改坏（引用不存在的 `env.current_step`、`env.get_board()`），原版被注释进字符串；`num_parallel=1` 路径会崩，默认并行路径仍可用。`pipeline.py` 需要 `pip install python-snappy` 才能 import。

## 新实现 `dqn_gomoku/`
- 9x9、`num_stack=1`（3 平面输入）、4 层 64 通道卷积 Q 网络。
- Double DQN + negamax 目标、输家回填 −1、目标 max 屏蔽非法动作、8 对称数据增强。
- 评估对固定对手（random / heuristic / 上一快照），前 2 手随机开局。
- 15 个单元测试：`/usr/local/bin/python3 -m unittest dqn_gomoku.tests.test_dqn`
- **必须用 `/usr/local/bin/python3`（3.11.1，PyCharm 用的解释器）**：3.11.1 的 `random.choice` 不接受 numpy 数组，代码里已规避，但测试要在这个解释器上跑。

## 训练记录
| 阶段 | 命令要点 | 步数/耗时 | 对随机 | 对启发式（100 局） |
|---|---|---|---|---|
| 1 | 纯自对弈，`--device mps` | 30 万步 / 104 min | 100% | 75% |
| 2 | `--init_ckpt` 阶段1 final，`--opponent_mix 0.5`，lr 2.5e-4 | 20 万步 / 69 min | 100% | 90–94% |

阶段 2 模型对阶段 1 模型 64%。最佳权重：`checkpoints/dqn_gomoku/9x9_stage2/final.pt`（`step_190008.pt` 略好）。
日志与曲线：`logs/dqn_gomoku/9x9/train_stage1.png`、`logs/dqn_gomoku/9x9_stage2/train_stage2.png`。
MPS 约 47 环境步/秒，CPU 慢 4.4 倍。

## 常用命令
```
/usr/local/bin/python3 -m dqn_gomoku.play --ckpt checkpoints/dqn_gomoku/9x9_stage2/final.pt
/usr/local/bin/python3 -m dqn_gomoku.evaluate --ckpt <pt> --opponent heuristic --games 100
/usr/local/bin/python3 -m dqn_gomoku.train --device mps [--init_ckpt <pt>] [--opponent_mix 0.5]
/usr/local/bin/python3 -m dqn_gomoku.plot --csv <train.csv> --out <png>
```

## 15x15 尝试（2026-09-25/26）
- 第一次：全连接输出头，50 万步，对启发式恒为 0，失败（归档在 `*/15x15_linearhead_failed`）。原因：`Linear(2*225→225)` 头无平移等变性，"堵四"要在 225 个位置各学一遍。
- 已给 `QNet` 加 `head='conv'`（全卷积头，默认），老 checkpoint 自动按 linear 加载。
- 第二次：conv 头 + 50% 对启发式对局，50 万步 5.3 小时 → 对随机 100%，对启发式 22%（100 局），输局仍是漏堵四。权重 `checkpoints/dqn_gomoku/15x15/final.pt`，曲线 `logs/dqn_gomoku/15x15/train_stage1.png`。
- 第二阶段热启动刚起步即被用户叫停（无 checkpoint，已删目录）。用户判断"训练方法有问题"，改天再议。

## 下次讨论的方向
- 纯 DQN（无搜索）在 15x15 上防守学得很慢，属于方法上限，不只是超参问题。
- 候选：(a) DQN 网络 + 浅层 MCTS 搜索（仓库已有 `parallel_uct_search` 可用）；(b) 缩小规模跑仓库原版 AlphaZero pipeline（需 `pip install python-snappy`，恢复原版 `uct_search`）；(c) 加深网络 / 增加历史平面 / 用 9x9 的 conv 层做迁移。

## 已知剩余问题 / 下一步候选
- 执白比执黑弱（82% vs 98%）。
- vs_prev 停在 0.35–0.6，单靠当前设置已接近平台；要更强需更大网络、更多步数或接 MCTS 搜索。
- 更大棋盘（13x13 / 15x15）需要重新训练，num_stack=1 的结构可直接复用。
- pygame 界面未重做（旧 `play.py` 依赖坏掉的 `uct_search`）。
- 备份：`../alpha_zero-main_backup_20260924.tar.gz`。

## GitHub（2026-11-02）
- 仓库：`Alepha188838884/alpha_zero-gomoku`，PRIVATE，已推送。本目录有独立 `.git`，不要从上层 `~/Desktop` 推送。git/gh 命令要加 `env -u GITHUB_TOKEN`。
- 未上传（与上游 michaelnny/alpha_zero 文件数一致、日期均为 2024-10-26，已写入 `.gitignore`）：`games/`、`checkpoints/go/`、`checkpoints/gomoku/`、`logs/go/`、`logs/gomoku/`。另忽略 `.idea/`。
- `README.md` 顶部已写明本人工作与上游的区分。2026-11-02 复测对启发式 100 局：9x9_stage2/final.pt 92%，15x15/final.pt 29%。
