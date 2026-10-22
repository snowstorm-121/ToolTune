# ToolTune：本地量化 LoRA 工具调用微调实验

在 Apple M5、16 GB 统一内存上，使用 **MLX 4-bit Qwen3-4B-Instruct-2507** 和 Glaive 数据，完成从严格数据处理、LoRA 训练到冻结 Test 对照的可审计实验。问题聚焦于单轮**是否调用工具、工具名及参数生成**；每项结果均可回溯到冻结契约、首次预测和复算记录。

> **冻结 Test 主结果：**同一 MLX 基座上，无 adapter 为 **89/190（46.84%）**，加载 Validation 选中的 adapter 为 **100/190（52.63%）**，严格决策匹配净增 **11/190（5.79 个百分点）**。这是预先冻结的**均衡诊断集**，不代表原始 Glaive 或生产流量准确率。

[完整 Test 报告](reports/T4-test.md) · [逐题成败转移](reports/evidence/t4/paired-transitions.jsonl) · [面试讲解与追问](reports/INTERVIEW.md) · [上游来源与许可](UPSTREAM.md)

## 结果速览

| 冻结 Test 指标 | 无 adapter | 选中 LoRA adapter |
|---|---:|---:|
| 严格决策匹配，含完整结束 | **89/190（46.84%）** | **100/190（52.63%）** |
| gold 调用严格匹配 | 66/95 | 71/95 |
| 调用召回 / 工具名正确 | 88/95 | 91/95 |
| 参数符合 schema，分母为全部 gold 调用 | 88/95 | 91/95 |
| 参考参数严格匹配 | 66/95 | 71/95 |
| 达到 256-token 上限而未完成 | 72/190 | 66/190 |

190 道题逐题配对：**17 题改善、6 题退化**。改善中，7 题是非调用回答在固定输出预算内完成，6 题是调用格式包裹恢复；另有 4 题与参考参数对齐。结果不能整体解释为更强的参数推理。选中模型仍有 **66/190** 题截断；没有计算显著性或置信区间。[原始输出重新计分的摘要](reports/evidence/t4/recomputed-comparison.json)和[冻结实验契约](reports/evidence/t4/frozen-contract.json)可核对分母、哈希及对照条件。

## 工程主链

`Glaive 原始对话 → 严格解析与隔离 → 完整对话/工具家族/近重复分组 → 官方 Instruct 模板与当前 assistant loss mask → 冻结 4-bit 基座上训练 LoRA → Validation 选模 → 冻结 Test 的 B0/SFT 对照`

- **数据可信。** 原始 112,960 条对话中，99,108 条通过严格解析；畸形角色、未知工具、参数 schema 错误等进入隔离记录。按完整来源对话、工具家族和既定近重复规则连通分组，审计所选跨集来源、组和规则内近重复交集。随后在原有 split 内确定性抽取均衡诊断集：Validation 100 调用 + 100 非调用，Test 95 + 95；不拆组、不补造标签。见 [数据协议](reports/T2-balanced-protocol.md) 与 [解析实现](src/tooltune/data.py)。
- **训练目标明确。** 使用[固定的官方 Instruct 模板](configs/chat_template.jinja)，只监督当前 assistant 的目标 token；历史消息和工具回复只作为输入，未来工具回复不进入目标。针对 MLX-LM 0.32.2 在 padding 边界多计一个目标 token 的实际问题，通过公开 loss 回调修正并用测试锁定。见 [格式化](src/tooltune/formatting.py)、[loss](src/tooltune/train.py) 与 [排障记录](reports/T1-resource-smoke.md)。
- **训练范围可核对。** 在最后 4 个 transformer block 的注意力 q/v 投影挂载 rank 8 LoRA：**327,680** 个可训练参数；基座张量哈希不变。正式运行 **200 次 optimizer 更新、800 个 microsteps、40,821 个监督 token 曝光**，实际遍历 4,000 条训练池中的 **800 条**，并未训练完整一轮。U100/U200 仅由 Validation 选择，选中 U200。见 [配置](configs/train.yaml)、[训练脚本](scripts/t3.py) 与 [Validation 报告](reports/T3-validation.md)。
- **公平评测与失败保留。** B0 和 U200 从同一 MLX 基座分别独立加载；Test 的 190 个逐题 prompt token 完全一致，均用贪心解码、EOS 停止和 256-token 上限。生成输入不含金标准；截断、解析失败和空输出保留分母，错误预测不自动重试。测试前锁定模型、adapter、模板、数据、解析器、指标和样本 ID 的实验契约，之后从首次原始预测离线复算。见 [评分实现](src/tooltune/metrics.py)、[Test 运行脚本](scripts/t4.py) 与 [核验结果](reports/evidence/t4/final-verification.json)。

## 能解释的成功与失败

- **正确调用：**Test 样本 `glaive-v2:112844:a2` 中，U200 完整输出 `play_music`，专辑与艺人参数均匹配参考答案。这是**文本形式的调用**，没有播放音乐。
- **真实退化：**`glaive-v2:037855:a4` 的日历标题与日期已给齐；B0 输出结构化调用，U200 只说“会创建事件”，因此计作漏调用。
- **指标边界：**`glaive-v2:081350:a2` 请求提取画作颜色的 Ruby 代码；U200 在预算内结束但拒绝给代码。它在“未调用工具”的动作指标中由错变对，**不代表回答质量改善**。
- **参考标签疑点：**`glaive-v2:095636:a4` 的参考调用含可见对话中未提供的密码字符串。按冻结参考照常计分，但不能把匹配它说成安全的参数抽取能力。

更多原文、退化原因和资源记录见 [Test 报告](reports/T4-test.md)。两组 held-out 调用题都只有**一个候选工具**；Test 的 95 个非调用题中仅 1 个带可用工具。项目因此没有证明复杂多工具路由、广泛的“有工具但不适配”拒绝能力、自然语言事实正确率或真实 API 执行成功率。局部只读工具执行[演示](scripts/demo_tools.py)使用构造调用文本，**不计入 Test**。

## 验证与复现

环境为 Apple Silicon、Python 3.12、[锁定依赖](uv.lock)。本机最终验收为 **68 项测试通过**、T2 的 **65 文件哈希契约通过**、两臂各 **190 条首次预测**、逐题 prompt token 一致且原始预测复算与摘要一致。模型 revision、原始数据 revision、许可证和本地兼容处理见 [UPSTREAM.md](UPSTREAM.md)。

仓库可直接进行不依赖已下载模型资产的代码测试：

```sh
uv sync --locked --python 3.12
.venv/bin/python -m pytest -q tests/test_data.py tests/test_sampling.py tests/test_contract.py tests/test_metrics.py tests/test_evaluation.py tests/test_demo_tools.py
```

上述测试在本机为 **57 passed**。完整实验复现还需按上游锁定 revision 恢复 MLX 权重、Glaive 原始数据、本地 manifest 和 adapter；这些资产、原始预测、训练日志及 `.venv` 均不在 Git 仓库中。取得资产后，可按 [T2 数据协议](reports/T2-balanced-protocol.md) → [T3 训练与 Validation](reports/T3-validation.md) → [T4 冻结 Test](reports/T4-test.md) 的命令和拒绝覆盖规则重建新实验；现有本地结果可用 `scripts/recompute_validation.py`、`scripts/recompute_test.py` 离线复算。仓库中的[小型证据快照](reports/evidence/t4/)用于审阅本次实测条件与结果，不冒充完整原始数据发布。

项目代码负责数据解析、训练编排、评分与审计；模型架构、量化与优化器由 MLX-LM 提供。面试介绍、完整数据流、核心概念、设计取舍及有依据的追问回答在 [INTERVIEW.md](reports/INTERVIEW.md)。
