# ToolTune 面试准备稿

这份材料围绕已经运行的项目。先记五组数：**112,960 条原始对话 → 99,108 条通过严格解析；训练池 4,000 决策，正式训练实际看过 800 条；Validation 200 题，Test 190 题；LoRA 327,680 个可训练参数、200 次 optimizer 更新；冻结 Test 严格匹配 B0 89/190、U200 100/190。** Validation 的 94/200→107/200 只用于选模，不和 Test 混报。

## 两分钟介绍

“ToolTune 是我在 Apple M5、16GB 机器上做的本地工具调用微调实验。目标是观察同一个量化 Qwen3-4B-Instruct-2507 基座，加载 LoRA 后能否更准确地决定是否调用工具，并给出符合参考答案的工具名和参数。

我先严格解析 Glaive 的对话和工具 schema，不能确定的记录隔离，不自动修标签；按完整对话、工具家族和近重复问法分组，冻结训练、Validation、Test。原始数据有 112,960 条对话，99,108 条通过解析。训练池 4,000 个决策样本；为了在本机资源内完成实验，正式运行 200 次 optimizer 更新、800 个 microsteps，实际只曝光了 800 个不同训练决策，所以我不会说训练完了整个 Glaive。

训练使用同一 MLX 4-bit 基座，冻结原权重，只在最后四层注意力 q/v 投影上训练 rank 8 LoRA，共 327,680 个参数。用官方 Instruct 模板，loss 只算当前 assistant 的目标 token，包括结束 token；历史消息和工具返回仅作为上下文。第 100、200 次更新在同一 Validation 上评测，200 次更新的 checkpoint 由预先固定的规则选中。Validation 严格决策匹配从无 adapter 的 94/200 到 107/200，其中新增正确调用多来自完整 tool_call 格式恢复。

最后我在选模前冻结的 190 题 Test 上，独立加载无 adapter 和选中 adapter，固定相同输入和解码，从原始输出离线复算。严格匹配从 89/190 提升到 100/190，逐题有 17 题改善、6 题退化。新增正确主要来自格式恢复和普通回答在 256 token 内结束，另有部分参数与参考答案对齐；仍有 66 题截断。我把参数参考匹配、schema 合法和真正执行工具分开，不把这 11 题净增说成复杂路由能力。”

## 一条样本的完整数据流

1. 原始 Glaive `system` 给工具定义，`chat` 给 USER、ASSISTANT、FUNCTION RESPONSE。`parse_conversation` 用严格 JSON 和有限的单引号 arguments 包裹语法解析；不使用 `eval`。不完整角色、未知工具、schema 错或畸形调用进隔离记录。依据：`src/tooltune/data.py`、`data/derived/v1/quarantine_*.jsonl`。
2. `decision_examples` 在每个 assistant 轮次截断对话：此前的 tool response 可以在输入上下文中，此轮之后的回复不能泄漏进输入。每个决策保留来源 ID、组 ID、gold 动作/工具/参数。按完整对话、工具家族、规范化问法和既定近重复规则连通分组后再分割。依据：`src/tooltune/data.py`、`data/manifests/v2-balanced/contract.json`。
3. `format_example` 用锁定的 Qwen 官方 Instruct 模板得到 prompt token 和完整目标 token。训练器的 `ChatDataset(mask_prompt=True)` 产生 prompt 边界，项目 loss 用移位后的 targets，只保留当前 assistant 目标位置 `[prompt_boundary, unpadded_length)`。其中结束 token 要监督，padding 不进入 loss。依据：`src/tooltune/formatting.py`、`src/tooltune/train.py`、`UPSTREAM.md`。
4. 梯度从这些 token 的交叉熵传到最后四层 q/v 的 LoRA A/B。基座 4-bit 权重冻结；micro batch 为 1，累积 4 个 microsteps 才做一次 Adam 更新。200 次更新共 800 次样本曝光，40,821 个监督 token 曝光，不等于 4,000 条训练池完整一轮。依据：`configs/train.yaml`、`reports/T3-validation.md`、本地 `artifacts/t3-r8-v1/formal/`。
5. 推理时只传工具定义和当前轮之前的消息，greedy 生成至 EOS 或 256 token；不约束 JSON，也不补修输出。解析器区分完整 `<tool_call>`、普通不调用文本和错误输出。对调用再检查工具名、参数 schema、参考参数严格匹配；非调用只评动作，不评文字事实。依据：`src/tooltune/evaluate.py`、`src/tooltune/metrics.py`。
6. Validation 只负责选 U200；Test 在选定后冻结契约，用同一基座分别独立加载无 adapter 和有 adapter 的推理进程。每道题的原始输出、prompt token、停止原因、分数和失败类别保存，离线重新计分。依据：`reports/T3-validation.md`、`reports/T4-test.md`、本地 `artifacts/t4-test-v1/`。

## 核心概念与设计取舍

- **量化与 LoRA：** 4-bit 说的是基座存储/计算表示；LoRA 用两个低秩矩阵表示所选线性层的增量，原权重不更新。rank 8 只控制增量的低秩瓶颈，不能推出计算量也下降到参数比例。当前安装的 MLX `scale=16` 是直接乘数，不等同于随意填写 PEFT `alpha`。Ollama GGUF 可作本地部署参考，不能与 MLX adapter 结果相减归因微调。
- **为什么只训当前 assistant：** 目标是该轮“调用/不调用与参数”。未来工具返回在决策时不可见；把它放进输入或 loss 会形成泄漏。历史工具返回只是上下文。一个真实 bug 是上游默认 loss 在有 padding 时多包含一个目标位置；项目通过公开 loss 回调把上界改为 `< unpadded_length`，有回归测试。
- **为什么严分组又均衡抽样：** 同一来源对话的多个轮次及工具家族/模板近重复若跨集，会高估泛化。原始 held-out 中调用稀少，经用户在正式模型评测前确认，两个 held-out split 内分别做均衡抽样，得到 Validation 100+100、Test 95+95。它们是诊断分布，不能外推原始流量准确率；分组规则也不保证没有语义或预训练泄漏。
- **为什么主指标严格：** 只有完整结束、正确动作、调用时工具名和参考参数都匹配且 schema 合法才算对。格式缺标签、截断、空输出都留在分母。参考答案可能欠明确，因此“未匹配”不自动等于业务错误；schema 合法也不保证真实服务可执行。
- **为什么只有 U200：** 在 Test 前固定按 Validation 严格匹配优先选；U100 为 103/200，U200 为 107/200。rank 16 是可选对照，这次未运行。Validation loss 到 U200 还在下降，但固定预算结束不能称收敛或最佳训练步数。

## 常见追问与有证据的回答

| 追问 | 回答要点与依据 |
|---|---|
| “你怎么证明微调真的生效？” | 16 个 LoRA 张量变化、327,680 个可训练参数，冻结基座哈希不变；新进程重载选中 adapter 的 16 个张量和原文件逐个匹配，Train 探针 logits 变化 11.3125。后者只证明生效，不证明质量。`reports/T3-validation.md`、`reports/evidence/t3/final-verification.json`。 |
| “200 步是不是只训练了 200 条？” | batch 1、梯度累积 4，所以 200 optimizer updates = 800 microsteps；实际记录 800 个不同决策、40,821 个监督 token。没跑满 4,000 条训练池。`reports/T3-validation.md`。 |
| “为什么不用已有 Ollama 模型直接训练？” | 本地 Ollama 是 GGUF 推理权重，MLX-LM LoRA 需要 MLX 格式。正式 B0 和 SFT 均用同一 pinned MLX 4-bit 基座和模板。`UPSTREAM.md`、T3/T4 契约。 |
| “Validation 进步主要是模型学会推理了吗？” | 不能这样说。B0→U200 的 9 个新正确调用中，7 个是原本缺少完整包裹，2 个是同一博客工具家族的字符串对齐；主要可观察到格式与参考对齐，复杂多工具路由没有测试。`reports/T3-validation.md`。 |
| “Test 的提升具体来自哪里？” | 严格匹配净增 11/190：17 改善、6 退化。7 个非调用改善来自 B0 截断而 U200 完成；调用的 10 个改善里 6 个是包裹格式恢复、4 个是参考参数对齐。调用另有 5 个退化，非调用 1 个退化。不能只报净增、不报退化和原因。`reports/T4-test.md`、本地 `artifacts/t4-test-v1/paired_transitions.jsonl`。 |
| “你有没有发现参考标签本身的问题？” | 有。Test `glaive-v2:095636:a4` 的参考参数给出一段可见历史中没有提供的明文密码；U200 恰好匹配，但不能当成安全参数抽取能力。另有税率 10% 的 `10` 与 `0.1` 语义表示歧义。仍按冻结参考计分，不事后改题。`reports/T4-test.md`、原始预测。 |
| “你如何防止泄漏和反复试 Test？” | 完整来源/家族/近重复分组，测试交集为零；生成输入剔除 gold 目标。T3 候选只用 Validation 选；T4 在首次 Test 前锁模型、adapter、代码、模板、Test ID 和解码，原始预测首次写入并只对未完成 ID 续跑。`data/manifests/v2-balanced/contract.json`、`artifacts/t4-test-v1/frozen_contract.json`。 |
| “你的调用准确率为什么不能代表 Agent？” | held-out 调用全部只有一个候选工具；多数非调用样本没有工具。项目只评单轮决策及参数，不做工具链执行、状态管理、业务成功率。可选本地执行器仅演示解析后安全分发。`reports/T2-balanced-protocol.md`、`src/tooltune/demo_tools.py`。 |
| “遇到哪些实际问题？” | Glaive arguments 的单引号外壳要有限语法解析；社区模型模板与官方 Instruct 模板不一致；上游 loss padding 上界多一位；下载曾停滞，保留失败日志后用直接 HTTP 完成。结果层还有截断、schema 缺参、参考年份欠明确和非调用动作分歧。`reports/T1-resource-smoke.md`、`reports/T3-validation.md`。 |
| “你会怎样继续改善？” | 先看 T4 错误分布；若截断多，固定新的输出预算并开新版本重新评估，原 Test 变为已见回归集。补独立审定的有工具但不适配/追问、多候选路由和真正执行成功场景，再考虑更长训练或 rank 对照；不在当前冻结 Test 上反复调参。`reports/T4-test.md`。 |

## 四类现场展示样本

| 场景 | 固定 ID 与讲法 |
|---|---|
| 正确调用 | Test `glaive-v2:112844:a2`：输入专辑和艺人，U200 生成完整 `play_music` 与参考参数；没有真的播放。 |
| 参数/动作错误 | Test `glaive-v2:037855:a4`：U200 说“我会创建日历”，却没有结构化调用。Test `glaive-v2:090538:a6`：额外加入 `genre=comedy`，参考参数不匹配，但前文谈过喜剧，解释有歧义。 |
| 不应重复调用 | Test `glaive-v2:090538:a8`：已有 `search_movies` 的工具返回，再回答 Bill Murray 片单；两个模型均未再次调用。这个子集仅有 1 个带工具的负例，不能外推。 |
| 信息不足/缺少输入 | Validation `glaive-v2:073945:a2`：查酒店时模型追问房型，来源参考直接调用；没有审定追问标签，只作定性讨论。Test `glaive-v2:081350:a2` 缺少画作图像，U200 提前结束但拒绝给 Ruby 代码；动作评分记“改善”，并不证明回答有帮助。 |

准备时先用两分钟稿说清数字与边界，再拿一条真实失败样本沿着“输入 → 原始输出 → 解析 → schema/参数 → 分母”解释。要区分三句话：**工具格式合法、参考参数吻合、真实工具执行成功**，它们不是同一个指标。
