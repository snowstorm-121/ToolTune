# T2 均衡评测协议验收（2026-10-22）

状态：用户确认的均衡方案已实施，T2完成。正式活动协议为 `v2-balanced`，配置为 `configs/evaluation-balanced.json`。T3/T4未启动，没有正式基线、选模或Test模型预测。

## 已授权变更及不变条件

原始v1哈希抽样只有Validation 5条、Test 14条调用。用户于2026-10-22确认改用均衡抽样；此决定发生在任何正式模型评测之前。

保留原始v1全部文件和哈希契约。v2沿用同一原始文件、解析器、512-token完整样本过滤、原生模板、seed=42及全部来源分组，不拆组或调换split。每个held-out split内分别按 `sha256([seed, sample_id])` 排序，取固定数量的真实call和no_call，再按同一哈希合并排序。训练集以链接复用v1，逐字节不变；没有补造、修补或按模型表现筛选金标签。

## 实际清单

| 项目 | Train | Validation | Test |
|---|---:|---:|---:|
| 决策数 |4,000|200|190|
| call |1,207|100|95|
| no_call |2,793|100|95|
| 独立来源对话 |3,919|176|165|
| 连通组 |393|172|163|
| 监督token |206,911|18,646|17,394|
| 总token（含目标） |1,326,198|65,263|61,630|
| 无可用工具的样本 |923|97|94|
| 1个可用工具的样本 |2,470|103|96|
| 2个可用工具的样本 |607|0|0|

Validation池中可用调用共100条，Test池中共95条，均全部保留；各自非调用池2,892/2,889条中确定性抽取100/95条。未凑满原计划300条Test，因为不补造调用、不破坏分组。

## 实测审计

- 重新读取原始来源生成候选池，类别计数与v1全量扫描一致。
- 再按反转的来源顺序独立解析、过滤和抽样，Validation/Test完整records字节与首次结果一致。
- 全部4,390条样本的原生ChatDataset/CacheDataset token和offset与项目格式化器一致；system/user/历史tool与assistant不受监督。
- 选中样本跨split来源、组、工具家族、规范化问法交集均为0；独立跨集词三元组Jaccard>=0.85扫描的近重复对数均为0。该结论不超出既定字符串规则，不代表语义/预训练层面无泄漏。
- 训练文件与v1逐字节相同；v1的48文件哈希契约仍通过，新契约覆盖65个文件，包括原始数据、模型/tokenizer文件、原v1契约、新配置、实现和选题清单。
- 构建及验证日志见 `artifacts/t2-balanced-prepare.log`；最终测试和独立校验见 `artifacts/final-balanced-verification.json`。

## 结果解释边界

这是**均衡诊断集**，不能将其总准确率解释为原始Glaive或生产流量的准确率。“全部不调用”的主指标在两个held-out集上均为50%。未来报告仍需提供调用召回、schema和参数严格匹配等指标的分子分母。

held-out调用全部只有1个候选工具，因此工具名准确率不能包装成复杂工具路由能力。Test中95条非调用仅1条带可用工具，Validation中100条非调用仅3条带工具；大多数负例是无工具普通对话，不能据此强称“有工具但不适配”场景已充分覆盖。

未建立独立审定的追问/无适配工具子集，相关专门指标仍为N/A。没有自动把问号/拒绝关键词变成金标签。普通no_call只检查是否调用，不证明自然语言答案正确；参数参考匹配不等于真实业务执行成功。多个决策可来自同一对话/组，后续不把所有行当独立观测计算置信区间。

## 本地入口

```sh
cd /Users/yyy/code/ToolTune
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/prepare_balanced.py --validate
.venv/bin/python -m pytest -q
```

在已完成v1的干净复现目录运行 `scripts/prepare_balanced.py` 可生成v2；命令拒绝覆盖已有v2。活动数据在 `data/derived/v2-balanced/`，来源/样本ID与契约在 `data/manifests/v2-balanced/`。原始v1和训练侧200+20烟测产物保留。此时没有新的训练运行，T1 adapter仍只是合成烟测产物。
