# Learning Module — 五阶段最终报告

日期：2026-09-24　范围：Phase 0–5 全部完成

学习模块位于仓库根的 `learning_module/`（Python，SQLite，子进程调用），harness 侧只有
`packages/core/agent-loop/src/learning-bridge.ts` 与 `agent.ts` / `index.ts` 中
Phase 0 固定的两个仪表点加一个 context provider 接缝。默认开关
`DSH_LEARNING_MODULE=off`。

---

## 一、各阶段测试结果

| 阶段 | 交付 | 验证结果 |
| --- | --- | --- |
| Phase 0 基础设施 | `experiences` 表、模式开关、两个空钩子、fail-silent 包裹 | 快照门 `34 failed / 75 passed / 7 skipped`（全部为 Windows 基线失败）；harness stderr 无模块输出 |
| Phase 1 状态编码与记录 | 两次调用状态编码器、经验身份三元组、`record_outcome` upsert、三层 reward | Python 套件；`audit_phase1.py` exit 0（字段齐全、取值在范围内）；`spotcheck_phase1.py` exit 0 |
| Phase 2 检索与价值 | Retriever（精确匹配→邻域放宽→衰减→置信度下限→top3）、价值引擎、`value`/`rewards` 列与回填脚本 | `acceptance_phase2.py` **26/26**；模块测试 128 passed |
| Phase 3 注入 | `context.py` 渲染带标注的参考块；经 `ctx.systemPrompt.context({ name: 'agent-loop:learning' })` 注入为可回放的会话快照 | agent-loop 学习相关 spec **49 passed**；`PHASE3_ACCEPTANCE.md` 记录 10 个真实会话的 35 个注入块（119–143 tokens，上限 200） |
| Phase 4 防作弊 | `validation.py`：改测试文件后通过则 reward×0.3 并标记；高 reward 小 diff 标记待抽查；`flagged` 查询接口 | `acceptance_phase4.py` **36/36**；Phase 4 新增 52 测试；模块累计 **181 passed** |
| Phase 5 Benchmark | 16 个固定任务、四臂 64 次真实运行、指标提取与报告 | 夹具门 **16/16**（出厂必失败、修好必通过）；库校验 **19/19**；`PHASE5_BENCHMARK.md` |

贯穿全程的门禁：`tsc -b packages/core/agent-loop` exit 0；`pnpm run test` 失败项
始终是改动前基线集合的子集（Windows 符号链接权限、`CreateProcessAsUserW` ACL、
`unknown tool "bash"`）；快照基线未变。

---

## 二、Benchmark 数据

16 个任务 × {`off`、`off-repeat`、`shadow`、`active`} = **64 次真实运行**，全部
`status=0`，无崩溃。`off-repeat` 是同一 `off` 模式的第二次运行，用来量化采样噪声。

### 关键指标

| 指标 | off (A) | shadow (B) | active (C) | C 相对 A |
| --- | --- | --- | --- | --- |
| 任务成功率 | 100% | 100% | 100% | 0.0 pp |
| 平均 tool call 数 | 8.63 | 8.44 | 8.44 | 2.2% 更少 |
| 平均 token 消耗 | 68 861 | 69 013 | 70 153 | **1.9% 更多** |
| 重复犯错率（含夹具种子） | 84.4% (27/32) | 75.8% (25/33) | 83.3% (25/30) | **1.2% 更低** |
| 重复犯错率（仅模型步骤） | 68.8% (11/16) | 64.7% (11/17) | 78.6% (11/14) | **−14.3%（更差）** |
| recovery speed（步） | 1.00 | 1.13 | 1.06 | −6.3%（更慢） |
| regression rate | 6.3% | 12.5% | 6.3% | 0.0% |
| 收到注入块的运行数 | 0 | 0 | 15 / 16 | — |
| 修改了评分用测试文件的运行数 | 0 | 0 | 0 | — |

### A 与 B：记录经验零副作用

- 16 次可比较运行中，A/B 有 **54 处字段差异**；
- 同一 `off` 模式重跑（A vs off-repeat）有 **58 处差异**——这就是噪声底；
- **54 < 58**：shadow 的差异不超过"同一模式跑两遍"的抖动，因此"记录经验不影响决策"
  成立。逐字节完全一致在未设 temperature 的采样模型上不可达，这一点在报告里明确写出，
  没有把噪声当作等价证据。

### 核心问题：重复犯错率

- 含夹具种子的口径：84.4% → 83.3%，**改善 1.2%**；
- 仅模型自身步骤的口径：68.8% → 78.6%，**变差 14.3%**；
- 阈值 10%。**两个口径都远低于门槛。**

**结论：`active` 模式在本 benchmark 上没有证明有效。** 报告未做任何美化：成功率三者
同为 100%（任务太短，天花板效应），token 反而增加 1.9%，recovery speed 略变慢，唯一
"变好"的 1.2% 落在噪声底（6%）之内，而剔除夹具种子后方向反转为更差。

### 为什么没测出效果（诚实归因）

1. **任务太容易**：16 个任务全部是单 bug 小工作区，A 组本身 100% 解出、平均 8.6 次调用，
   没有留给经验去省的空间。
2. **重复犯错是跨任务的长尾**：唯一对模块有利的指标要求同一失败在后续任务中再现，而
   这些任务彼此独立、失败种类只有 5 类，样本里没有出现"同一个坑反复踩"的模式。
3. **Phase 3 实测：模型完全不引用经验块**：10 个真实会话中，模型从未提及注入的历史经验
   （唯一一次关键词命中还是指 pytest 缓存警告）。这一点在 Phase 5 得到独立确认。
4. **两个已知设计缺陷未修**（按裁定留到本阶段评估）：
   - 检索只按 `state_error_type` 选行，同一动作会同时出现在"成功做法"和"避免做法"两组里；
   - reward 衡量的是"这一步是否推进了任务"，不是"这个动作是否合理"，于是 `run_test`
     这类必要动作会被标成负值并被建议避免。
   在本 benchmark 上，注入内容因此既不可靠也不可执行——改善为 0 与这两点一致。
5. **样本量**：每任务每臂 1 次运行（外加 off-repeat 作为噪声底），6% 以内的差异都不可区分。

---

## 三、是否建议正式启用 active

**不建议。**

| 选项 | 建议 |
| --- | --- |
| 默认开启 `active` | **否**。核心指标未达 10% 门槛，token 反而上升，且注入内容存在自相矛盾（同一动作同时被推荐和劝阻）。 |
| 保留 `off` 为默认 | **是**。当前默认值即为 `off`，不需要改动。 |
| 以 `shadow` 继续采集 | **可以**，且已验证零副作用（差异在噪声底内）。若要在真实项目上积累数据，`shadow` 是安全选项。 |
| 重新评估的前置条件 | ① 修掉两个设计缺陷（检索键粒度、reward 语义）；② 用**真实且重复**的失败场景构建 benchmark（同一坑跨任务再现），而不是 16 个独立小题；③ 每臂至少 3 次重复以压低噪声。三者具备后再跑一次本 benchmark。 |

`active` 的开关、注入格式、预算与 fail-silent 行为都已按规格实现并验证，因此"随时可
开启、开启也不会有害"这一点是成立的——**不能成立的只是"开启会有收益"**。

---

## 四、复现方式

```sh
# 夹具门：确认 16 个任务出厂必失败、修好必通过（不花模型调用）
node --import tsx/esm .learning-tools/verify_benchmark_fixtures.ts

# 四臂运行（每个任务每臂一次；off-repeat 用于噪声底）
node --import tsx/esm .learning-tools/benchmark_run.ts <scratch> --arms off,off-repeat,shadow,active

# 报告
node --import tsx/esm .learning-tools/benchmark_report.ts <scratch> --out <scratch>/PHASE5_REPORT.md
node --import tsx/esm .learning-tools/finish_phase5_report.ts <scratch>

# 三臂经验库独立校验
python .learning-tools/validate_benchmark_stores.py <scratch> off shadow active
```

需要 `DEEPSEEK_API_KEY`、一个可启动 headless profile 的 `DSH_HOME`（见
`.learning-tools/README.md`）。64 次运行约 35 分钟。
