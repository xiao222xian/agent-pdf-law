# STA2HTM 项目技术交接文档（Dify + GPUStack）

## 1. 项目目标
本项目用于构建一个 STA2HTM 时序分析问答助手。

核心输入：
- `setup/hold 主 JSON`（main）
- `*.clk.json`（clock 维度）
- `*.blk.json`（block 维度）

核心能力：
- 自动校验三份 JSON 一致性。
- 基于用户问题 + 三份 JSON + RAG 知识库输出分析结论。
- 支持异常输入兜底（无效问题、文件不一致、JSON 解析失败）。

---

## 2. 当前目录结构与关键资产

项目路径：`/home/lyg/STA2HTM`

关键目录：
- `RAG/`：原始蒸馏 YAML 知识。
- `RAG_md/`：YAML 转成 markdown 的版本。
- `RAG_chunks/`：按 `id` 拆分后的检索粒度文档（推荐入库）。
- `dify_assets/`：工作流代码节点与提示词资产。
- `tools/`：自动化脚本（例如 YAML 按 id 拆分）。

关键文件（当前在用）：
- 代码节点脚本：`dify_assets/code_node_sta2htm.py`
- 成功分支提示词：`dify_assets/llm_prompt_success_rag_rawfiles.txt`
- 失败分支提示词：`dify_assets/llm_prompt_fail.txt`
- 拆分脚本：`tools/split_rag_yaml_to_chunks.py`

---

## 3. 系统架构与执行流

Dify Workflow（推荐）:
1. `用户输入`
2. `文档提取器`（main）
3. `文档提取器 2`（clk）
4. `文档提取器 3`（blk）
5. `代码执行`
6. `条件分支`（按 `validation_ok`）
7. True 分支：`知识检索` -> `LLM1` -> `直接回复`
8. False 分支：`LLM2` 或 `直接回复2`

说明：
- 代码节点负责“硬规则校验”和轻打包。
- LLM1 负责“按用户问题进行分析生成”。
- 建议失败分支优先用 `直接回复2`（不依赖 LLM，稳定）。

---

## 4. 代码节点设计说明（重点）

文件：`dify_assets/code_node_sta2htm.py`

### 4.1 输入变量
- `arg1`: main JSON 文本（来自文档提取器1）
- `arg2`: clk JSON 文本（来自文档提取器2）
- `arg3`: blk JSON 文本（来自文档提取器3）
- `arg4`: 用户问题（可选，建议绑定 `sys.query`）

### 4.2 输出变量
- `validation_ok` (Boolean)
- `validation_error` (String)
- `analysis_payload` (Object, 浅层)
- `analysis_payload_json` (String, 完整 payload)

### 4.3 代码逻辑
1. 解析输入 JSON 文本：
- 支持普通 JSON。
- 支持被 ```json code fence 包裹的文本。

2. 结构校验：
- main 必须包含：`group/mode/check/corners/per_corner_summary/endpoints`
- clk 必须包含：`group/mode/check/corners/clocks`
- blk 必须包含：`group/mode/check/corners/blocks`

3. 一致性校验：
- `group/mode/check` 三文件一致。
- `corners` 三文件一致。

4. 用户问题硬拦截（可选但推荐开启）：
- 当 `arg4` 非空时：
  - 空、过短、纯数字、无 STA 关键词 -> `query_invalid`。
  - 直接 `validation_ok=False`，走失败分支。

5. 输出深度控制：
- `analysis_payload` 只保留浅层 `meta/consistency/counts`，避免 Dify 报错 `object too deep`。
- `analysis_payload_json` 存完整三份原始 JSON（字符串），供 LLM1 解析。

### 4.4 为什么这样设计
- 避免在代码节点做过多计算，减少逻辑耦合。
- 把分析责任交给 LLM，便于根据问题上下文灵活回答。
- 通过 `analysis_payload_json` 绕过 Dify 对象深度限制。

---

## 5. 提示词策略说明

### 5.1 成功分支提示词
文件：`dify_assets/llm_prompt_success_rag_rawfiles.txt`

核心约束：
- 从 `analysis_payload_json.raw_files.*` 自行读取并计算结论。
- 输入无效时只输出单句澄清，立即停止。
- `check` 类型不一致（用户问 hold 但数据是 setup）时立即停止。
- 必须给证据路径与具体数值。
- 禁止输出模板占位语句和 `<think>`。

### 5.2 失败分支提示词
文件：`dify_assets/llm_prompt_fail.txt`

核心约束：
- 若 `validation_error` 以 `query_invalid` 开头：只输出一句澄清。
- 若是文件/格式错误：给出简明错误解释和下一步修正建议。

---

## 6. 知识库（RAG）处理流程

### 6.1 原始数据
- `RAG/*.yaml` 三类知识：concepts / diagnostics / pitfalls。

### 6.2 自动拆分
脚本：`tools/split_rag_yaml_to_chunks.py`

功能：
- 按 `- id: "..."` 自动切分。
- 输出到：`RAG_chunks/concepts|diagnostics|pitfalls`。
- 当前总量：42 个 chunk（20 + 12 + 10）。

执行命令：
```bash
python3 /home/lyg/STA2HTM/tools/split_rag_yaml_to_chunks.py \
  --input-dir /home/lyg/STA2HTM/RAG \
  --output-dir /home/lyg/STA2HTM/RAG_chunks
```

### 6.3 Dify 入库建议
- 优先直接上传 `RAG_chunks` 下的 md 文件，不推荐用大块 CSV。
- 分段参数建议：
  - 分段标识符：`\n\n`
  - 分段长度：500~800
  - 重叠：80~150
  - Top-K：6
- 若有 rerank，建议开启。

### 6.4 Embedding 使用说明
- Embedding 模型配置在“知识库索引”层，不在 LLM 节点。
- 切换 embedding 后必须重建索引，否则不生效。

---

## 7. 常见问题与排障

### 7.1 输入 `1` 仍输出长分析
原因：只靠提示词拦截不稳定。
处理：
- 代码节点启用 `arg4` + query 硬拦截。
- 条件分支基于 `validation_ok`。

### 7.2 失败分支报模型凭据错误
报错示例：`Model xxx credentials is not initialized.`
处理：
- 失败分支改为 `直接回复2`（推荐）。
- 或给 LLM2 配置可用模型凭据。

### 7.3 代码节点报 `Depth limit 5 reached, object too deep`
原因：对象输出层级过深。
处理：
- `analysis_payload` 保持浅层。
- 深层原始 JSON 放入字符串 `analysis_payload_json`。

### 7.4 检索结果跑偏
处理：
- 用 `RAG_chunks` 小粒度文档入库。
- 提高 Top-K。
- Query 增加 check/mode 约束词（如 hold/func/min delay）。

### 7.5 模板化空洞输出（复读“发现+证据”）
处理：
- 在 LLM1 提示词禁止模板占位语句。
- 强制每条发现输出证据路径 + 数值。

---

## 8. 接手者快速启动清单（1小时内）

1. 打开 Dify Workflow，确认节点拓扑与本文件一致。
2. 确认代码节点脚本为：`dify_assets/code_node_sta2htm.py` 最新版。
3. 确认代码节点输入已绑定 `arg1/arg2/arg3/arg4`。
4. 确认输出变量 4 个：`validation_ok/validation_error/analysis_payload/analysis_payload_json`。
5. 确认 LLM1 提示词使用：`llm_prompt_success_rag_rawfiles.txt`。
6. 确认失败分支可用（优先直接回复2，避免凭据问题）。
7. 确认知识检索绑定新知识库（由 `RAG_chunks` 入库）。
8. 用 sample 三文件做回归：
   - 正常问题：应输出结构化分析。
   - 输入 `1`：应被拦截，仅返回澄清句。
   - 故意错配文件：应走失败分支。

---

## 9. 后续优化建议（可选）

1. 在代码节点新增可选“轻统计”字段（worst_corner、dominant_clock）作为辅助，不改变主逻辑。
2. 失败分支完全去 LLM 化（固定模板回复），进一步提升稳定性。
3. 增加自动回归脚本，固定 5 条测试问题做发布前检查。
4. 引入 query 重写层（提取 check/mode/intent）提升检索精度。

---

## 10. 关键文件索引

- `dify_assets/code_node_sta2htm.py`
- `dify_assets/llm_prompt_success_rag_rawfiles.txt`
- `dify_assets/llm_prompt_fail.txt`
- `tools/split_rag_yaml_to_chunks.py`
- `RAG_chunks/`
- `input_sample/`

---

交接建议：
- 先保证“稳定可跑”（失败分支 direct reply）。
- 再逐步优化检索与提示词，不要同时大改代码节点和提示词。
