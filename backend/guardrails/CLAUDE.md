# Guardrails — 双护栏

## 模块清单

| 文件 | 功能 |
|------|------|
| `rules.py` | 规则定义（10 种注入模式 + 6 种输出风险 + LLM 语义 Prompt）|
| `query_guard.py` | 查询护栏（正则快速匹配 + LLM 语义二次判断）|
| `output_guard.py` | 输出护栏（正则匹配 + LLM 语义判断）|

## 查询护栏规则（10 种）

| 规则 | 类别 | 严重度 |
|------|------|--------|
| prompt_injection_ignore | injection | high |
| prompt_injection_override | injection | high |
| shell_exec | dangerous_cmd | high |
| sql_injection | injection | high |
| excessive_length | input_validation | low |
| encoded_injection | injection | medium |
| jailbreak_dan | injection | high |
| roleplay_bypass | injection | medium |
| system_prompt_leak | injection | high |
| multilingual_injection | injection | high |
| chunked_injection | injection | medium |

## 输出护栏规则（6 种）

| 规则 | 类别 | 严重度 |
|------|------|--------|
| sensitive_info | sensitive | high |
| harmful_content | harmful | high |
| internal_system_leak | sensitive | high |
| pii_leak | sensitive | high |
| code_injection | harmful | high |

## LLM 语义判断

规则匹配后，可选调用 LLM 做二次语义判断（`GUARDRAIL_LLM_ENABLED=true`）。
