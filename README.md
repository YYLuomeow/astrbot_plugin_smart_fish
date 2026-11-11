# 聪明的鱼（astrbot_plugin_smart_fish）

面向 OneBot v11（aiocqhttp）的 AstrBot 群聊插件：先判断消息是不是在跟机器人说话，再决定是否回复。回复统一通过 `event.request_llm(...)` 交给 AstrBot 系统 LLM 链路生成，人格、模型、上下文都由 AstrBot 统一管理。

同时提供一个可被模型自主调用的工具，用来查询基元律动（TokenRhythm）的钱包余额和最近用量。

## 功能

- **消息判断**：先用主动唤醒词 / 主人消息快速放行，其余消息交给判断模型（LLM 判断或 JEV 判断），概率达到阈值才回复。
- **回复防抖**：等待用户把话说完，期间有新消息会取消正在生成的回复，合并后重新生成。
- **认主**：可配置主人 QQ 号和昵称，主人消息可跳过判断，并注入认主提示词。
- **群聊上下文**：判断和回复都会携带最近若干条群聊记录。
- **图片消息**：可先用对话模型把图片转成文字，再按唤醒词决定是否回复；转述结果按图片内容哈希缓存，重启后仍然有效。
- **戳一戳**：被戳后可反戳，并可选结合上下文回复。
- **决策记录面板**：插件页面展示最近的判断记录，包括图片完整转述和机器人最终回复，最多保留 100 条。
- **余额查询工具**：模型可通过 `query_tokenrhythm_wallet` 查询基元律动钱包余额和最近花费、Token、调用次数与主要模型。

## 安装

1. 将本目录放到 AstrBot 的 `data/plugins/astrbot_plugin_smart_fish`。
2. 重启 AstrBot，或在 WebUI 插件管理中重载本插件。
3. 在 WebUI 插件配置中按需填写下方配置项。

依赖：AstrBot `>=4.29,<5`，平台适配器 `aiocqhttp`。

## 配置说明

### 主动唤醒词 `bot_aliases`

消息中出现任意一个词时直接视为在对机器人说话，不经过判断模型。默认包含 `鱼来`、`鱼呢`、`小鲸鱼`、`大肥鱼`。

### 消息判断 `judge`

| 配置项 | 说明 |
| --- | --- |
| `enabled` | 是否启用消息判断 |
| `mode` | `llm` 使用对话模型判断，`decision` 使用决策模型平台判断 |
| `judge_context_count` | 判断时携带的最近群聊条数，0 表示只判断当前消息 |
| `threshold` | 判定为「在和我说话」的概率阈值，默认 0.55 |
| `llm_provider_id` | LLM 判断使用的模型，留空使用当前会话模型 |
| `platform` | 决策平台：`tokenrhythm`（基元律动）或 `typesafe` |
| `tokenrhythm_*` / `typesafe_*` | 对应平台的 API 地址、API Key、模型 ID，留空使用默认值 |
| `decision_question` | 决策模型使用的判断问题 |
| `prompt` | LLM 判断提示词，支持 `{context}` 与 `{message}` 占位符 |

### 图片回复 `image_to_text`

| 配置项 | 说明 |
| --- | --- |
| `enabled` | 是否启用图转文 |
| `provider_id` | 图转文模型，留空使用当前会话模型（需支持图片输入） |
| `prompt` | 图转文提示词 |
| `wake_words` | 转述文字包含其中任意一个词时回复 |
| `cache_enabled` | 是否缓存图转文结果 |
| `cache_size` | 缓存条数上限，超出后按写入顺序淘汰 |
| `cache_clear` | 开启并保存后立即清空缓存，随后自动关闭 |

### 回复防抖 `debounce`

`enabled` 启用防抖，`seconds` 为静默等待秒数，期间收到新消息会重置计时。

### 主人设置 `master`

`ids` 为主人 QQ 号（留空使用 AstrBot 全局管理员），`names` 为对应昵称，`always_reply` 让主人消息跳过判断，`extra_prompt` 为注入的认主提示词。

### 戳一戳 `poke`

`poke_back` 控制是否反戳，`reply` 控制是否结合上下文回复。

### 余额查询 `wallet`

| 配置项 | 说明 |
| --- | --- |
| `enabled` | 启用余额查询工具 |
| `cookie` | 基元律动网页登录 Cookie（密文保存），未填写时工具直接拒绝查询 |
| `admin_only` | 仅管理员可查询，其他人询问会被拒绝 |

Cookie 获取方式：浏览器登录 [tokenrhythm.studio](https://tokenrhythm.studio)，打开开发者工具的 Network 面板，任选一个 `/api/` 请求，复制请求头中的完整 `Cookie` 值。Cookie 属于登录态，过期后需要重新复制。

## 余额查询工具

工具名 `query_tokenrhythm_wallet`，由模型在用户询问余额、剩余金额、最近花费或用量时自主调用：

- 参数 `days`（整数，默认 7，范围 1-90）：统计最近 N 天（含今天）的用量。
- 数据来源：
  - 余额：`GET https://tokenrhythm.studio/api/wallet/summary`
  - 用量：`GET https://tokenrhythm.studio/api/usage/multimodal/overview`（`fundingScope=WALLET`，按模型分组）
- 返回内容：可用余额（含冻结与欠款仅在大于 0 时显示）、区间花费、调用次数、输入 / 输出 Token，以及花费占比最高的三个模型。
- Cookie 过期或网络异常时返回明确的失败提示，不会静默失败。

示例输出：

```text
钱包余额 96.19 元（赠送 46.19 元、充值 50.00 元）。
最近 7 天（2026-11-05 ~ 2026-11-11）花费 8.84 元，调用 1278 次，Token 1663.6 万（输入 1610.6 万 / 输出 52.9 万）。
主要模型：DeepSeek V4.1 Flash 1017 6.25 元（71%）、Qwen3.8 Flash 2.59 元（29%）、GLM-5.3 Flash 0.01 元（0%）。
```

## 决策记录面板

WebUI 扩展功能菜单中打开本插件页面即可查看最近的判断记录，包括判断结果、判断方式、图片完整转述和机器人回复，页面每 5 秒自动刷新。

后端接口：

- `GET /astrbot_plugin_smart_fish/decision/records`：最近的判断记录
- `POST /astrbot_plugin_smart_fish/decision/records/clear`：清空记录
