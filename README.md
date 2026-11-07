# Customer Email Automation with Python and Streamlit

这是上周 Dify Customer Email Automation 的 Python 重建版。输入客户姓名、邮件主题、正文，先分类，再从对应公司政策中检索，最后让 LLM 根据检索结果撰写回复草稿。

## 启动

在 Terminal 中运行：

```bash
cd customer-email-automation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

打开终端显示的本地地址，通常是 http://localhost:8501。主页面仅显示邮件输入及生成的回复；连接配置及历史记录位于设置中，知识库在后台检索，不展示政策原文。展开侧栏的 Settings → Connection，输入 HKUST GenAI 订阅的 Primary key 或 Secondary key（任一个）。后台默认使用 `gpt-5.6-luna` 和 `text-embedding-3-small`，页面无需选择模型；如需更改，可修改 `.env` 中的 `CHAT_MODEL` 和 `EMBEDDING_MODEL`。应用使用学校网关 `https://hkust.azure-api.net/hkust-genai/v1` 并自动添加学校要求的 `api-key` 请求头。无需填写 Azure deployment 或 API version。

也可以把 `.env.example` 复制为 `.env`，在其中填入自己的配置。不要提交或分享含密钥的文件。没有密钥时也能查看界面及当前会话已有的历史记录，但不能分类、生成向量或生成模型回复。

## 与原流程的对应

| 原 Dify 节点 | Python 重建 |
| --- | --- |
| USER INPUT | `app.py` 的 Streamlit 表单：name、subject、body |
| DETERMINE EMAIL ISSUE | `rag.py` 中 `classify()`，调用聊天模型 |
| 两个 KNOWLEDGE RETRIEVAL | `load_chunks()` 读取两份 Word 文档，`embed()` 生成向量，`retrieve()` 在对应类别中按余弦相似度取 Top K |
| LLM / LLM 2 | `generate()` 共用一个回复函数，接收所选分支的政策片段 |
| OUTPUT | 主页面显示回复并下载 TXT；Settings → History 保存当前会话中成功生成的邮件及回复 |

流程可概括为：

```text
两份政策 Word → 文本清理 → FAQ/政策分块 → Embeddings → 会话内向量索引
客户姓名 + 主题 + 正文 → LLM 分类 → 主题和正文向量化
→ 在选定类别中检索 Top K → 检索文本加入提示词 → LLM 回复 → 页面展示
```

Retrieval 是从政策中寻找相关片段；Augmentation 是把片段加入提示词；Generation 是让模型依据片段写回复。这里只用 NumPy 做向量检索：两份小文档不需要额外的向量数据库。

## 知识库与修改说明

`data/` 保留课程目录中的两份原始政策文件：Shipping / Delivery Issues 和 Refund / Return / Exchange。导出的 `.ifpkg` 只有流程配置和知识库 ID，不包含 Dify 的向量索引，所以需要重新建立索引。

- 保留原来的两个业务类别。额外处理 mixed（同时涉及物流和退款）及 other（超出范围），避免所有邮件被强制分入两类。
- 原流程按类别名称检索，重建版按邮件主题和正文检索；姓名只用于称呼。
- 从 Word 中提取政策与 FAQ，移除 Customer Service Templates 部分，避免将示例“退款已处理”当成订单事实。原文中的 RTF 转义已解码，例如 `1\\'962` 恢复为 `1–2`。
- 使用 OpenAI embeddings 加余弦排序，未复刻原流程的 Qwen reranker。相似度不是置信概率；默认阈值 0.20 只是演示起点，需要用样例校准。
- 向量保存在当前 Streamlit 会话内；政策、密钥或 embedding model 改变会重建，刷新会话后可能需要重新计算。邮件和密钥不写入应用文件。
- 没有检索结果时返回固定英文转人工回复，不让 LLM 编造政策。即使有检索结果，LLM 回复仍应人工核对。
- 没有真实订单系统，不承诺退款已批准、包裹已发货或已经执行操作。页面只生成草稿。

## 使用与历史记录

1. 在主页面填写客户姓名、主题及邮件正文，然后点击 Generate reply。
2. 物流场景可输入 `My package is late. How can I track it?`；退货场景可输入 `I received my order 10 days ago and would like to return it for a refund.`。
3. 成功生成后，Settings → History 按最新在前显示邮件主题和香港时间。点击主题即可恢复原邮件和对应回复，不会重新调用 API。失败的请求不会创建历史记录。
4. 历史记录仅保留在当前 Streamlit 会话内，不写入磁盘，不包含 API key。浏览器刷新、新建会话或服务重启后不会保留。
5. Top K 固定为 4，相似度阈值固定为 0.20；主页面不显示模型、检索参数及政策原文。

这些是建议演示场景，不代表真实 API 调用已全部通过。安装后的本地测试不消耗 API 配额：

```bash
python -m unittest discover -s tests -v
```

测试验证原政策提取、分支隔离、余弦排序、空检索、生成提示词的上下文传递，以及 Streamlit 表单、错误提示、历史记录保存及恢复；模型使用替身。最终分类及回复质量需要在配置真实 API 后验证。

## 官方参考

- [HKUST API 连接说明](https://itso.hkust.edu.hk/services/general-it-services/generative-ai-tools/hkust-genai-api-service/integration-ai-tools)
- [HKUST 支持的模型](https://itso.hkust.edu.hk/services/general-it-services/generative-ai-tools/hkust-genai-api-service/supported-models)

- [OpenAI vector embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI chat completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
