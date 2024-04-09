## ai-knowledge-base
基于langchain实现的增强检索问答知识库

## 功能
加载本地文档(支持txt、pdf、docx、csv等)存储到向量数据库用于知识问答，优先从本地向量数据库检索答案，找不到答案时从互联网搜索

## 选项
- verbose
开启详细输出
- multi
使用multi_query，从不同角度发散提问，耗token
- use_persist
文档持久化到本地向量库
- document_path
文档目录，支持txt、pdf、docx、csv等

## 版本
- python 3.11.9
- langchain==0.1.12
- langchain-community==0.0.28
- chromadb==0.4.24

## GPT KEY
可以从[gpt4free](https://github.com/chatanywhere/GPT_API_free)申请免费版本,免费版本仅支持gpt-3.5-turbo
