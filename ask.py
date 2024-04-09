from typing import Any, Dict, Optional, List
from langchain_core.documents import Document
from langchain_community.llms.openai import OpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.language_models import BaseLanguageModel
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.agents import create_react_agent,AgentExecutor
from langchain.agents.tools import Tool
from functools import partial
from langchain import hub
from langchain.agents import load_tools
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.indexes import VectorstoreIndexCreator
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.storage import InMemoryStore
import os
from dotenv import load_dotenv

# 免费key只能用gpt-3.5-turbo
os.environ["OPENAI_API_KEY"] = "KEY"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.com.cn/v1"
load_dotenv()

# from langchain_core.vectorstores import VectorStore
# 重写VectorstoreIndexCreator相关类方法，使用MultiQueryRetriever
class VectorStoreIndexWrapper1(VectorStoreIndexWrapper):
    verbose = False
    multi = False
    def __init__(self,vectorstore,verbose,multi):
        super().__init__(vectorstore=vectorstore)
        self.verbose = verbose
        self.multi = multi
    def query(
            self,
            question: str,
            llm: Optional[BaseLanguageModel] = None,
            retriever_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs: Any,
    ):
        """Query the vectorstore."""
        llm = llm or OpenAI(temperature=0)
        retriever_kwargs = retriever_kwargs or {}
        if self.multi:
            from langchain.retrievers.multi_query import MultiQueryRetriever  # MultiQueryRetriever工具
            if self.verbose:
                # 设置查询的日志记录
                import logging
                logging.basicConfig()
                logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
            retriever = MultiQueryRetriever.from_llm(retriever=self.vectorstore.as_retriever(**retriever_kwargs),llm=llm)
        else:
            retriever = self.vectorstore.as_retriever(**retriever_kwargs)
        from langchain.prompts import PromptTemplate
        # Build prompt
        template = """<指令>要求Final Answer直接返回context已知内容。</指令><已知内容>{context}</已知内容><问题>{question}</问题>。"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template, )
        chain = RetrievalQA.from_chain_type(
            llm, retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}, verbose=self.verbose
        )
        return chain.invoke({"query": question})


class VectorstoreIndexCreator1(VectorstoreIndexCreator):
    verbose = False
    multi = False
    def __init__(self,vectorstore_cls,embedding,text_splitter,vectorstore_kwargs,verbose,multi):
        super().__init__(vectorstore_cls=vectorstore_cls,embedding=embedding,text_splitter=text_splitter,vectorstore_kwargs=vectorstore_kwargs)
        self.verbose = verbose
        self.multi = multi
    def from_documents(self, documents: List[Document]) -> VectorStoreIndexWrapper1:
        """Create a vectorstore index from documents."""
        sub_docs = self.text_splitter.split_documents(documents)
        vectorstore = self.vectorstore_cls.from_documents(
            sub_docs, self.embedding, **self.vectorstore_kwargs
        )
        return VectorStoreIndexWrapper1(vectorstore=vectorstore,verbose=self.verbose,multi=self.multi)


class QA:
    def __init__(self, document_path='./files/', use_persist=True, verbose=False, multi=False):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000
        )
        # 文档目录，支持多文件
        self.document_path = document_path
        # 将文档进行向量持久化到本地
        self.use_persist = use_persist
        self.verbose = verbose
        self.multi = multi
    def create_directory_loader(self,file_type):
        # https://zhuanlan.zhihu.com/p/675798969
        loaders = {
            '.txt': TextLoader,
            '.pdf': PyMuPDFLoader,
            '.xml': UnstructuredXMLLoader,
            '.csv': CSVLoader,
            '.docx': Docx2txtLoader,
            '.pptx': UnstructuredPowerPointLoader,
        }

        return DirectoryLoader(
            path=self.document_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type], # 使用加载数据的方式
            show_progress=True,  # 显示进度
            use_multithreading=True,  # 使用多线程
            silent_errors=True,  # 遇到错误继续
        )
    def load_documents(self):
        pdf_loader = self.create_directory_loader('.pdf')
        xml_loader = self.create_directory_loader('.xml')
        txt_loader = self.create_directory_loader('.txt')
        csv_loader = self.create_directory_loader('.csv')
        docx_loader = self.create_directory_loader('.docx')
        pptx_loader = self.create_directory_loader('.pptx')

        loaders = [txt_loader,pdf_loader,csv_loader,xml_loader,docx_loader,pptx_loader]

        return loaders
    def create_index(self):
        # 创建一个InMemoryStore的实例
        store = InMemoryStore()
        set_llm_cache(InMemoryCache())
        # 创建一个OpenAIEmbeddings的实例，这将用于实际计算文档的嵌入
        underlying_embeddings = OpenAIEmbeddings()
        # 创建一个CacheBackedEmbeddings的实例。这将为underlying_embeddings提供缓存功能，嵌入会被存储在上面创建的InMemoryStore中。
        # 为缓存指定了一个命名空间，以确保不同的嵌入模型之间不会出现冲突。
        embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings,  # 实际生成嵌入的工具
            store,  # 嵌入的缓存位置
            namespace=underlying_embeddings.model  # 嵌入缓存的命名空间
        )
        # 把文档在向量数据库持久化
        vectorstore_persist_directory = "VectorData"
        if self.use_persist and os.path.exists(vectorstore_persist_directory):
            print('从本地加载向量库数据')
            vectorstore = Chroma(persist_directory=vectorstore_persist_directory, embedding_function=embedder)
            index = VectorStoreIndexWrapper1(vectorstore=vectorstore,verbose=self.verbose,multi=self.multi)
        else:
            loaders = self.load_documents()

            if self.use_persist:
                print('文档使用向量库持久化到本地')
                index = VectorstoreIndexCreator1(
                    vectorstore_cls=Chroma,
                    embedding=embedder,
                    # text_splitter=CharacterTextSplitter(chunk_size=200, chunk_overlap=50),
                    text_splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10),
                    vectorstore_kwargs={"persist_directory": vectorstore_persist_directory},
                    verbose=self.verbose,
                    multi=self.multi
                ).from_loaders(loaders)
            else:
                print('不使用向量库持久化')
                index = VectorstoreIndexCreator1(verbose=self.verbose, multi=self.multi).from_loaders(loaders)

        return index

    def agent_init(self,index):
        # 添加工具
        tools_c = [
            Tool(
                name="Vector_search",
                func=partial(index.query, llm=self.llm, verbose=self.verbose, retriever_kwargs={"search_kwargs": {"k": 1}}),
                description="Utilize this tool when the user asks for similarity search and when he explicitly asks you to use Vector_search. Input should be in the form of a question containing full context",
            ),
        ]
        # 引用其他公共工具
        tools_p = load_tools([
            "llm-math", # 计算
            "ddg-search",  # duckduckgo引擎
        ], llm=self.llm)
        tools = tools_c + tools_p
        prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=self.verbose)

        return agent_executor

    def query(self, query):
        query = f'优先使用Vector_search，如果没有检索到答案，再使用ddg-search。output必须使用中文。问题：{query}'
        index = self.create_index()
        agent_executor = self.agent_init(index)
        result = agent_executor.invoke({"input": query})
        return result

if __name__ == '__main__':
    res = QA(verbose=True,multi=True).query("公司的wifi是？")
    print(res.get('output'))

