from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


class RAGChat:
    """
    实现基于 RAG (检索增强生成) 的对话系统
    """

    def __init__(self, db_path: str):
        """
        初始化 RAG 对话系统

        参数:
        - db_path (str): 向量数据库的路径
        """
        # 1. 初始化嵌入模型
        self.embeddings = OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434"
        )

        # 2. 加载向量数据库
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings
        )

        # 3. 初始化 LLM
        self.llm = ChatOllama(
            model="qwen2.5:7b",
            base_url="http://localhost:11434",
            temperature=0.7
        )

        # 4. 设置对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"  # 明确指定存储 "answer" 键的值
        )

        # 添加提示词模板
        template = """你是一个智能助手，请根据以下已知信息回答问题：
        已知信息：{context}
        问题：{question}
        请给出详细且准确的回答："""
        
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        # 5. 创建对话链
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer",
            combine_docs_chain_kwargs={"prompt": self.prompt}
        )

    def chat(self, query: str) -> str:
        """
        进行对话交互

        参数:
        - query (str): 用户输入的问题

        返回:
        - str: AI 的回答
        """
        result = self.qa_chain.invoke({"question": query})
        
        # 打印检索到的内容
        # print("\n检索到的相关内容：")
        # for i, doc in enumerate(result["source_documents"], 1):
        #     print(f"文档 {i}: {doc.page_content}")
        #     print("-" * 50)
        
        return result["answer"]


# 使用示例
if __name__ == "__main__":
    # 初始化 RAG 对话系统
    rag_chat = RAGChat("./chroma_db_for LLM2")

    # 测试对话
    while True:
        query = input("\n请输入您的问题（输入 'q' 退出）: ")
        if query.lower() == 'q':
            break
        try:
            response = rag_chat.chat(query)
            print("\nAI 回答:", response)
        except Exception as e:
            import traceback
            print(f"发生错误: {str(e)}")
            traceback.print_exc()

