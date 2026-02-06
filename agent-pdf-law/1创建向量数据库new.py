from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os

def create_vector_database(file_name: str, db_name: str):
    """
    创建向量数据库的函数。
    
    参数:
    - file_name (str): 输入文件名，包含文档内容的路径。
    - db_name (str): 向量数据库名称，保存的目录路径。
    
    返回:
    - vectorstore (Chroma): 创建的向量数据库对象。
    """
    
    # 1. 读取文件内容
    with open(file_name, 'r', encoding='utf-8') as file:
        docs = file.read()

    # 2. 使用字符递归分割器将文档进行分割
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    docs = text_splitter.split_text(docs)

    # 3. 使用 Ollama 的嵌入模型
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        base_url="http://localhost:11434",
    )

    # 4. 创建向量数据库
    documents = [Document(page_content=doc) for doc in docs]
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=f"./{db_name}"
    )

    return vectorstore

if __name__ == "__main__":
    file_name = "./test.md"
    db_name = "chroma_db_for_LLM2"  # 确保目录名中没有空格，以避免潜在的问题
    # 调用函数创建并保存向量数据库
    vectorstore = create_vector_database(file_name, db_name)
    
    # 测试
    query = "進階元件類偏好設定？"
    results = vectorstore.similarity_search(query, k=3)
    print(results)