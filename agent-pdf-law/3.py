# 导入核心库
import os
from PyPDF2 import PdfReader

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# ===================== 极简配置项（仅需修改这2个！）=====================
# 脚本所在目录（避免 Windows 反斜杠转义问题）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 1. 本地法务PDF知识库路径：默认用「考核素材_法务文档」文件夹（加载该目录下全部 PDF）
PDF_PATH = os.path.join(_SCRIPT_DIR, "考核素材_法务文档")
# 2. 本地Chroma向量库存储路径（首次自动创建，后续无需重新解析PDF）
CHROMA_DB_PATH = os.path.join(_SCRIPT_DIR, "chroma_legal_db")
# ===================== 固定配置（适配Ollama）=====================
# 若出现 "unable to allocate CPU buffer" 说明内存不足，请改用更小模型（如 1.5b）或关闭其他程序
OLLAMA_MODEL = "qwen2.5:1.5b"  # 内存紧张用 1.5b；内存充足可改为 qwen2.5:3b 或 qwen2.5:7b
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 默认本地地址
EMBEDDING_MODEL = "bge-m3"  # 使用 Ollama 的 bge-m3 嵌入（需先拉取：ollama pull bge-m3）
RETRIEVE_TOP_K = 3  # 检索最相关的 3 个知识库片段


# ===================== 步骤1：解析本地PDF，提取文本 =====================
def load_pdf_text(pdf_path):
    """解析PDF，支持单文件/多文件/文件夹，过滤无文本的PDF"""
    all_text = ""
    if os.path.isfile(pdf_path) and pdf_path.endswith(".pdf"):
        pdf_files = [pdf_path]
    elif os.path.isdir(pdf_path):
        pdf_files = [os.path.join(pdf_path, f) for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    else:
        raise ValueError("请输入有效的PDF文件/文件夹路径！")

    for pdf_file in pdf_files:
        try:
            reader = PdfReader(pdf_file)
            page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
            if not page_texts:
                print(f"警告：{pdf_file} 无可提取文本（可能是图片型/加密PDF）")
                continue
            text = "\n".join(page_texts)
            all_text += text + "\n\n"
            print(f"成功解析PDF：{pdf_file}，提取文本长度：{len(text)}字符")
        except Exception as e:
            print(f"解析PDF失败：{pdf_file}，错误：{e}")
    return all_text


# ===================== 步骤2：法务文本切分（保证法条语义完整）=====================
def split_legal_text(raw_text):
    """针对中文法务文本的切分策略，优先按法条分隔符切分"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个片段500字符（适配法条长度）
        chunk_overlap=50,  # 片段重叠50字符，保证上下文衔接
        length_function=len,
        separators=["\n\n", "\n", "。", "；", "，", "、", "(", ")", "【", "】"]  # 法务文本专属分隔符
    )
    chunks = text_splitter.split_text(raw_text)
    print(f"文本切分完成，生成 {len(chunks)} 个法务知识片段")
    return chunks


# ===================== 步骤3：初始化本地Chroma向量库 =====================
def init_legal_vector_db(chunks, embeddings, db_path):
    """首次创建向量库，后续直接加载，无需重复解析PDF"""
    if os.path.exists(db_path) and len(os.listdir(db_path)) > 0:
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
        try:
            n = vectordb._collection.count()
            print(f"成功加载本地向量库：{db_path}，含 {n} 个知识片段")
        except Exception:
            print(f"成功加载本地向量库：{db_path}")
    else:
        vectordb = Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=db_path)
        if hasattr(vectordb, "persist"):
            vectordb.persist()
        print(f"成功创建并持久化向量库：{db_path}，存入 {len(chunks)} 个知识片段")
    return vectordb


# ===================== 步骤4：构建法务RAG问答链（核心）=====================
def build_legal_rag_chain():
    """整合所有流程，构建端到端法务RAG问答链"""
    # 1. 加载并解析PDF
    raw_text = load_pdf_text(PDF_PATH)
    if not raw_text:
        raise Exception("未从任何PDF中提取到有效文本，请检查PDF文件！")
    # 2. 切分法务文本
    text_chunks = split_legal_text(raw_text)
    # 3. 初始化向量化模型（Ollama bge-m3，与项目其他脚本一致）
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    print(f"成功初始化向量化模型：{EMBEDDING_MODEL}")
    # 4. 初始化向量库
    vectordb = init_legal_vector_db(text_chunks, embeddings, CHROMA_DB_PATH)
    # 5. 初始化 Ollama 对话模型
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,  # 极低温度，保证法务回答准确、无编造
    )
    print(f"成功对接 Ollama 本地模型：{OLLAMA_MODEL}")
    # 6. 你的专属法务提示词模板（一字未改，严格按要求）
    LEGAL_PROMPT = """
你是专业的法务智能助手，严格依据以下提供的法务PDF知识库内容回答问题，不得编造任何未提及的法律条文或信息：
已知法务知识库内容：{context}
用户的法律问题：{question}
请给出**详细、准确、符合法律条文原文**的回答，条理清晰优先分点说明：
    """.strip()
    prompt = PromptTemplate(template=LEGAL_PROMPT, input_variables=["context", "question"])
    # 7. 构建检索器（相似性检索）
    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVE_TOP_K})

    # 定义核心问答函数
    def legal_qa(question):
        # 检索相关法务知识库
        context_docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in context_docs])
        # 拼接提示词，调用 Ollama 模型生成回答
        input_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(input_prompt)
        answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
        return answer, context_docs

    print(f"✅ 法务 RAG 智能助手构建完成（基于 Ollama/{OLLAMA_MODEL}），可开始提问！")
    return legal_qa


# ===================== 主函数：交互式问答 =====================
if __name__ == "__main__":
    print("=" * 60)
    print(f"🔍 本地法务 RAG 智能助手（Ollama/{OLLAMA_MODEL}）启动中...")
    print("=" * 60)
    # 构建问答链
    legal_qa = build_legal_rag_chain()
    print("=" * 60)
    print("💡 输入法律问题即可查询（输入q/quit/退出 关闭助手）")
    print("=" * 60)

    # 持续交互式问答
    while True:
        question = input("\n请输入你的法律问题：")
        if question.lower() in ["q", "quit", "退出"]:
            print("👋 法务智能助手已关闭，感谢使用！")
            break
        if not question.strip():
            print("⚠️  问题不能为空，请重新输入！")
            continue
        # 生成回答并打印
        try:
            answer, context_docs = legal_qa(question)
            print("\n" + "=" * 40 + " 法务助手专业回答 " + "=" * 40)
            print(answer)
            # 打印检索到的知识库原文（便于验证回答准确性，法务场景必备）
            print("\n" + "=" * 40 + " 检索到的法务知识库原文 " + "=" * 40)
            for i, doc in enumerate(context_docs):
                doc_content = doc.page_content
                display_content = doc_content[:600] + "..." if len(doc_content) > 600 else doc_content
                print(f"\n📚 相关知识片段 {i + 1}：")
                print(display_content)
        except Exception as e:
            print(f"❌ 回答生成失败，错误：{e}")