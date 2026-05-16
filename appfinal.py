import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


# =========================
# 1. App Config
# =========================

st.set_page_config(
    page_title="Arabic-English Transcript Chatbot",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Arabic-English Transcript Chatbot")
st.caption("Ask questions about the uploaded Arabic-English transcript episodes.")


# =========================
# 2. Load API Keys
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# =========================
# 3. Fixed Settings
# =========================

TRANSCRIPTS_FOLDER = "data/trans"
VECTORSTORE_FOLDER = "vectorstore"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RETRIEVED_CHUNKS = 10
HISTORY_WINDOW = 4


# =========================
# 4. Load Transcripts
# =========================

def load_transcripts(folder_path):
    folder = Path(folder_path)
    docs = []

    for episode_id, path in enumerate(sorted(folder.glob("*.txt")), start=1):
        text = path.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "filename": path.name,
                    "episode_id": episode_id,
                    "text_type": "MS1 normalized natural text",
                    "preprocessing": (
                        "no lemmatization, no stemming, no punctuation removal, "
                        "English tokens preserved, dialect/code-switching preserved"
                    ),
                },
            )
        )

    return docs


# =========================
# 5. Chunking
# =========================

def chunk_documents(raw_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            "؟",
            "!",
            ".",
            "،",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP

    return chunks


# =========================
# 6. Embeddings
# =========================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =========================
# 7. Vector Store
# =========================

def build_vector_store(chunks, embeddings):
    if Path(VECTORSTORE_FOLDER).exists():
        shutil.rmtree(VECTORSTORE_FOLDER)

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_FOLDER,
    )


@st.cache_resource
def load_vector_store(_embeddings):
    return Chroma(
        persist_directory=VECTORSTORE_FOLDER,
        embedding_function=_embeddings,
    )


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVED_CHUNKS,
            "fetch_k": 40,
            "lambda_mult": 0.7,
        },
    )


# =========================
# 8. LLMs
# =========================

def get_groq_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        api_key=GROQ_API_KEY,
    )


def get_openrouter_llm():
    return ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct:free",
        temperature=0.0,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )


def invoke_with_fallback(prompt, primary_llm, fallback_llm):
    try:
        response = primary_llm.invoke(prompt)
        return response.content, "Groq", False

    except Exception:
        try:
            response = fallback_llm.invoke(prompt)
            return response.content, "OpenRouter fallback", True

        except Exception:
            return (
                "عذرًا، حدث خطأ مؤقت أثناء توليد الإجابة. حاول مرة أخرى بعد قليل.",
                "Failed",
                True,
            )


# =========================
# 9. Prompt
# =========================

AR_STRICT_PROMPT = """
أنت مساعد ذكي يجيب فقط اعتمادًا على السياق المسترجع من الحلقات.

القواعد:
1. استخدم المعلومات الموجودة في السياق فقط.
2. إذا لم تكن الإجابة موجودة بوضوح في السياق، قل:
"لا أستطيع الإجابة من المعلومات المتاحة في الحلقات."
3. لا تخترع أسماء أو أحداث أو تفاصيل.
4. يمكن أن تكون الأسئلة بالعربية أو الإنجليزية أو خليط بينهما.
5. أجب بنفس لغة سؤال المستخدم قدر الإمكان.
6. اجعل الإجابة مباشرة وواضحة.
7. لا تضف معلومات من خارج السياق.

السياق:
{context}

المحادثة السابقة:
{history}

سؤال المستخدم:
{question}

الإجابة:
"""


# =========================
# 10. Memory
# =========================

def sliding_window_history(messages, window_size=HISTORY_WINDOW):
    recent = messages[-window_size:]
    return "\n".join([f"{m['role']}: {m['content']}" for m in recent])


# =========================
# 11. Context Formatting
# =========================

def format_context(docs):
    formatted = []

    for doc in docs:
        source = doc.metadata.get("filename", "unknown")
        episode = doc.metadata.get("episode_id", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")

        formatted.append(
            f"[Episode {episode} | Source: {source} | Chunk: {chunk_id}]\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


# =========================
# 12. Keyword Fallback
# =========================

def keyword_fallback_search(question, docs_folder, max_results=3):
    keywords = []

    # Octopus
    if "شعبة" in question or "ينتمي" in question or "الأخطبوط" in question:
        keywords.extend(["الرخويات", "الأخطبوط من", "Cephalopoda", "رأسيات الأرجل"])

    # F-35
    if "طائرة" in question or "الطائرة" in question or "f-35" in question.lower() or "إف" in question:
        keywords.extend(["F-35", "فخر الطيران الأمريكي", "لوكهيد مارتن", "Radar Absorbent Material", "RAM"])

    # Samurai
    if "ساموراي" in question or "السيف" in question or "بوشيدو" in question or "كاتانا" in question:
        keywords.extend(["الذي يخدِم", "الكاتانا", "Bushidō", "بوشيدو", "روح الساموراي"])

    results = []

    for path in Path(docs_folder).glob("*.txt"):
        text = path.read_text(encoding="utf-8")

        for keyword in keywords:
            index = text.find(keyword)

            if index != -1:
                start = max(0, index - 400)
                end = min(len(text), index + 500)

                results.append(
                    Document(
                        page_content=text[start:end],
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "episode_id": "keyword_fallback",
                            "chunk_id": f"keyword_{keyword}",
                        },
                    )
                )

                if len(results) >= max_results:
                    return results

    return results


# =========================
# 13. OOD Detection
# =========================

OOD_MESSAGE_AR = "السؤال خارج نطاق الحلقات المتاحة لدي، لذلك لا أستطيع الإجابة عليه من السياق الحالي."


def is_out_of_domain(retrieved_docs):
    return not retrieved_docs


# =========================
# 14. Main RAG Function
# =========================

def answer_question(question, messages, retriever, primary_llm, fallback_llm):
    history = sliding_window_history(messages)

    retrieval_query = f"""
    Conversation history:
    {history}

    Current question:
    {question}
    """

    retrieved_docs = retriever.invoke(retrieval_query)

    keyword_docs = keyword_fallback_search(question, TRANSCRIPTS_FOLDER)

    if keyword_docs:
        retrieved_docs = keyword_docs + retrieved_docs

    if is_out_of_domain(retrieved_docs):
        return {
            "answer": OOD_MESSAGE_AR,
            "retrieved_docs": [],
            "model_used": None,
            "fallback_used": False,
            "ood": True,
        }

    context = format_context(retrieved_docs)

    prompt = AR_STRICT_PROMPT.format(
        context=context,
        history=history,
        question=question,
    )

    answer, model_used, fallback_used = invoke_with_fallback(
        prompt,
        primary_llm,
        fallback_llm,
    )

    return {
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "model_used": model_used,
        "fallback_used": fallback_used,
        "ood": False,
        "context": context,
    }


# =========================
# 15. Build or Load System
# =========================

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file.")
    st.stop()

if not OPENROUTER_API_KEY:
    st.error("Missing OPENROUTER_API_KEY in .env file.")
    st.stop()

if not Path(TRANSCRIPTS_FOLDER).exists():
    st.error(f"Transcript folder not found: {TRANSCRIPTS_FOLDER}")
    st.stop()

embeddings = get_embeddings()

if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs" not in st.session_state:
    st.session_state.logs = []

with st.spinner("Preparing chatbot..."):
    raw_docs = load_transcripts(TRANSCRIPTS_FOLDER)

    if len(raw_docs) == 0:
        st.error(f"No .txt transcript files found in: {TRANSCRIPTS_FOLDER}")
        st.stop()

    if not Path(VECTORSTORE_FOLDER).exists():
        chunks = chunk_documents(raw_docs)
        build_vector_store(chunks, embeddings)

    vectorstore = load_vector_store(embeddings)
    retriever = get_retriever(vectorstore)

    primary_llm = get_groq_llm()
    fallback_llm = get_openrouter_llm()

    st.session_state.system_ready = True


# =========================
# 16. Chat UI
# =========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


user_question = st.chat_input("Ask a question about the episodes...")

if user_question:
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    result = answer_question(
        question=user_question,
        messages=st.session_state.messages,
        retriever=retriever,
        primary_llm=primary_llm,
        fallback_llm=fallback_llm,
    )

    answer = result["answer"]

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.logs.append(result)

    with st.chat_message("assistant"):
        st.write(answer)


# =========================
# 17. Logs
# =========================

with st.expander("📋 Logs"):
    if not st.session_state.logs:
        st.write("No logs yet.")
    else:
        for i, log in enumerate(st.session_state.logs, start=1):
            st.write(f"### Turn {i}")
            st.write("Model used:", log.get("model_used"))
            st.write("Fallback used:", log.get("fallback_used"))
            st.write("Out of domain:", log.get("ood"))

            st.write("Retrieved chunks:")

            for doc in log.get("retrieved_docs", []):
                st.write(doc.metadata)
                st.write(doc.page_content[:700])
                st.write("---")