# MS3 Technical Report: Arabic-English RAG Chatbot

## 1. Data and Text Representation
The system uses three original MS1 transcript episodes: Octopus, F-35, and Samurai. The transcripts are stored as `.txt` files under `data/trans`. The text is loaded as normalized natural text without lemmatization, stemming, punctuation removal, or English-token removal. This preserves Arabic dialectal variation and Arabic-English code-switching such as `F-35`, `RAM`, `Cephalopoda`, and `Bushidō`.

## 2. Chunking and Traceability
The transcripts are split using LangChain's `RecursiveCharacterTextSplitter`. The final chunking configuration uses a chunk size of 500 characters and an overlap of 100 characters. This setting was selected to keep retrieved passages focused while preserving continuity across transcript segments. Each chunk stores metadata including `filename`, `source`, `episode_id`, `chunk_id`, `chunk_size`, and `chunk_overlap`, making every answer traceable to its source episode.

## 3. Embeddings and Vector Store
The system uses `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` through `HuggingFaceEmbeddings`. This multilingual model supports semantic retrieval over Arabic and English mixed text. Chunks are indexed in Chroma, a local vector database. Retrieval uses Maximal Marginal Relevance (MMR) with `k=10`, `fetch_k=40`, and `lambda_mult=0.7` to balance relevance and diversity.

## 4. RAG Generation and Prompting
The chatbot uses a strict Arabic grounding prompt. The prompt instructs the model to answer only from retrieved context and to reject unsupported questions with a fixed message. This reduces hallucination and improves controllability. The system was designed to support different prompt styles; the final demo uses the strict Arabic prompt because it performed best for Arabic transcript questions.

## 5. Multi-turn Memory
The chatbot supports multi-turn interaction using Streamlit `session_state`. A sliding-window memory strategy keeps the last four messages. The current question is combined with recent conversation history during retrieval, allowing the system to resolve follow-up questions such as "وما نقطة قوتها؟" after the user asks about the F-35.

## 6. Robustness and Fallback
The primary LLM is `llama-3.1-8b-instant` through Groq. The fallback LLM is `meta-llama/llama-3.1-8b-instruct:free` through OpenRouter. If the primary API fails, the system automatically tries the fallback model. If both fail, it returns a safe error message instead of crashing.

## 7. Out-of-Domain Handling
The system uses retrieval-based out-of-domain detection. If no relevant transcript context is retrieved, the chatbot returns: "السؤال خارج نطاق الحلقات المتاحة لدي، لذلك لا أستطيع الإجابة عليه من السياق الحالي." The strict prompt also prevents the model from answering when the retrieved context does not clearly support the answer.

## 8. Interface and Logs
The Streamlit interface provides a simple chat experience. It displays the conversation and includes an expandable logs section showing the model used, fallback status, OOD status, retrieved chunk metadata, and retrieved text. These logs help verify that answers are grounded in transcript context.

## 9. Limitations
The system depends on retrieval quality. Very broad questions may fail if they do not identify the intended episode. OOD detection is simple and can be improved using similarity thresholds. Automatic batch evaluation can be added using the MS2 QA JSON files, while the final demo focuses on a stable interactive RAG system with visible retrieval logs.
