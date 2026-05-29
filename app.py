"""
Literature RAG v5.2 (Web-based Streamlit Version)
==================================================
설치 필요:
    pip install streamlit pymupdf sentence-transformers numpy torch openai google-generativeai
"""

import streamlit as st
import sys, re, json as _json, urllib.request, pickle, gc, math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

# =============================================================================
# 기본 파라미터 및 가속 장치 설정
# =============================================================================
EMBED_MODEL = "BAAI/bge-m3"
INDEX_FILE = "index.pkl"
TOP_K = 15
MAX_PER_SRC = 3
FINAL_K = 8
MAX_TOKENS = 2000

LLM_LIST = ["OpenAI GPT", "Google Gemini", "Ollama (Local)"]
OAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
GEM_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"]
OLL_MODELS = ["llama3.2", "llama3.1", "mistral", "gemma2"]


def get_acceleration_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# ── [핵심 캐싱] 웹 서버 메모리에 딱 1번만 로드하여 동료들과 공유 ──
@st.cache_resource
def load_embedding_model():
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer
    dev = get_acceleration_device()
    return SentenceTransformer(EMBED_MODEL, device=dev)


def encode_chunks(model, chunks, batch=8):
    all_embs = []
    dev = get_acceleration_device()
    for i in range(0, len(chunks), batch):
        emb = model.encode(chunks[i:i + batch], batch_size=batch,
                           convert_to_numpy=True, show_progress_bar=False,
                           normalize_embeddings=True)
        all_embs.append(emb.astype(np.float32))
        gc.collect()
        if dev == "mps":
            torch.mps.empty_cache()
        elif dev == "cuda":
            torch.cuda.empty_cache()
    return np.vstack(all_embs)


def tokenize(text):
    return re.findall(r'[a-zA-Z0-9\.%°\-]+', text.lower())


# =============================================================================
# SimpleIndex (하이브리드 결합 아키텍처 그대로 유지)
# =============================================================================
class SimpleIndex:
    def __init__(self):
        self.embeddings = None
        self.documents = []
        self.sources = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.doc_term_freqs = []
        self.doc_frequencies = defaultdict(int)
        self.k1 = 1.5
        self.b = 0.75

    def add(self, docs, embs, source):
        embs = embs.astype(np.float32)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
        self.documents.extend(docs)
        self.sources.extend([source] * len(docs))
        for doc in docs:
            tokens = tokenize(doc)
            self.doc_lengths.append(len(tokens))
            tf = defaultdict(int)
            for t in tokens: tf[t] += 1
            self.doc_term_freqs.append(tf)
            for t in tf.keys(): self.doc_frequencies[t] += 1
        self.avg_doc_len = np.mean(self.doc_lengths) if self.doc_lengths else 0.0

    def search_vector(self, q_emb, k):
        if self.embeddings is None or len(self.documents) == 0: return []
        qn = (q_emb / (np.linalg.norm(q_emb) + 1e-9)).astype(np.float32)
        sims = self.embeddings @ qn
        dists = 1.0 - sims
        k = min(k, len(self.documents))
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        return [(i, float(dists[i])) for i in idx]

    def search_bm25(self, query_text, k):
        N = len(self.documents)
        if N == 0: return []
        q_tokens = tokenize(query_text)
        scores = np.zeros(N, dtype=np.float32)
        for token in q_tokens:
            df = self.doc_frequencies[token]
            if df == 0: continue
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for i in range(N):
                tf = self.doc_term_freqs[i].get(token, 0)
                if tf == 0: continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[i] / (self.avg_doc_len + 1e-9)))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        k = min(k, N)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(i, float(scores[i])) for i in top_indices if scores[i] > 0]

    def hybrid_search(self, query_text, q_emb, top_k):
        vector_res = self.search_vector(q_emb, k=top_k * 2)
        bm25_res = self.search_bm25(query_text, k=top_k * 2)
        rrf_constant = 60
        rrf_scores = defaultdict(float)
        for rank, (idx, _) in enumerate(vector_res, 1): rrf_scores[idx] += 1.0 / (rrf_constant + rank)
        for rank, (idx, _) in enumerate(bm25_res, 1): rrf_scores[idx] += 1.0 / (rrf_constant + rank)
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        return [(self.documents[i], self.sources[i], 1.0 - rrf_scores[i]) for i in sorted_indices]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "documents": self.documents, "sources": self.sources,
                         "doc_lengths": self.doc_lengths, "avg_doc_len": self.avg_doc_len,
                         "doc_term_freqs": self.doc_term_freqs, "doc_frequencies": self.doc_frequencies}, f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f: d = pickle.load(f)
        idx = cls()
        idx.embeddings = d["embeddings"]
        idx.documents = d["documents"]
        idx.sources = d["sources"]
        idx.doc_lengths = d.get("doc_lengths", [])
        idx.avg_doc_len = d.get("avg_doc_len", 0.0)
        idx.doc_term_freqs = d.get("doc_term_freqs", [])
        idx.doc_frequencies = d.get("doc_frequencies", defaultdict(int))
        return idx


# =============================================================================
# PDF 및 LLM 백엔드 로직
# =============================================================================
def extract_pdf(f_bytes, filename, max_chars=150_000) -> str:
    try:
        import fitz
        doc = fitz.open(stream=f_bytes, filetype="pdf")
        parts, total = [], 0
        for page in doc:
            t = page.get_text()
            if t.strip():
                parts.append(t.strip());
                total += len(t)
                if total >= max_chars: break
        doc.close()
        text = "\n\n".join(parts)
        for m in ["References\n", "REFERENCES\n", "Bibliography\n"]:
            i = text.find(m);
            if i != -1: text = text[:i]
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        if len(text) / max(len(parts), 1) < 50: return "__GARBAGE__"
        return text
    except Exception:
        return ""


def chunk(text: str) -> list:
    SECTION_RE = re.compile(
        r'^\s*\d+\.?\d*\.?\s*(Abstract|Introduction|Experimental|Method|Result|Discussion|Conclusion|Preparation|Simulation|Summary)',
        re.IGNORECASE | re.MULTILINE)
    if len(text) > 150_000: text = text[:150_000]
    secs = [m.start() for m in SECTION_RE.finditer(text)]
    chunks, start, prev = [], 0, -1
    while start < len(text):
        if start == prev: start += 700; continue
        prev = start;
        end = min(start + 800, len(text))
        if end < len(text):
            mid = start + 400
            sb = next((s for s in secs if mid <= s < end), None)
            if sb:
                end = sb
            else:
                b = text.rfind("\n\n", mid, end)
                if b != -1: end = b + 2
        c = text[start:end].strip()
        if len(c) >= 80: chunks.append(c)
        start = end - 100
    return chunks


def call_llm(prompt, provider, model, api_key=""):
    if provider == "OpenAI GPT":
        import openai
        r = openai.OpenAI(api_key=api_key).chat.completions.create(
            model=model, messages=[{"role": "system", "content": "You are a scientific assistant."},
                                   {"role": "user", "content": prompt}], max_tokens=MAX_TOKENS, temperature=0.1)
        return r.choices[0].message.content
    elif provider == "Google Gemini":
        import google.generativeai as gai
        gai.configure(api_key=api_key)
        return gai.GenerativeModel(model, generation_config=gai.types.GenerationConfig(max_output_tokens=MAX_TOKENS,
                                                                                       temperature=0.1)).generate_content(
            prompt).text
    else:
        payload = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r:
            return _json.loads(r.read())["response"]


# =============================================================================
# Streamlit Web UI 구성
# =============================================================================
st.set_page_config(page_title="NdFeB RAG 웹 분석기", layout="wide")
st.title("⚗️ NdFeB Literature Hybrid RAG Web v5.2")
st.caption("실험과학자를 위한결정립계 확산 공정(GBDP) 문헌 분석 시스템")

# 왼쪽 사이드바 설정 영역
with st.sidebar:
    st.header("🔑 API 및 모델 구성")
    provider = st.selectbox("LLM Provider", LLM_LIST)

    if provider == "OpenAI GPT":
        model = st.selectbox("Model", OAI_MODELS)
        api_key = st.text_input("OpenAI API Key", type="password")
    elif provider == "Google Gemini":
        model = st.selectbox("Model", GEM_MODELS)
        api_key = st.text_input("Gemini API Key", type="password")
    else:
        model = st.selectbox("Model", OLL_MODELS)
        api_key = ""
        st.info("동료 PC에서 'ollama serve'가 구동 중이어야 호환됩니다.")

    st.markdown("---")
    st.header("📁 지식 베이스(인덱스) 관리")

    # 서버 내부의 고정된 index.pkl 자동 로드 시도
    idx_path = Path(INDEX_FILE)
    if idx_path.exists():
        if 'index' not in st.session_state:
            st.session_state.index = SimpleIndex.load(idx_path)
        st.success(f"✓ 통합 지식 베이스 로드 완료 ({st.session_state.index.count()} chunks)")
    else:
        st.warning("⚠️ index.pkl 파일을 서버 스크립트와 같은 폴더에 배치해 주세요.")

# 메인 화면 레이아웃 분할
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("❓ 문헌 질의 (Question)")
    user_query = st.text_area("영구자석 공정 관련 질문을 입력하세요:",
                              placeholder="예: What compositions are used in GBDP to enhance coercivity?", height=100)

    # 예시 질문 선택 드롭다운
    selected_example = st.selectbox("💡 빠른 예시 질문 선택", ["-- 선택하세요 --"] + EXAMPLES)
    if selected_example != "-- 선택하세요 --":
        user_query = selected_example

    if st.button("▶ 질의 실행 (Ask)", type="primary"):
        if not user_query.strip():
            st.error("질문 내용을 입력해 주세요.")
        elif 'index' not in st.session_state:
            st.error("인덱스 지식 베이스가 활성화되지 않았습니다.")
        elif provider in ["OpenAI GPT", "Google Gemini"] and not api_key.strip():
            st.error("선택한 LLM의 API Key를 입력해 주세요.")
        else:
            with st.spinner("하이브리드 RRF 매칭 및 인용 분석 중..."):
                try:
                    # 1. 전역 가속 임베딩 로더 호출
                    emb_model = load_embedding_model()
                    q_emb = emb_model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)[0]

                    # 2. 융합 앙상블 검색
                    results = st.session_state.index.hybrid_search(user_query, q_emb, TOP_K)

                    if not results:
                        st.warning("관련 구절을 찾지 못했습니다.")
                    else:
                        src_cnt = defaultdict(int);
                        sel = []
                        for doc, src, score in results:
                            if src_cnt[src] < MAX_PER_SRC:
                                sel.append((doc, src, score));
                                src_cnt[src] += 1
                            if len(sel) >= FINAL_K: break

                        # 컨텍스트 조립
                        grouped = defaultdict(list)
                        for doc, src, _ in sel: grouped[src].append(doc)
                        ctx_parts = [f"=== Source: {s} ===\n" + "\n\n[...]\n\n".join(cks) for s, cks in grouped.items()]
                        context = "\n\n---\n\n".join(ctx_parts)

                        # 프롬프트 조립 (언어 룰 강제 결합)
                        prompt = (
                            "You are a scientific literature analyst for NdFeB magnets.\n"
                            "RULES:\n"
                            "1. Answer ONLY from the passages below.\n"
                            "2. Do NOT use your own knowledge.\n"
                            "3. Cite source filename after every claim.\n"
                            "4. ANSWER LANGUAGE RULE: You MUST respond in the SAME LANGUAGE as the user's question. "
                            "If the question is in Korean, reply in Korean. If in English, reply in English.\n\n"
                            f"Question: {user_query}\n\nPassages:\n{context}\n\nAnswer:"
                        )

                        # 3. LLM 결과 수신
                        answer = call_llm(prompt, provider, model, api_key)

                        st.session_state.answer = answer
                        st.session_state.sources = list(src_cnt.keys())
                        st.session_state.debug_logs = [f"rrf_score={s:.4f} [{src}] {d[:60]}..." for d, src, s in sel]
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {str(e)}")

    # 결과 표기 영역
    if 'answer' in st.session_state:
        st.subheader("📝 인용 검증 답변 (Answer)")
        st.info(st.session_state.answer)

with col_right:
    st.subheader("📌 매칭 문헌 출처 (Sources)")
    if 'sources' in st.session_state and st.session_state.sources:
        for src in st.session_state.sources:
            st.markdown(f"📄 `{src}`")
    else:
        st.write("인용된 논문이 여기에 표시됩니다.")

    st.markdown("---")
    st.subheader("🔍 매칭 로그 디버그")
    if 'debug_logs' in st.session_state:
        for log in st.session_state.debug_logs:
            st.caption(log)