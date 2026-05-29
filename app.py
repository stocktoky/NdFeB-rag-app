"""
Literature RAG v5.4 (Web-based Streamlit Version)
==================================================
설치 필요:
    pip install streamlit pymupdf sentence-transformers numpy torch openai google-generativeai
"""

import streamlit as st
import sys, re, json as _json, urllib.request, pickle, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

# =============================================================================
# 글로벌 환경 변수 및 고정 파라미터 수립
# =============================================================================
EMBED_MODEL   = "all-MiniLM-L6-v2"
INDEX_FILE    = "index.pkl"
TOP_K         = 15
MAX_PER_SRC   = 3
FINAL_K       = 8
MAX_TOKENS    = 2000

LLM_LIST   = ["OpenAI GPT", "Google Gemini", "Ollama (Local)"]
OAI_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
GEM_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-2.0-flash"]
OLL_MODELS = ["llama3.2", "llama3.1", "mistral", "gemma2"]

EXAMPLES = [
    "What is NdFeB permanent magnet?",
    "What are the methods for making source materials for GBDP?",
    "How are diffusion sources prepared for grain boundary diffusion?",
    "What is PCAG alloy and how is it prepared?",
    "What compositions are used in GBDP to enhance coercivity?",
    "How does CILFM affect Pr-rich shell formation?",
    "What role does TaF5 play in the two-step GBDP?",
    "GBDP에 대해서 설명해줘",
    "입계계면 확산 공정에 대해서 설명해줘",
    "GBDP 효과를 통해서 보자력이 향상되는 이유를 설명해줘",
    "please explain the advantage of the GBDP in terms of an increase of coercivity",
    "NdFeB 영구자석의 소재의 중희토류 이슈에 대해서 설명해줘",
]

# =============================================================================
# 글로벌 자원 가속화 및 캐싱 레이어
# =============================================================================
@st.cache_resource
def load_embedding_model():
    """서버 가동 시 메모리에 임베딩 가중치를 단 1회만 적재하도록 강제 보장"""
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")

def encode_chunks(model, chunks, batch=8):
    all_embs = []
    for i in range(0, len(chunks), batch):
        emb = model.encode(chunks[i:i+batch], batch_size=batch,
                           convert_to_numpy=True, show_progress_bar=False,
                           normalize_embeddings=True)
        all_embs.append(emb.astype(np.float32))
        gc.collect()
    return np.vstack(all_embs)

# =============================================================================
# SimpleIndex 아키텍처
# =============================================================================
class SimpleIndex:
    def __init__(self):
        self.embeddings = None
        self.documents  = []
        self.sources    = []

    def add(self, docs, embs, source):
        embs = embs.astype(np.float32)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])
        self.documents.extend(docs)
        self.sources.extend([source] * len(docs))

    def count(self): 
        return len(self.documents)

    def search(self, q_emb, k):
        if self.embeddings is None or len(self.documents) == 0: return []
        qn = (q_emb / (np.linalg.norm(q_emb) + 1e-9)).astype(np.float32)
        sims = self.embeddings @ qn
        dists = 1.0 - sims
        k = min(k, len(self.documents))
        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]
        return [(self.documents[i], self.sources[i], float(dists[i])) for i in idx]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "documents": self.documents, "sources": self.sources}, f, protocol=4)
        return path.stat().st_size / 1024 / 1024

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f: d = pickle.load(f)
        idx = cls()
        idx.embeddings = d["embeddings"]
        idx.documents  = d["documents"]
        idx.sources    = d["sources"]
        return idx

# =============================================================================
# PDF 및 청킹 프로세서
# =============================================================================
def extract_pdf(f_bytes, max_chars=150_000) -> str:
    try:
        import fitz
        doc = fitz.open(stream=f_bytes, filetype="pdf")
        parts, total = [], 0
        for page in doc:
            t = page.get_text()
            if t.strip():
                parts.append(t.strip()); total += len(t)
                if total >= max_chars: break
        doc.close()
        text = "\n\n".join(parts)
        for m in ["References\n", "REFERENCES\n", "Bibliography\n"]:
            i = text.find(m)
            if i != -1: text = text[:i]
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        if len(text) / max(len(parts), 1) < 50: return "__GARBAGE__"
        return text
    except Exception: return ""

def chunk(text: str) -> list:
    SECTION_RE = re.compile(r'^\s*\d+\.?\d*\.?\s*(Abstract|Introduction|Experimental|Method|Result|Discussion|Conclusion|Preparation|Simulation|Summary)', re.IGNORECASE | re.MULTILINE)
    if len(text) > 150_000: text = text[:150_000]
    secs = [m.start() for m in SECTION_RE.finditer(text)]
    chunks, start, prev = [], 0, -1
    while start < len(text):
        if start == prev: start += 700; continue
        prev = start; end = min(start + 800, len(text))
        if end < len(text):
            mid = start + 400
            sb = next((s for s in secs if mid <= s < end), None)
            if sb: end = sb
            else:
                b = text.rfind("\n\n", mid, end)
                if b != -1: end = b + 2
        c = text[start:end].strip()
        if len(c) >= 80: chunks.append(c)
        start = end - 100
    return chunks

# =============================================================================
# LLM 통신 및 Query Expansion 인터페이스
# =============================================================================
def call_llm(prompt, provider, model, api_key=""):
    if provider == "OpenAI GPT":
        import openai
        r = openai.OpenAI(api_key=api_key).chat.completions.create(
            model=model, messages=[{"role":"system","content":"You are a scientific assistant."}, {"role":"user","content":prompt}], max_tokens=MAX_TOKENS, temperature=0.1)
        return r.choices[0].message.content
    elif provider == "Google Gemini":
        import google.generativeai as gai
        gai.configure(api_key=api_key)
        return gai.GenerativeModel(model, generation_config=gai.types.GenerationConfig(max_output_tokens=MAX_TOKENS, temperature=0.1)).generate_content(prompt).text
    else:
        payload = _json.dumps({"model":model,"prompt":prompt,"stream":False}).encode()
        req = urllib.request.Request("http://localhost:11434/api/generate", data=payload, headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=120) as r: return _json.loads(r.read())["response"]

def expand_query(query, provider, model, api_key=""):
    prompt = (
        "You are a materials science literature search expert.\n"
        "Generate 4 alternative search queries for the question below.\n"
        "Use terminology found in NdFeB magnet research papers.\n"
        "Return ONLY a JSON array of 4 strings.\n\n"
        f"Question: {query}\n\n"
        'Example: ["query1", "query2", "query3", "query4"]'
    )
    queries = [query]
    try:
        raw = call_llm(prompt, provider, model, api_key)
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            extras = _json.loads(m.group())
            queries += [q.strip() for q in extras if q.strip()]
    except Exception: pass
    
    q = query.lower()
    rules = {
        "gbdp":            ["grain boundary diffusion process", "gbd source coating heat treatment", "diffusion process NdFeB"],
        "입계계면":          ["grain boundary diffusion process", "gbdp microstructure modification", "intergranular phase"],
        "확산 공정":         ["diffusion kinetics mechanism", "heat treatment temperature duration", "diffusion depth"],
        "source material": ["diffusion source preparation", "PCAG alloy melt-spinning powder", "diffusion source precursor"],
        "coercivity":      ["coercivity enhancement NdFeB", "μ0Hc improvement shell microstructure", "demagnetization curve"],
        "보자력":           ["coercivity improvement mechanism", "remanence Jr coercivity Hcj trade-off", "core-shell structured grain"],
        "making":          ["preparation melt-spinning fabrication", "alloy powder synthesis", "melt spun ribbon fabrication"],
        "composition":     ["alloy composition at.% wt.%", "rare earth elemental concentration", "stoichiometric Nd2Fe14B matrix"],
        "중희토류":          ["heavy rare earth elements HREE", "dy dyprosium tb terbium substitution", "hree saving technology"],
        "heavy rare":      ["heavy rare earth grain boundary diffusion", "dy/tb elemental mapping", "critical raw materials reduction"],
        "what is":         ["NdFeB sintered magnet introduction", "NdFeB permanent magnet properties", "anisotropic sintered magnet"],
    }
    for kw, adds in rules.items():
        if kw in q: queries += adds
        
    seen, out = set(), []
    for q2 in queries:
        if q2 not in seen: seen.add(q2); out.append(q2)
    return out[:6]

# =============================================================================
# 웹 프론트엔드 UI 컴포넌트 레이어 (Streamlit 호환 컴파일)
# =============================================================================
st.set_page_config(page_title="NdFeB RAG 웹 분석기", layout="wide")

# CSS 커스텀 다크 테마 주황 포인트 적용
st.markdown("""
    <style>
    .stApp { background-color: #1c1c1c; color: #e5e0d8; }
    .stButton>button { background-color: #d4700a !important; color: white !important; font-weight: bold; border: none !important; }
    .stButton>button:hover { background-color: #e8890f !important; }
    div[data-testid="stExpander"] { background-color: #272727 !important; border: 1px solid #484848 !important; }
    </style>
""", unsafe_allow_html=True)  # <--- 반드시 unsafe_allow_html=True 로 수정해 주세요!

st.title("⚗️ NdFeB Literature RAG Web v5.4")
st.caption("Experimental Magnet Science Intelligence — Pure English Output Enforcement")

# 좌측 사이드바 영역 컨트롤
with st.sidebar:
    st.header("🤖 모델 및 API 설정")
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
        st.info("Ollama 사용 시 로컬 호스트에서 'ollama serve'가 켜져 있어야 합니다.")

    st.markdown("---")
    st.header("📁 통합 지식 베이스")
    
    idx_path = Path(INDEX_FILE)
    if idx_path.exists():
        if 'index' not in st.session_state:
            st.session_state.index = SimpleIndex.load(idx_path)
        # 안전한 파이썬 표준 len 함수 분기로 변경하여 대시보드 에러 전면 차단
        total_chunks = len(st.session_state.index.documents)
        st.success(f"✓ 지식 베이스 연동 완료 ({total_chunks} Chunks)")
    else:
        st.warning("⚠️ 폴더 내에 index.pkl 파일이 보이지 않습니다. 파일을 업로드해 주세요.")

# 메인 UI 분할 스플리터 대체 구성
col_main, col_side = st.columns([2, 1])

with col_main:
    st.subheader("❓ 문헌 질의 (Question Area)")
    
    # 예시 질문 셀렉트 박스
    selected_example = st.selectbox("💡 빠른 검색 프리셋 질문 선택", ["-- 직접 입력 --"] + EXAMPLES)
    
    if selected_example != "-- 직접 입력 --":
        init_query = selected_example
    else:
        init_query = ""
        
    user_query = st.text_area("소재 공정 조건 및 수치 질문을 입력하세요:", value=init_query, placeholder="Enter your question here...", height=100)

    if st.button("▶ 문헌 추론 가동 (Ask)", type="primary"):
        if not user_query.strip():
            st.error("질문 내용을 작성해 주세요.")
        elif 'index' not in st.session_state:
            st.error("지식 베이스 인덱스가 존재하지 않습니다.")
        elif provider in ["OpenAI GPT", "Google Gemini"] and not api_key.strip():
            st.error("해당 공급자의 API Key가 누락되었습니다.")
        else:
            with st.spinner("BGE-M3 구조 기반 컨텍스트 교차 검색 및 학술 검증 중..."):
                try:
                    # 임베딩 전역 가속화 캐시 자원 할당
                    emb_model = load_embedding_model()
                    q_emb = emb_model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)[0]
                    
                    # 검색 결과 파싱
                    results = st.session_state.index.search(q_emb, TOP_K)
                    
                    if not results:
                        st.warning("일치하는 구절이 인덱스에 존재하지 않습니다.")
                    else:
                        src_cnt = defaultdict(int); sel = []
                        for doc, src, dist in results:
                            if src_cnt[src] < MAX_PER_SRC:
                                sel.append((doc, src, dist)); src_cnt[src] += 1
                            if len(sel) >= FINAL_K: break
                        
                        # 컨텍스트 빌딩
                        grouped = defaultdict(list)
                        for doc, src, _ in sel: grouped[src].append(doc)
                        ctx_parts = [f"=== Source: {s} ===\n" + "\n\n[...]\n\n".join(cks) for s, cks in grouped.items()]
                        context = "\n\n---\n\n".join(ctx_parts)
                        
                        # 다국어 간섭 원천 배제 영어 고정 조항 시스템 프롬프트 투영
                        prompt = (
                            "You are a scientific literature analyst for NdFeB magnets.\n\n"
                            "RULES:\n"
                            "1. Answer ONLY from the passages below.\n"
                            "2. Do NOT use your own knowledge.\n"
                            "3. Cite source filename after every claim.\n"
                            "4. Extract specific values/compositions/methods exactly.\n"
                            "5. STRICT OUTPUT LANGUAGE RULE: Regardless of the language of the user's question, you MUST write the entire answer in 100% PURE ENGLISH. "
                            "Do NOT use any Korean words, particles, or endings (e.g., 은/는, 이/가, ~입니다, ~했습니다). "
                            "Do NOT use any Chinese characters (e.g., 磁, 等, 閞, 使用) or Japanese characters. "
                            "The entire response must be written in standard, clean, academic English syntax and vocabulary.\n"
                            "6. If not found: 'The indexed papers do not contain information about [topic]. Papers searched: [filenames].'\n\n"
                            f"Question: {user_query}\n\n"
                            f"Passages ({len(src_cnt)} papers):\n\n{context}\n\n"
                            "Answer (cite sources):"
                        )
                        
                        answer = call_llm(prompt, provider, model, api_key)
                        
                        st.session_state.answer = answer
                        st.session_state.sources = list(src_cnt.keys())
                        st.session_state.debug_logs = [f"dist={d:.3f} [{src}] {c[:60]}..." for c, src, d in sel]
                        
                except Exception as e:
                    st.error(f"연산 런타임 오류: {str(e)}")

    st.markdown("---")
    if 'answer' in st.session_state:
        st.subheader("📝 검증 답변 출력 (Answer - 100% English)")
        st.write(st.session_state.answer)

with col_side:
    st.subheader("📌 인용 문헌 파일 (Sources)")
    if 'sources' in st.session_state and st.session_state.sources:
        for src in st.session_state.sources:
            st.markdown(f"📄 `{src}`")
    else:
        st.caption("인용된 소스 파일이 여기에 나타납니다.")
        
    st.markdown("---")
    st.subheader("🔍 임베딩 매칭 벡터 로그")
    if 'debug_logs' in st.session_state:
        for log in st.session_state.debug_logs:
            st.caption(log)
