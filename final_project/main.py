"""
Tupick — 조각투자 서비스 AI (RAG MVP)
-------------------------------------------------
Streamlit 앱 하나로 수집 → 인덱싱 → 검색 → 리포트 생성까지 수행합니다.

# ✅ 기능
- URL 다건 입력 후 본문 크롤링(robots.txt는 별도 확인 필요)
- 문단 청크 + 메타데이터 스키마로 정제
- bge-m3 임베딩 + FAISS 벡터 인덱스 로컬 저장/로드
- 하이브리드 검색(BM25 유사 키워드 + 벡터 검색) 간단 버전
- Ollama(로컬 LLM) 또는 OpenAI API 중 사용 가능
- 근거 각주 + 안전 문구 포함 리포트 생성

# 📦 요구 라이브러리(예시)
pip install streamlit requests beautifulsoup4 sentence-transformers faiss-cpu rank-bm25 pydantic python-dotenv

# 🔧 사전 준비
1) 모델
   - (권장) Ollama 설치 후: `ollama pull llama3.1:8b-instruct` 또는 Qwen2.5-7B-Instruct
   - OpenAI를 쓸 경우 .env에 OPENAI_API_KEY 세팅
2) 실행: `streamlit run app.py`

# ⚠️ 주의
- 각 사이트의 약관/robots.txt와 저작권을 준수하세요.
- 본 앱은 정보 제공용 데모입니다. 투자 권유가 아닙니다.
"""

import os
import re
import json
import time
import uuid
import math
import faiss
import queue
import base64
import hashlib
import requests
import streamlit as st

from io import BytesIO
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pydantic import BaseModel
from dotenv import load_dotenv

# ----------------------------
# Config
# ----------------------------
load_dotenv()
def _default_data_dir():
    # Use ASCII-only path on Windows to avoid Faiss unicode bug
    env = os.getenv("TUPICK_DATA_DIR")
    if env:
        return env
    if os.name == "nt":
        return r"C:\tupick_data"  # safe ASCII path
    return os.path.join(os.getcwd(), "tupick_data")

DATA_DIR = _default_data_dir()
INDEX_DIR = os.path.join(DATA_DIR, "index")
DOCS_PATH = os.path.join(DATA_DIR, "docs.jsonl" )
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "BAAI/bge-m3")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_SIZE = 550
CHUNK_OVERLAP = 60

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ----------------------------
# Data Schemas
# ----------------------------
class DocChunk(BaseModel):
    id: str
    source: str
    category: str
    title: str
    section: str
    url: str
    as_of_date: str
    text: str
    language: str = "ko-KR"

# ----------------------------
# Utils
# ----------------------------

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_whitespace(text)
    tokens = text.split(" ")
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + size]
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return [c for c in chunks if len(c) > 20]


def fetch_url(url: str, timeout: int = 20, use_js: bool = False) -> Tuple[str, str]:
    """Return (title, main_text).
    1) Try requests+BS4
    2) (optional) Fallback to Playwright JS render for SPA/blocked pages
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "ko,ko-KR;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.text.strip() if soup.title else url
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        body = normalize_whitespace(soup.get_text(" "))
        # If content too small, try JS rendering if allowed
        if use_js and len(body) < 800:
            raise RuntimeError("Body too small; try JS render")
        return title, body
    except Exception as e:
        if not use_js:
            raise
        # Playwright fallback
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                ctx = browser.new_context(locale="ko-KR", user_agent=headers["User-Agent"])
                page = ctx.new_page()
                page.goto(url, wait_until="networkidle", timeout=timeout*1000)
                title = page.title() or url
                body = page.locator("body").inner_text(timeout=timeout*1000)
                browser.close()
                body = normalize_whitespace(body)
                return title, body
        except Exception as e2:
            raise RuntimeError(f"fetch_url failed for {url}: {e2}")


def guess_category_from_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "tessa" in host:
        return "미술품/명품"
    if "art" in host:
        return "미술품"
    if "music" in host or "mucis" in host or "mucic" in host:
        return "저작권/음원"
    if "kasa" in host or "lucent" in host:
        return "부동산"
    return "조각투자/일반"


def now_date() -> str:
    import datetime as dt
    return dt.datetime.now().strftime("%Y-%m-%d")


def save_jsonl(items: List[Dict[str, Any]], path: str):
    with open(path, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# ----------------------------
# Embeddings & Index
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    model = SentenceTransformer(EMB_MODEL_NAME)
    return model


def build_faiss_index(embeddings: List[List[float]]) -> faiss.IndexFlatIP:
    import numpy as np
    arr = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(arr)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)
    return index


def search_faiss(index: faiss.IndexFlatIP, query_vec, top_k: int = TOP_K):
    import numpy as np
    q = np.array([query_vec], dtype="float32")
    faiss.normalize_L2(q)
    scores, idx = index.search(q, top_k)
    return scores[0].tolist(), idx[0].tolist()

# ----------------------------
# Reranking (BM25 simple)
# ----------------------------
class SimpleHybrid:
    def __init__(self, texts: List[str]):
        tokenized = [t.split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.texts = texts
    def score(self, query: str, candidates: List[int], k: int = TOP_K) -> List[int]:
        # Re-rank the candidate indices by BM25 score w.r.t query
        doc_scores = self.bm25.get_scores(query.split())
        pairs = [(i, doc_scores[i]) for i in candidates]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [i for (i, _) in pairs[:k]]

# ----------------------------
# LLM Backends
# ----------------------------
LLM_SYSTEM_PROMPT = (
    "너는 ‘조각투자 도메인 어시스턴트’야. 제공된 CONTEXT만을 근거로 한국어로 답변해. "
    "항상 [사용자 성향]의 리스크, 예산, 목표를 반드시 리포트에 반영해야 한다. "
    "숫자/기간/수익률 등은 컨텍스트에 있는 내용만 사용하고, 없으면 ‘자료 부족’이라고 말해. "
    "투자 권유처럼 단정하지 말고 정보 제공 관점으로 작성해. "
    "답변은 반드시 아래 리포트 템플릿 구조로 작성한다:\n"
    "A. 핵심요약 (3줄 이내, 사용자 성향 1줄 반영)\n"
    "B. 수익·비용 구조 요약 (근거 문장 끝에 [번호])\n"
    "C. 예산별 전략 (사용자 예산 기준 분할/티켓/유동성)\n"
    "D. 리스크 포인트 (사용자 리스크 선호 반영)\n"
    "E. 목표 적합성 평가 (사용자 목표 기준, [번호])\n"
    "F. 유사상품 비교 (있으면, [번호])\n"
    "G. 출처 (번호, 제목, URL)\n"
    "H. 안전 문구 (‘본 서비스는 정보 제공용이며, 수익 보장을 하지 않습니다.’)\n"
)



def build_user_prompt(
    user_query: str,
    passages: List[Dict[str, Any]],
    risk: str,
    budget: int,
    horizon: str
) -> str:
    context_lines = []
    for i, p in enumerate(passages, start=1):
        title = p.get("title", "Untitled")
        url = p.get("url", "")
        date = p.get("as_of_date", "")
        text = p.get("text", "")
        context_lines.append(f"[{i}] {title} | {date} | {url}\n{text}")
    context_str = "\n\n".join(context_lines)

    user_prompt = (
        f"[사용자 성향]\n리스크={risk}, 예산={budget}원, 목표={horizon}\n\n"
        f"[질문]\n{user_query}\n\n"
        f"[CONTEXT]\n{context_str}\n"
    )
    return user_prompt



def call_ollama(model: str, system: str, prompt: str, host: str = OLLAMA_HOST) -> str:
    url = f"{host}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")


def call_openai(model: str, system: str, prompt: str, api_key: str) -> str:
    import json as _json
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, data=_json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ----------------------------
# Persistence helpers
# ----------------------------

def index_paths() -> Tuple[str, str]:
    faiss_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "meta.json")
    return faiss_path, meta_path


def save_index(index: faiss.IndexFlatIP, meta: Dict[str, Any]):
    faiss_path, meta_path = index_paths()
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    try:
        faiss.write_index(index, faiss_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # Windows + non-ASCII path bug: fallback to C:	upick_data
        if os.name == "nt":
            fallback_base = r"C:\tupick_data"
            fb_index_dir = os.path.join(fallback_base, "index")
            os.makedirs(fb_index_dir, exist_ok=True)
            fb_faiss = os.path.join(fb_index_dir, "faiss.index")
            fb_meta = os.path.join(fb_index_dir, "meta.json")
            faiss.write_index(index, fb_faiss)
            with open(fb_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            try:
                import streamlit as _st
                _st.warning(f"기본 경로에 쓰기 실패하여 ASCII 경로로 저장했습니다: {fb_faiss}")
            except Exception:
                pass
        else:
            raise


def load_index() -> Tuple[Any, Dict[str, Any]]:
    faiss_path, meta_path = index_paths()
    # primary
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        index = faiss.read_index(faiss_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return index, meta
    # fallback for Windows ASCII path
    if os.name == "nt":
        fb_index_dir = os.path.join(r"C:\tupick_data", "index")
        fb_faiss = os.path.join(fb_index_dir, "faiss.index")
        fb_meta = os.path.join(fb_index_dir, "meta.json")
        if os.path.exists(fb_faiss) and os.path.exists(fb_meta):
            index = faiss.read_index(fb_faiss)
            with open(fb_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
            return index, meta
    return None, {}


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Tupick — 조각투자 AI(MVP)", layout="wide")
st.title("Tupick — 조각투자 서비스 AI · RAG MVP")
st.caption("본 서비스는 정보 제공용 데모입니다. 수익을 보장하지 않습니다.")

with st.sidebar:
    st.header("데이터 수집 & 인덱싱")
    st.write("수집할 URL을 줄바꿈으로 입력하세요.")
    urls_text = st.text_area("URLs", height=200, placeholder="https://tessa.art/faq\nhttps://www.musicow.com/about/guide\nhttps://www.kasa.co.kr/faq")
    source_name = st.text_input("소스 식별자(서비스명)", value="generic")
    use_js = st.checkbox("JS 렌더링(Playwright) 사용", value=True, help="뮤직카우/카사처럼 SPA/차단된 페이지 대응")
    if st.button("수집→인덱싱 실행", type="primary"):
        urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        if not urls:
            st.error("URL을 입력해 주세요.")
        else:
            with st.spinner("수집 및 인덱싱 중..."):
                docs = []
                for u in urls:
                    try:
                        title, body = fetch_url(u, use_js=use_js)
                        cat = guess_category_from_url(u)
                        for ch in chunk_text(body):
                            did = f"{source_name}_{uuid.uuid4().hex[:12]}"
                            docs.append(
                                DocChunk(
                                    id=did,
                                    source=source_name,
                                    category=cat,
                                    title=title,
                                    section="본문",
                                    url=u,
                                    as_of_date=now_date(),
                                    text=ch,
                                ).model_dump()
                            )
                    except Exception as e:
                        st.warning(f"수집 실패: {u} — {e}")
                if docs:
                    save_jsonl(docs, DOCS_PATH)
                    st.success(f"수집 완료: {len(docs)} 청크 저장")

            # 인덱싱
            with st.spinner("임베딩 및 인덱싱 중..."):
                all_docs = load_jsonl(DOCS_PATH)
                texts = [d["text"] for d in all_docs]
                embedder = get_embedder()
                vecs = embedder.encode(texts, normalize_embeddings=True).tolist()
                index = build_faiss_index(vecs)
                save_index(index, {"count": len(texts)})
                st.success(f"인덱싱 완료: {len(texts)} 개 문서")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("질의·리포트 생성")
    user_query = st.text_area("질문을 입력하세요", height=120, placeholder="예) 테사 수수료 구조와 분배 주기를 요약해줘. 초보자 기준으로 간단히, 100만원 예산 전략도 포함")

    persona_col1, persona_col2, persona_col3 = st.columns(3)
    with persona_col1:
        risk = st.selectbox("리스크 선호", ["낮음", "중간", "높음"], index=1)
    with persona_col2:
        budget = st.number_input("예산(원)", min_value=0, value=1_000_000, step=100_000)
    with persona_col3:
        horizon = st.selectbox("목표", ["단기현금흐름", "중기수익", "장기성장"], index=0)

    if st.button("검색→생성", type="primary"):
        index, meta = load_index()
        if index is None:
            st.error("인덱스가 없습니다. 좌측에서 먼저 수집→인덱싱을 실행하세요.")
        else:
            all_docs = load_jsonl(DOCS_PATH)
            texts = [d["text"] for d in all_docs]

            embedder = get_embedder()
            qvec = embedder.encode([user_query], normalize_embeddings=True)[0]
            scores, idxs = search_faiss(index, qvec, top_k=max(TOP_K, 8))

            # Hybrid re-rank
            rer = SimpleHybrid(texts)
            re_idx = rer.score(user_query, idxs, k=TOP_K)

            top_passages = [all_docs[i] for i in re_idx]

            # Build prompt
            enriched_query = (
                f"[사용자 성향] 리스크={risk}, 예산={budget}원, 목표={horizon}\n"
                f"[질문] {user_query}"
            )
            prompt = build_user_prompt(
                          user_query=user_query,
                          passages=top_passages,
                          risk=risk,
                          budget=budget,
                          horizon=horizon
                        )

            # Call LLM
            try:
                if OPENAI_API_KEY:
                    answer = call_openai(OPENAI_MODEL, LLM_SYSTEM_PROMPT, prompt, OPENAI_API_KEY)
                else:
                    answer = call_ollama(OLLAMA_MODEL, LLM_SYSTEM_PROMPT, prompt, OLLAMA_HOST)
            except Exception as e:
                st.error(f"LLM 호출 오류: {e}")
                answer = ""

            if answer:
                st.markdown("### 답변")
                st.write(answer)

                # Download report as Markdown
                md = f"# Tupick 리포트\n\n**작성일:** {now_date()}\n\n{answer}\n"
                b = md.encode("utf-8")
                st.download_button("리포트(MD) 다운로드", data=b, file_name="tupick_report.md", mime="text/markdown")

with col2:
    st.subheader("인덱스 상태")
    index, meta = load_index()
    if index is not None:
        st.success(f"인덱스 로드됨 — 문서 수: {meta.get('count')}")
    else:
        st.info("아직 인덱스가 없습니다.")

    st.markdown("#### 유용한 팁")
    st.markdown(
    "- SPA(자바스크립트 렌더) 페이지는 ‘JS 렌더링’ 체크를 켜세요. "
    "처음엔 `pip install playwright` 후 `playwright install chromium` 필요.\n"
    "- 질의에 서비스명/키워드를 넣으면 검색 정확도가 올라갑니다.\n"
    "- 같은 사이트는 한 번에 3~10개 URL 정도만 먼저 수집해보세요.\n"
    "- 분배주기/수수료/리스크 등 규칙 단어를 포함하면 리포트 품질이 좋아집니다."
)
st.divider()

st.caption(
    "ⓘ 컴플라이언스: 각 사이트의 약관 및 robots.txt를 준수해야 하며, 본 데모는 교육/연구용입니다."
)
