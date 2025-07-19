# rag_chat.py
import numpy as np
import json
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class SemanticSearcher:
    def __init__(self, model_name='jhgan/ko-sroberta-multitask'):
        print("Initializing SemanticSearcher...")
        self.model = SentenceTransformer(model_name)
        self.texts, self.embeddings = self._load_data()
        self.index = self._build_faiss_index()
        print("SemanticSearcher initialized successfully.")

    def _load_data(self, embeddings_path="faq_embeddings.npy", texts_path="faq_texts.json"):
        """미리 생성된 임베딩과 텍스트 데이터를 불러옵니다."""
        try:
            print(f"Loading embeddings from {embeddings_path}")
            embeddings = np.load(embeddings_path)
            print(f"Loading texts from {texts_path}")
            with open(texts_path, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            return texts, embeddings
        except FileNotFoundError:
            print("Error: Embedding or text file not found.")
            print("Please run 'build_embeddings.py' first to generate the necessary files.")
            return [], None

    def _build_faiss_index(self):
        """FAISS 인덱스를 빌드하여 빠른 검색을 준비합니다."""
        if self.embeddings is not None:
            print("Building FAISS index...")
            d = self.embeddings.shape[1]  # 벡터의 차원
            index = faiss.IndexFlatL2(d)
            index.add(self.embeddings)
            return index
        return None

    def search(self, query, top_k=5):
        """사용자 질문과 가장 유사한 문서를 검색합니다."""
        if self.index is None:
            return []
        print(f"Encoding query: '{query}'")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        print(f"Searching top-{top_k} results...")
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 검색된 인덱스에 해당하는 텍스트를 반환합니다.
        results = [self.texts[i] for i in indices[0]]
        print(f"Found results: {results}")
        return results

# 전역 변수로 Searcher 인스턴스 생성 (Streamlit 앱에서 재사용)
searcher = SemanticSearcher()

def rag_answer(query, api_key, history: list):
    """
    대화 기록과 시맨틱 서치를 함께 사용하여 RAG 답변을 생성합니다.
    """
    print(f"Received query for RAG: '{query}'")
    # 1. 검색 (Search) - 현재 질문에 가장 관련성 높은 문서를 찾습니다.
    relevant_docs = searcher.search(query, top_k=3)
    context = "\n".join(relevant_docs) if relevant_docs else "관련 정보를 찾을 수 없습니다."
    print(f"Context for prompt:\n{context}")

    # 2. 프롬프트 구성 (Prompt Engineering)
    # 시스템 메시지와 대화 기록, 그리고 새로운 정보를 포함한 프롬프트를 구성합니다.
    messages = history.copy()
    
    # 새로운 사용자 질문과 검색된 컨텍스트를 함께 추가합니다.
    prompt_with_context = f"[참고 정보]\n{context}\n\n[사용자 질문]\n{query}\n"
    messages.append({"role": "user", "content": prompt_with_context})

    # 3. 생성 (Generation)
    print("Generating answer with OpenAI...")
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,  # 전체 대화 기록을 전달
            temperature=0.5,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content.strip()
        print(f"Generated answer: {answer}")
        return answer
    except Exception as e:
        print(f"Error during OpenAI API call: {e}")
        return "답변을 생성하는 동안 오류가 발생했습니다. API 키 설정과 네트워크 연결을 확인해주세요."