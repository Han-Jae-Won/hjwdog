# Stanford Dogs 품종 분류 프로젝트

## 실행 방법
pip install -r requirements.txt
1. 데이터 준비: `data/Images/` 아래에 Stanford Dogs 원본 이미지 폴더 구조로 위치
2. 학습:
    ```
    python -m src.train
    ```
3. Streamlit 서비스 실행:
    ```
    python -m streamlit run streamlit_app.py
    ```
