import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import json
import io
from openai import OpenAI
from rag_chat import rag_answer
import speech_recognition as sr
from gtts import gTTS
import os
import cv2
import tempfile
import random
import zipfile
import pandas as pd
from datetime import datetime
import requests
import googlemaps

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="강아지 만능 솔루션",
    page_icon="🐶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for max-width
st.markdown(
    """
    <style>
    .main .block-container {
        max-width: 1920px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 환경/모델/데이터 경로 설정 ---
MODEL_PATH = "dog_breed_model.pth"
CLASS_NAMES_PATH = "class_names.json"
BREED_DATA_PATH = "dog_breeds_data.json"
DIARY_PATH = "pet_diary.json"
DOG_IMAGES_DIR = "C:/Users/hjw83/prj/stanford_dogs_project/data/Images"
IMG_SIZE = 224

# --- OpenAI API KEY 설정 ---
openai_api_key = st.secrets.get("openai_api_key")

# --- 캐시된 함수들 (데이터 로딩) ---
@st.cache_resource
def load_display_breed_names():
    with open(CLASS_NAMES_PATH, encoding='utf-8') as f:
        raw_class_names = json.load(f)
        display_names = [name.split('-', 1)[1].replace('_', ' ') for name in raw_class_names]
        st.write(f"[DEBUG] Loaded display class names: {len(display_names)} items")
        return display_names

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, encoding='utf-8') as f:
        class_names = json.load(f)
        st.write(f"[DEBUG] Loaded class names: {len(class_names)} items")
        return class_names

@st.cache_resource
def load_model(num_classes):
    # ... (이전과 동일)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

@st.cache_data
def load_breed_data():
    try:
        with open(BREED_DATA_PATH, "r", encoding="utf-8") as f:
            raw_breed_data = json.load(f)
            processed_breed_data = {}
            for key, value in raw_breed_data.items():
                # Extract English name from key like "푸들 (Poodle)"
                import re
                match = re.search(r'\((.*?)\)', key)
                if match:
                    english_name = match.group(1)
                    processed_breed_data[english_name] = value
                else:
                    processed_breed_data[key] = value # Fallback for keys without English name
            return processed_breed_data
    except FileNotFoundError:
        st.error(f"File not found: {BREED_DATA_PATH}")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"JSON decoding error in {BREED_DATA_PATH}: {e}")
        return {}
    except Exception as e:
        st.error(f"Error loading breed data: {e}")
        return {}

@st.cache_resource
def load_display_breed_names():
    breed_data = load_breed_data()
    return sorted(list(breed_data.keys()))

@st.cache_data
def get_all_dog_images():
    # ... (이전과 동일)
    all_images = []
    if not os.path.isdir(DOG_IMAGES_DIR):
        return []
    for breed_folder in os.listdir(DOG_IMAGES_DIR):
        breed_path = os.path.join(DOG_IMAGES_DIR, breed_folder)
        if os.path.isdir(breed_path):
            for img_name in os.listdir(breed_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append({
                        "path": os.path.join(breed_path, img_name),
                        "breed": breed_folder.split('-', 1)[1].replace("_", " ")
                    })
    return all_images

# --- 새로운 함수: 건강 일지 데이터 관리 ---
def load_diary_data():
    if os.path.exists(DIARY_PATH):
        with open(DIARY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_diary_data(data):
    with open(DIARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- 전처리 및 예측 함수 ---
# ... (이전과 동일)
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

def predict(img):
    class_names = load_class_names()
    model = load_model(len(class_names))
    input_tensor = preprocess(img)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idxs = torch.topk(probs, 3)
    
    results = []
    for i in range(3):
        breed = class_names[top_idxs[0][i]]
        prob = top_probs[0][i].item()
        results.append({"breed": breed, "probability": prob})
    return results

# --- 세션 상태 초기화 ---
# ... (이전과 동일, diary_data 추가)
if "prediction_results" not in st.session_state:
    st.session_state.prediction_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "당신은 반려동물, 특히 강아지에 대한 지식이 풍부한 전문가입니다. 주어진 '참고 정보'를 바탕으로 사용자의 질문에 대해 친절하고 상세하게 답변해주세요. 만약 참고 정보에 질문과 관련된 내용이 없다면, 정보가 부족하여 답변하기 어렵다고 솔직하게 말해주세요. 절대로 정보를 지어내지 마세요."}
    ]
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = ""
if "diary_data" not in st.session_state:
    st.session_state.diary_data = load_diary_data()

# --- 사이드바 UI ---
with st.sidebar:
    # ... (이전과 동일)
    st.title("🐶 견종 분석기")
    st.markdown("---")
    uploaded_file = st.file_uploader("강아지 이미지를 업로드하세요.", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        st.image(uploaded_file, caption="업로드한 이미지", use_column_width=True)
        img = Image.open(uploaded_file).convert('RGB')
        
        with st.spinner("이미지를 분석 중입니다..."):
            st.session_state.prediction_results = predict(img)
        
        top_breed = st.session_state.prediction_results[0]['breed']
        st.success(f"**분석 완료!**")
        st.info(f"가장 유력한 품종은 **'{top_breed}'** 입니다.")
        st.markdown("---")

    st.header("📝 앱 안내")
    st.write("""
    1.  **이미지 업로드**: 사이드바에서 강아지 사진을 업로드하여 품종을 분석하세요.
    2.  **기능 선택**: 메인 화면의 탭을 눌러 다양한 기능을 이용해보세요.
    """)

# --- 메인 화면 UI ---
st.title("강아지 만능 솔루션 🐾")

tab_list = [
    "🔍 상세 분석", "💬 전문가 상담", "🆚 품종 비교", "🎁 맞춤 추천", 
    "🎨 AI 이미지 생성", "❓ 퀴즈 게임", "🗺️ 주변 장소", "📔 성장/건강 일지", "🗣️ 행동/소리 분석", "🤖 미로 게임"
]
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tab_list)

# --- 탭 1~6: 기존 기능들 ... (코드는 생략, 이전과 동일) ---
with tab1:
    # 상세 분석 탭 ...
    st.header("🔍 상세 분석 결과")
    if st.session_state.prediction_results:
        st.subheader("📊 품종 예측 확률 (Top 3)")
        for result in st.session_state.prediction_results:
            st.write(f"**{result['breed']}**: {result['probability']*100:.2f}%")
        
        st.markdown("---")
        
        st.subheader("🎬 동영상에서 특정 품종 강아지 찾기")
        dog_breed = st.session_state.prediction_results[0]['breed']
        st.info(f"분석 기준 품종: **{dog_breed}**")

        video_file = st.file_uploader("동영상을 업로드하세요 (mp4, mov, avi)", type=["mp4", "avi", "mov"], key="video_upload")
        frame_interval = st.number_input("프레임 추출 간격(초)", min_value=1, max_value=10, value=1, help="설정된 초마다 한 프레임씩 검사합니다.")

        if video_file:
            st.video(video_file)
            if st.button("동영상 분석 시작"):
                # ... (동영상 분석 로직)
                if video_file:
                    # Save uploaded video to a temporary file
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video_file.read())
                    video_path = tfile.name

                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("동영상을 열 수 없습니다. 지원되는 형식인지 확인해주세요.")
                        tfile.close()
                        os.unlink(video_path)
                        st.stop()

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st.write(f"총 프레임 수: {total_frames}, FPS: {fps:.2f}")

                    detected_frames = []
                    frame_count = 0
                    target_breed_found = False

                    progress_text = "동영상 분석 중..."
                    my_bar = st.progress(0, text=progress_text)

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        # Process frame only at specified interval
                        if frame_count % (int(fps * frame_interval)) == 0:
                            # Convert OpenCV BGR frame to PIL RGB image
                            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            predictions = predict(pil_img)

                            if predictions and predictions[0]['breed'] == dog_breed:
                                detected_frames.append(frame)
                                target_breed_found = True
                                st.write(f"프레임 {frame_count}에서 '{dog_breed}' 발견!")
                                st.image(pil_img, caption=f"프레임 {frame_count} - {dog_breed} 발견!", use_column_width=True)

                        # Update progress bar
                        progress = min(frame_count / total_frames, 1.0)
                        my_bar.progress(progress, text=f"{progress_text} ({int(progress*100)}%)")

                    cap.release()
                    tfile.close()
                    os.unlink(video_path) # Clean up temporary file

                    my_bar.empty() # Remove progress bar

                    if not target_breed_found:
                        st.info(f"동영상에서 '{dog_breed}' 품종을 찾지 못했습니다.")
                    else:
                        st.success(f"동영상 분석 완료! '{dog_breed}' 품종이 발견된 프레임들을 확인하세요.")
                else:
                    st.warning("동영상 파일을 먼저 업로드해주세요.")
    else:
        st.info("⬅️ 사이드바에서 강아지 이미지를 먼저 업로드해주세요.")

with tab2:
    st.header("💬 전문가 상담 (AI 챗봇)")
    st.write("반려동물에 대한 궁금증을 AI 전문가에게 물어보세요!")

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        if message["role"] != "system": # System messages are not displayed to the user
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("무엇이든 물어보세요!"):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                # Call rag_answer function from rag_chat.py
                full_response = rag_answer(prompt, openai_api_key, st.session_state.chat_history)
                st.markdown(full_response)
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

with tab3:
    st.header("🆚 품종 비교하기")
    st.write("두 가지 품종을 선택하여 특징을 비교해 보세요.")

    display_class_names = load_display_breed_names()
    if display_class_names:
        col1, col2 = st.columns(2)
        breed1 = col1.selectbox("첫 번째 품종 선택", display_class_names, key="breed_compare_1")
        breed2 = col2.selectbox("두 번째 품종 선택", display_class_names, index=min(1, len(display_class_names)-1), key="breed_compare_2")

        st.write(f"[DEBUG] Selected Breed 1: {breed1}, Selected Breed 2: {breed2}")

        if breed1 and breed2:
            st.subheader(f"{breed1} vs {breed2}")
            breed_data = load_breed_data()
            
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown(f"### {breed1}")
                if breed1 in breed_data:
                    st.write(f"**크기:** {breed_data[breed1].get('기본특징', '정보 없음')}") # Using basic characteristic as size for now
                    st.write(f"**성격:** {breed_data[breed1].get('성격', '정보 없음')}")
                    st.write(f"**건강상 유의점:** {breed_data[breed1].get('건강상 유의점', '정보 없음')}")
                    st.write(f"**털 관리:** {breed_data[breed1].get('털 관리', '정보 없음')}")
                    st.write(f"**운동/활동:** {breed_data[breed1].get('운동/활동', '정보 없음')}")
                else:
                    st.info("해당 품종에 대한 상세 정보가 없습니다.")

            with col_info2:
                st.markdown(f"### {breed2}")
                if breed2 in breed_data:
                    st.write(f"**크기:** {breed_data[breed2].get('기본특징', '정보 없음')}")
                    st.write(f"**성격:** {breed_data[breed2].get('성격', '정보 없음')}")
                    st.write(f"**건강상 유의점:** {breed_data[breed2].get('건강상 유의점', '정보 없음')}")
                    st.write(f"**털 관리:** {breed_data[breed2].get('털 관리', '정보 없음')}")
                    st.write(f"**운동/활동:** {breed_data[breed2].get('운동/활동', '정보 없음')}")
                else:
                    st.info("해당 품종에 대한 상세 정보가 없습니다.")
    else:
        st.info("품종 정보를 로드할 수 없습니다.")

with tab4:
    st.header("🎁 맞춤형 강아지 제품 추천")
    st.write("우리 강아지에게 딱 맞는 제품을 추천해 드립니다.")

    st.subheader("강아지 정보 입력")
    dog_name = st.text_input("강아지 이름", key="rec_dog_name")
    dog_breed_rec = st.selectbox("강아지 품종", load_class_names(), key="rec_dog_breed")
    dog_age = st.number_input("강아지 나이 (년)", min_value=0, max_value=30, value=1, key="rec_dog_age")
    dog_weight_rec = st.number_input("강아지 체중 (kg)", min_value=0.1, value=5.0, step=0.1, key="rec_dog_weight")

    if st.button("제품 추천받기", key="get_recommendations_btn"):
        if dog_name and dog_breed_rec:
            st.subheader(f"{dog_name}를 위한 맞춤 추천")
            st.write(f"**{dog_breed_rec}** 품종, **{dog_age}살**, **{dog_weight_rec}kg**의 {dog_name}에게 추천하는 제품입니다.")
            
            st.markdown("**사료 추천:**")
            st.info("활동량과 나이에 맞는 고품질 사료 (예: 로얄캐닌, 내추럴발란스)")
            
            st.markdown("**간식 추천:**")
            st.info("치석 제거에 도움이 되는 덴탈껌, 훈련용 보상 간식")

            st.markdown("**장난감 추천:**")
            st.info("내구성이 좋고 안전한 터그놀이 장난감, 노즈워크 장난감")

            st.markdown("**용품 추천:**")
            st.info("편안한 쿠션, 미끄럼 방지 식기, 안전한 하네스")
        else:
            st.warning("강아지 이름과 품종을 입력해주세요.")

with tab5:
    st.header("🎨 AI 강아지 이미지 생성")
    st.write("원하는 강아지 이미지를 AI로 생성해 보세요!")

    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. secrets.toml 파일에 'openai_api_key'를 설정해주세요.")
    else:
        prompt_image = st.text_input("생성하고 싶은 강아지 이미지에 대한 설명을 입력하세요 (예: '푸른 잔디밭에서 뛰어노는 골든 리트리버')", key="image_gen_prompt")
        image_size = st.selectbox("이미지 크기", ["1024x1024", "1024x1792", "1792x1024"], key="image_gen_size")
        num_images = st.slider("생성할 이미지 개수", min_value=1, max_value=4, value=1, key="image_gen_num")

        if st.button("이미지 생성하기", key="generate_image_btn"):
            if prompt_image:
                with st.spinner("이미지를 생성 중입니다... 잠시만 기다려 주세요."):
                    try:
                        client = OpenAI(api_key=openai_api_key)
                        response = client.images.generate(
                            model="dall-e-3", # 또는 "dall-e-3" (API 키에 따라)
                            prompt=prompt_image,
                            size=image_size,
                            n=num_images
                        )
                        for img_data in response.data:
                            st.image(img_data.url, caption=prompt_image, use_column_width=True)
                    except Exception as e:
                        st.error(f"이미지 생성 중 오류가 발생했습니다: {e}")
                        st.info("OpenAI API 키가 유효한지, 그리고 DALL-E 모델 사용 권한이 있는지 확인해주세요.")
            else:
                st.warning("이미지 설명을 입력해주세요.")

with tab6:
    st.header("❓ 강아지 품종 맞추기 퀴즈")
    st.write("강아지 품종에 대한 지식을 테스트해 보세요!")

    all_dog_images = get_all_dog_images()
    if not all_dog_images:
        st.warning("퀴즈에 사용할 강아지 이미지를 로드할 수 없습니다. 'data/Images' 폴더를 확인해주세요.")
    else:
        if st.session_state.current_quiz is None:
            # Select a random image for the quiz
            quiz_image_info = random.choice(all_dog_images)
            correct_breed = quiz_image_info['breed']
            
            # Generate incorrect options
            class_names = load_class_names()
            incorrect_options = random.sample([b for b in class_names if b != correct_breed], min(3, len(class_names) - 1))
            options = incorrect_options + [correct_breed]
            random.shuffle(options)

            st.session_state.current_quiz = {
                "image_path": quiz_image_info['path'],
                "correct_breed": correct_breed,
                "options": options
            }
            st.session_state.quiz_feedback = ""

        quiz = st.session_state.current_quiz
        st.image(quiz['image_path'], caption="이 강아지의 품종은 무엇일까요?", use_column_width=True)

        selected_option = st.radio("정답을 선택하세요:", quiz['options'], key="quiz_radio")

        col_quiz_btn1, col_quiz_btn2 = st.columns(2)
        if col_quiz_btn1.button("정답 확인", key="check_quiz_btn"):
            if selected_option == quiz['correct_breed']:
                st.session_state.quiz_feedback = "🎉 정답입니다!" 
                st.session_state.quiz_score += 1
            else:
                st.session_state.quiz_feedback = f"❌ 오답입니다. 정답은 **{quiz['correct_breed']}** 입니다."
            st.session_state.current_quiz = None # Reset quiz for next round
            st.rerun()

        if col_quiz_btn2.button("다음 문제", key="next_quiz_btn"):
            st.session_state.current_quiz = None
            st.session_state.quiz_feedback = ""
            st.rerun()

        st.markdown(st.session_state.quiz_feedback)
        st.write(f"현재 점수: {st.session_state.quiz_score}점")

# --- 탭 7: 주변 장소 찾기 (프로토타입) ---
# --- 탭 7: 주변 장소 찾기 (구글 맵 API 연동) ---
with tab7:
    st.header("🗺️ 내 주변 반려견 동반 장소 찾기")
    st.write("구글 맵 API를 이용하여 내 주변의 반려견 동반 장소를 검색합니다.")

    # --- API 키 확인 ---
    google_maps_api_key = st.secrets.get("GOOGLE_MAPS_API_KEY")
    if google_maps_api_key:
        st.info(f"API 키를 성공적으로 읽었습니다. (시작: {google_maps_api_key[:5]}...)")

    if not google_maps_api_key:
        st.error("구글 맵 API 키가 설정되지 않았습니다. 아래 가이드를 따라 키를 설정해주세요.")
        st.info("""
        **🔑 구글 맵 API 키 설정 방법**

        1.  **Google Cloud Platform 접속 및 로그인**: [https://console.cloud.google.com](https://console.cloud.google.com)
        2.  **새 프로젝트 생성 또는 기존 프로젝트 선택**
        3.  **API 활성화**:
            -   좌측 메뉴에서 'API 및 서비스' > '라이브러리'로 이동합니다.
            -   '**Maps JavaScript API**'와 '**Places API**'를 검색하여 각각 '사용 설정'합니다.
        4.  **API 키 발급**:
            -   좌측 메뉴에서 'API 및 서비스' > '사용자 인증 정보'로 이동합니다.
            -   '사용자 인증 정보 만들기' > 'API 키'를 클릭하여 키를 발급받습니다.
            -   **보안을 위해 발급받은 키의 'API 제한사항'을 설정하는 것을 강력히 권장합니다.** (HTTP 리퍼러 제한: `http://localhost:8501/*`)
        5.  **secrets.toml 파일 생성**: 프로젝트 폴더 안에 `.streamlit` 폴더를 만들고, 그 안에 `secrets.toml` 파일을 생성합니다.
        6.  **파일에 키 저장**:
            ```toml
            # .streamlit/secrets.toml
            GOOGLE_MAPS_API_KEY = "여기에_복사한_API_키를_붙여넣으세요"
            ```
        7.  **앱 재실행**: 파일을 저장한 후, Streamlit 앱을 다시 실행하면 API 키가 적용됩니다.
        """)
    else:
        # --- 검색 UI ---
        col1, col2 = st.columns([2, 1])
        location = col1.text_input("중심 위치를 입력하세요", placeholder="예: 강남역, 해운대 해수욕장")
        category_options = {
            "카페": "cafe",
            "식당": "restaurant",
            "동물병원": "veterinary_care",
            "공원": "park",
            "숙소": "lodging"
        }
        selected_category_display = col2.selectbox("장소 종류", list(category_options.keys()))
        category_type = category_options[selected_category_display]
        radius = st.slider("검색 반경 (km)", min_value=1, max_value=20, value=5, step=1)

        if st.button("🔍 검색하기"):
            if not location:
                st.warning("중심 위치를 입력해주세요.")
            else:
                with st.spinner(f"'{location}' 주변의 '{selected_category_display}' 장소를 검색 중입니다..."):
                    gmaps = googlemaps.Client(key=google_maps_api_key)

                    try:
                        # 1. 중심 위치의 위도/경도 얻기
                        geocode_result = gmaps.geocode(location)
                        if not geocode_result:
                            st.error(f"'{location}'의 위치를 찾을 수 없습니다. 정확한 주소나 장소 이름을 입력해주세요.")
                            st.stop()
                        
                        center_lat = geocode_result[0]['geometry']['location']['lat']
                        center_lon = geocode_result[0]['geometry']['location']['lng']
                        
                        # 2. 주변 장소 검색 (Places API - Nearby Search)
                        places_result = gmaps.places_nearby(
                            location=(center_lat, center_lon),
                            radius=radius * 1000, # km를 미터로 변환
                            type=category_type, # 구글 Places API 타입
                            keyword="애견동반" # 키워드 검색
                        )

                        results = places_result.get('results', [])

                        if not results:
                            st.warning("검색 결과가 없습니다. 다른 키워드나 반경으로 시도해보세요.")
                        else:
                            st.success(f"총 {len(results)}개의 장소를 찾았습니다!")
                            
                            # --- 결과 표시 (지도 + 목록) ---
                            places_data = []
                            for place in results:
                                # Fetch place details for phone number
                                place_details = gmaps.place(place_id=place['place_id'], fields=['formatted_phone_number'])
                                phone_number = place_details['result'].get('formatted_phone_number', 'N/A') if 'result' in place_details else 'N/A'

                                places_data.append({
                                    'place_name': place.get('name'),
                                    'category_name': place.get('types', [])[0] if place.get('types') else 'N/A',
                                    'address_name': place.get('vicinity'),
                                    'lat': place['geometry']['location']['lat'],
                                    'lon': place['geometry']['location']['lng'],
                                    'place_url': f"https://www.google.com/maps/search/?api=1&query={place.get('name')}&query_place_id={place.get('place_id')}",
                                    'phone_number': phone_number
                                })
                            
                            # Convert places_data to JSON string for JavaScript
                            places_json = json.dumps(places_data, ensure_ascii=False)

                            # Get center coordinates for the map
                            map_center_lat = center_lat
                            map_center_lon = center_lon

                            # Generate HTML for the custom map
                            map_html = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <title>Google Map</title>
                                <meta name="viewport" content="initial-scale=1.0, user-scalable=no">
                                <meta charset="utf-8">
                                <style>
                                    #map {{
                                        height: 100%;
                                    }}
                                    html, body {{
                                        height: 100%;
                                        margin: 0;
                                        padding: 0;
                                    }}
                                </style>
                            </head>
                            <body>
                                <div id="map" style="height: 500px; width: 100%;"></div>
                                <script>
                                    var map;
                                    var places = {places_json}; // Data from Streamlit
                                    var centerLat = {map_center_lat};
                                    var centerLon = {map_center_lon};

                                    function initMap() {{
                                        map = new google.maps.Map(document.getElementById('map'), {{
                                            center: {{lat: centerLat, lng: centerLon}},
                                            zoom: 13 // Adjust zoom level as needed
                                        }});

                                        places.forEach(function(place) {{
                                            var marker = new google.maps.Marker({{
                                                position: {{lat: place.lat, lng: place.lon}},
                                                map: map,
                                                title: place.place_name
                                            }});

                                            var contentString = `
                                                <div>
                                                    <strong>${{place.place_name}}</strong><br>
                                                    주소: ${{place.address_name}}<br>
                                                    전화번호: ${{place.phone_number}}<br>
                                                    <a href="${{place.place_url}}" target="_blank">구글맵에서 보기</a>
                                                </div>
                                            `;

                                            var infoWindow = new google.maps.InfoWindow({{
                                                content: contentString
                                            }});

                                            marker.addListener('click', function() {{
                                                infoWindow.open(map, marker);
                                            }});
                                        }});
                                    }}
                                </script>
                                <script async defer
                                    src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap">
                                </script>
                            </body>
                            </html>
                            """
                            st.components.v1.html(map_html, height=500)

                            # Reuse existing logic for displaying the list below the map
                            places_df = pd.DataFrame(places_data) # Recreate DataFrame for list display
                            st.subheader("검색된 장소 목록")
                            for i, row in places_df.iterrows():
                                st.markdown(f"**{row['place_name']}**")
                                st.write(f"*카테고리*: {row['category_name']}")
                                st.write(f"*주소*: {row['address_name']}")
                                st.write(f"*전화번호*: {row['phone_number']}")
                                st.markdown(f"[구글맵에서 보기]({row['place_url']})", unsafe_allow_html=True)
                                st.markdown("---")
                    except Exception as e:
                        st.error(f"구글 맵 API 호출 중 오류가 발생했습니다: {e}")
                        st.info("API 키가 유효한지, 그리고 'Maps JavaScript API'와 'Places API'가 활성화되어 있는지 확인해주세요.")

# --- 탭 8: 성장/건강 일지 ---
with tab8:
    st.header("📔 성장/건강 일지")
    st.write("반려견의 성장 과정과 건강 상태를 기록하고 관리하세요.")

    # 수정 모드를 위한 세션 상태 초기화
    if "editing_index" not in st.session_state:
        st.session_state.editing_index = None

    # --- 입력/수정 폼 ---
    entry_to_edit = {}
    if st.session_state.editing_index is not None:
        try:
            entry_to_edit = st.session_state.diary_data[st.session_state.editing_index]
        except IndexError:
            st.session_state.editing_index = None # Invalid index, reset

    expanded_state = st.session_state.editing_index is not None

    with st.expander("➕ 새 기록 추가 / ✏️ 기록 수정", expanded=expanded_state):
        form_key = "diary_entry_form"
        if st.session_state.editing_index is not None:
            form_key = f"diary_edit_form_{st.session_state.editing_index}"

        with st.form(key=form_key):
            default_date = datetime.strptime(entry_to_edit.get('date'), '%Y-%m-%d').date() if 'date' in entry_to_edit else datetime.now().date()
            default_weight = entry_to_edit.get('weight', 0.0)
            default_feed = entry_to_edit.get('feed', 0)
            default_memo = entry_to_edit.get('memo', '')

            col1, col2, col3 = st.columns(3)
            date = col1.date_input("날짜", default_date)
            weight = col2.number_input("체중 (kg)", min_value=0.0, value=default_weight, format="%.2f")
            feed = col3.number_input("식사량 (g)", min_value=0, value=default_feed)
            memo = st.text_area("메모 (예: 예방접종, 특이사항 등)", value=default_memo)

            col_buttons = st.columns(2)
            if st.session_state.editing_index is not None:
                submit_button = col_buttons[0].form_submit_button(label="수정 완료")
                cancel_button = col_buttons[1].form_submit_button(label="수정 취소")
            else:
                submit_button = col_buttons[0].form_submit_button(label="기록 저장")
                cancel_button = None

            if submit_button:
                new_entry = {
                    'date': date.strftime('%Y-%m-%d'),
                    'weight': weight,
                    'feed': feed,
                    'memo': memo
                }
                if st.session_state.editing_index is not None:
                    st.session_state.diary_data[st.session_state.editing_index] = new_entry
                    st.success("기록이 성공적으로 수정되었습니다!")
                    st.session_state.editing_index = None
                else:
                    st.session_state.diary_data.append(new_entry)
                    st.success("일지가 성공적으로 저장되었습니다!")

                st.session_state.diary_data.sort(key=lambda x: x['date'], reverse=True)
                save_diary_data(st.session_state.diary_data)
                st.rerun()

            if cancel_button:
                st.session_state.editing_index = None
                st.rerun()

    st.markdown("---")

    # --- 기록 보기 및 관리 (st.dataframe 사용) ---
    st.subheader("기록 보기 및 관리")
    
    if not st.session_state.diary_data:
        st.info("아직 작성된 일지가 없습니다. 새 기록을 추가해보세요.")
    else:
        diary_df = pd.DataFrame(st.session_state.diary_data)
        diary_df['date_dt'] = pd.to_datetime(diary_df['date'])
        diary_df = diary_df.sort_values(by='date_dt', ascending=False).reset_index(drop=True) # Sort by date descending

        # 검색 기능
        search_term = st.text_input("메모 내용 검색", placeholder="예: '병원' 또는 '산책'", key="diary_search_input")
        if search_term:
            diary_df = diary_df[diary_df['memo'].str.contains(search_term, case=False, na=False)]

        if diary_df.empty and search_term:
            st.warning("검색 결과가 없습니다.")
        elif diary_df.empty:
            st.info("아직 작성된 일지가 없습니다. 새 기록을 추가해보세요.")
        else:
            # Display dataframe
            st.dataframe(diary_df.drop(columns=['date_dt']), use_container_width=True, height=300)

            # Selectbox for edit/delete
            # Create a list of display options that includes the original index for easy lookup
            display_options_with_index = []
            for original_idx, row in diary_df.iterrows():
                display_options_with_index.append({
                    "label": f"{row['date']} - {row['memo'][:50]}...", # Truncate memo for display
                    "original_index": original_idx
                })

            # Use a regular selectbox for display
            selected_option_label = st.selectbox(
                "수정 또는 삭제할 기록을 선택하세요.",
                options=[opt["label"] for opt in display_options_with_index],
                index=0 if display_options_with_index else None,
                key="diary_record_selector"
            )

            # Find the original index based on the selected label
            selected_original_index = None
            if selected_option_label:
                for opt in display_options_with_index:
                    if opt["label"] == selected_option_label:
                        selected_original_index = opt["original_index"]
                        break

            if selected_original_index is not None:
                col_action = st.columns(2)
                if col_action[0].button("선택된 기록 수정", key="edit_selected_record"):
                    st.session_state.editing_index = selected_original_index
                    st.rerun()
                
                if col_action[1].button("선택된 기록 삭제", key="delete_selected_record"):
                    # Ensure we delete from the original, unsorted/unfiltered list
                    del st.session_state.diary_data[selected_original_index]
                    save_diary_data(st.session_state.diary_data)
                    st.success("기록이 성공적으로 삭제되었습니다!")
                    st.rerun()

    # --- 체중 변화 그래프 ---
    st.subheader("체중 변화 그래프")
    if not st.session_state.diary_data:
        st.info("기록이 없어 그래프를 표시할 수 없습니다.")
    else:
        diary_df_for_graph = pd.DataFrame(st.session_state.diary_data)
        if 'weight' in diary_df_for_graph.columns and not diary_df_for_graph.empty:
            diary_df_for_graph['date_dt'] = pd.to_datetime(diary_df_for_graph['date'])
            diary_df_for_graph = diary_df_for_graph.sort_values(by='date_dt') # Sort by date ascending for graph
            st.line_chart(diary_df_for_graph.set_index('date_dt')['weight'])
        else:
            st.info("체중 기록이 없어 그래프를 표시할 수 없습니다.")

# --- 탭 9: 행동/소리 분석 (체험형) ---
with tab9:
    st.header("🗣️ 행동/소리 분석 (재미로 해보기)")
    st.warning("이 기능은 실제 AI 분석이 아닌, 재미를 위한 체험용 기능입니다.")

    analysis_type = st.radio("분석할 유형을 선택하세요.", ["짖는 소리 분석", "꼬리 행동 분석"])

    uploaded_media = None
    if analysis_type == "짖는 소리 분석":
        uploaded_media = st.file_uploader("강아지 짖는 소리 파일을 업로드하세요 (mp3, wav)", type=['mp3', 'wav'])
    else:
        uploaded_media = st.file_uploader("강아지 꼬리 행동 동영상을 업로드하세요 (mp4, mov)", type=['mp4', 'mov'])

    if uploaded_media:
        if st.button("분석하기"):
            with st.spinner("AI가 열심히 분석 중... (척하고 있습니다)"):
                import time
                time.sleep(3) # 분석하는 척 딜레이
                
                if analysis_type == "짖는 소리 분석":
                    results = [
                        "'맛있는 간식을 달라!'는 강력한 요구의 짖음이네요! 🍖",
                        "'낯선 사람이 온 것 같아요! 아빠, 확인해봐요!' 경계심이 느껴져요. 🚨",
                        "'산책 갈 시간이에요! 신난다!' 기쁨과 흥분이 가득 담겨있어요. 🌳",
                        "'조금 심심한 것 같아요. 저랑 놀아주세요.' 관심을 원하는 소리에요. 🎾"
                    ]
                else: # 꼬리 행동 분석
                    results = [
                        "꼬리를 높이 들고 빠르게 흔드는 걸 보니, 자신감이 넘치고 매우 신이 난 상태 같아요! 😄",
                        "꼬리가 수평보다 살짝 아래에 있고 부드럽게 흔들리네요. 편안하고 안정적인 기분인 것 같아요. 😊",
                        "꼬리를 다리 사이로 내린 것을 보니, 조금 무섭거나 불안한 상황인가 봐요. 안심시켜주세요. 😟",
                        "꼬리를 뻣뻣하게 세우고 짧고 빠르게 흔드는 것은, 무언가에 집중하고 경계하고 있다는 신호일 수 있어요. 🤔"
                    ]
                
                st.success("**분석 완료!**")
                st.info(random.choice(results))

with tab10:
    st.header("🤖 강화 학습 시뮬레이션: 미로 게임")
    st.write("미리 학습된(또는 최단 경로를 아는) 에이전트가 미로를 탐색하는 과정을 시뮬레이션합니다.")

    import maze_game
    import matplotlib.pyplot as plt

    col_maze1, col_maze2 = st.columns(2)
    maze_width = col_maze1.slider("미로 너비", min_value=10, max_value=50, value=20, step=2)
    maze_height = col_maze2.slider("미로 높이", min_value=10, max_value=50, value=20, step=2)

    if st.button("새 미로 생성 및 시뮬레이션 시작"):
        # Generate maze
        maze = maze_game.generate_maze(maze_width, maze_height)

        # Define start and end points (ensure they are not walls)
        start = (1, 1) # Default start
        end = (maze_height - 2, maze_width - 2) # Default end

        # Ensure start and end are not walls
        if maze[start[0], start[1]] == 1:
            for r in range(maze_height):
                for c in range(maze_width):
                    if maze[r, c] == 0:
                        start = (r, c)
                        break
                if maze[start[0], start[1]] == 0: break

        if maze[end[0], end[1]] == 1:
            for r in range(maze_height -1, -1, -1):
                for c in range(maze_width -1, -1, -1):
                    if maze[r, c] == 0:
                        end = (r, c)
                        break
                if maze[end[0], end[1]] == 0: break

        # Find path using BFS (simulating learned agent)
        path = maze_game.find_path_bfs(maze, start, end)

        if path:
            st.success("경로를 찾았습니다! 에이전트가 미로를 탐색합니다.")
            fig = maze_game.visualize_maze(maze, path, start, end)
            st.pyplot(fig, use_container_width=True) # Use use_container_width for better fit
            plt.close(fig) # Close figure to prevent memory issues
        else:
            st.error("경로를 찾을 수 없습니다. 미로를 다시 생성해 보세요.")

    st.info("이 시뮬레이션은 강화 학습 에이전트가 미로를 탐색하는 '결과'를 보여줍니다. 실제 학습 과정은 포함하지 않습니다.")
