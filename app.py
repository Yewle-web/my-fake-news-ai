import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI 가짜뉴스 판별 시스템", page_icon="🤖")

st.title("🤖 AI 가짜뉴스 판별 시스템")
st.write("영어 뉴스 기사 본문을 입력하시면 AI가 **진짜 뉴스(TRUE)**인지 **가짜 뉴스(FAKE)**인지 판별해 드립니다.")

# 버퍼 오버플로우 및 인코딩 에러를 완벽히 방지하는 안전 로드 함수
def read_csv_safe(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp949']
    for enc in encodings:
        try:
            # engine='python' 및 on_bad_lines='skip'으로 깨진 데이터 통과
            return pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
        except Exception:
            continue
    return pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore', engine='python', on_bad_lines='skip')

@st.cache_resource
def train_model():
    fake_df = read_csv_safe("Fake.csv")
    true_df = read_csv_safe("True.csv")
    
    # text 컬럼 확인
    if 'text' not in fake_df.columns or 'text' not in true_df.columns:
        fake_text = fake_df.iloc[:, 0]
        true_text = true_df.iloc[:, 0]
    else:
        fake_text = fake_df['text']
        true_text = true_df['text']
        
    df_fake = pd.DataFrame({'text': fake_text, 'label': 1})
    df_true = pd.DataFrame({'text': true_text, 'label': 0})
    
    # 빠른 학습과 안정적인 구동을 위해 상위 1,000개씩 사용
    df = pd.concat([df_fake.head(1000), df_true.head(1000)], axis=0).reset_index(drop=True)
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000)),
        ('clf', LogisticRegression())
    ])
    
    model.fit(df['text'].astype(str), df['label'])
    return model

try:
    with st.spinner("AI 모델을 학습하는 중입니다..."):
        model = train_model()
    st.success("AI 준비 완료!")
except Exception as e:
    st.error(f"파일을 읽는 중 에러가 발생했습니다: {e}")
    model = None

st.divider()

news_text = st.text_area("📰 검사할 뉴스 기사 본문을 여기에 붙여넣으세요:", height=200)

if st.button("🔍 가짜뉴스 검사하기"):
    if not news_text.strip():
        st.warning("뉴스 기사 내용을 입력해주세요!")
    elif model is not None:
        pred = model.predict([news_text])[0]
        prob = model.predict_proba([news_text])[0]
        fake_prob = prob[1] * 100
        
        st.subheader("📊 검사 결과")
        if pred == 1:
            st.error(f"🚨 **가짜 뉴스(FAKE)**일 확률이 높습니다! ({fake_prob:.1f}%)")
        else:
            st.success(f"✅ **진짜 뉴스(TRUE)**일 확률이 높습니다! (가짜뉴스 확률: {fake_prob:.1f}%)")
            
        st.progress(int(fake_prob))
