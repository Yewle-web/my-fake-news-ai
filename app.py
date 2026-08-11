import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI 가짜뉴스 판별 시스템", page_icon="🤖")

st.title("🤖 AI 가짜뉴스 판별 시스템")
st.write("영어 뉴스 기사 본문이나 제목을 입력하시면 AI가 **진짜 뉴스(TRUE)**인지 **가짜 뉴스(FAKE)**인지 판별해 드립니다.")

# 텍스트 노이즈 전처리 함수
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def read_csv_safe(file_path):
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp949']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc, engine='python', on_bad_lines='skip')
        except Exception:
            continue
    return pd.read_csv(file_path, encoding='utf-8', encoding_errors='ignore', engine='python', on_bad_lines='skip')

@st.cache_resource
def train_model():
    # FA-KES 데이터셋 로드
    df = read_csv_safe("FA-KES-Dataset.csv")
    
    # 컬럼명이 정확히 일치하지 않을 경우를 대비한 안전 로직
    title_col = 'article_title' if 'article_title' in df.columns else df.columns[1]
    content_col = 'article_content' if 'article_content' in df.columns else df.columns[2]
    label_col = 'labels' if 'labels' in df.columns else df.columns[-1]
    
    # 제목과 본문 결합 및 결측치 처리
    df['full_text'] = df[title_col].astype(str).fillna('') + " " + df[content_col].astype(str).fillna('')
    
    # 라벨 정리 (FA-KES 라벨: 0 = FAKE, 1 = TRUE)
    # 모델 출력을 위해 1 = FAKE, 0 = TRUE로 맞춤
    df['target'] = df[label_col].apply(lambda x: 1 if str(x).strip() == '0' else 0)
    
    # 텍스트 정제 적용
    df['clean_text'] = df['full_text'].apply(clean_text)
    
    # TF-IDF 및 머신러닝 모델 학습
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(C=2.0, max_iter=1000))
    ])
    
    model.fit(df['clean_text'], df['target'])
    return model

try:
    with st.spinner("FA-KES 고품질 데이터로 AI를 학습시키는 중입니다..."):
        model = train_model()
    st.success("고성능 AI 모델 준비 완료!")
except Exception as e:
    st.error(f"학습 중 오류 발생: {e}")
    model = None

st.divider()

news_text = st.text_area("📰 검사할 뉴스 기사(제목 또는 본문)를 입력하세요:", height=200)

if st.button("🔍 가짜뉴스 검사하기"):
    if not news_text.strip():
        st.warning("뉴스 기사 내용을 입력해주세요!")
    elif model is not None:
        cleaned_input = clean_text(news_text)
        pred = model.predict([cleaned_input])[0]
        prob = model.predict_proba([cleaned_input])[0]
        fake_prob = prob[1] * 100
        
        st.subheader("📊 검사 결과")
        if pred == 1:
            st.error(f"🚨 **가짜 뉴스(FAKE)**일 확률이 높습니다! ({fake_prob:.1f}%)")
        else:
            st.success(f"✅ **진짜 뉴스(TRUE)**일 확률이 높습니다! (가짜뉴스 확률: {fake_prob:.1f}%)")
            
        st.progress(int(fake_prob))
