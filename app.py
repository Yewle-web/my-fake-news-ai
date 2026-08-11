import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="AI 가짜뉴스 판별 시스템", page_icon="🤖")

st.title("🤖 AI 가짜뉴스 판별 시스템")
st.write("영어 뉴스 기사 본문을 입력하시면 AI가 **진짜 뉴스(TRUE)**인지 **가짜 뉴스(FAKE)**인지 판별해 드립니다.")

# 텍스트 노이즈 정제 (특수문자, 특수기호 제거로 정확도 향상)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

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
    fake_df = read_csv_safe("Fake.csv")
    true_df = read_csv_safe("True.csv")
    
    # text 컬럼 추출
    fake_text = fake_df['text'] if 'text' in fake_df.columns else fake_df.iloc[:, 0]
    true_text = true_df['text'] if 'text' in true_df.columns else true_df.iloc[:, 0]
        
    df_fake = pd.DataFrame({'text': fake_text, 'label': 1})
    df_true = pd.DataFrame({'text': true_text, 'label': 0})
    
    # 💡 데이터 학습량 1,000개 ➡️ 10,000개로 대폭 확대 (정확도 대폭 상승)
    df = pd.concat([df_fake.head(10000), df_true.head(10000)], axis=0).reset_index(drop=True)
    
    # 텍스트 정제
    df['clean_text'] = df['text'].apply(clean_text)
    
    # 💡 n-gram (1단어~2단어 조합 분석) 및 max_features 10,000개로 확대
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    
    model.fit(df['clean_text'], df['label'])
    return model

try:
    with st.spinner("데이터 20,000건을 고성능 학습 중입니다... (약 10~20초 소요)"):
        model = train_model()
    st.success("고성능 AI 모델 준비 완료!")
except Exception as e:
    st.error(f"모델 학습 중 에러 발생: {e}")
    model = None

st.divider()

news_text = st.text_area("📰 검사할 뉴스 기사 본문을 여기에 붙여넣으세요:", height=200)

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
