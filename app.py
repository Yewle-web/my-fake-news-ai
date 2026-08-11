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

# 텍스트 노이즈 정제 함수
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
    df_fakes = read_csv_safe("FA-KES-Dataset.csv")
    
    # 제목(article_title)과 본문(article_content) 결합
    title_col = 'article_title' if 'article_title' in df_fakes.columns else df_fakes.columns[1]
    content_col = 'article_content' if 'article_content' in df_fakes.columns else df_fakes.columns[2]
    label_col = 'labels' if 'labels' in df_fakes.columns else df_fakes.columns[-1]
    
    df_fakes['full_text'] = df_fakes[title_col].fillna('') + " " + df_fakes[content_col].fillna('')
    
    # FA-KES 라벨 정리: 0 = FAKE(가짜), 1 = TRUE(진짜)
    # 기존 코드 호환을 위해 Fake = 1, True = 0 으로 변환
    df_fakes['target_label'] = df_fakes[label_col].apply(lambda x: 1 if str(x).strip() == '0' else 0)
    
    train_df = pd.DataFrame({
        'text': df_fakes['full_text'],
        'label': df_fakes['target_label']
    })
    
    # 추가로 True.csv 파일이 레포지토리에 있다면 함께 병합하여 성능 보강
    try:
        true_df = read_csv_safe("True.csv")
        t_text = true_df['text'] if 'text' in true_df.columns else true_df.iloc[:, 0]
        df_true_extra = pd.DataFrame({'text': t_text.head(1000), 'label': 0})
        train_df = pd.concat([train_df, df_true_extra], axis=0).reset_index(drop=True)
    except Exception:
        pass  # True.csv가 없어도 FA-KES 단독으로 학습 진행
        
    train_df['clean_text'] = train_df['text'].apply(clean_text)
    
    # 머신러닝 파이프라인
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(C=2.0, max_iter=1000))
    ])
    
    model.fit(train_df['clean_text'], train_df['label'])
    return model

try:
    with st.spinner("FA-KES 고품질 데이터셋으로 AI 모델을 학습 중입니다..."):
        model = train_model()
    st.success("고성능 AI 모델 준비 완료!")
except Exception as e:
    st.error(f"모델 학습 중 에러 발생: {e}")
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
            st.success(f"✅ **
