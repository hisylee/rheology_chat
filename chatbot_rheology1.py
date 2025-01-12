#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Transformers

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# 캐시된 모델 로드
@st.cache_resource
def load_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

# 캐시된 데이터셋 로드
@st.cache_data
def load_dataset():
    try:
        df = pd.read_excel('rheology_dataset_em.xlsx', header=0)
        df['embedding'] = df['embedding'].apply(json.loads)
        return df
    except Exception as e:
        st.error(f"데이터셋을 로드하는 동안 오류가 발생했습니다: {e}")
        return pd.DataFrame()

# 모델 및 데이터 로드
model = load_model()
df = load_dataset()

# UI 헤더
st.header('식품물성학 챗봇')
st.write("식품 물성 관련 질문을 해주세요.")
st.caption('(아직 학습 중이오니 어색한 답변 이해부탁드려요.)')

# 세션 상태 초기화
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# 입력 폼
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('질문 입력:', '')
    submitted = st.form_submit_button('전송')

# 질문 제출 후 처리
if submitted and user_input:
    if df.empty:
        st.error("데이터셋이 비어 있습니다. 데이터 파일을 확인하세요.")
    else:
        try:
            embedding = model.encode(user_input)

            # 코사인 유사도 계산
            df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
            answer = df.loc[df['distance'].idxmax()]

            st.session_state.past.append(user_input)
            if answer['distance'] < 0.7:
                response = '어떻게 대답해야할 지 모르겠어요. 더 공부하도록 할께요 ㅠㅠ'
                st.session_state.generated.append(response)
            else:
                st.session_state.generated.append(answer['A'])

        except Exception as e:
            st.error(f"처리 중 오류가 발생했습니다: {e}")

# 이전 대화 기록 표시
for i in range(len(st.session_state['past'])):
    j = len(st.session_state['past']) - i - 1
    st.write(f"**Q: {st.session_state['past'][j]}**")
    st.write(f"**A: {st.session_state['generated'][j]}**")
    st.markdown('---')


# In[ ]:




