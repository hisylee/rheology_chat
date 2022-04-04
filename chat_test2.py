#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
#from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    #df = pd.read_csv('rheology_dataset_em.csv')
    df = pd.read_excel('rheology_dataset_em.xlsx', header = 0)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('식품물성학 챗봇')
#st.markdown("[세종대학교 식품가공학 연구실](http://home.sejong.ac.kr/~suyonglee/)")
st.write("식품 물성 관련 질문을 해주세요")
st.caption('(아직 학습 중이오니 이해부탁드려요.)')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('Questions: ', '')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['A'])

for i in range(len(st.session_state['past'])):
     st.write(f"Q: {st.session_state['past'][i]}")
     #st.write(st.session_state['past'][i])
     if len(st.session_state['generated']) > i:
         st.write(f"A: {st.session_state['generated'][i]}")

# for i in range(len(st.session_state['past'])):
#     message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#     if len(st.session_state['generated']) > i:
#         message(st.session_state['generated'][i], key=str(i) + '_bot')

