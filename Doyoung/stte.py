import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

# bert_base = 'bert-base-nli-mean-tokens'
all_mini = 'all-MiniLM-L6-v2'
# sentence_t5 = 'Sentence-T5'

model = SentenceTransformer(all_mini)

sentence_embeddings = pd.read_csv('feature_synopsis_minilm.csv').set_index('title_id').values
titles = pd.read_csv('../assets/titles_200p.csv').drop_duplicates()
popular_title = titles.loc[lambda x : x.popularity > 10000]['title_id'].tolist()
st.title('Text similarity using text embedding')
st.text('If you describe the plot you want, I recommend a comic/anime with similar content')

sentence = st.text_input("What is your favorite anime/comic")
wv = model.encode([sentence])

rec_df = pd.Series(cosine_similarity(wv, sentence_embeddings).flatten(), index = titles.title_id, name = 'similarity').sort_values(ascending = False).loc[lambda x : x.index.isin(popular_title)].to_frame()
df = rec_df.reset_index().merge(titles[['title_id','title_english','title_romaji','synopsis']], how = 'left', on = 'title_id').head(20)

if sentence != '':
    st.dataframe(df)


st.caption(f'Data provided by AniList: https://github.com/AniList/ApiV2-GraphQL-Docs')
st.caption('text embedding algorithm : https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1')
st.caption(f'View git repo: https://github.com/doyoung-umich/pj_otaku')