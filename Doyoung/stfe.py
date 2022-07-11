import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from sklearn.metrics.pairwise import cosine_similarity


st.title('Content-based filtering algorithm')

genre = pd.read_csv('feature_genre.csv', index_col = 'title_id')
synopsis = pd.read_csv('feature_synopsis.csv', index_col = 'title_id')
tag = pd.read_csv('feature_tags.csv', index_col = 'title_id').loc[lambda x : x.index.isin(genre.index)]

features = {
    'genre':genre,
    'synopsis':synopsis,
    'tag':tag
}


    
def aggrid_interactive_table(df: pd.DataFrame):
    """Creates an st-aggrid interactive table based on a dataframe.

    Args:
        df (pd.DataFrame]): Source dataframe

    Returns:
        dict: The selected row
    """
    options = GridOptionsBuilder.from_dataframe(
        df, enableRowGroup=True, enableValue=True, enablePivot=True
    )

    options.configure_side_bar()

    options.configure_selection("single")
    selection = AgGrid(
        df,
        enable_enterprise_modules=True,
        gridOptions=options.build(),
        theme="light",
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
    )

    return selection



titles = pd.read_csv('../assets/titles_200p.csv', index_col = 'title_id')[['title_english','title_romaji','type','genres','synopsis']].loc[genre.index]


keyword = st.text_input("What is your favorite anime/comic")
if keyword == '':
    _titles = titles
else:
    _titles = titles.fillna('').loc[lambda x : (x.title_english.apply(lambda x : x.lower()).str.contains(keyword.lower())) | (x.title_romaji.apply(lambda x : x.lower()).str.contains(keyword.lower()))]

selection = aggrid_interactive_table(df=_titles.reset_index())

options = st.multiselect(
     'Choose features',
     ['genre','synopsis','tag'])

title_feature = []
if len(options) != 0:
    for option in options:
        title_feature.append(features[option])
    latent_mat = pd.concat(title_feature, axis = 1).fillna(0)

else:
    latent_mat = pd.concat(list(features.values()), axis = 1).fillna(0)
    
titles_sim = pd.DataFrame(cosine_similarity(latent_mat), index = latent_mat.index, columns = latent_mat.index)


if selection:
    try:
        selected_title = selection["selected_rows"][0]['title_romaji']
        selected_id = selection["selected_rows"][0]['title_id']
        st.write(f"Titles similar to {selected_title}:")
        sim_list = titles_sim.loc[selected_id].sort_values(ascending = False)[1:21].to_frame().rename(columns = {selected_id:'similarity'})
        st.dataframe(sim_list.reset_index().merge(titles, how = 'left', on = 'title_id'))
    except:
        pass
    