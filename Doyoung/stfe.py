import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

st.title('Content-based filtering algorithm')
titles_sim = pd.read_csv('latent_sim.csv').set_index('title_id')
titles = pd.read_csv('../assets/titles_200p.csv', index_col = 'title_id')[['title_english','title_romaji','type','genres','synopsis']].loc[titles_sim.index]

titles_sim.columns = titles_sim.index

keyword = st.text_input("What is your favorite anime/comic")
if keyword == '':
    _titles = titles
else:
    _titles = titles.fillna('').loc[lambda x : (x.title_english.apply(lambda x : x.lower()).str.contains(keyword)) | (x.title_romaji.apply(lambda x : x.lower()).str.contains(keyword))]
    
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


selection = aggrid_interactive_table(df=_titles.reset_index())
if selection:
    try:
        selected_title = selection["selected_rows"][0]['title_romaji']
        selected_id = selection["selected_rows"][0]['title_id']
        st.write(f"Titles similar to {selected_title}:")
        sim_list = titles_sim.loc[selected_id].sort_values(ascending = False)[1:21].to_frame().rename(columns = {selected_id:'similarity'})
        st.dataframe(sim_list.reset_index().merge(titles, how = 'left', on = 'title_id'))
    except:
        pass
    