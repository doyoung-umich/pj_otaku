import json
from ast import literal_eval
import streamlit as st
import pandas as pd
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy import sparse
import datetime 
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from ubfilter import UserBasedFiltering


if 'yes' not in st.session_state:
    st.session_state['yes'] = False

placeholder = st.empty()
if st.session_state['yes'] == False:
    
    text = """
    **Welcome to Project Otaku!**
    
    Please note that:
    * We will **not** collect private information from you(e.g., name, email, gender, IP address, cookies, location, or other demographic information).
    * We will collect clickstream data that contains the session id(which is fully anonymized), AB test id, timestamp, favorite title_id, and title_id stored in your bucket. We will only use this to conduct a statistical test(AB test for different recommender algorithms) and the evaluation of the algorithms(to calculate metrics such as CTR, DCG, and Precision@k) for our capstone project.
    * We want to be completely transparent on the data collecting process and willing to share detailed information on how the clickstream data is collected. For more information, please check out the pdf below:
    https://github.com/doyoung-umich/pj_otaku/blob/main/otaku_database.pdf
    """
    with placeholder.container():
        st.markdown(text)

        yes = st.button('I understand and wish to proceed')
        st.session_state['yes'] = yes

    
    
if st.session_state['yes']:
    placeholder.empty()


    f = open('../../.Import/dburl.json')
    dburl = json.load(f)
    f.close()


    cred = credentials.Certificate('../../.Import/project-otaku.json')


    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, dburl)
    except:
        firebase_admin.initialize_app(cred, dburl)

    if 'session_id' not in st.session_state:
        dbdir = db.reference('sessions')
        if dbdir.get() == None:
            session_id = 1
        else:
            session_id = dbdir.get()['session_id'] + 1

        dbdir.update({'session_id':session_id})
        st.session_state['session_id'] = session_id
        st.session_state['abtest_id'] = np.random.choice([0,1])

        dbdir = db.reference('user_sessions')

        session_info = {'session_id':str(st.session_state['session_id']),
                        'abtest_id':str(st.session_state['abtest_id']),
                        'timestamp':str(datetime.datetime.now())}


        dbdir.push().set(session_info)


    dbdir = db.reference('test')


    st.title('Project Otaku: comic/anime recommender system')
    
    # if st.session_state['abtest_id'] == 0:
    #     st.write('cbf')
    # else:
    #     st.write('ubf')
    
    
    if 'dataset' not in st.session_state:
        st.session_state['titles'] = pd.read_csv('titles_200p_synopsis_cleaned.csv', index_col = 'title_id')


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



    keyword = st.text_input("What is your favorite anime/comic")
    if keyword == '':
        _titles = st.session_state['titles'].drop(['cover_image_url','adult'], axis = 1)
    else:
        _titles = st.session_state['titles'].fillna('').loc[lambda x : (x.title_english.apply(lambda x : x.lower()).str.contains(keyword.lower())) | (x.title_romaji.apply(lambda x : x.lower()).str.contains(keyword.lower()))].drop(['cover_image_url','adult'], axis = 1)

    selection = aggrid_interactive_table(df=_titles.reset_index())


    if 'selected_titles' not in st.session_state:
        if selection['selected_rows'] != []:
            st.session_state['selected_titles'] = pd.DataFrame(selection['selected_rows'][0], index = [0])
        else:
            st.session_state['selected_titles'] = pd.DataFrame(columns = _titles.columns)
    else:
        if selection['selected_rows'] != []:
            st.session_state['selected_titles'] = pd.concat([st.session_state['selected_titles'], pd.DataFrame(selection['selected_rows'][0], index = [0])]).drop_duplicates()
        else:
            pass

    # st.dataframe(st.session_state['selected_titles'])


    selected_titles_id = [s.loc['title_id'] for idx, s in st.session_state['selected_titles'].iterrows()]
    selected_titles_name = [s.loc['title_romaji'] for idx, s in st.session_state['selected_titles'].iterrows()]


    # st.write(selected_titles_id)
    # st.write(selected_titles_name)


    selected_titles = st.multiselect('Selected Titles', 
                                     selected_titles_name, 
                                     default = selected_titles_name,
                                     )

    if 'title_id' in st.session_state['selected_titles'].columns:
        st.session_state['selected_titles'] = st.session_state['selected_titles'].loc[lambda x : x.title_romaji.isin(selected_titles)]
        final_selected_title_id = st.session_state['selected_titles'].loc[lambda x : x.title_romaji.isin(selected_titles)].title_id.tolist()
        final_selected_title_name = st.session_state['selected_titles'].loc[lambda x : x.title_romaji.isin(selected_titles)].title_romaji.tolist()

    algo = st.radio(
     "algorithm:",
     ('CBF', 'UBF'))
    if algo == 'CBF':
        st.session_state['abtest_id'] = 0
    else:
        st.session_state['abtest_id'] = 1
    
####################### initializing algorithms #########################

    if st.session_state['abtest_id'] == 0:
        ##############################
        #                            #
        #  content based filtering   #
        #                            #
        ##############################
        
        if 'titles_sim' not in st.session_state:
            st.session_state['titles_sim'] = sparse.load_npz("latent_sim.npz")
            st.session_state['title_idx_num'] = pd.read_csv('title_idx_num.csv')
            
        # titles_sim = pd.DataFrame(cosine_similarity(latent_mat), index = latent_mat.index, columns = latent_mat.index)
        
    else:
        ##############################
        #                            #
        #    user based filtering    #
        #                            #
        ##############################
        
        if 'ubf' not in st.session_state:
            st.session_state['ubf'] = UserBasedFiltering() # initialize module


    if 'button' not in st.session_state:
        st.session_state['button'] = 0    

    if selected_titles:

        # selected_title = selection["selected_rows"][0]['title_romaji']

        if len(selection['selected_rows']) != 0:
            selected_id = selection["selected_rows"][0]['title_id']
        else:
            selected_id = -1
        st.write(f"You might also like:")
        
####################### actual algorithms #########################

        if st.session_state['abtest_id'] == 0:
            ##############################
            #                            #
            #  content based filtering   #
            #                            #
            ##############################

            idx = st.session_state['title_idx_num'].loc[lambda x : x.title_id.isin(final_selected_title_id)].index
            r = st.session_state['titles_sim'][idx]
            result_list = (-(r.toarray()**3).sum(axis = 0)).argsort()
            result_list = result_list[~np.isin(result_list, idx)][:50]
            
            result = st.session_state['title_idx_num'].loc[result_list].merge(st.session_state['titles'], how = 'left', on = 'title_id')
            # sim_list = st.session_state['titles_sim'].loc[final_selected_title_id]
            # wa = sim_list.values
            # wa[np.where(wa < 0)] = 0
            # sim_list = pd.DataFrame(wa**3, columns = st.session_state['latent_mat'].index).sum()
            # sim_list = sim_list.sort_values(ascending = False).to_frame()
            # # .sort_values(ascending = False).iloc[1:21].to_frame().rename(columns = {selected_id:'similarity'})
            # result = sim_list.reset_index().merge(st.session_state['titles'], how = 'left', on = 'title_id')
            fst_pattern = str(final_selected_title_name).replace(',','|').replace('[','').replace("'",'').replace(']','').replace('| ','|').lower()
            # st.write(fst_pattern)
            result = result.loc[lambda x : ~x.title_romaji.str.lower().str.contains(fst_pattern)].head(50)
          
            
            
        else:
            ##############################
            #                            #
            #    user based filtering    #
            #                            #
            ##############################
            
            # top_10_similar_user_ids = st.session_state['ubf'].get_similar_users_from_titles(final_selected_title_id)
            # st.session_state['ubf'].evaluate_by_overlap_titles(top_10_similar_user_ids)
            # recommended_titles = st.session_state['ubf'].recommend_unread_titles(50, top_10_similar_user_ids,final_selected_title_id, method="refer_others")        
            
            result = pd.DataFrame( columns = ['title_id','distances'])            
            for title in final_selected_title_id:
                result = pd.concat([result, st.session_state['ubf'].recommend_from_other_user_histories(title, output_neighbors=51).loc[lambda x : ~x.title_id.isin(result.title_id)]])
                
            result = result.sort_values('distances').loc[lambda x : ~x.title_id.isin(final_selected_title_id)].head(50)
            result = result.merge(st.session_state['titles'], how = 'left', on = 'title_id')
        
        
####################### pushing query data #########################        
        
        if 'selected_id' not in st.session_state:
            st.session_state['selected_id'] = selected_id
            dbdir = db.reference('query')

            update_query = {'session_id':st.session_state['session_id'],
                            'query':st.session_state['selected_id'],
                            'timestamp':str(datetime.datetime.now())}

            dbdir.push().set(update_query)

        else:
            if st.session_state['selected_id'] != selected_id and selected_id != -1:
                st.session_state['selected_id'] = selected_id
                dbdir = db.reference('query')

                update_query = {'session_id':st.session_state['session_id'],
                                'query':st.session_state['selected_id'],
                                'timestamp':str(datetime.datetime.now())}

                dbdir.push().set(update_query)
            else:
                pass


        dbdir = db.reference('test')

        buttons = []
        # st.write(len(result))


####################### displaying the result #########################
        
        for idx, s in result.iterrows():
            try:
                if type(s.loc['title_english']) == float:
                    st.subheader(f"{s.loc['title_romaji']}")
                else:
                    st.subheader(f"{s.loc['title_english']} / {s.loc['title_romaji']}")
                try:
                    st.text(s.loc['genres'][1:-1])
                except:
                    pass
                col1, col2 = st.columns([1, 3])
                with col1:
                    if s.loc['adult'] == False:
                        pic_url = literal_eval(s.loc['cover_image_url'])['large']
                    else:
                        pic_url = 'nsfw.png'
                    st.image(pic_url)
                with col2:
                    st.markdown(s.synopsis)

                # st.button("❤️",key = idx, on_click = onclick(buttons, idx))
                buttons.append(st.button("❤️",key = idx))
            except:
                buttons.append(False)
            
        if 'bucket' not in st.session_state:
            st.session_state['bucket'] = []

####################### pushing click data #########################            

        for i, b in enumerate(buttons): 
            if buttons[i]:
                st.session_state['button'] = i+1
                st.write(f"Button {st.session_state['button']} was clicked.")
                update = {'session_id':str(st.session_state['session_id']),
                          'abtest_id':str(st.session_state['abtest_id']),
                          'title_id':str(int(result.reset_index().iloc[i].loc['title_id'])), 
                          'relevant_index': str(st.session_state['button']), 
                          'timestamp': str(datetime.datetime.now())}
                
                # st.write(update)
                dbdir.push().set(update)
                chosen = st.session_state['titles'].loc[int(update['title_id'])].to_dict()
                st.session_state['bucket'].append(chosen)

####################### bucket #########################                

    with st.sidebar:
        st.subheader('Your bucket')
        if 'bucket' in st.session_state:

            # csv = pd.DataFrame(st.session_state['bucket'])
            if len(st.session_state['bucket']) > 0:
                st.download_button(
                     label="Download your bucket as CSV",
                     data = pd.DataFrame(st.session_state['bucket']).to_csv(),
                     file_name='your_bucket.csv'
                 )


            for i in st.session_state['bucket']:
                if type(i['title_english']) == float:
                    st.subheader(f"{i['title_romaji']}")
                else:
                    st.subheader(f"{i['title_english']} / {i['title_romaji']}")
                if i['adult'] == False:
                    pic_url = literal_eval(i['cover_image_url'])['large']
                else:
                    pic_url = 'nsfw.png'
                st.image(pic_url, width = 200)

        else:
            pass
    st.caption(f'Data provided by AniList: https://github.com/AniList/ApiV2-GraphQL-Docs')
    st.caption(f'View git repo: https://github.com/doyoung-umich/pj_otaku')
