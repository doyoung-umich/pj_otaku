
import sys
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from joypy import joyplot
import sweetviz as sv
from tqdm import tqdm
from collections import ChainMap
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy import sparse
pd.set_option("max_columns", 200)
from fuzzywuzzy import fuzz

QUERY_USER_ID = 1

class UserBasedFiltering:
    def __init__(self):
        # Load various data first
        # self.df_titles = pd.read_csv("../assets/titles_2000p.csv")
        # self.df_titles_genre = pd.read_csv("../assets/ryota_title_genre_2000p.csv")
        # self.df_mlist = pd.read_csv("../assets/media_list_all_users.csv")
        # self.df_mlist_genre = pd.read_csv("../assets/ryota_media_list_genre.csv")
        # self.df_user_genre_dist = pd.read_csv("../assets/ryota_user_genre_dist.csv")
        self.mat_title_user = sparse.load_npz("title_user_truncated.npz")
        self.titlle_idx_list = list(np.load("title_user_idx_truncated.npy"))
        self.model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)
        self.model.fit(self.mat_title_user)

    
    def recommend_from_other_user_histories(self, q_title_id, output_neighbors=10):
        '''
        Query by the given title_id. Refers to the title:user matrix
        '''
        q_title_idx = self.titlle_idx_list.index(q_title_id)
        distances, indices = self.model.kneighbors(self.mat_title_user[q_title_idx], n_neighbors=output_neighbors+1) # output_neighbors+1 because it always puts q_title_id as result
#         indices = indices[indices != q_title_idx] # remove queried title_id from result

        titlle_idx_arr = np.array(self.titlle_idx_list)
        recommended_title_ids = titlle_idx_arr[indices].reshape(-1)
        rt = pd.DataFrame([recommended_title_ids.tolist(), distances[0].tolist()]).T
        rt.columns = ['title_id','distances']
        
        return rt
