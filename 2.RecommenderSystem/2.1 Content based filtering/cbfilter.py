import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

class ContentBasedFiltering:
    def __init__(self):
        self.sim_mat = None
        self.titles_df = pd.read_csv('/home/dy0904k/assets/titles_200p_cleaned.csv')
        self.title_romaji_map = self.titles_df.set_index('title_id')['title_romaji'].to_dict()
        self.popular_titles = self.titles_df.loc[lambda x : x.popularity > 10000]['title_id'].tolist()
        self.similarity_metric = None
        
    def create_sim_mat(self, df, method = 'cosine_similarity'):
        """
        Create a similarity matrix with a title-feature dataframe using chosen method.
        a title-feature dataframe should be formatted as follows:
        - title_ids in the indexes
        - features(e.g., genres, synopsis, etc.) in the columns
        
        *parameters
        - df(Pandas DataFrame object): title-feature dataframe
        - method(String): ['cosine_similarity', ' manhattan_distances', 'euclidean_distances']
        
        *attributes
        - self.sim_mat(Pandas DataFrame object): similarity matrix created from title-feature dataframe
        """
        
        df = df.loc[lambda x : x.index.isin(self.titles_df.title_id)]
        if method == 'cosine_similarity':
            sim_mat = cosine_similarity(df)
        elif method == 'manhattan_distances':
            sim_mat = manhattan_distances(df)
        elif method == 'euclidean_distances':
            sim_mat = euclidean_distances(df)
        else:
            raise ValueError("method not in {'cosine_similarity', 'manhattan_distances', 'euclidean distances'}")
        
        self.similarity_metric = method
        self.sim_mat = pd.DataFrame(sim_mat, index = df.index, columns = df.index)
        
    def check_sanity(self, title_id, max_num = 20, in_romaji = True, only_popular = True):
        """
        Check the sanity of the system with the chosen title_id.
        The system will push similar titles to given title_id
        
        *parameters
        - title_id(Integer): title_id of a title
        - max_num(Integer): number of titles the system will push
        - in_romaji(Boolean): if True, the result will be presented with title_romajis instead of title_ids
        - only_popular(Boolean): if True, the system will push titles whose popularity exceeds 10,000
        
        *return
        - sim_rank(Pandas DataFrame object): a list of similar titles to given title_id sorted by similarity
        """
        sim_mat = self.sim_mat
        if only_popular == True:
            sim_mat = self.sim_mat[self.popular_titles]
            
        if self.similarity_metric == 'cosine_similarity':
            ascending = False
        else:
            ascending = True
        
        sim_rank = sim_mat.loc[title_id].sort_values(ascending = ascending)[1:max_num + 1].to_frame()
        if in_romaji == True:
            sim_rank.index = sim_rank.index.map(self.title_romaji_map)
            sim_rank.columns = [self.title_romaji_map[title_id]]
        return sim_rank
