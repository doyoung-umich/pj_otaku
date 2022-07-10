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
    
    def evaluate_rmse(self, train_df, test_df):
        """
        return rmse of the system
        train_df and test_df should contain following columns from reviews dataset:
        ['title_id','user_id','score']
        
        
        *parameter
        - train_df(Pandas DataFrame object)
        - test_df(Pandas DataFrame object)
        
        *return
        - rmse(Float)
        - test_result(Pandas DataFrame object): a dataframe that contains the information about 
                                                actual scores and predicted scores for each titles for each users
        """
        if self.similarity_metric != 'cosine_similarity':
            raise ValueError("rmse evaluation only supports cosine similarity method")
        
        try:
            train = train_df[['user_id','title_id','score']]
            test = test_df[['user_id','title_id','score']]
        except:
            raise ValueError("train_df and test_df should contain following columns from reviews dataset: ['title_id','user_id','score']")
        
        train = train.loc[lambda x : x.title_id.isin(self.sim_mat.index)]
        test_mod = test.loc[lambda x : ~x.user_id.isin(set(test.user_id) - set(train.user_id)) & x.title_id.isin(self.sim_mat.index)]
        test_users = test_mod.user_id.unique()
        test_result = pd.DataFrame(columns = ['user_id', 'title_id', 'score_actual', 'score_predicted'])

        for user in tqdm(test_users):
            user_title_sim = self.sim_mat.loc[train.loc[lambda x : x.user_id == user]['title_id']].sort_values('title_id')
            user_score = train.loc[lambda x : x.user_id == user][['title_id','score']].sort_values('title_id')
            weight_sum = user_title_sim.sum()
            score_prediction = (user_title_sim.T.values @ user_score['score'].values)/weight_sum.values.flatten()
            score_prediction_dict = dict(zip(self.sim_mat.index, score_prediction))
            user_test_titles = test_mod.loc[lambda x : x.user_id == user]
            temp_df = user_test_titles[['user_id','title_id', 'score']]
            temp_df['score_predicted'] = temp_df['title_id'].map(score_prediction_dict)
            temp_df = temp_df.rename(columns = {'score':'score_actual'})
            test_result = pd.concat([test_result, temp_df]).fillna(0)
        mse = mean_squared_error(y_true= test_result['score_actual'].values, y_pred=test_result['score_predicted'].values)
        rmse = np.sqrt(mse)
        
        return rmse, test_result
    
    def evaluate_precision_recall(self, train_df, test_df, relevant_threshold = 70, num_push = 10):
        """
        return average precision and recall
        
        *parameter
        - train_df(Pandas DataFrame object)
        - test_df(Pandas DataFrame object)
        - relevant_threshold(Integer): the least amount of score a title should receive to be considered a 'relevant title'
        - num_push(Integer): the number of similar titles that the system push for each relevant title_id in the training dataset
        
        *return
        - avg_precision(Float)
        - avg_recall(Float)
        - precision_list(Pandas Series object): list of precision received from each user in the test dataset
        - recall_list(Pandas Series object): list of recall received from each user in the test dataset
        """
        train = train_df.loc[lambda x : x.title_id.isin(self.sim_mat.index)]
        test_mod = test_df.loc[lambda x : ~x.user_id.isin(set(test_df.user_id) - set(train.user_id)) & x.title_id.isin(self.sim_mat.index)]
        test_users = test_mod.user_id.unique()
        
        precision = {}
        recall = {}
        
        if self.similarity_metric == 'cosine_similarity':
            ascending = False
        else:
            ascending = True
        
        for user in tqdm(test_users):
            user_title_sim = self.sim_mat.loc[train.loc[lambda x : (x.user_id == user) & (x.score >= relevant_threshold)]['title_id'].drop_duplicates()].sort_values('title_id')
            user_score = train.loc[lambda x : (x.user_id == user) & (x.score >= relevant_threshold)][['title_id','score']].sort_values('title_id')
            rec_title = []
            for title in user_title_sim.index:
                rec_title += user_title_sim.loc[title].sort_values(ascending = ascending).index.tolist()[1:num_push + 1]

            rec_title = list(set(rec_title) - set(user_title_sim.index))
            user_test_titles = test_mod.loc[lambda x : (x.user_id == user) & (x.score >= relevant_threshold)].title_id.tolist()
            try:
                precision[user] = len(pd.Series(rec_title).loc[lambda x : x.isin(user_test_titles)])/len(rec_title)
                recall[user] = len(pd.Series(rec_title).loc[lambda x : x.isin(user_test_titles)])/len(user_test_titles)
            except:
                pass
        
        avg_precision = pd.Series(precision).mean()
        avg_recall = pd.Series(recall).mean()
        return avg_precision, avg_recall, pd.Series(precision), pd.Series(recall)
