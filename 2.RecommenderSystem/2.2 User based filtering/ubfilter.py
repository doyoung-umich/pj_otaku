import sys
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy import sparse


class UserBasedFiltering:
    def __init__(self):
        '''
        Initializes UserBasedFiltering with necessary data to run it efficiently
        '''
        # Load various data first
        self.df_titles = pd.read_csv("/mnt/disks/sdb/home/dy0904k/assets/titles_2000p.csv")
        self.df_titles_genre = pd.read_csv("/mnt/disks/sdb/home/dy0904k/assets/ryota_title_genre_2000p.csv")
        self.df_mlist = pd.read_csv("/mnt/disks/sdb/home/dy0904k/assets/media_list_all_users.csv")
        self.df_mlist_genre = pd.read_csv("/mnt/disks/sdb/home/dy0904k/assets/ryota_media_list_genre.csv")
        self.df_user_genre_dist = pd.read_csv("/mnt/disks/sdb/home/dy0904k/assets/ryota_user_genre_dist.csv")
        self.mat_title_user = sparse.load_npz("/mnt/disks/sdb/home/dy0904k/assets/ryota_title_user.npz")
        self.titlle_idx_list = list(np.load("/mnt/disks/sdb/home/dy0904k/assets/ryota_title_user_idx.npy"))
        self.model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20)
        self.model.fit(self.mat_title_user)

    def get_similar_users_from_user_id(self, start_col, dist_metric="cosine_similarity", query_user_id=1, ascending=False):
        '''
        query similar users from user_id
        :params
            start_col: index of the column to start the similarity calculation from
            dist_metric: distance metric to be used
            query_user_id: querying user_id
            ascending: whether to sort similarity matrix based on the similarity scores in an ascending order
        :returns
            list of top 10 similar user_ids
        '''
        # Calculate user similarities
        df = self.df_user_genre_dist
        id_list = df["user_id"]
        if dist_metric=="euclidean_distances":
            df_sim_mat = pd.DataFrame(euclidean_distances(df.iloc[:, start_col:]))
        elif dist_metric=="manhattan_distances":
            df_sim_mat = pd.DataFrame(manhattan_distances(df.iloc[:, start_col:]))
        else:
            df_sim_mat = pd.DataFrame(cosine_similarity(df.iloc[:, start_col:]))
        df_sim_mat.index = id_list
        df_sim_mat.columns = id_list

        similar_users = df_sim_mat[query_user_id].sort_values(ascending=ascending).reset_index()
        top_10_similar_user_ids = list(similar_users.iloc[1:11, 0])
        return top_10_similar_user_ids


    def get_similar_users_from_titles(self, q_titles, threshold=50):
        '''
        query similar users from list of favorite title_ids
        :params
            q_titles: list of titles to query
            threshold: how many titles that a user needs to have in his/her list in order to be considered
        :returns
            list of top 10 similar user_ids
        '''
        # refer to users with more than 50 titles -> more stable genre distribution
        df_user_mlist_count = self.df_mlist_genre[["user_id", "mlist_count"]]
        df_user_mlist_count = df_user_mlist_count[df_user_mlist_count["mlist_count"]>threshold]
        ref_user_ids = df_user_mlist_count["user_id"].values

        # limit df_user_genre_dist to users with more than threshold n titles in their media list
        df_user_genre_dist_thresh = self.df_user_genre_dist[self.df_user_genre_dist["user_id"].isin(ref_user_ids)]

        # create genre map from title_id list
        df_titles_genre_ex = self.df_titles_genre[self.df_titles_genre["title_id"].isin(q_titles)]
        df_titles_genre_ex = df_titles_genre_ex.sum(axis=0) / len(df_titles_genre_ex)

        # get the genre distribution values and work out cosine similarity
        ex_genre_dist = df_titles_genre_ex.iloc[1:].values.reshape(1,-1)
        user_genre_dist = df_user_genre_dist_thresh.iloc[:,1:].values
        res = cosine_similarity(user_genre_dist, ex_genre_dist)
        res = res.reshape(-1)
        high_sim_idx = np.argsort(res)[-10:]
        top_10_similar_user_ids = ref_user_ids[high_sim_idx]
        return top_10_similar_user_ids


    def evaluate_by_overlap_titles(self, similar_user_ids, query_user_id=1):
        '''
        Work out the average ratio of titles overlap and use it as direct evaluation metric
        Higher the ratio of overlap = better similarity calculation
        :params
            similar_user_ids: list of similar user_ids
            query_user_id: user_id of the querying user. Default set to global variable QUERY_USER_ID
        :returns
            avgerage of overlap ratio
        '''
        df = self.df_mlist
        overlap_ratios = []
        df_q = df[df["user_id"]==query_user_id]
        q_u_titles = list(df_q["title_id"])

        for user_id in similar_user_ids:
            df_sim = df[df["user_id"]==user_id]
            sim_u_titles = list(df_sim["title_id"])
            overlap = list(set(sim_u_titles) & set(q_u_titles))
            overlap_ratios.append(len(overlap) / len(sim_u_titles))
        avg_overlap_ratio = sum(overlap_ratios) / len(overlap_ratios)
        return avg_overlap_ratio


    def recommend_unread_titles(self, n_titles, similar_user_list, query_user_id=1, method="refer_others"):
        '''
        It retrieves the media list of similar users and then recommend based on specified logic

        :params
            df_titles: titles df
            df_mlist: media list df
            n_titles: how many titles to recommend
            query_user: querying user_id
            similar_user_list: list of similar user_ids
            method: which method to make recommendation
        :returns
            list of title_id as recommendation
        '''
        # get title_ids that the querying user hasn't read but similar users have
        df_mlist_similar_user = self.df_mlist[self.df_mlist["user_id"].isin(similar_user_list)]
        df_mlist_q_user = self.df_mlist[self.df_mlist["user_id"]==query_user_id]
        q_users_titles = list(df_mlist_q_user["title_id"])
        df_mlist_similar_user_not_read = df_mlist_similar_user[~df_mlist_similar_user["title_id"].isin(q_users_titles)]

        if method=="refer_popularity":
            # refer_popularity method: get "favorites" count of the unread titles and return top n titles
            unread_list = list(df_mlist_similar_user_not_read["title_id"].unique())
            df_recommend_list = self.df_titles[self.df_titles["title_id"].isin(unread_list)]
            df_recommend_list = df_recommend_list[["title_id", "favorites"]].sort_values(by="favorites", ascending=False).iloc[:n_titles]
            recommend_list = list(df_recommend_list["title_id"])
        else:
            # refer_others method: get count of titles and return top n titles
            df_recommend_list = df_mlist_similar_user_not_read.groupby("title_id").size().sort_values(ascending=False).iloc[:n_titles]
            recommend_list = list(df_recommend_list.index)
        return recommend_list


    def recommend_from_other_user_histories(self, q_title_id, output_neighbors=10):
        '''
        Query by the given title_id. Refers to the title:user matrix
        :params
            q_title_id: querying user_id
            output_neighbors: n_neighbors parameter to set with NearestNeighbors
        :returns
            list of title_id recommendations

        '''
        q_title_idx = self.titlle_idx_list.index(q_title_id)
        distances, indices = self.model.kneighbors(self.mat_title_user[q_title_idx], n_neighbors=output_neighbors+1) # output_neighbors+1 because it always puts q_title_id as result
        indices = indices[indices != q_title_idx] # remove queried title_id from result

        titlle_idx_arr = np.array(self.titlle_idx_list)
        recommended_title_ids = titlle_idx_arr[indices].reshape(-1)
        return recommended_title_ids


