import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import matplotlib.pyplot as plt


def plot_images(path, character_ids):
    '''
    Shows first 10 images in 2x5 frame
    :params
        path: path to the image
        character_ids: id of characters to draw
    :returns
        it doesn't return but plots the images
    
    '''
    rows, columns = 2, 5
    imgs = []
    character_ids = character_ids[:10+1]

    # get the actual images
    for id in character_ids:
        try:
            img = Image.open(path + str(id) + ".png")
            imgs.append(np.array(img))
        except:
            pass

    # iterate over axis and show
    fig, axes = plt.subplots(rows, columns, figsize=(8,4))
    for img, ax in zip(imgs, axes.flatten()):
        ax.imshow(img, cmap="gray")
    plt.show()


class ImageBasedRecommendation:
    def __init__(self, query_path, version):
        print("model version: ", version)
        self.embedding_flat_np = np.load("/mnt/disks/sdb/home/dy0904k/assets/character_images/models_and_embeddings/image_embedding_"+version+".npy") 
        self.embedding_ids = np.load("/mnt/disks/sdb/home/dy0904k/assets/character_images/models_and_embeddings/image_embedding_character_ids_"+version+".npy") 
        self.df_characters = pd.read_csv("/mnt/disks/sdb/home/dy0904k/assets/characters_200p.csv")
        self.query_path = query_path

        # create character_based similarity matrix
        self.df_chara_sim_mat = pd.DataFrame(cosine_similarity(self.embedding_flat_np))
        self.df_chara_sim_mat.index = self.embedding_ids
        self.df_chara_sim_mat.columns = self.embedding_ids

        # create title_based similarity matrix
        np_embedding_id_concat = np.c_[self.embedding_ids.astype(int), self.embedding_flat_np] # create character_id:embeddings table
        df_embedding = pd.DataFrame(np_embedding_id_concat)
        df_embedding.rename(columns={0:"character_id"}, inplace=True)
        df_characters_unique = self.df_characters.drop_duplicates(subset="character_id") 
        df_title_char = df_characters_unique[["title_id", "character_id"]] # get character:title reference table
        df_merged = pd.merge(df_title_char, df_embedding, how="inner", on="character_id")
        df_title_embedding_avg = df_merged.groupby("title_id").mean() # merge and calculate "average" of image features
        self.df_title_sim_mat = pd.DataFrame(cosine_similarity(df_title_embedding_avg.iloc[:, 1:])) # similarity calculation of titles
        self.df_title_sim_mat.index = df_title_embedding_avg.index
        self.df_title_sim_mat.columns = df_title_embedding_avg.index

    def recommend_titles_from_similar_characters(self, query_character_id, top_n):
        '''
        recommends titles from similar characters to the queried character
        :params
            query_character_id: character id of the queried character
            top_n: how many similar characters to get
        :returns
            list of recommended titles

        '''
        # show the querying character
        df_q = self.df_characters[self.df_characters["character_id"]==query_character_id]
        print("Queried character: ", df_q["character_name"].unique(), " who appears in: ", df_q["title_romaji"].unique())
        img = Image.open(self.query_path + str(query_character_id) + ".png")
        plt.imshow(np.array(img), cmap="gray")
        plt.show()

        # get similar character
        df = self.df_chara_sim_mat[str(query_character_id)].sort_values(ascending=False)
        df_top = df[1:top_n+1]
        top_ids = list(df_top.index.astype(int))
        plot_images(self.query_path, top_ids)
        
        # print character names
        print("Similar characters (in the order of appearance)")
        for chara_id in top_ids:
            df_recc = self.df_characters[self.df_characters["character_id"]==int(chara_id)]
            print("Character: ", df_recc["character_name"].unique(), " who appears in : ", df_recc["title_romaji"].unique())

        # get titles that each similar character appears in
        df_res = self.df_characters[self.df_characters["character_id"].isin(top_ids)]
        df_res = df_res.drop_duplicates(subset="character_name")
        recc_title_ids = df_res["title_id"].unique()
        
        return recc_title_ids

    def recommend_titles_from_similar_image_embedding(self, query_title_id, top_n):
        '''
        recommends titles from similar characters to the queried character
        :params
            query_title_id: title id of the queried title
            top_n: how many similar titles to get
        :returns
            list of recommended titles
        '''
        # query title and pull out similar titles
        df = self.df_title_sim_mat[query_title_id].sort_values(ascending=False)
        df_top = df[0:top_n+1]
        top_ids = df_top.index

        # get images for comparison
        for idx, id in enumerate(top_ids):
            df = self.df_characters[self.df_characters["title_id"]==id]
            if idx == 0:
                print("Querying title: ", df["title_romaji"].unique())
            else:
                print("Similar title: ", df["title_romaji"].unique())
            character_ids = df["character_id"].unique()
            plot_images(self.query_path, character_ids)

        return list(top_ids[1:]) # exclude first element as it's the queried title
