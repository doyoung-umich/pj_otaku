# Project Otaku

### Project Goal and Motivation
The main goal of this project is to develop a recommender system that recommends a title of an Anime or “Manga” (Japanese comics), depending on the preferences of the querying user and the type of algorithm implemented.

### Repo folder structure
* 1.EDA
  - **henry_notebook_01.ipynb**: Descriptive statistics, correlation in genres, anime vs. manga comparison, wordcloud charts
  - **Network analysis of studios and staff.ipynb**: The network analysis of the studios and staff
  - **Structural analysis of synopsis BERT.ipynb**: Exploration of synopsis text - visualizations in the vector space
* 2.RecommenderSystem
  - **2.1 Content based filtering**
    - **cbf_walkthrough.ipynb**: An example usage of cbfilter.py
    - **cbfilter.py**: A recommendation module powered by content-based filtering
    - **Content Based Filtering Algorithm-Walkthrough.ipynb**: Feature engineering, Creating a title-latent factor matrix, Calculating the similarity between titles
  - **2.2 User based filtering**
    - **Recommendation Module.ipynb**: A recommendation module powered by user-based filtering and an example usage of the module
    - **Structural analysis of user data**: Data processing, calculate user similarity by genre probability distribution
  - **2.3 Image embedding**
    - **Character image recommendation.ipynb**: Data processing, model training, creating image embedding, creating a recommender module powered by image similarity
* 3.App and Evaluation
  - **abtest.ipynb**: Data retrieval from the realtime database, statistical tests, and calculation of evaluation metrics
  - **streamlit_app.py**: Development of a web application

### Example usages
#### Content-based filtering algorithm
```python
# refer to 2.RecommenderSystem/2.1 Content based filtering/cbf_walkthrought.ipynb
import pandas as pd
import numpy as np
from cbfilter import ContentBasedFiltering

"""
To use the ContentBasedFiltering class, first you should create a title-feature dataframe

A title-feature dataframe is a Pandas DataFrame object that takes title_ids as an index and their features as column
"""
title_feature = pd.read_csv('title_feature_matrix.csv')

cbf = ContentBasedFiltering()

#use create_sim_mat method to build similarity matrix
cbf.create_sim_mat(title_feature, method = 'cosine_similarity')

# Check the sanity of the system with the chosen title_id.
cbf.check_sanity(title_id = 30002, max_num = 10, in_romaji = True, only_popular = True)
```
- Example outcome
<img src="https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/cbf.png" width="300" height="300">

#### User-based filtering algorithm
```python
# refer to 2.RecommenderSystem/2.2 User based filtering/Recommendation Module.ipynb
import pandas as pd
import numpy as np

# Initialize
ubf = UserBasedFiltering()

# Load titles data for checking purposes
df_titles = pd.read_csv("titles.csv")
```
##### Recommendation from user_id
```python
# calculate similarity and similar user ids
top_10_similar_user_ids = ubf.get_similar_users_from_user_id(start_col=1, dist_metric="cosine_similarity", query_user_id=QUERY_USER_ID)

# make recommendation
recommended_titles = ubf.recommend_unread_titles(10, top_10_similar_user_ids, method="refer_popularity")

# show recommendations
display(df_titles[df_titles["title_id"].isin(recommended_titles)].head(3))
```
- Example outcome
<img src="https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/ubf_userid.png" width="300" height="300">

##### Recommendation from list of title_ids
```python
# example query title_id list
ex_titles_romance = [72451, 97852, 85135, 101583, 87395, 59211, 132182, 30145, 41514, 86481]

top_10_similar_user_ids = ubf.get_similar_users_from_titles(ex_titles_romance)

# make recommendation
recommended_titles = ubf.recommend_unread_titles(10, top_10_similar_user_ids, method="refer_others")

# show recommendations
display(df_titles[df_titles["title_id"].isin(recommended_titles)].head(3))
```
- Example outcome
<img src="https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/ubf_titleid.png" width="300" height="300">

##### Recommendation from a title, but refering to title-user matrix
```python
query_title_id = 105778 # Chainsaw man (popular, recent, dark fantasy) -> SPYxFAMILY 
res = ubf.recommend_from_other_user_histories(query_title_id)

# show recommendations
display(df_titles[df_titles["title_id"].isin(res)])
```
- Example outcome
<img src="https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/ubf_titleusermatrix.png" width="300" height="300">

#### Drawing similarity
```python
# refer to 2.RecommenderSystem/2.3 User based filtering/Character image recommendation.ipynb
ibr_search = ImageBasedRecommendation("../assets/character_images/character_images_grayscale/")
```
##### Character image similarity
```python
res = ibr_search.recommend_titles_from_similar_characters(query_character_id=137304, top_n=10)
print(res)
```
- Example outcome
<img src="https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/img_embedding_character_sim.png" width="300" height="300">

##### Title similarity
```python
res = ibr_search.recommend_titles_from_similar_image_embedding(query_title_id=30002, top_n=3)
print(res)
```
- Example outcome
<img src="https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/img_embedding_title_sim.png" width="300" height="300">