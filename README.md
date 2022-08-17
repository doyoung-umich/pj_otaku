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
* Content-based filtering algorithm
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
![ example outcome of check_sanity](https://github.com/doyoung-umich/pj_otaku/blob/main/Sample%20Images/cbf.png "example outcome of check_sanity")

* User-based filtering algorithm
```python

```