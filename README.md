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
  - **2.2 User based filtering
    - **Recommendation Module.ipynb**: A recommendation module powered by user-based filtering and an example usage of the module
    - **Structural analysis of user data**: Data processing, calculate user similarity by genre probability distribution
  - **2.3 Image embedding
    - **Character image recommendation.ipynb:
* 3.App and Evaluation

### The algorithms
* Content-based algorithm
```python

```
