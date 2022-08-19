import pandas as pd
import time
from random import randint
from IPython.display import display
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
import chromedriver_binary # you'll need to specify the version beforehand. For my PC it was "pip install chromedriver_binary==103.0.5060.53"



# # ============================== user images ==============================
# print("\n")
# print("~~~~~~~~~~~~~~~ start getting user avatar images ~~~~~~~~~~~~~~~")
#
# # Load users table
# df_users_cleaned = pd.read_csv("dataset/users_200p_cleaned.csv", encoding="ISO-8859-1")
# display(df_users_cleaned.head())
#
# # Selenium settings
# d = webdriver.Chrome()
#
# # list of urls and user_ids
# avatar_urls = list(df_users_cleaned["avatar"].values)
# user_ids = list(df_users_cleaned["user_id"].values)
# # avatar_urls = avatar_urls[40:45]
# # user_ids = user_ids[40:45]
# # print(avatar_urls, user_ids)
#
# for id, url in zip(user_ids, avatar_urls):
#
#     # randomized sleep for courtesy
#     sleep_sec = randint(1, 3)
#     time.sleep(sleep_sec)
#     print("sleep sec: ", sleep_sec, " user_id: ", id)
#
#     try:
#         d.get(url)
#         image = d.find_element(By.TAG_NAME, "img")
#         if image:
#             with open(f"dataset/user_images/{id}.png", "wb") as file:
#                 file.write(image.screenshot_as_png)
#     except:
#         print("retrieval error at user_id: ", id)
#
# d.quit()


# # ============================== title images ==============================
#
# def extract_url(url_txt):
#     '''
#     Function to extract the url text. "cover_image_url" column values consist of nested dicts, so we need to extract just the url string
#     :param url_txt: mixed string object of non-url text and url text
#     :return: extracted url string
#     '''
#     match = re.findall("'\S*'", url_txt)
#     url_extract = re.findall("\'([^']+)'", match[1])
#     return url_extract[0]
#
#
# print("\n")
# print("~~~~~~~~~~~~~~~ start getting title thumbnail images ~~~~~~~~~~~~~~~")
#
# # Load titles table
# df_titles_cleaned = pd.read_csv("dataset/titles_200p_cleaned.csv")
# display(df_titles_cleaned.head())
#
# # Selenium settings
# d = webdriver.Chrome()
#
# # list of urls and title_ids
# thumbnail_urls = list(df_titles_cleaned["cover_image_url"].values)
# title_ids = list(df_titles_cleaned["title_id"].values)
#
# thumbnail_urls = [extract_url(url) for url in thumbnail_urls]
# print(thumbnail_urls[:5], title_ids[:5])
#
# for id, url in zip(title_ids, thumbnail_urls):
#
#     # randomized sleep for courtesy
#     sleep_sec = randint(1,5)
#     time.sleep(sleep_sec)
#     print("sleep sec: ", sleep_sec, " user_id: ", id)
#
#     try:
#         d.get(url)
#         image = d.find_element(By.TAG_NAME, "img")
#         if image:
#             with open(f"dataset/title_images/{id}.png", "wb") as file:
#                 file.write(image.screenshot_as_png)
#     except:
#         print("retrieval error at title_id: ", id)
#
# d.quit()


# ============================== character images ==============================
print("\n")
print("~~~~~~~~~~~~~~~ start getting character images ~~~~~~~~~~~~~~~")

# Load users table
folder_path = "/Users/xxxxxxxxx/folder/"
image_folder_path = "/Users/xxxxxxxxx/folder/images/"
df_characters = pd.read_csv(folder_path+"characters_498_titles")

# list of urls and character_ids
character_urls = list(df_characters["character_image_url"].values)
character_ids = list(df_characters["character_id"].values)

# Selenium settings
d = webdriver.Chrome()

for id, url in zip(character_ids, character_urls):

    # randomized sleep for courtesy
    sleep_sec = randint(1, 3)
    time.sleep(sleep_sec)
    print("sleep sec: ", sleep_sec, " character_id: ", id)

    try:
        d.get(url)
        image = d.find_element(By.TAG_NAME, "img")
        if image:
            with open(f"{image_folder_path}{id}.png", "wb") as file:
                file.write(image.screenshot_as_png)
    except:
        print("retrieval error at user_id: ", id)

d.quit()