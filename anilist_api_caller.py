import requests
from IPython.display import display
import pandas as pd
import time
from random import randint

# doc regarding the query format:
# https://anilist.gitbook.io/anilist-apiv2-docs/overview/graphql/pagination

# This is the query that is based around "media" entity
# We will retrieve basic_info/reviews/tags/studios/staff for each title
# title and basic_info is one_to_one, but the rest have one_to_many relationship with a title
query = """
query ($page: Int, $perPage: Int) {
    Page (page: $page, perPage: $perPage) {
        media(sort: SCORE_DESC) {
            id
            title {
                english
                romaji
            }
            type
            duration
            startDate {
                year
            }
            chapters
            volumes
            status
            countryOfOrigin
            isAdult
            genres
            averageScore
            meanScore
            popularity
            favourites
            stats {
                scoreDistribution {
                    score
                    amount
                }
                statusDistribution {
                    status
                    amount
                }
            }
            rankings {
                type
                rank
            }
            description
            coverImage {
                large
            }
            
            studios {
                nodes {
                    id
                    name
                }
            }
            
            staff {
                nodes {
                    id
                    name {
                        full
                    }
                    languageV2
                }
            }

            tags {
                id
                name
                category
                rank
            }

            reviews {
                edges {
                    node {
                        id
                        media {
                            id
                            title {
                                english
                                romaji
                            }
                        }
                        user {
                            id
                        }
                        rating
                        summary
                        score
                        ratingAmount
                        body
                    }
                }
            }
        }
    }
}
"""

# query to get user info
users_query = """
query ($page: Int, $perPage: Int) {
    Page (page: $page, perPage: $perPage) {
        users {
            id
            about
            avatar {
                medium
            }
            favourites {
                anime {
                    nodes {
                        id
                        type
                        title {
                            english
                            romaji
                        }
                    }
                }
                manga {
                    nodes {
                        id
                        type
                        title {
                            english
                            romaji
                        }
                    }
                }
            }
        }
    }
}
"""


# query to get user's read/watch list
list_query = """
query ($userId: Int, $type: MediaType) {
    MediaListCollection (userId: $userId, type: $type) {
        lists {
            entries {
                id
                userId
                mediaId
                status
                repeat
            }
        }
    }
}
"""


# query to get character image urls
character_query = """
query ($page: Int, $perPage: Int) {
    Page (page: $page, perPage: $perPage) {
        media(type: MANGA, sort: POPULARITY_DESC) {
            id
            title {
                english
                romaji
            }
            characters {
                nodes {
                    id
                    name {
                        full
                    }
                    image {
                        large
                    }
                }
            }
        }
    }
}
"""



def process_reviews(reviews, review_array):
    '''
    func to handle "reviews" table
    :param reviews: review content data. Actual info are contained inside "nodes"
    :param review_array: array object that was declared globally prior to calling this function
    :return: doesn't return anything, but modifies global review_array where each element is a dict of review data
    '''
    nodes = reviews["edges"]
    for node in nodes:
        dict_r = {}
        n = node["node"]
        dict_r["review_id"] = n["id"]
        dict_r["title_id"] = n["media"]["id"]
        dict_r["title_english"] = n["media"]["title"]["english"]
        dict_r["title_romaji"] = n["media"]["title"]["romaji"]
        if n["user"] is None: # there are a few reviews without user ids
            dict_r["user_id"] = "Unknown"
        else:
            dict_r["user_id"] = n["user"]["id"]
        dict_r["score"] = n["score"] # The score that the reviewer gave to the title
        dict_r["rating"] = n["rating"] # how many users liked the review (ex: if 10 out of 148 users liked this review, this field is 10)
        dict_r["ratingCount"] = n["ratingAmount"] # total number of users who evaluated this review (ex: if 10 out of 148 users liked this review, this field is 148)
        dict_r["text_summary"] = n["summary"]
        dict_r["text_body"] = n["body"]
        review_array.append(dict_r)


def process_tags(tags, title_id, title, tags_array):
    '''
    func to handle "tags" table
    :param tags: tag information data for each title
    :param title_id: title_id is given so that the resulting data is able to have title-tag structure
    :param title: (same as above)
    :param tags_array: array object that was declared globally prior to calling this function
    :return: doesn't return anything, but modifies global tags_array where each element is a dict of title and its associated tag data
    '''
    for tag in tags:
        dict_tags = {}
        dict_tags["tag_id"] = tag["id"]
        dict_tags["tag_name"] = tag["name"]
        dict_tags["tag_category"] = tag["category"]
        # how tag rank is decided -> https://anilist.co/forum/thread/1991
        dict_tags["tag_rank"] = tag["rank"] # The relevance ranking of the tag out of the 100 for this media
        dict_tags["title_id"] = title_id
        dict_tags["title_english"] = title["english"]
        dict_tags["title_romaji"] = title["romaji"]
        tags_array.append(dict_tags)


def process_studios(studios, title_id, title, studios_array):
    '''
    func to handle "studios" table
    :param studios: names of studios that were involved in making certain anime title
    :param title_id: title_id is given so that the resulting data is able to have title-studio structure
    :param title: (same as above)
    :param studios_array: array object that was declared globally prior to calling this function
    :return: doesn't return anything, but modifies global studios_array where each element is a dict of title and its associated studio data
    '''
    for node in studios["nodes"]:
        dict_sd = {}
        dict_sd["title_id"] = title_id
        dict_sd["title_english"] = title["english"]
        dict_sd["title_romaji"] = title["romaji"]
        dict_sd["studio_id"] = node["id"]
        dict_sd["studio_name"] = node["name"]
        studios_array.append(dict_sd)


def process_staff(staff, title_id, title, staff_array):
    '''
    func to handle "staff" table
    :param staff: names of staff who were involved in making certain anime title
    :param title_id: title_id is given so that the resulting data is able to have title-staff structure
    :param title: (same as above)
    :param staff_array: array object that was declared globally prior to calling this function
    :return: doesn't return anything, but modifies global staff_array where each element is a dict of title and its associated staff data
    '''
    for node in staff["nodes"]:
        dict_st = {}
        dict_st["title_id"] = title_id
        dict_st["title_english"] = title["english"]
        dict_st["title_romaji"] = title["romaji"]
        dict_st["staff_id"] = node["id"]
        dict_st["staff_name"] = node["name"]["full"]
        dict_st["staff_lang"] = node["languageV2"] # language that the staff speaks
        staff_array.append(dict_st)


def process_title(media, titles_array, review_array, tags_array, studios_array, staff_array):
    '''
    func to handle "titles" table
    :param media: media (means anime or manga) information. All the information about 1 title is inside here
    :param titles_array: array object that was declared globally prior to calling this function
    :param review_array: (same as above)
    :param tags_array: (same as above)
    :param studios_array: (same as above)
    :param staff_array: (same as above)
    :return: doesn't return anything, but modifies global titles_array where each element is a dict of various data about 1 title
    '''
    for t in media:
        dict_t = {}
        dict_t["title_id"] = t["id"]
        dict_t["title_english"] = t["title"]["english"]
        dict_t["title_romaji"] = t["title"]["romaji"]
        dict_t["type"] = t["type"] # newly added to distinguish anime/manga
        dict_t["duration"] = t["duration"] # newly added to indicate how long each episodes are for an anime title
        dict_t["start_year"] = t["startDate"]["year"]
        dict_t["chapters"] = t["chapters"] # The amount of chapters the manga has when complete
        dict_t["volume"] = t["volumes"] # The amount of volumes the manga has when complete
        dict_t["publishing_status"] = t["status"] # The current releasing status of the media
        dict_t["country"] = t["countryOfOrigin"]
        dict_t["adult"] = t["isAdult"] # If the media is intended only for 18+ adult audiences
        dict_t["genres"] = t["genres"]
        dict_t["average_score"] = t["averageScore"] # A weighted average score of all the user"s scores of the media
        dict_t["mean_score"] = t["meanScore"] # Mean score of all the user"s scores of the media
        dict_t["popularity"] = t["popularity"] # The number of users with the media on their list
        dict_t["favorites"] = t["favourites"] # The amount of user"s who have favourited the media
        for s in t["stats"]["scoreDistribution"]:
            score = s["score"]
            dict_t["score_%s"%score] = s["amount"]
        for s in t["stats"]["statusDistribution"]:
            status = s["status"]
            dict_t["count_%s"%status] = s["amount"]
        for r in t["rankings"]:
            rank_type = r["type"]
            dict_t["ranking_%s"%rank_type] = r["rank"] # The ranking of the media in a particular time span and format compared to other media
        dict_t["synopsis"] = t["description"]
        dict_t["cover_image_url"] = t["coverImage"]
        process_reviews(t["reviews"], review_array)
        process_tags(t["tags"], t["id"], t["title"], tags_array)
        process_studios(t["studios"], t["id"], t["title"], studios_array)
        process_staff(t["staff"], t["id"], t["title"], staff_array)
        titles_array.append(dict_t)


def process_favorites(favorites, user_id, favorites_array):
    '''
    func to handle users "favorites" table
    :param favorites: names of one user"s favorite manga/anime. Actual info are contained inside "nodes"
    :param user_id: user_id is given so that the resulting data is able to have user-favorite_title structure
    :param favorites_array: array object that was declared globally prior to calling this function
    :return: doesn't return anything, but modifies global favorites_array where each element is a dict of favorite titles of a user
    '''
    # loop over anime favorites
    for node in favorites["anime"]["nodes"]:
        dict_f = {}
        dict_f["user_id"] = user_id
        dict_f["title_id"] = node["id"]
        dict_f["title_english"] = node["title"]["english"]
        dict_f["title_romaji"] = node["title"]["romaji"]
        dict_f["type"] = node["type"]
        favorites_array.append(dict_f)
    # loop over manga favorites
    for node in favorites["manga"]["nodes"]:
        dict_f = {}
        dict_f["user_id"] = user_id
        dict_f["title_id"] = node["id"]
        dict_f["title_english"] = node["title"]["english"]
        dict_f["title_romaji"] = node["title"]["romaji"]
        dict_f["type"] = node["type"]
        favorites_array.append(dict_f)


def process_users(user, users_array, favorites_array):
    '''
    func to handle "users" table
    :param user: info of 1 user
    :param users_array: globally declared array
    :param favorites_array: globally declared array
    :return: doesn't return anything, but modifies global users_array where each element is a dict of user information
    '''
    for u in user:
        dict_u = {}
        dict_u["user_id"] = u["id"]
        dict_u["about"] = u["about"]
        dict_u["avatar"] = u["avatar"]["medium"]
        process_favorites(u["favourites"], u["id"], favorites_array)
        users_array.append(dict_u)


def process_lists(media_list, list_array):
    for l in media_list:
        for entry in l["entries"]:
            dict_l = {}
            dict_l["list_id"] = entry["id"]
            dict_l["user_id"] = entry["userId"]
            dict_l["title_id"] = entry["mediaId"]
            dict_l["status"] = entry["status"]
            dict_l["repeat"] = entry["repeat"]
            list_array.append(dict_l)


def process_characters(media, chara_array):
    for m in media:
        for node in m["characters"]["nodes"]:
            dict_ch = {}
            dict_ch["title_id"] = m["id"]
            dict_ch["title_english"] = m["title"]["english"]
            dict_ch["title_romaji"] = m["title"]["romaji"]
            dict_ch["character_id"] = node["id"]
            dict_ch["character_name"] = node["name"]["full"]
            dict_ch["character_image_url"] = node["image"]["large"]
            chara_array.append(dict_ch)


def data_to_csv(data_array, csv_title, index=True):
    '''
    func to save array to csv
    :param data_array: array that will be converted to pandas df
    :param csv_title: name given to csv file
    :param index: boolean to decide if you want to keep the df index
    :return: does not return anything but saves the csv file under given directory
    '''
    print(f"~~~~~~~~~~~ creating {csv_title} csv ~~~~~~~~~~~")
    cols_titles = data_array[0].keys()
    print("col titles: ", cols_titles)
    df = pd.DataFrame(data_array, columns=cols_titles)
    print("len of titles table: ", len(df))
    display(df.head())
    file_title = "./dataset/" + csv_title + ".csv"
    print("file title: ", file_title)
    df.to_csv(file_title, index=index) # False for titles


# # ================================ call API and create title-based data ================================
#
# TITLES = []
# REVIEWS = []
# TAGS = []
# STUDIOS = []
# STAFF = []
#
# print(f"~~~~~~~~~~~ start 'titles' query processing ~~~~~~~~~~~")
# for page in range(1,201):
#     print("page: ", page)
#     sleep_sec = randint(1,10)
#     time.sleep(sleep_sec)
#     print("sleep sec: ", sleep_sec)
#     variables = {
#         "page": page, # 1-20
#         "perPage": 50 # max is 50
#         # "averageScore_greater": 20 # 74 (should take about 750 titles)
#     }
#     url = "https://graphql.anilist.co"
#     response = requests.post(url, json={"query": query, "variables": variables})
#     json_obj = response.json()
#     media = json_obj["data"]["Page"]["media"]
#     process_title(media, TITLES, REVIEWS, TAGS, STUDIOS, STAFF)
#     # print("TITLES: ", TITLES)
#     # print("STAFF: ", STAFF)
#
# # save data to csv
# data_to_csv(TITLES, "titles_200p", index=False)
# data_to_csv(REVIEWS, "reviews_200p", index=False)
# data_to_csv(TAGS, "tags_200p", index=False)
# data_to_csv(STUDIOS, "studios_200p", index=False)
# data_to_csv(STAFF, "staff_200p", index=False)


# ================================ call API and create user-based data ================================


# USERS = []
# FAVORITES = []
#
# print(f"~~~~~~~~~~~ start 'users' query processing ~~~~~~~~~~~")
# for page in range(1,3): # 200 pages x 50 users per page -> 10,000 users
#     print("page: ", page)
#     sleep_sec = randint(1,10)
#     time.sleep(sleep_sec)
#     print("sleep sec: ", sleep_sec)
#     variables = {
#         "page": page, # 1-20
#         "perPage": 50, # max is 50
#     }
#     url = "https://graphql.anilist.co"
#     response = requests.post(url, json={"query": users_query, "variables": variables})
#     json_obj = response.json()
#     print(json_obj)
#     users = json_obj["data"]["Page"]["users"]
#     print(users)
#     process_users(users, USERS, FAVORITES)
#     # print("USERS: ", USERS)
#     # print("FAVORITES: ", FAVORITES)
#
#
# # save data to csv
# data_to_csv(USERS, "users_200p", index=False)
# data_to_csv(FAVORITES, "favorites_200p", index=False)



# # ================================ call API and create user's read/watch list data ================================
#
#
# MEDIALIST = []
#
# print(f"~~~~~~~~~~~ start 'list' query processing ~~~~~~~~~~~")
# for user_id in range(5001, 10001): # 5,000 users
#     for mediatype in ["ANIME", "MANGA"]:
#         print("user_id: ", user_id, " media: ", mediatype)
#         sleep_sec = randint(1,3)
#         time.sleep(sleep_sec)
#         print("sleep sec: ", sleep_sec)
#         variables = {
#             "userId": user_id,
#             "type": mediatype
#         }
#         url = "https://graphql.anilist.co"
#         response = requests.post(url, json={"query": list_query, "variables": variables})
#         json_obj = response.json()
#         try:
#             media_list = json_obj["data"]["MediaListCollection"]["lists"]
#             process_lists(media_list, MEDIALIST)
#             # print("MEDIALIST: ", MEDIALIST)
#         except:
#             print("error at: ", user_id)
#
# # save data to csv
# data_to_csv(MEDIALIST, "media_list_5001_10000_users", index=False)



# ================================ call API and retrieve character image url data ================================


CHARACTERS = []

print(f"~~~~~~~~~~~ start 'characters' query processing ~~~~~~~~~~~")
for page in range(1,200):
    print("page: ", page)
    sleep_sec = randint(1,3)
    time.sleep(sleep_sec)
    print("sleep sec: ", sleep_sec)
    variables = {
        "page": page,
        "perPage": 50
    }
    url = "https://graphql.anilist.co"
    response = requests.post(url, json={"query": character_query, "variables": variables})
    json_obj = response.json()
    media = json_obj["data"]["Page"]["media"]
    process_characters(media, CHARACTERS)
    # print("CHARACTERS: ", CHARACTERS)

# save data to csv
data_to_csv(CHARACTERS, "characters_200p", index=False)
