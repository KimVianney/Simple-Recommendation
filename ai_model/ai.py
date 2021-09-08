import pandas as pd 
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


path = "/home/vianney/Desktop/code/Recommendation system/Movieflix/datasets"

# Create data frames for both datasets
credits_df = pd.read_csv(path + "/tmdb_credits.csv")
movies_df = pd.read_csv(path + "/tmdb_movies.csv")

credits_df.columns = ['id', 'title', 'cast', 'crew']

#Merge the datasets on id column
movies_df = movies_df.merge(credits_df, on="id")

# Columns  to be used in recommender
features = ["cast", "crew", "keywords", "genres"]

#Converting the data into a usable structure
for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

# print(movies_df[features].head(10))
# Extract director from crew column
def get_director(crew):
    for i in crew:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

# Extract top 3 items in  a list
def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]

        return names
    return []

# Get director from our data frame
movies_df["director"] = movies_df["crew"].apply(get_director)

# Get first 3 items of columns shown in features
features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

movies_df['title'] = movies_df['original_title']
# print(movies_df[['title','cast', 'director', 'keywords', 'genres']].head())

# Remove all whitespace and make them lowercase
def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, list):
            return str.lower(row.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)

# Make titles to lower case
movies_df['title'] = movies_df['title'].str.lower()

# Create column containing all metadata to feed into vectorizer
def create_soup(features):
    return ' '.join(features['keywords']) + ' '+' '.join(features['cast']) + ' ' + features['director']+' '+' '.join(features['genres'])

movies_df["soup"] = movies_df.apply(create_soup, axis=1)
#print(movies_df["soup"].head())


count_vectorizer = CountVectorizer(stop_words="english")
count_matrix  = count_vectorizer.fit_transform(movies_df["soup"])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset indices for dataframe
movies_df = movies_df.reset_index()

# Reverse mapping of movie titles. Easily make recommendations using movie title
indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()



# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim2):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:21]

    movies_indices = [ind[0] for ind in similarity_scores]
    movie_titles = movies_df["title"].iloc[movies_indices]
    movie_genres = movies_df["genres"].iloc[movies_indices]
    movie_ids = movies_df["id"].iloc[movies_indices]
    description = movies_df["overview"].iloc[movies_indices]

    recommendation_data = pd.DataFrame(columns=['Name', 'Genres'])
    recommendation_data["Movie_Id"] = movie_ids
    recommendation_data["Name"] = movie_titles
    recommendation_data["Genres"] = movie_genres
    recommendation_data["Description"] = description

    return recommendation_data

def  results(movie_name):
    movie_name = movie_name.lower()

    find_movie = movies_df

    if movie_name not in movies_df['title'].unique():
        return 'Movie not in Database'

    else:
        recommendations = get_recommendations(movie_name)
        return recommendations.to_dict('records')

