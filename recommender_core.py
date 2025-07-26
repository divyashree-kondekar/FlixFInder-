# recommender_core.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import ast
import requests # For TMDb API calls
from gtts import gTTS # For Text-to-Speech
import os # For saving/managing audio files
import streamlit as st # Ensure this is imported for caching

TMDB_API_KEY = '71e491a012824f19bbdbd71689f95225' # <<< Make sure this is your NEW, VALID API KEY!

# --- 1. Load and Merge Data (TMDb) ---
try:
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    credits.rename(columns={'movie_id': 'id'}, inplace=True)
    df = movies.merge(credits, on='id')
except FileNotFoundError:
    st.error("TMDb dataset files not found. Please ensure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the 'data/' directory.")
    st.stop() # Stop the app if essential files are missing


# --- 2. Process JSON Columns (TMDb) ---
# Ensure these functions handle potential non-string inputs or empty lists gracefully
def convert(obj):
    if pd.isna(obj) or obj == '[]':
        return []
    L = []
    try:
        for i in ast.literal_eval(obj):
            if isinstance(i, dict) and 'name' in i:
                L.append(i['name'])
    except (ValueError, SyntaxError):
        pass # Silently ignore malformed strings
    return L


def convert_cast(obj):
    if pd.isna(obj) or obj == '[]':
        return []
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if isinstance(i, dict) and 'name' in i:
                if counter < 3: # Limit to top 3 cast members
                    L.append(i['name'])
                    counter += 1
                else:
                    break
    except (ValueError, SyntaxError):
        pass # Silently ignore malformed strings
    return L


def fetch_director(obj):
    if pd.isna(obj) or obj == '[]':
        return []
    L = []
    try:
        for i in ast.literal_eval(obj):
            if isinstance(i, dict) and i.get('job') == 'Director':
                L.append(i['name'])
                break # Assuming one director per movie for simplicity
    except (ValueError, SyntaxError):
        pass # Silently ignore malformed strings
    return L


df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)
df['cast'] = df['cast'].apply(convert_cast)
df['crew'] = df['crew'].apply(fetch_director)


# --- 3. Clean Text Data and Create 'tags' Column (TMDb) ---
def collapse(L):
    # Ensure L is a list before attempting to iterate
    if not isinstance(L, list):
        return []
    L1 = []
    for i in L:
        # Ensure i is a string before replacing spaces
        if isinstance(i, str):
            L1.append(i.replace(" ", ""))
    return L1


df['genres'] = df['genres'].apply(collapse)
df['keywords'] = df['keywords'].apply(collapse)
df['cast'] = df['cast'].apply(collapse)
df['crew'] = df['crew'].apply(collapse)

df['overview'] = df['overview'].fillna("")

df['tags'] = df['overview'] + " " + df['genres'].apply(lambda x: " ".join(x)) + " " + \
             df['keywords'].apply(lambda x: " ".join(x)) + " " + \
             df['cast'].apply(lambda x: " ".join(x)) + " " + \
             df['crew'].apply(lambda x: " ".join(x))

df['tags'] = df['tags'].apply(lambda x: x.lower())

# --- 4. Final DataFrame for Content-Based Recommender ---
# IMPORTANT: Keeping 'original_title' as requested
df_final = df[['id', 'original_title', 'tags', 'genres', 'popularity', 'vote_average','release_date','vote_count','overview']].copy()

# Add 'genres_names' by converting 'genres' column to list of names if not already
# This re-uses the 'convert' function logic or similar for string-formatted genre lists
# Ensure 'genres' is a string before literal_eval
df_final['genres_list'] = df_final['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_final['genres_names'] = df_final['genres_list'].apply(lambda x: [g['name'] for g in x] if all(isinstance(g, dict) and 'name' in g for g in x) else x)


# --- 5. Text Vectorization (Content-Based) ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df_final['tags']).toarray()

# --- 6. Cosine Similarity Calculation (Content-Based) ---
similarity = cosine_similarity(vectors)

@st.cache_data # Added cache for poster URL fetching
# --- TMDb Poster Fetching Function ---
def get_movie_poster_url(tmdb_id, image_size='w500'):
    base_url = "https://api.themoviedb.org/3/movie/"
    url = f"{base_url}{tmdb_id}?api_key={TMDB_API_KEY}"  # Use the global API key

    try:
        response = requests.get(url, timeout=10) # Added timeout
        response.raise_for_status()
        data = response.json()

        if data and 'poster_path' in data and data['poster_path']:
            poster_path = data['poster_path']
            return f"https://image.tmdb.org/t/p/{image_size}{poster_path}"
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching movie poster for TMDB ID {tmdb_id}: {e}")
        return None

@st.cache_data # Added cache for recommendations
# --- Recommendation Functions (Content-Based) ---
def recommend(movie_title, selected_genre=None):
    # IMPORTANT: Using 'original_title' as requested
    movie_index_list = df_final[df_final['original_title'].str.lower() == movie_title.lower()].index
    if movie_index_list.empty:
        return []
    movie_index = movie_index_list[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    recommended_movies = []
    count = 0
    for i in movies_list:
        if i[0] == movie_index:
            continue
        rec_movie_title = df_final.iloc[i[0]]['original_title'] # IMPORTANT: Using 'original_title'
        rec_movie_genres = df_final.iloc[i[0]]['genres_names'] # Use genres_names for consistency

        if selected_genre:
            normalized_selected_genre = selected_genre.replace(" ", "").lower()
            normalized_rec_genres = [g.replace(" ", "").lower() for g in rec_movie_genres]
            if normalized_selected_genre in normalized_rec_genres:
                recommended_movies.append(rec_movie_title)
                count += 1
        else:
            recommended_movies.append(rec_movie_title)
            count += 1
        if count == 5:
            break
    return recommended_movies

@st.cache_data # Added cache for recommendations
def recommend_top_by_genre(selected_genre):
    recommended_movies = []
    normalized_selected_genre = selected_genre.replace(" ", "").lower()

    genre_movies = df_final[df_final['genres_names'].apply(
        lambda x: normalized_selected_genre in [g.replace(" ", "").lower() for g in x]
    )]
    if genre_movies.empty:
        return []
    top_genre_movies = genre_movies.sort_values(by='popularity', ascending=False)
    recommended_movies = top_genre_movies['original_title'].head(5).tolist() # IMPORTANT: Using 'original_title'
    return recommended_movies


# --- Collaborative Filtering Setup (MovieLens ml-latest-small) ---
# Initialize ml_movie_titles to an empty list by default
ml_movie_titles = []
ml_title_to_index = {} # Initialize ml_title_to_index as well
knn_model_ml = None # Initialize knn_model_ml as well
movie_user_sparse_matrix_ml = None # Initialize sparse matrix

try:
    ratings_ml_small = pd.read_csv('dataforcollab/ratings.csv')
    movies_ml_small = pd.read_csv('dataforcollab/movies.csv')
    links_ml_small = pd.read_csv('dataforcollab/links.csv')

    df_ml_full = pd.merge(ratings_ml_small, movies_ml_small, on='movieId')
    df_ml_full = pd.merge(df_ml_full, links_ml_small[['movieId', 'tmdbId']], on='movieId', how='left')
    df_ml_full.dropna(subset=['tmdbId'], inplace=True)
    df_ml_full['tmdbId'] = df_ml_full['tmdbId'].astype(int)

    # Ensure 'title' in MovieLens data is stripped for clean matching
    df_ml_full['title'] = df_ml_full['title'].str.strip()

    user_movie_matrix_ml = df_ml_full.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_user_sparse_matrix_ml = csr_matrix(user_movie_matrix_ml.transpose().values)

    knn_model_ml = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn_model_ml.fit(movie_user_sparse_matrix_ml)

    ml_movie_titles = user_movie_matrix_ml.columns.tolist()
    ml_title_to_index = {title: i for i, title in enumerate(ml_movie_titles)}
except FileNotFoundError:
    st.warning("MovieLens dataset files for collaborative filtering not found in 'dataforcollab/'. Collaborative filtering features may be limited.")
    # If files are not found, ml_movie_titles remains an empty list, preventing NameError.
except Exception as e:
    st.error(f"An error occurred during MovieLens data loading or processing: {e}. Collaborative filtering features may be limited.")
    ml_movie_titles = [] # Ensure it's an empty list on other errors too
    ml_title_to_index = {}
    knn_model_ml = None
    movie_user_sparse_matrix_ml = None


@st.cache_data # Added cache for recommendations
# --- Recommendation Function (Collaborative) ---
def collaborative_recommend(movie_title_ml, n_recommendations=5):
    movie_title_ml_stripped = movie_title_ml.strip()
    # Check if ml_title_to_index is populated and model is fitted before using it
    if not ml_title_to_index or movie_title_ml_stripped not in ml_title_to_index or knn_model_ml is None or movie_user_sparse_matrix_ml is None:
        return []
    movie_idx = ml_title_to_index[movie_title_ml_stripped]

    distances, indices = knn_model_ml.kneighbors(
        movie_user_sparse_matrix_ml[movie_idx],
        n_neighbors=n_recommendations + 1
    )

    recommended_movies_info = []
    for i in range(1, len(indices[0])):
        ml_rec_movie_idx = indices[0][i]
        ml_rec_movie_title = ml_movie_titles[ml_rec_movie_idx]

        ml_rec_movie_id_match = movies_ml_small[movies_ml_small['title'] == ml_rec_movie_title]['movieId']
        if ml_rec_movie_id_match.empty:  # Fallback if MovieLens title mapping fails
            continue
        ml_rec_movie_id = ml_rec_movie_id_match.iloc[0]

        tmdb_id_match = links_ml_small[links_ml_small['movieId'] == ml_rec_movie_id]['tmdbId']

        if not tmdb_id_match.empty:
            rec_tmdb_id = int(tmdb_id_match.iloc[0])
            # IMPORTANT: Get original_title from df_final for display consistency
            tmdb_title_match = df_final[df_final['id'] == rec_tmdb_id]['original_title']


            if not tmdb_title_match.empty:
                rec_title_tmdb = tmdb_title_match.iloc[0]
                recommended_movies_info.append({
                    'title': rec_title_tmdb, # This 'title' key will be used by display_recommendations
                    'tmdb_id': rec_tmdb_id
                })
    return recommended_movies_info

# --- Text-to-Speech Function ---
# This function was added in a previous step for the voice assistant output
def text_to_audio(text, filename="temp/response.mp3"): # Changed default path to 'temp/'
    """Converts text to speech and saves it as an MP3 file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"Error in TTS: {e}")
        return None


# @st.cache_data # REMOVED CACHING FOR DEBUGGING PURPOSES
# --- MODIFIED: Function to Fetch Movies by Various TMDb Categories ---
def get_movies_by_category(category='trending', time_window='day', n_movies=15):
    """
    Fetches movies from TMDb API based on specified category.
    Category can be 'trending', 'now_playing', 'upcoming', 'top_rated'.
    time_window is only used for 'trending' ('day' or 'week').
    """
    base_api_url = "https://api.themoviedb.org/3/"

    endpoint = ""
    if category == 'trending':
        if time_window not in ['day', 'week']:
            print("Invalid time_window for 'trending'. Must be 'day' or 'week'.")
            return []
        endpoint = f"trending/movie/{time_window}"
    elif category == 'now_playing':
        endpoint = "movie/now_playing"
    elif category == 'upcoming':
        endpoint = "movie/movie/upcoming" # Corrected endpoint for upcoming
    elif category == 'top_rated':
        endpoint = "movie/top_rated"
    else:
        print(f"Invalid category: {category}")
        return []

    url = f"{base_api_url}{endpoint}?api_key={TMDB_API_KEY}"

    movies_info = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data and 'results' in data:
            for movie in data['results'][:n_movies]:
                movies_info.append({
                    'title': movie.get('title'), # TMDb's title field
                    'tmdb_id': movie.get('id')
                })
        return movies_info

    except requests.exceptions.RequestException as e:
        print(f"Error fetching movies for category '{category}': {e}")
        return []


@st.cache_data
## --- TMDb Genre ID Mapping (Re-introduced) ---
def get_tmdb_genre_id_map(api_key):
    """Fetches TMDb movie genre names and their corresponding IDs."""
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url,timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and 'genres' in data:
            return {genre['name']: genre['id'] for genre in data['genres']}
    except requests.exceptions.RequestException as e:
        print(f"Error fetching TMDb genres: {e}")
    return {}

tmdb_genre_name_to_id = get_tmdb_genre_id_map(TMDB_API_KEY)

@st.cache_data # Added cache for API calls
# --- NEW: Function to get mood-based movies from TMDb API ---
def get_mood_based_movies_from_tmdb(mood_genres_names, certification_level=None, certification_country='US', n_movies=10):
    """
    Fetches movies from TMDb API based on genre names and optional certification.
    """
    if not mood_genres_names:
        return []

    # Convert genre names to TMDb IDs
    genre_ids = []
    for genre_name in mood_genres_names:
        genre_id = tmdb_genre_name_to_id.get(genre_name)
        if genre_id:
            genre_ids.append(str(genre_id))

    if not genre_ids:
        return []

    # Construct TMDb Discover API URL
    base_url = "https://api.themoviedb.org/3/discover/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'with_genres': ','.join(genre_ids),
        'sort_by': 'popularity.desc',
        'vote_count.gte': 10, # Reduced vote_count.gte to be less restrictive for testing
        'language': 'en-US'
    }

    if certification_level and certification_country: # Corrected: certification_level is now a parameter
        params['certification_country'] = certification_country
        params['certification.lte'] = certification_level


    print(f"TMDb API Request URL: {base_url}")
    print(f"TMDb API Request Params: {params}")


    movies_info = []
    try:
        response = requests.get(base_url, params=params,timeout=10)
        print(f"TMDb API Response Status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        print(f"TMDb API Response Data (first 200 chars): {str(data)[:200]}...")

        if data and 'results' in data:
            for movie in data['results'][:n_movies]:
                movies_info.append({
                    'title': movie.get('title'), # TMDb's title field
                    'tmdb_id': movie.get('id')
                })
        return movies_info

    except requests.exceptions.RequestException as e:
        print(f"Error fetching mood-based movies from TMDb: {e}")
        return []



# @st.cache_data # REMOVED CACHING FOR DEBUGGING PURPOSES
# --- NEW: Function to get detailed movie info from TMDb, including videos ---
def get_movie_details_from_tmdb(tmdb_id):
    """
    Fetches detailed movie information, including production details and videos, from TMDb.
    """
    if not tmdb_id:
        print("DEBUG: get_movie_details_from_tmdb called with None tmdb_id.")
        return None

    # Use 'append_to_response=videos' to get trailer data in the same call
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=videos,credits"

    print(f"DEBUG: Fetching detailed movie info for TMDB ID: {tmdb_id}")
    print(f"DEBUG: Detail API URL: {url}")

    try:
        response = requests.get(url, timeout=10)
        print(f"DEBUG: TMDb Detail API Response Status: {response.status_code}")
        response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        print(f"DEBUG: TMDb Detail API Response Data (first 500 chars): {str(data)[:500]}...")


        # Extract relevant details
        details = {
            'title': data.get('title'), # TMDb's title field
            'overview': data.get('overview'),
            'poster_path': data.get('poster_path'),
            'backdrop_path': data.get('backdrop_path'),
            'release_date': data.get('release_date'),
            'vote_average': data.get('vote_average'),
            'vote_count': data.get('vote_count'),
            'runtime': data.get('runtime'),
            'tagline': data.get('tagline'),
            'genres': [g['name'] for g in data.get('genres', [])],
            'spoken_languages': [lang['english_name'] for lang in data.get('spoken_languages', [])],
            'production_companies': [comp['name'] for comp in data.get('production_companies', [])],
            'videos': data.get('videos', {}).get('results', []), # List of video objects
            'cast': [member['name'] for member in data.get('credits', {}).get('cast', [])[:5]], # Top 5 cast
            'director': next((crew['name'] for crew in data.get('credits', {}).get('crew', []) if crew['job'] == 'Director'), 'N/A')
        }
        return details

    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error fetching movie details for TMDB ID {tmdb_id}: {e}")
        print(f"ERROR: Response text: {e.response.text if e.response else 'N/A'}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Connection error fetching movie details for TMDB ID {tmdb_id}: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"ERROR: Timeout error fetching movie details for TMDB ID {tmdb_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"ERROR: General request error fetching movie details for TMDB ID {tmdb_id}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred in get_movie_details_from_tmdb for TMDB ID {tmdb_id}: {e}")
        return None
