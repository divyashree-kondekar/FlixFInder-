# 🎬 FlixFinder – Your Personalized Movie Recommendation System!

Welcome to **FlixFinder**, a personalized movie recommendation system designed to help users discover amazing films based on their interests, mood, or previous favorites!

This application uses **content-based filtering**, **collaborative filtering**, **TMDb API**, and **voice input** to generate intelligent and diverse recommendations.

---

## 🚀 Features

✨ **Modern Netflix-like UI**  
✨ **Content-Based Filtering** – Find movies similar to what you already love  
✨ **Collaborative Filtering** – Discover movies loved by people with similar tastes  
✨ **Mood-Based Recommendations** – Choose a mood, get matching movies  
✨ **Dynamic Lists** – Trending, Top Rated, Now Playing & Upcoming  
✨ **Movie Detail View** – See trailer, overview, rating, cast, and more  
✨ **Voice Assistant** – Speak your movie preferences (e.g., “Recommend a comedy movie”)  
✨ **Text-to-Speech Responses** – System talks back your results!  

---

## 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Backend Logic | Python, Pandas, NumPy |
| Machine Learning | Scikit-learn (Cosine Similarity, KNN) |
| Dataset | TMDb 5000 Movie Dataset + MovieLens |
| UI Framework | Streamlit |
| Voice Features | SpeechRecognition, gTTS |
| APIs | TMDb API |


## 📁 Folder Structure

flixfinder/
│
├── app.py # Main Streamlit app
├── recommender_core.py # All recommendation logic + TMDb integration
├── requirements.txt # Python dependencies
├── data/ # TMDb 5000 movies dataset files
│ └── tmdb_5000_movies.csv
│ └── tmdb_5000_credits.csv
├── dataforcollab/ # MovieLens dataset for collaborative filtering
│ └── ratings.csv
│ └── movies.csv
│ └── links.csv
├── temp/ # Temporary audio files for TTS
├── profilepic.png # Developer image (About Us tab)
└── README.md 

## 📊 Dataset Used

FlixFinder uses two primary datasets to power its recommendations:

---

### 🗃️ 1. TMDb 5000 Movie Dataset (`data/`)

- **Files:**
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`

- **Source:** [Kaggle – TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

- **Usage:**
  - Used for content-based recommendations
  - Extracts movie metadata (title, genres, overview, cast, crew)
  - Enriched using the **TMDb API** for posters, trailers, ratings, etc.

---

### 🧠 2. MovieLens Dataset (`dataforcollab/`)

- **Files:**
  - `ratings.csv`
  - `movies.csv`
  - `links.csv`

- **Source:** [MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/)

- **Usage:**
  - Used for collaborative filtering
  - Matches MovieLens movies to TMDb movies using `links.csv`

---

## NOTE:
Details button will work across tab 1 , tab 4 and tab 7 only due to streamlit rerun Issue!!


😎 Developer - Divyashree Kondekar!
💛 Passionate about MovieTech, and building cool apps!
🌟 Made with ❤️ as part of internship/project submission...




