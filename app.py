#app.py
import streamlit as st
import pandas as pd
from ast import literal_eval
import requests
import os
import re
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder

# Import all necessary functions and variables from your recommender_core.py
from recommender_core import (
    df_final, recommend, recommend_top_by_genre,
    collaborative_recommend, get_movie_poster_url,
    ml_movie_titles, text_to_audio,
    get_movies_by_category,
    TMDB_API_KEY,
    tmdb_genre_name_to_id,
    get_mood_based_movies_from_tmdb,
    get_movie_details_from_tmdb
)


# --- Initialize session state variables robustly ---
def initialize_session_state():
    if "last_command" not in st.session_state:
        st.session_state["last_command"] = ""
    if "current_audio_played" not in st.session_state:
        st.session_state["current_audio_played"] = False
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "main"  # Can be "main" or "movie_details"
    if "selected_movie_id" not in st.session_state:
        st.session_state["selected_movie_id"] = None
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []



# IMPORTANT: Using 'original_title' as requested for consistency with df_final
all_df_movie_titles = df_final['original_title'].tolist()

# --- Streamlit UI Setup ---
st.set_page_config(
    layout="wide",
    page_title="Flix Finder",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)

# --- Custom CSS for a more Netflix-like feel ---
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #141414; /* Netflix dark background */
        color: #FFFFFF; /* White text */
        font-family: 'Inter', sans-serif; /* Using Inter font */
    }

    /* Header styling */
    h1 {
        color: #E50914; /* Netflix Red */
        text-align: center;
        font-size: 3.5em;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    h2 {
        color: #FFFFFF;
        text-align: center;
        margin-top: 1.5em;
        margin-bottom: 1em;
        font-size: 2em;
    }
    h3 {
        color: #FFFFFF;
        font-size: 1.5em;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }

    /* Markdown text styling */
    .stMarkdown {
        text-align: center;
        font-size: 1.1em;
        color: #D3D3D3;
    }

    /* Button styling */
    .stButton>button {
        background-color: #E50914; /* Netflix Red */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        font-weight: bold;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    .stButton>button:hover {
        background-color: #F40612; /* Slightly brighter red on hover */
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5);
    }

    /* Input fields */
    .stTextInput>div>div>input {
        background-color: #282828;
        color: #FFFFFF;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 10px;
    }
    .stSelectbox>div>div>div {
        background-color: #282828;
        color: #FFFFFF;
        border: 1px solid #444444;
        border-radius: 8px;
    }

    /* Columns for movie posters */
    .st-emotion-cache-nahz7x { /* This class might change with Streamlit updates, but targets columns */
        padding: 10px !important;
        border-radius: 10px;
        background-color: #1F1F1F; /* Slightly lighter background for movie cards */
        margin: 5px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out;
    }
    .st-emotion-cache-nahz7x:hover {
        transform: translateY(-5px);
    }

    /* Image styling */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }

    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
    }

    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #141414;
        color: #D3D3D3;
        text-align: center;
        padding: 10px 0;
        font-size: 0.9em;
        border-top: 1px solid #282828;
    }
</style>
""", unsafe_allow_html=True)

# --- Main Title and Tagline ---
# Define the custom red color for headings
HEADING_COLOR = "#E50914"


# --- Helper function to display recommendations with posters ---
# MODIFIED: Added clickability for movie posters
def display_recommendations(movie_list, source_type="content"):
    if not movie_list:
        st.info("No recommendations found for your query. Try different input or check spelling.")
        return

    num_cols = 5
    cols = st.columns(num_cols)

    for i, movie_info in enumerate(movie_list):
        with cols[i % num_cols]:  # Distribute movies across the columns
            title = ""
            tmdb_id = None

            if isinstance(movie_info, dict):
                # For TMDb API results (trending, mood-based) and collaborative filtering
                title = movie_info.get('title', "")
                tmdb_id = movie_info.get('tmdb_id')
                if tmdb_id is None:  # Fallback for compatibility if 'id' is used directly
                    tmdb_id = movie_info.get('id')
                # DEBUG: Print info for dictionary-based movie_info
                # st.write(f"DEBUG: Dict input - Title: {title}, TMDb ID: {tmdb_id}, Source: {source_type}")
            else:  # For content/genre recs (if just title string passed)
                title = movie_info
                # IMPORTANT: Using 'original_title' for lookup in df_final and stripping whitespace
                tmdb_id_match = df_final[df_final['original_title'].str.lower() == title.lower().strip()]['id']
                if not tmdb_id_match.empty:
                    tmdb_id = tmdb_id_match.iloc[0]
                # DEBUG: Print info for string-based movie_info
                # st.write(f"DEBUG: String input - Title: {title}, Derived TMDb ID: {tmdb_id}, Source: {source_type}")

            # Display title even if ID is missing
            st.markdown(f"<h3 style='font-size:1.1em; text-align:center;'>{title}</h3>", unsafe_allow_html=True)

            # Create a clickable area for the poster
            if tmdb_id:
                poster_url = get_movie_poster_url(tmdb_id, image_size='w342')

                # Use a button with a unique key to make the poster clickable
                if st.button(f"ℹ️ Details", key=f"details_button_{source_type}_{tmdb_id}",
                             help=f"Click for details of {title}"):
                    st.session_state.update({
                        "current_page": "movie_details",
                        "selected_movie_id": tmdb_id
                    })
                    st.rerun()  # <<< FIXED: Safe rerun after update

                # Display the image inside the button's logical area
                if poster_url:
                    st.image(poster_url, use_container_width=True, caption="")
                else:
                    st.image("https://placehold.co/342x513/282828/FFFFFF?text=No+Poster", use_container_width=True,
                             caption="No Poster Available")
            else:
                # This block is executed if tmdb_id is None
                st.image("https://placehold.co/342x513/282828/FFFFFF?text=ID+Missing", use_container_width=True,
                         caption="ID Missing")
                # DEBUG: Indicate why ID is missing
                # st.write(f"DEBUG: No TMDb ID found for '{title}' (Source: {source_type}). Cannot show details.")


# --- NEW FUNCTION: Display Movie Details Page ---
def display_movie_details_page():
    st.markdown(f"<h2 style='text-align: center; color: {HEADING_COLOR};'>🎬 Movie Details</h2>", unsafe_allow_html=True)

    if st.button("← Back to Recommendations"):
        st.session_state.current_page = "main"
        st.session_state.selected_movie_id = None
        st.rerun()

    movie_id = st.session_state.selected_movie_id
    if not movie_id:
        st.error("No movie selected for details.")
        return

    with st.spinner("Fetching movie details..."):
        # DEBUG: Print the movie_id being passed to get_movie_details_from_tmdb
        # st.write(f"DEBUG: Fetching details for TMDb ID: {movie_id}")
        movie_details = get_movie_details_from_tmdb(movie_id)

    if movie_details:
        st.title(movie_details.get('title'))

        # Display poster and main details side-by-side
        col_poster, col_details = st.columns([1, 2])

        with col_poster:
            poster_url = get_movie_poster_url(movie_id, image_size='w500')
            if poster_url:
                st.image(poster_url, use_container_width=True, caption=movie_details.get('title'))
            else:
                st.image("https://placehold.co/500x750/282828/FFFFFF?text=No+Poster", use_container_width=True,
                         caption="No Poster Available")

        with col_details:
            if movie_details.get('tagline'):
                st.markdown(f"<h3 style='color: #FFD700;'><i>{movie_details.get('tagline')}</i></h3>",
                            unsafe_allow_html=True)

            st.markdown(f"**Overview:** {movie_details.get('overview', 'N/A')}")
            st.markdown(f"**Genres:** {', '.join(movie_details.get('genres', ['N/A']))}")
            st.markdown(f"**Release Date:** {movie_details.get('release_date', 'N/A')}")

            # Format rating to one decimal place
            vote_avg = movie_details.get('vote_average')
            if vote_avg is not None:
                st.markdown(f"**Rating:** {vote_avg:.1f}/10 ({movie_details.get('vote_count', 0)} votes)")
            else:
                st.markdown(f"**Rating:** N/A")

            st.markdown(f"**Runtime:** {movie_details.get('runtime', 'N/A')} minutes")
            st.markdown(f"**Director:** {movie_details.get('director', 'N/A')}")
            st.markdown(f"**Top Cast:** {', '.join(movie_details.get('cast', ['N/A']))}")
            st.markdown(f"**Production Companies:** {', '.join(movie_details.get('production_companies', ['N/A']))}")
            st.markdown(f"**Spoken Languages:** {', '.join(movie_details.get('spoken_languages', ['N/A']))}")

        st.markdown("---")

        # --- Trailer Section ---
        st.subheader("🎬 Movie Trailer")
        # Find a YouTube trailer
        youtube_videos = [v for v in movie_details.get('videos', []) if
                          v.get('site') == 'YouTube' and v.get('type') == 'Trailer']

        if youtube_videos:
            # Prefer the first official trailer if available
            main_trailer_key = None
            for video in youtube_videos:
                if "trailer" in video.get('name', '').lower():
                    main_trailer_key = video['key']
                    break
            if main_trailer_key is None and youtube_videos:  # Fallback to any YouTube trailer if no explicit 'trailer' name
                main_trailer_key = youtube_videos[0]['key']

            if main_trailer_key:
                trailer_url = f"https://www.youtube.com/watch?v={main_trailer_key}"
                st.video(trailer_url)
            else:
                st.info("No primary trailer found, but other videos might be available.")
                # DEBUG: Indicate if no main trailer but other videos exist
                # if movie_details.get('videos', []):
                #     st.write(f"DEBUG: Found {len(movie_details.get('videos', []))} videos, but no main trailer.")
        else:
            st.info("No trailer available for this movie.")
            # DEBUG: Indicate if no videos found at all
            # st.write("DEBUG: No videos found at all for this movie.")

    else:
        st.error("Could not load movie details. Please try again or select another movie.")
        if st.session_state.selected_movie_id:
            st.write(f"Attempted to fetch details for TMDb ID: {st.session_state.selected_movie_id}")
            # DEBUG: Add more detail if movie_details is None
            # st.write("DEBUG: get_movie_details_from_tmdb returned None.")


initialize_session_state()  # Call the initialization function at the start
# --- Main App Logic (Conditional Rendering) ---
if st.session_state.current_page == "movie_details":
    display_movie_details_page()
    st.stop()

# --- Main Title and Tagline (only shown on main page) ---
st.markdown(
    "<h1 style='text-align: center; color: #E50914; font-size: 3.5em; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);'>Flix Finder! <span style='font-size:0.8em; vertical-align: middle;'>❤️</span></h1>",
    unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Your Personalized Movie Recommendation Hub!!</h3>",
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Discover your next favorite movie with our intelligent recommendation system...</p>",
    unsafe_allow_html=True)

# --- Navigation Tabs (only shown on main page) ---
# Updated tab names to match the user's provided code (tab6, tab7, tab8)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Top by Genre!",
    "Content-Based!",
    "Collaborative Filtering!",
    "Voice Recommendation Flixa!",
    "TMDB Trending!",
    "Discover by Movie!",
    "Find Your Vibe!",
    "About Us!"
])

# --- Tab 1: Top Movies by Genre (Using recommend_top_by_genre) ---
with tab1:
    st.markdown(f"<h2 style='text-align: center; color:#E50914;'>🎬 Top Movies by Genre!</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Explore the most popular movies within your favorite categories. Perfect for when you know what genre you're in the mood for!")

    # Get unique genres from your df_final for the selectbox
    # Use 'genres_names' which is already processed in recommender_core.py
    all_genres_raw = df_final['genres_names'].explode().unique()  # Use genres_names here
    all_genres = sorted([g for g in all_genres_raw if isinstance(g, str)])
    selected_genre_for_top = st.selectbox(
        "Select a Genre:",
        ['-- Select Genre --'] + all_genres,
        key='genre_select_tab1'  # Unique key for this widget
    )

    if selected_genre_for_top != '-- Select Genre --':
        with st.spinner(f"Fetching top movies in {selected_genre_for_top}..."):
            top_genre_movies = recommend_top_by_genre(selected_genre_for_top)
            display_recommendations(top_genre_movies, source_type="genre")
    else:
        st.info("👈 Please select a genre from the dropdown.")

# --- Tab 2: Find Movies Similar to What You Like (Content-Based) ---
with tab2:
    st.markdown(f"<h2 style='text-align: center; color:#E50914;'>🔍 Find Similar Movies!</h2>",
                unsafe_allow_html=True)
    st.markdown("Tell us a movie you enjoyed, and we'll find others with similar plot, cast, and themes.")

    content_movie_title = st.text_input(
        "Enter a movie title (e.g., *Avatar*, *The Dark Knight Rises*):",
        placeholder="e.g., Inception",
        key='content_movie_input_tab2'  # Unique key
    )

    content_genre_filter_option = st.selectbox(
        "Optionally filter these similar movies by a specific genre:",
        ['Any Genre'] + all_genres,
        key='content_genre_filter_tab2'  # Unique key
    )

    if st.button("Get Content-Based Recommendations", key='content_button_tab2'):
        if content_movie_title:
            genre_filter_for_content = None if content_genre_filter_option == 'Any Genre' else content_genre_filter_option
            with st.spinner(f"Finding similar movies to '{content_movie_title}'..."):
                similar_movies = recommend(content_movie_title, selected_genre=genre_filter_for_content)
                if similar_movies:
                    st.subheader(f"Recommendations based on '{content_movie_title}'")
                    display_recommendations(similar_movies, source_type="content")
                else:
                    st.error(
                        f"Could not find '{content_movie_title}' or recommendations for it. Please check spelling or try another movie.")
        else:
            st.warning("Please enter a movie title to get content-based recommendations.")

# --- Tab 3: What Other Users Liked (Collaborative Filtering) ---
with tab3:
    st.markdown(f"<h2 style='text-align: center; color:#E50914;'>🤝 What Other Users Liked!</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Discover movies that users with tastes similar to yours have enjoyed. This helps you find hidden gems!")
    st.markdown("*(Note: This uses a separate MovieLens dataset. Movie titles might slightly differ.)*")

    # For collaborative, the user needs to enter a MovieLens title
    # Let's try to provide a selection of popular ML movies to guide the user
    # Filter out empty strings if any, and sort for better display
    popular_ml_movies = sorted([m for m in ml_movie_titles if m.strip()])

    collab_movie_title = st.selectbox(
        "Select a MovieLens movie that you or similar users might have rated:",
        ['-- Select a MovieLens Movie --'] + popular_ml_movies,
        key='collab_movie_select_tab3'  # Unique key
    )

    if st.button("Get Collaborative Recommendations", key='collab_button_tab3'):
        if collab_movie_title != '-- Select a MovieLens Movie --':
            with st.spinner(f"Finding movies liked by users who liked '{collab_movie_title}'..."):
                collab_recs = collaborative_recommend(collab_movie_title)
                if collab_recs:
                    st.subheader(f"Movies liked by users who enjoyed '{collab_movie_title}'")
                    display_recommendations(collab_recs, source_type="collaborative")
                else:
                    st.info(
                        "No collaborative recommendations found. Please ensure the selected movie has sufficient ratings for recommendations.")
        else:
            st.warning("Please select a MovieLens movie to get collaborative recommendations.")
# --- Tab 4: Voice Assistant ---

with tab4:
    st.markdown(f"<h2 style='text-align: center; color:#E50914;'>🎙️ Speak to Your Voice Assistant-Flixa!</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "Click the microphone button, speak your command (e.g., 'recommend a comedy movie', 'find movies similar to Inception'), then click again to stop recording.")
    st.markdown("Try commands like: ")
    st.markdown("- `recommend a **comedy** movie`")
    st.markdown("- `find movies similar to **Avatar**`")
    st.markdown(
        "- `what are some good **action** movies from MovieLens?` (You'll need to specify an exact MovieLens title here, e.g., 'Toy Story (1995)')")

    # Microphone recorder component
    audio_bytes = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        callback=None,
        args=(),
        kwargs=(),
        key="mic_recorder"
    )

    user_command_text = ""  # Initialize to empty string

    # Check if a new audio recording was just completed
    if audio_bytes and 'bytes' in audio_bytes and audio_bytes['bytes']:
        st.session_state["current_audio_played"] = False  # Reset flag for new command
        # Save the audio bytes to a temporary WAV file
        with st.spinner("Transcribing your speech..."):
            try:
                os.makedirs("temp", exist_ok=True)
                audio_file_path = "temp/recorded_audio.wav"
                with open(audio_file_path, "wb") as f:
                    f.write(audio_bytes['bytes'])

                r = sr.Recognizer()
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = r.record(source)
                    user_command_text = r.recognize_google(audio_data)
                    st.session_state["last_command"] = user_command_text
                    st.success(f"You said: \"{user_command_text}\"")

            except sr.UnknownValueError:
                response_text = "I'm sorry, I couldn't understand your audio. Please try speaking more clearly."
                st.error(response_text)

                audio_file_path = text_to_audio(response_text)
                if audio_file_path:
                    st.audio(audio_file_path, format='audio/mp3', start_time=0, autoplay=True)
                    st.session_state["current_audio_played"] = True  # Set flag after playing error audio
                else:
                    st.warning("Could not generate audio for the error message.")

                user_command_text = ""
                st.session_state["last_command"] = ""  # Clear last command if not understood
            except sr.RequestError as e:
                response_text = f"Could not request results from speech recognition service. Please check your internet connection: {e}"
                st.error(response_text)

                audio_file_path = text_to_audio(response_text)
                if audio_file_path:
                    st.audio(audio_file_path, format='audio/mp3', start_time=0, autoplay=True)
                    st.session_state["current_audio_played"] = True  # Set flag after playing error audio
                else:
                    st.warning("Could not generate audio for the error message.")

                user_command_text = ""
                st.session_state["last_command"] = ""
            except Exception as e:
                response_text = f"An unexpected error occurred during transcription: {e}"
                st.error(response_text)

                audio_file_path = text_to_audio(response_text)
                if audio_file_path:
                    st.audio(audio_file_path, format='audio/mp3', start_time=0, autoplay=True)
                    st.session_state["current_audio_played"] = True  # Set flag after playing error audio
                else:
                    st.warning("Could not generate audio for the error message.")

                user_command_text = ""
                st.session_state["last_command"] = ""
            finally:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
    else:
        # If no NEW audio, use the last command from session state
        user_command_text = st.session_state.get("last_command", "")

    # This block processes the command and plays audio ONLY if not already played for this command
    if user_command_text and not st.session_state["current_audio_played"]:  # ADDED CHECK HERE
        with st.spinner("Processing your command..."):
            command_lower = user_command_text.lower()
            response_text = "I'm sorry, I couldn't understand your request. Please try rephrasing."
            audio_file_path = None

            if "similar to" in command_lower:
                match = re.search(r'similar to (.*?)(?:\.|\?|movie)?$', command_lower)
                if match:
                    movie_query = match.group(1).strip()
                    genre_match = re.search(
                        r'(action|comedy|drama|sci-fi|thriller|romance|animation|horror|family|documentary|fantasy|crime|mystery|adventure|war|western|music|history|biography)\s+movie',
                        command_lower)
                    genre_filter = genre_match.group(1) if genre_match else None

                    recommended_titles = recommend(movie_query, selected_genre=genre_filter)
                    if recommended_titles:
                        response_text = f"Okay, here are some movies similar to {movie_query}: {', '.join(recommended_titles[:5])}."
                        st.subheader(f"Movies similar to '{movie_query}'")
                        display_recommendations(recommended_titles, source_type="voice_content")
                    else:
                        st.error(f"I couldn't find movies similar to {movie_query}. Please try another title.")
                else:
                    response_text = "Please specify which movie you'd like similar recommendations for."

            elif "movie from movielens" in command_lower or "movie on movielens" in command_lower or "recommend movies like" in command_lower and "from movielens" in command_lower:
                match = re.search(r'like (.*?)(?:\s+from movielens|\.|\?|$)', command_lower)
                if match:
                    movie_query_ml = match.group(1).strip()
                    collab_recs = collaborative_recommend(movie_query_ml)
                    if collab_recs:
                        response_text = f"Based on user preferences, here are movies similar to {movie_query_ml}: {', '.join([r['title'] for r in collab_recs[:5]])}."
                        st.subheader(f"Movies similar to '{movie_query_ml}' (Collaborative)")
                        display_recommendations(collab_recs, source_type="voice_collab")
                    else:
                        st.info(
                            f"I couldn't find collaborative recommendations for {movie_query_ml}. Please ensure it's a valid MovieLens title like 'Toy Story (1995)'.")
                else:
                    response_text = "Please specify a MovieLens movie title, e.g., 'recommend movies like Toy Story (1995) from MovieLens'."

            elif any(genre.lower() in command_lower for genre in all_genres):
                found_genre = None
                for genre in all_genres:
                    if genre.lower() in command_lower:
                        found_genre = genre
                        break
                if found_genre:
                    top_movies = recommend_top_by_genre(found_genre)
                    if top_movies:
                        response_text = f"Here are some top {found_genre} movies: {', '.join(top_movies[:5])}."
                        st.subheader(f"Top movies in '{found_genre}'")
                        display_recommendations(top_movies, source_type="voice_genre")
                    else:
                        st.info(f"I couldn't find top movies in the {found_genre} genre.")
                else:
                    st.info("Please specify a valid genre like action, comedy, or sci-fi.")
            else:
                st.info("I'm not sure if you're asking for a genre, or similar movies. Please be more specific.")

            # Convert response text to speech and play ONLY if not already played
            audio_file_path = text_to_audio(response_text)

            if audio_file_path and not st.session_state["current_audio_played"]:
                st.audio(audio_file_path, format='audio/mp3', start_time=0, autoplay=True)
                st.session_state["current_audio_played"] = True  # Set flag after playing success audio
            elif not audio_file_path:
                st.error("Failed to generate audio response.")

# --- Tab 5: Dynamic Lists  ---
with tab5:
    st.markdown(f"<h2 style='text-align: center; color:#E50914;'>🎬 Discover Movies Dynamically!</h2>",
                unsafe_allow_html=True)
    st.markdown("Explore movies that are currently trending, playing in theaters, coming soon, or highly rated.")

    category_display_name = st.selectbox(
        "Select a Movie List:",
        ("Trending - Daily", "Trending - Weekly", "Now Playing", "Upcoming", "Top Rated"),
        key="dynamic_list_select"
    )

    if st.button(f"Show {category_display_name}", key="show_dynamic_list_button"):
        with st.spinner(f"Fetching {category_display_name} movies..."):
            api_category = ""
            time_window_param = None

            if "Trending" in category_display_name:
                api_category = "trending"
                time_window_param = "day" if "Daily" in category_display_name else "week"
            elif category_display_name == "Now Playing":
                api_category = "now_playing"
            elif category_display_name == "Upcoming":
                api_category = "upcoming"
            elif category_display_name == "Top Rated":
                api_category = "top_rated"

            # Call the unified function
            dynamic_movies = get_movies_by_category(
                category=api_category,
                time_window=time_window_param,
                n_movies=15  # Fetch 15 movies
            )

            if dynamic_movies:
                st.subheader(f"Top {len(dynamic_movies)} {category_display_name} Movies:")
                display_recommendations(dynamic_movies, source_type="dynamic_list")
            else:
                st.info(f"Could not fetch {category_display_name} movies at this time. Please try again later.")

# Tab: Discover by Movie (now tab6)
with tab6:
    st.markdown(f"<h2 style='text-align: center; color:#E50914;'>🔍 Discover Movies by Name</h2>",
                unsafe_allow_html=True)
    st.write("Enter a movie title to get personalized recommendations based on its content.")

    # 1. Text input for the search query
    search_query = st.text_input("Start typing a movie name:", placeholder="e.g., Inception",
                                 key="movie_search_input_autocomplete")

    found_movie_title = None  # Variable to store the final selected movie title

    if search_query:
        # IMPORTANT: Using 'original_title' for consistency and stripping whitespace
        filtered_titles = [
                              title for title in all_df_movie_titles
                              if search_query.lower().strip() in title.lower().strip()
                          ][:20]  # Limit to top 20 suggestions for performance

        if filtered_titles:
            # 3. Display suggestions in a selectbox
            st.session_state.current_selected_movie = st.selectbox(
                "Select a movie from the suggestions:",
                options=['-- Select --'] + filtered_titles,  # Add a placeholder option
                key="movie_autocomplete_selectbox"
            )

            if st.session_state.current_selected_movie != '-- Select --':
                found_movie_title = st.session_state.current_selected_movie
        else:
            st.info("No movie titles match your search. Try another name or adjust your spelling.")
    else:
        st.info("Start typing a movie title to see suggestions.")
        st.write("Popular examples: Avatar, The Dark Knight Rises, Minions")

    if st.button("Get Suggestions", key="get_suggestions_button_autocomplete"):
        if found_movie_title:
            st.subheader(f"Recommendations for '{found_movie_title}':")
            # --- TO DISPLAY SEARCHED MOVIE'S DETAILS ---
            st.markdown("#### Your Selected Movie:")
            # IMPORTANT: Using 'original_title' for consistency with df_final
            selected_movie_data = df_final[df_final['original_title'] == found_movie_title].iloc[0]
            col_poster, col_details = st.columns([1, 2])
            with col_poster:
                movie_poster_url = get_movie_poster_url(selected_movie_data['id'])
                if movie_poster_url:
                    st.image(movie_poster_url, caption=found_movie_title, width=150)
                else:
                    st.write(found_movie_title)

            with col_details:
                release_year = 'N/A'
                if 'release_date' in selected_movie_data and pd.notnull(selected_movie_data['release_date']):
                    try:
                        release_year = pd.to_datetime(selected_movie_data['release_date']).year
                    except ValueError:
                        pass

                st.markdown(f"**Release Year:** {release_year}")

                try:
                    # Using 'genres_names' which is already processed in recommender_core.py
                    if 'genres_names' in selected_movie_data and selected_movie_data['genres_names']:
                        genre_names = ", ".join(selected_movie_data['genres_names'])
                    else:
                        # Fallback if 'genres_names' isn't directly populated or is empty
                        genres_list = literal_eval(selected_movie_data['genres'])
                        genre_names = ", ".join([g['name'] for g in genres_list])
                    st.markdown(f"**Genres:** {genre_names if genre_names else 'N/A'}")
                except (ValueError, SyntaxError, TypeError):
                    st.markdown(f"**Genres:** N/A")

                st.markdown(
                    f"**Average Rating:** {selected_movie_data['vote_average']:.1f} / 10 (from {selected_movie_data['vote_count']} votes)")
                st.markdown(f"**Overview:** {selected_movie_data['overview']}")

            st.markdown("---")

            with st.spinner("Finding recommendations..."):
                recommended_movies = recommend(found_movie_title)

            if recommended_movies:
                display_recommendations(recommended_movies, source_type="search_based")
            else:
                st.warning(f"Could not find recommendations for '{found_movie_title}' or something went wrong.")
        else:
            st.warning("Please select a movie from the suggestions or type a valid movie name.")

# Tab: Find Your Vibe (Mood Matcher) (now tab7)
with tab7:
    st.markdown(f"<h2 style='text-align: center; color: {HEADING_COLOR};'>🌈 Find Your Movie Vibe!</h2>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Select your current mood, and we'll suggest movies that match your feeling from TMDb.</p>",
        unsafe_allow_html=True)

    MOOD_TO_GENRES = {
        "Happy / Uplifting 😊": ["Comedy", "Animation", "Family", "Music", "Romance"],
        "Exciting / Action-Packed 💥": ["Action", "Adventure", "Science Fiction", "Thriller", "War"],
        "Relaxing / Chill 😌": ["Documentary", "Comedy", "History"],
        "Thought-Provoking / Deep 🤔": ["Drama", "Mystery", "Science Fiction", "Crime"],
        "Scary / Suspenseful 👻": ["Horror", "Thriller", "Mystery"],
        "Romantic 💕": ["Romance", "Drama", "Comedy"]
    }

    mood_options = list(MOOD_TO_GENRES.keys())
    selected_mood = st.selectbox(
        "How are you feeling today?",
        options=['-- Select your mood --'] + mood_options,
        key="mood_selector_tmdb"
    )

    if selected_mood != '-- Select your mood --':
        st.subheader(f"Movies for a '{selected_mood}' mood:")

        target_genres_names = MOOD_TO_GENRES[selected_mood]

        certification_level = None
        certification_country = 'US'

        if selected_mood == "Romantic 💕":
            certification_level = 'PG-13'
            st.info("Romantic movie suggestions are filtered to be generally suitable for audiences 16+.")

        with st.spinner(f"Finding movies for a '{selected_mood}' vibe from TMDb..."):
            mood_based_recommendations = get_mood_based_movies_from_tmdb(
                target_genres_names,
                certification_level=certification_level,
                certification_country=certification_country
            )

        if mood_based_recommendations:
            display_recommendations(mood_based_recommendations, source_type="mood_based_tmdb")
        else:
            st.info(
                f"Could not find movies for a '{selected_mood}' mood with the applied filters. Try a different mood.")
    else:
        st.info("Please select a mood to get movie suggestions.")

# Tab: About Us (now tab8)

with tab8:
    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.markdown(f"<h2 style='text-align: center; color: {HEADING_COLOR};'>🎬 About Flix Finder!</h2>",
                    unsafe_allow_html=True)
        st.markdown(
            """
            <p style='text-align: center;'>
            Welcome to **Flix Finder**, your ultimate destination for discovering your next favorite movie!
            This application leverages various recommendation techniques to provide you with personalized movie suggestions.
            </p>
            """, unsafe_allow_html=True
        )

        st.divider()

        st.markdown(f"<h3 style='text-align: center; color: {HEADING_COLOR};'>Our Features:</h3>",
                    unsafe_allow_html=True)

        feature_col1, feature_col2 = st.columns(2)

        features = [
            ("🎬 Top by Genre", "Explore the most popular movies within specific genres."),
            ("🔍 Content-Based",
             "Get suggestions for movies similar to ones you already love, based on their plot, cast, and keywords."),
            ("🤝 Collaborative Filtering",
             "Discover movies recommended by users with similar tastes, powered by the MovieLens dataset."),
            ("🎙️ Voice Assistant", "Interact with Flix Finder using your voice to get recommendations."),
            ("🚀 Dynamic Lists",
             "Stay updated with trending, now playing, upcoming, and top-rated movies directly from TMDb."),
            ("📝 Discover by Name",
             "Quickly find detailed information and recommendations for any movie in our database by typing its name."),
            ("😍 Find your Vibe",
             "Mood Based Suggestions!!!")
        ]

        for i, (icon_title, description) in enumerate(features):
            target_col = feature_col1 if i % 2 == 0 else feature_col2
            with target_col:
                st.markdown(f"#### {icon_title}")
                st.write(description)
                st.markdown("---")

        st.divider()

        st.markdown(f"<h3 style='text-align: center; color: {HEADING_COLOR};'>Technologies Used:</h3>",
                    unsafe_allow_html=True)

        st.markdown(
            """
            <div style='display: flex; justify-content: center;'>
                <ul style='list-style-position: inside; text-align: left; padding-left: 0; margin-left: 0;'>
                    <li><b>Streamlit:</b> For building the interactive web application.</li>
                    <li><b>Python:</b> The core programming language.</li>
                    <li><b>Pandas & NumPy:</b> For data manipulation and analysis.</li>
                    <li><b>Scikit-learn:</b> For machine learning models (Cosine Similarity, Nearest Neighbors).</li>
                    <li><b>Requests:</b> For fetching movie data and posters from The Movie Database (TMDb) API.</li>
                    <li><b>SpeechRecognition & gTTS:</b> For the voice assistant functionality.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True
        )

        st.divider()

        st.markdown(f"<h3 style='text-align: center; color: {HEADING_COLOR};'>Developer:</h3>",
                    unsafe_allow_html=True)

        dev_img_col, dev_text_col = st.columns([0.5, 3])
        with dev_img_col:
            # Replaced the icon with a placeholder image
            st.image("profilepic.png", width=100)  # Placeholder image
        with dev_text_col:
            st.markdown("<p style='font-size: 1.2em; margin-top: 10px;'>**Divyashree Kondekar**</p>",
                        unsafe_allow_html=True)
            st.markdown(
                "<p style='font-style: italic;'>Developed with ❤️ by Divyashree as an internship project.</p>",
                unsafe_allow_html=True)

        st.markdown(
            "<p style='text-align: center; padding-top: 20px;'>Feel free to explore and enjoy your movie discovery journey!</p>",
            unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer" style="text-align: center; padding-top: 20px; color: #777;">
    Built with ❤️ by Divyashree!!
</div>
""", unsafe_allow_html=True)
