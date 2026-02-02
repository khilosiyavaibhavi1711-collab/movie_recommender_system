import streamlit as st
import pickle
import requests
import numpy as np
import pandas as pd


# --- CONFIG ---
st.set_page_config(page_title="Movie Mentor Pro", layout="wide")


# --- LOAD DATA ---
@st.cache_data
def load_data():
    movies = pickle.load(open('movies.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    svm_model = pickle.load(open('svm_model.pkl', 'rb'))
    return movies, similarity, svm_model


movies, similarity, svm_model = load_data()


# --- HELPER FUNCTIONS ---
@st.cache_data
def fetch_details(movie_title):
    api_key = "83f43d02"
    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={api_key}&plot=short"
    details = {"poster": "https://placehold.co/500x750?text=No+Poster", "year": "N/A", "rating": "N/A",
               "plot": "No plot available."}
    try:
        data = requests.get(url, timeout=5).json()
        if data.get('Response') == 'True':
            details["poster"] = data.get('Poster') if data.get('Poster') != 'N/A' else details["poster"]
            details["year"] = data.get('Year', 'N/A')
            details["rating"] = data.get('imdbRating', 'N/A')
            details["plot"] = data.get('Plot', "No plot available.")
    except:
        pass
    return details


def recommend(movie_title):
    # This is where the sorting and enumerate logic lives!
    index = movies[movies['title'] == movie_title].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    results = []
    for i in distances[1:6]:
        m_data = movies.iloc[i[0]]
        details = fetch_details(m_data.title)

        # Use SVM to predict if it's a "Top Pick"
        features = [[m_data.popularity, m_data.vote_count]]
        prediction = svm_model.predict(features)

        details['title'] = m_data.title
        details['is_top_pick'] = True if prediction[0] == 1 else False
        results.append(details)
    return results


# --- UI ---
st.title(
    "Your Movie Mentor🎬 "
)

# Professional Tagline
st.markdown("""
    <h3 style='text-align: left; color: yellow; font-style: italic;'>
    "Your Next Favorite Movie is Just One Click Away."
    </h3>
    """, unsafe_allow_html=True)

st.write("Popcorn ready? Let our AI find your perfect match.")

selected_movie = st.selectbox("Select Your Favourite Movie :", movies['title'].values)

if st.button('Get Recommendations'):
    recs = recommend(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            # Show SVM badge
            if recs[i]['is_top_pick']:
                st.markdown("🔥 **TOP PICK**")
            else:
                st.markdown("🍿 **SIMILAR**")

            st.image(recs[i]['poster'])
            st.write(f"**{recs[i]['title']}**")
            st.caption(f"{recs[i]['year']} | ⭐ {recs[i]['rating']}")
            with st.expander("Read Plot"):
                st.write(recs[i]['plot'])

# --- ADD THIS IN THE SIDEBAR OR BELOW THE MAIN BUTTON ---
if st.button('🎲 Surprise Me!'):
    # Filter movies that the SVM considers "Top Picks" (Quality Label = 1)
    top_picks = movies[movies['quality_label'] == 1]

    # Select one at random
    random_movie = top_picks.sample(1).iloc[0]

    st.balloons()
    st.subheader(f"Our AI suggests: {random_movie.title}")

    # Fetch details for the surprise movie
    details = fetch_details(random_movie.title)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(details['poster'])
    with col2:
        st.write(f"**Year:** {details['year']}")
        st.write(f"**Rating:** {details['rating']}")
        st.write(details['plot'])

        import lzma
        import pickle
        import streamlit as st


        @st.cache_resource
        def load_data():
            # Load normal files
            movies = pickle.load(open('movies.pkl', 'rb'))
            svm_model = pickle.load(open('svm_model.pkl', 'rb'))

            # Load the compressed file (The one with the WinRAR icon)
            with lzma.open('similarity.pkl.xz', 'rb') as f:
                similarity = pickle.load(f)

            return movies, similarity, svm_model