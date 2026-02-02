
# 🎬 Movie Mentor Pro: Hybrid AI Recommender

Movie Mentor Pro is a sophisticated movie recommendation engine that combines **Natural Language Processing (NLP)** and **Machine Learning Classification** to provide high-quality cinematic suggestions.



## 🚀 Key Features
* **Hybrid Engine:** Uses Cosine Similarity for content matching and SVM for quality filtering.
* **Top Pick Badges:** Automatically identifies "Must-Watch" films using an SVC (Support Vector Classifier).
* **Live Posters:** Fetches real-time movie posters and metadata via the OMDb API.
* **Compressed Data:** Utilizes LZMA compression (`.xz`) to handle large similarity matrices efficiently.
* **Interactive UI:** Built with Streamlit for a seamless user experience.

## 🧠 How it Works
1.  **Stage 1: Vectorization & Similarity:**
    The system processes movie metadata (genres, cast, crew, keywords) and converts them into vectors using `CountVectorizer`. It then calculates the **Cosine Similarity** to find movies with similar themes.
    
2.  **Stage 2: SVM Classification:**
    A Support Vector Machine (SVM) model is trained on movie popularity and vote counts to classify movies into "Top Picks" (High Quality) or "Standard" results.

3.  **Stage 3: Deployment:**
    The app is deployed on Streamlit Cloud, utilizing a compressed similarity matrix to optimize memory usage.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Libraries:** Pandas, Scikit-Learn, NLTK, Streamlit
* **API:** OMDb API (Open Movie Database)
* **Compression:** LZMA / Pickle

## 📥 Installation & Usage
To run this project locally:

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/movie-recommender-pro.git](https://github.com/khilosiyavaibhavi1711-collab/movie-recommender-pro.git)
