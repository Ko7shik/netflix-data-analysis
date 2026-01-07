import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("netflix_titles.csv")

# Keep only required columns and drop missing values
df = df[['title', 'description']].dropna().reset_index(drop=True)

# Convert text descriptions into TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a mapping of movie titles to index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, num_recommendations=5):
    if title not in indices:
        return f"'{title}' not found in dataset."

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:num_recommendations + 1]
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

# Example run
if __name__ == "__main__":
    print("Recommendations for 'Breaking Bad':\n")
    print(recommend("Breaking Bad"))