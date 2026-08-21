# ==========================================
# MOVIE RECOMMENDATION SYSTEM
# ==========================================

# 1. Import libraries

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 2. Load Dataset
# ==========================================

data = pd.read_csv("movies.csv")


print("First 5 movies:")
print(data.head())


# ==========================================
# 3. Check Dataset
# ==========================================

print("\nDataset shape:")
print(data.shape)


print("\nColumn names:")
print(data.columns)


# ==========================================
# 4. Handle Missing Values
# ==========================================

data["genre"] = data["genre"].fillna("")

data["keywords"] = data["keywords"].fillna("")


# ==========================================
# 5. Combine Movie Information
# ==========================================

data["combined"] = (
    data["genre"] + " " + data["keywords"]
)


print("\nCombined information:")
print(data[["title", "combined"]].head())


# ==========================================
# 6. Convert Text into Numbers
# ==========================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)


feature_matrix = vectorizer.fit_transform(
    data["combined"]
)


print("\nFeature matrix shape:")
print(feature_matrix.shape)


# ==========================================
# 7. Calculate Similarity
# ==========================================

similarity = cosine_similarity(
    feature_matrix
)


print("\nSimilarity matrix shape:")
print(similarity.shape)


# ==========================================
# 8. Movie Recommendation Function
# ==========================================

def recommend_movies(movie_name):

    # Find movie index

    movie_index = data[
        data["title"].str.lower() == movie_name.lower()
    ].index


    # Check if movie exists

    if len(movie_index) == 0:

        print("\nMovie not found.")

        return


    movie_index = movie_index[0]


    # Get similarity scores

    similarity_scores = list(
        enumerate(similarity[movie_index])
    )


    # Sort movies by similarity

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )


    # Display recommendations

    print("\nRecommended Movies:")

    count = 0


    for index, score in similarity_scores:

        # Don't recommend the same movie

        if index == movie_index:
            continue


        print(
            data.iloc[index]["title"],
            "-> Similarity:",
            round(score, 2)
        )


        count += 1


        if count == 5:
            break


# ==========================================
# 9. Take User Input
# ==========================================

movie_name = input(
    "\nEnter a movie name: "
)


# ==========================================
# 10. Recommend Movies
# ==========================================

recommend_movies(movie_name)
