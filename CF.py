import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
# import numpy as np
# import warnings
# from sklearn.decomposition import TruncatedSVD

books = pd.read_csv('BX-Books.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
books.columns = ['ISBN', 'BookTitle', 'BookAuthor', 'Year-of-Publication', 'Publisher', 'Image-URL-S',
                 'Image-URL-M', 'Image-URL-L']
book_titles = books['BookTitle']
ratings = pd.read_csv('ratings_test.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
ratings.columns = ['user_id', 'ISBN', 'rating']
# users = pd.read_csv('BX-Users.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
# users.columns = ['user_id', 'location', 'age']
book_ratings = pd.merge(ratings, books, on='ISBN')
col = ['BookAuthor', 'Year-of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
book_ratings = book_ratings.drop(col, axis=1)
print(book_ratings)
book_ratings = book_ratings.drop_duplicates(['user_id', 'BookTitle'])
book_ratings = book_ratings.pivot(index='BookTitle', columns='user_id', values='rating').fillna(0)
# italy_users_mf = italy_users.pivot(index='user_id', columns='BookTitle', values='rating').fillna(0)
print(book_ratings)
book_ratings.to_csv('output.csv', sep=';', encoding="latin-1", index=False)
sparse_matrix = csr_matrix(book_ratings.values)
print(sparse_matrix)
model = NearestNeighbors()
model.fit(sparse_matrix)
# query = np.random.choice(book_ratings.shape[0])
# print('query:', query)
# print(book_ratings.iloc[query, :])

# k = book_ratings.iloc[query, :]
# j = np.array(k, dtype=pd.Series)

distances, indices = model.kneighbors(book_ratings, n_neighbors=5)

# for i in range(0, len(distances.flatten())):
#     if i == 0:
#         print("Recommendations for ", book_ratings.index[query], " : \n")
#     else:
#         print(i, " : ", book_ratings.index[indices.flatten()[i]])
for i in range(0, len(distances)):
    print("Recommendations for ", i, " ", book_ratings.index[i], " : \n")
    for j in indices[i]:
        print(j, " : ", book_ratings.index[j])

#   matrix factorization user-based filtering
# X = italy_users_mf.values.T
# SVD = TruncatedSVD(n_components=12, random_state=17)
# matrix = SVD.fit_transform(X)
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# corr = np.corrcoef(matrix)
# titles = italy_users_mf.columns
# mf_list = list(titles)
# rec = mf_list.index("Zeke and Ned")
# corr_rec = corr[rec]
# result = list(titles[(corr_rec < 1.0) & (corr_rec > 0.9)])
# print(result)