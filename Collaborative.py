import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

# preparing all data
books = pd.read_csv('BX-Books.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
books.columns = ['ISBN', 'BookTitle', 'BookAuthor', 'Year-of-Publication', 'Publisher', 'Image-URL-S',
                 'Image-URL-M', 'Image-URL-L']

# data split into training and testing
books_train, books_test = train_test_split(books, test_size=0.2)
books_train = books_train.reset_index()


ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
ratings.columns = ['user_id', 'ISBN', 'rating']
users = pd.read_csv('BX-Users.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
users.columns = ['user_id', 'location', 'age']
book_ratings = pd.merge(ratings, books_train, on='ISBN')
col = ['BookAuthor', 'Year-of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
book_ratings = book_ratings.drop(col, axis=1)

#  performing knn algorithm on users from italy to try on a small number of users 
italy_users = pd.merge(book_ratings, users, left_on='user_id', right_on='user_id', how='left')
italy_users = italy_users[italy_users['location'].str.contains("italy")]
italy_users = italy_users.drop(['age'], axis=1)
italy_users = italy_users.drop_duplicates(['user_id', 'BookTitle'])
italy_users = italy_users.pivot(index='BookTitle', columns='user_id', values='rating').fillna(0)
italy_users.to_csv('op.csv', sep=';', encoding="latin-1", index=False)
print(italy_users)

#  creating sparse matrix for knn
sparse_matrix = csr_matrix(italy_users.values)
# print(sparse_matrix)

# knn item based
# model = NearestNeighbors(metric='cosine', algorithm='brute')
model = NearestNeighbors()
model.fit(sparse_matrix)


distances, indices = model.kneighbors(italy_users, n_neighbors=5)

#   printing 5 neighbours of each book
for i in range(0, len(distances)):
    print("Recommendations for ", i, " ", italy_users.index[i], " : \n")
    for j in indices[i]:
        print(j, " : ", italy_users.index[j])

