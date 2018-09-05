import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#   Loading Data
#   books = pd.read_csv(io.StringIO(downloaded.getvalue().decode('latin-1')), sep=';',
#   escapechar='\\', error_bad_lines=False)

books = pd.read_csv('test2.csv', sep=';', escapechar='\\', error_bad_lines=False,  encoding="latin-1")
books.columns = ['ISBN', 'BookTitle', 'BookAuthor', 'Year-of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M',
                 'Image-URL-L']
books_train, books_test = train_test_split(books, test_size=0.2)
# print(books)
# print(books_train)

books_train = books_train.reset_index()
books_train = books_train.drop(['Year-of-Publication', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)
# books_train.to_csv('op.csv', sep=';', encoding="latin-1", index=False)
vector = TfidfVectorizer(analyzer='word', ngram_range=(1, 5), stop_words='english')
vector_c = CountVectorizer()

#  get tf_idf matrix for title similarity
title_tf = vector.fit_transform(books_train['BookTitle'])
title_tf_idf = vector.fit_transform(books_train['BookTitle'])

#  calculate cosine similarity matrix based on titles
title_similarity = cosine_similarity(title_tf_idf, title_tf_idf)

#  similarity based on Book Author
authors_list = []
authors = books_train['BookAuthor']
for a in authors:
    authors_list.append(str.lower(a.replace(" ", "")))

author_cv = vector_c.fit_transform(authors_list)
print(author_cv)
author_similarity = cosine_similarity(author_cv, author_cv)

#  similarity based on book publisher
publishers_list = []
publishers = books_train['Publisher']
for p in publishers:
    publishers_list.append(str.lower(p.replace(" ", "")))

publisher_cv = vector_c.fit_transform(publishers_list)
publisher_similarity = cosine_similarity(publisher_cv, publisher_cv)

# Total similarity
# total_similarity = title_similarity + author_similarity + publisher_similarity
total_similarity = title_similarity + author_similarity

#  sort in order of most similar and recommend the top 5 similar

for i, row in books_train.iterrows():
    print(row['BookTitle'], " By: ", row['BookAuthor'], " Publisher: ", row['Publisher'])
    print("\n")
    best_similar = total_similarity[i].argsort()[::-1]
    best_similar = best_similar[1:9]
    print("10 Recommended Books: ")
    for n in best_similar:
        print(books_train.BookTitle[n], " by: ", books_train.BookAuthor[n], " Publisher: ", books_train.Publisher[n])
    print("\n")
