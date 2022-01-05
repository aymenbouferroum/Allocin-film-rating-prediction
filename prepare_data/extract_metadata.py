# Extracting metadata from allocine website :
#       - the release year of the film
#       - The name of the film
# Extracting from the corpus :
#       - Number of reviews
#       - Max, Min, Med note (per film)

import ast
from statistics import mean, stdev
from collections import Counter
from urllib.request import urlopen
from bs4 import BeautifulSoup

"""
    A function that is able to read data file,
    Every line in the input file takes the following structure : 
    {'movie_id': 'xx', 'review_id': 'xx', 'commentaire': "xx", 'len_commentaire': xx, 'note': 'xx'}
"""


def readFile(filename):
    data = []
    for line in open(filename, 'r', encoding='UTF-8'):
        line = ast.literal_eval(line)
        data.append(line)
    return data


documents = readFile("../data/train_comments.txt")
print(len(documents))
# get a list of movies ids
movies_id = []
for i in range(len(movies_id)):
    movies_id.append(movies_id[i].get("movie_id"))

# get number of reviews for each film
numb_reviews = Counter(movies_id)

movies = {}
arr2 = []
tested_ids = []

for i in range(len(documents)):

    movie_id = documents[i].get("movie_id")
    if movie_id not in tested_ids:  # Test if the movie_id was processed before
        movies["movie_id"] = movie_id

        # Getting MOVIE NAME,YEAR OF THE RELEASE from ALLOCINE website
        url = 'https://www.allocine.fr/film/fichefilm_gen_cfilm=' + movie_id + '.html'
        soup = BeautifulSoup(urlopen(url), features="lxml").title.get_text()
        try:
            movie_title = soup.split('- film')[0]
            release_year = soup.split('- film')[1].replace(" - AlloCiné", "")

            print(movie_title)
        except:
            movie_title = soup.split('- AlloCiné')[0]
            release_year = "none"

        movies["movie_title"] = movie_title
        movies["release_year"] = release_year
        movies["number_of_reviews"] = numb_reviews.get(movie_id)
        notes = []
        for j in range(len(movies_id)):
            if (movies_id[j].get("movie_id") == movie_id):
                notes.append(float(movies_id[j].get("note").replace(",", ".")))
        try:
            movies["Max_note"] = max(notes)
            movies["Min_note"] = min(notes)
            movies["Median_note"] = mean(notes)
            movies["stdev"] = stdev(notes)
        except:
            movies["stdev"] = 0
        arr2.append(movies.copy())
        tested_ids.append(movie_id)

print(arr2[0])
f = open('output/train_movies_states.txt', 'w')

# Writing results into a .txt file
for i in range(len(arr2)):
    print(arr2[i], file=f)

print("Total number of films : ", len(numb_reviews), file=f)

f.close()
