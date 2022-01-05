"""
    xml file parsing script to extract data into a .txt file
"""

import xml.etree.ElementTree as ET

tree = ET.parse("../data/train.xml")
root = tree.getroot()

comments = {}
arr = []

for comment in root.findall('comment'):
    movie_id = comment.find('movie').text
    review_id = comment.find('review_id').text
    commentaire = comment.find('commentaire').text
    len_commentaire = len(str(commentaire).split())
    note = comment.find('note').text

    comments["movie_id"] = movie_id
    comments["review_id"] = review_id
    comments["commentaire"] = commentaire
    comments["len_commentaire"] = len_commentaire
    comments["note"] = note
    arr.append(comments.copy())

fi = open('../data/train_comments.txt', 'w', encoding='utf-8')

for i in range(len(arr)):
    print(arr[i], file=fi)
fi.close()
