"""
   This script is used to create x_test.txt after applying stop_words, lemmatization, tokenization
"""
import unicodedata
from nltk.corpus import stopwords
import ast
from keras.preprocessing.text import text_to_word_sequence
from stop_words import get_stop_words
import spacy

nlp = spacy.load('fr_core_news_lg') # load NLTK french stop words

# The list of the created stop words
sw_list2 = ["film", "a", "abord", "afin", "ah", "ai", "qu", "aie", "ailleurs", "ainsi", "ait", "allaient", "allo",
            "allons", "allô", "alors", "anterieur", "anterieure", "anterieures", "apres", "après", "as", "au",
            "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "auraient", "aurait", "auront", "aussi", "autre",
            "autrefois", "autrement", "autres", "autrui", "aux", "auxquelles", "auxquels", "avaient", "avais", "avait",
            "avant", "avec", "avoir", "avons", "ayant", "bah", "bas", "basee", "bat", "bigre", "boum", "brrr", "c'",
            "car", "ce", "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui",
            "celui-ci", "celui-là", "cent", "certain", "certaine", "certaines", "certains", "certes", "ces", "cet",
            "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher", "chers", "chez", "chiche",
            "chut", "chère", "chères", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac",
            "clic", "combien", "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre",
            "couic", "crac", "c’", "d'", "da", "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis",
            "dernier", "derniere", "derriere", "derrière", "des", "desormais", "desquelles", "desquels", "deux",
            "deuxième", "deuxièmement", "devant", "devers", "devra", "different", "differentes", "differents",
            "différent", "différente", "différentes", "différents", "dire", "directe", "directement", "dit", "dite",
            "dits", "divers", "diverse", "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit",
            "doivent", "donc", "dont", "douze", "douzième", "dring", "du", "duquel", "durant", "dès", "désormais", "d’",
            "effet", "egale", "egalement", "egales", "eh", "elle", "elle-même", "elles", "elles-mêmes", "en", "encore",
            "enfin", "entre", "envers", "environ", "es", "est", "et", "etaient", "etais", "etait", "etant", "etc",
            "etre", "eu", "euh", "eux", "eux-mêmes", "exactement", "excepté", "extenso", "exterieur", "fais",
            "faisaient", "faisant", "fait", "façon", "feront", "fi", "flac", "floc", "font", "gens", "ha", "hein",
            "hem", "hep", "hi", "ho", "holà", "hop", "hormis", "hors", "hou", "houp", "hue", "hui", "huit", "huitième",
            "hum", "hurrah", "hé", "i", "il", "ils", "importe", "j'", "je", "jusqu", "jusque", "juste", "j’", "l'",
            "la", "laisser", "laquelle", "las", "le", "lequel", "les", "lesquelles", "lesquels", "leur", "leurs",
            "longtemps", "lors", "lorsque", "lui", "lui-meme", "lui-même", "là", "lès", "l’", "m'", "ma", "maint",
            "maintenant", "mais", "malgre", "me", "meme", "memes", "mes", "mien", "mienne", "miennes", "miens", "mille",
            "moi", "moi-meme", "moi-même", "moindres", "mon", "moyennant", "même", "mêmes", "m’", "n'", "na", "naturel",
            "naturelle", "naturelles", "ne", "necessaire", "necessairement", "neuvième", "ni", "nombreuses", "nombreux",
            "non", "nos", "notamment", "notre", "nous", "nous-mêmes", "nôtre", "nôtres", "n’", "o", "oh", "ohé", "ollé",
            "olé", "on", "ont", "onze", "onzième", "ore", "ou", "ouias", "oust", "ouste", "outre", "ouvert", "ouverte",
            "ouverts", "où", "paf", "pan", "par", "parce", "parle", "parlent", "parler", "parmi", "parseme", "partant",
            "particulier", "particulière", "passé", "pendant", "pense", "permet", "personne", "peut", "peuvent", "peux",
            "pff", "pfft", "pfut", "pif", "pire", "plein", "plouf", "plus", "plusieurs", "plutôt", "possessif",
            "possessifs", "possible", "possibles", "pouah", "pour", "pourrais", "pourrait", "pouvait", "prealable",
            "precisement", "premier", "première", "premièrement", "pres", "probable", "probante", "procedant", "proche",
            "près", "psitt", "pu", "puis", "puisque", "pur", "pure", "qu'", "quand", "quant", "quant-à-soi", "quanta",
            "quarante", "quatorze", "quatre", "quatre-vingt", "quatrième", "quatrièmement", "que", "quel", "quelconque",
            "quelle", "quelles", "quelqu'un", "quelque", "quelques", "quels", "qui", "quiconque", "quinze", "quoi",
            "quoique", "qu’", "rare", "rarement", "rares", "relative", "rend", "rendre", "restant", "reste", "restent",
            "retour", "revoici", "revoilà", "s'", "sa", "sacrebleu", "sait", "sapristi", "sauf", "se", "sein", "seize",
            "selon", "semblable", "semblaient", "semble", "semblent", "sent", "sept", "septième", "sera", "seraient",
            "serait", "seront", "ses", "seul", "seule", "si", "sien", "sienne", "siennes", "siens", "sinon", "six",
            "sixième", "soi", "soi-même", "soit", "soixante", "son", "sont", "sous", "souvent", "specifique",
            "specifiques", "speculatif", "stop", "strictement", "subtiles", "suffisant", "suffisante", "suffit", "suis",
            "suit", "suivant", "suivante", "suivantes", "suivants", "suivre", "superpose", "sur", "surtout", "s’", "t'",
            "ta", "tac", "tant", "te", "tel", "telle", "tellement", "telles", "tels", "tenant", "tend", "tenir",
            "tente", "tes", "tic", "tien", "tienne", "tiennes", "tiens", "toc", "toi", "toi-même", "ton", "touchant",
            "toujours", "tous", "tout", "toute", "toutefois", "toutes", "treize", "trente", "trois", "troisième",
            "troisièmement", "trop", "tsoin", "tsouin", "tu", "té", "t’", "un", "une", "unes", "uniformement", "unique",
            "uniques", "uns", "va", "vais", "vas", "vers", "via", "vif", "vifs", "vingt", "vivat", "vive", "vives",
            "vlan", "voici", "voilà", "vont", "vos", "votre", "vous", "vous-mêmes", "vu", "vé", "vôtre", "vôtres",
            "zut", "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "été", "être", "ô"]

sw_list = get_stop_words("fr") + ["c'est", "j'ai", "qu'il", "qu'elle", "qu'ils", "qu'elles", "d'un",
                                  "d'une", "c'est", "où", "film", "n'ai", "à", "qu'", "n'", "déjà", "c'", "l'", "où",
                                  "s'", "qu'", "j'", "d'", "ainsi", "»", "«", "m'", "être", "'", "y'a", "y'", "c’est",
                                  "j’ai", "qu’il", "qu’elle", "qu’ils", "qu’elles", "d’un",
                                  "d’une", "c’est", "où", "film", "n’ai", "à", "qu’", "n’", "déjà", "c’", "l’", "où",
                                  "s’", "qu’", "j’", "d’", "ainsi", "»", "«", "m’", "être", "’", "y’a", "y’", "entre",
                                  "ça", "même", "meme", "mème", "scénario", "personnage", "acteur", "scène", "la",
                                  "celui"] + stopwords.words(
    'french')
for w in sw_list:
    if w in ["personne", "personnes", "seulement", "personnes", "nouveau", "nouveaux", "aucun", "encore", "force",
             "haut", "moins", "pas"]:
        sw_list.remove(w)


def readFile(filename):
    trainList = []
    for line in open(filename, 'r', encoding='UTF-8'):
        line = ast.literal_eval(line).get("commentaire")
        trainList.append(line)
    return trainList


def tokenization(testList):
    filtered_corpus = []
    for i in range(len(testList)):
        try:
            word_tokens = text_to_word_sequence(unicodedata.normalize("NFKD", testList[i]))
            lemma_word = []
            doc = nlp(' '.join(word_tokens))
            for token in doc:
                lemma_word.append(token.lemma_)

            filtered_comment = []
            for w in lemma_word:
                if w not in sw_list2:
                    filtered_comment.append(w)
            if filtered_comment == []:
                print("here is an empty ", i, "  ", testList[i])
                filtered_corpus.append(["....."])
            else:
                filtered_corpus.append(filtered_comment)
            print(i, " of ", len(testList))
        except:
            print("here is the exception", i, "  ", testList[i], "\n and is the next", testList[i + 1])
            filtered_corpus.append(["....."])
            pass
    return filtered_corpus


def join_list(filtered_corpus):
    resulted_list = []
    for i in range(len(filtered_corpus)):
        resulted_list.append(' '.join(filtered_corpus[i]))
    return resulted_list


file = readFile("../data/test_comments.txt")
print("original length: ", len(file))
token = tokenization(file)

vector_corpus = join_list(token)
tokens_output = open('../data/x_test.txt', 'w', encoding='utf-8')

print("length vector_corppus ", len(vector_corpus))
for i in range(len(vector_corpus)):  # range(len(vector_corpus)):
    print(vector_corpus[i], file=tokens_output)
tokens_output.close()
