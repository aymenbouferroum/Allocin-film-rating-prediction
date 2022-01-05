import os
from collections import Counter, OrderedDict
import itertools
import matplotlib.pyplot as plt

sw_list = ["film", "a", "abord", "afin", "ah", "ai", "qu", "aie", "ailleurs", "ainsi", "ait", "allaient", "allo",
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
           "zut", "à", "â", "ça", "ès", "étaient", "étais", "était", "étant", "été", "être", "ô", "à"]
words = []


def read_file(link):
    with open(link, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                words.append(word.lower())

    # words = [x for x in words if not (x.isdigit()
    #                                   or x[0] == '-' and x[1:].isdigit())]  # pour eliminer les chiffres
    return words


def read_file_without_stopwords(link):
    with open(link, "r", encoding="utf-8") as f:
        for line in f:
            for word in line.split():
                if word.lower() not in sw_list:
                    words.append(word.lower())
    return words


if __name__ == '__main__':
    words = read_file_without_stopwords("../data/x_test.txt")
    cou = dict(Counter(words))
    op = open("temp.txt", "w", encoding='utf-8')
    print(cou, file=op)
    ordered = dict(OrderedDict(sorted(cou.items(), key=lambda x: x[1], reverse=True)))
    out = dict(itertools.islice(ordered.items(), 30))
    print(out)
    plt.bar(out.keys(), out.values())
    plt.xlabel('Mots')
    plt.xticks(rotation=90)
    plt.ylabel("Nombre d'apparitions")
    plt.title("Nombre d'apparitions des mots dans le corpus")
    # plt.savefig("output/train_text_frequence_deleted.png")
    plt.show()
    print("Done !")
    op.close()
    os.remove("temp.txt")
