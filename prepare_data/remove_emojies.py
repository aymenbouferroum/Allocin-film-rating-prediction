import emoji
import regex as re


def read_file(link):
    arr1 = []
    with open(link, "r", encoding="utf-8") as f:
        for line in f:
            arr1.append(line.rstrip('\n'))
    return arr1


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def split_count(text):
    emoji_list = []
    data = re.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI['fr'] for char in word):
            emoji_list.append(word)

    return emoji_list


arr2 = []
file = open('../data/x_train.txt', 'w', encoding='utf-8')
if __name__ == '__main__':
    data = read_file("./data/x_train.txt")
    for i in range(len(data)):
        arr2.append(remove_emoji(data[i]))
        print(arr2[i], file=file)
    for i in range(len(data)):
        if split_count(arr2[i]):
            print(split_count(arr2[i]))
