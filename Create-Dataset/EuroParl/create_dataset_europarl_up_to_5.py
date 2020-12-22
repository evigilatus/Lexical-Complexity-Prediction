
import nltk, glob, re, pandas as pd
from nltk.corpus import stopwords

path = 'en_text/*.txt'
files = glob.glob(path)
stop_words = set(stopwords.words('english'))
word_cnt = dict()
sentence_word = set()
sentence_word_up_to_5 = set()

for file in files:

    in_file = open(file, errors="ignore")
    text = in_file.read()
    in_file.close()

    sentences = nltk.sent_tokenize(text)

    for sent in sentences:
        words = nltk.word_tokenize(sent)
        tagged_words = nltk.pos_tag(words, tagset="universal")

        for word in tagged_words:

            if not re.match('^[a-zA-Z]+[-]*[a-zA-Z]+$', word[0]):
                continue

            if word[1] == "NOUN" and word[0].lower() not in stop_words:

                sentence_word.add((sent, word[0]))

for item in sentence_word:

    if item[1].lower() in word_cnt and word_cnt.get(item[1].lower()) >= 5:
        continue

    if item[1].lower() not in word_cnt:
        word_cnt[item[1].lower()] = 1
    elif word_cnt.get(item[1].lower()) < 5:
        word_cnt[item[1].lower()] = word_cnt.get(item[1].lower()) + 1

    sentence_word_up_to_5.add(item)

sentence_word_up_to_5 = sorted(sentence_word_up_to_5, key=lambda x: x[1].lower())

df = pd.DataFrame(sentence_word_up_to_5, columns = ["sentence", "word"])
df["frequency"] = df.apply(lambda row: word_cnt.get(row["word"].lower()), axis=1)

df.to_csv("europarl_single_word_up_to_5.tsv", sep = '\t', index = False)
