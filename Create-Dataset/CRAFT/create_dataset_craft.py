
import nltk, glob, re, pandas as pd
from nltk.corpus import stopwords

path = 'craft_raw/*.txt'
files = glob.glob(path)
stop_words = set(stopwords.words('english'))
word_cnt = dict()
sentence_word = set()

for file in files:

    in_file = open(file, errors="ignore")
    text = in_file.read()
    in_file.close()

    sentences = nltk.sent_tokenize(text)

    for sent in sentences:
        words = nltk.word_tokenize(sent)
        tagged_words = nltk.pos_tag(words, tagset="universal")

        for word in tagged_words:

            if not re.match('^[a-zA-Z]+$', word[0]):
                continue

            if word[1] == "NOUN" and word[0].lower() not in stop_words:

                if word[0].lower() not in word_cnt:
                    word_cnt[word[0].lower()] = 1
                elif (sent, word[0]) not in sentence_word:
                    word_cnt[word[0].lower()] = word_cnt.get(word[0].lower()) + 1

                sentence_word.add((sent, word[0]))

sentence_word = sorted(sentence_word, key=lambda x: x[1].lower())

df = pd.DataFrame(sentence_word, columns = ["sentence", "word"])
df["frequency"] = df.apply(lambda row: word_cnt.get(row["word"].lower()), axis=1)

df.to_csv("craft_single_word.tsv", sep = '\t', index = False)
