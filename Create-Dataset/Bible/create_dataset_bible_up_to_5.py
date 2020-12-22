
import nltk, csv, pandas as pd
from nltk.corpus import stopwords

in_file = open("bible.txt")
text = in_file.read()
in_file.close()

stop_words = set(stopwords.words('english'))
sentence_word = set()
word_cnt = dict()

sentences = nltk.sent_tokenize(text)

for sent in sentences:
    words = nltk.word_tokenize(sent)
    tagged_words = nltk.pos_tag(words, tagset="universal")

    for word in tagged_words:
        if word[1] == "NOUN" and word[0].lower() not in stop_words:

            if word[0].lower() in word_cnt and word_cnt.get(word[0].lower()) >= 5:
                continue

            if word[0].lower() not in word_cnt:
                word_cnt[word[0].lower()] = 1
            elif word_cnt.get(word[0].lower()) < 5 and (sent.replace('\n', ' '), word[0]) not in sentence_word:
                word_cnt[word[0].lower()] = word_cnt.get(word[0].lower()) + 1

            sentence_word.add((sent.replace('\n', ' '), word[0]))


sentence_word = sorted(sentence_word, key=lambda x: x[1].lower())

df = pd.DataFrame(sentence_word, columns = ["sentence", "word"])
df["frequency"] = df.apply(lambda row: word_cnt.get(row["word"].lower()), axis=1)

df.to_csv("bible_single_word_up_to_5.tsv", sep = '\t', index = False)
