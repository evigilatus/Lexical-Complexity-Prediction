# Semeval 2021, Task 1

## Проектно предложение по ИИОЗ и ПОЕЕ

Семевал 2021 т1 е задача за предвиждане сложността на дума и на дума в контекст. Данните са взети от различни домейни и са класифицирани в скала на сложност 1:5. Оригиналната статия е Shardlow et al. (2020) и описва класификатор чрез линейна регресия на базата на многоизмерен ембединг, както и чрез употребата на евристични предиктори като дължина на думата, брой срички и честота на срещане в универсалния домейн, който е представляван от универсалния индекс за извършвани търсения на Гугъл. Предложението за бъдещо развитие включва и проверка на резултатите, които биха се получили с невронна мрежа като Бърт.

Предложението на нашата група е да последваме препоръката за употреба на невронни мрежи и да се опитаме да подобрим резултатите на оригиналния класификатор чрез трениране на Бърт върху данните от задачата. Тъй като предоставения датасет може да се окаже недостатъчен за постигане на задоволителни резултати, част от усилията ни ще са насочени към разширяване на данните от външни източници. Допълнително, ще се опитаме да подобрим оригиналния класификатор чрез добавяне на нови евристични стратегии. Това ще ни позволи да анотираме събраните данни по бърз и ефикасен начин.

Има няколко варианта за събиране на допълнителни данни. Първо, оригиналната статия прави ограничена подборка от няколко корпуса (Библията, Еуропарл 2005, CRAFT). Няма причина поради която да не можем да употребим целите оригинални корпуси. Второ, Еуропарл 2005 не е последната версия на корпуса и можем да използваме допълнителните данни в по-късните издания, последното от които е Еуропарл 2012. Трето, можем да потърсим сходни датасетове или да скрейпнем от Интернет материал за такива на тематики сходни с оригиналните корпуси.

Смятаме, че има и относително леснопостижими начини да подобрим точността на класификатора от оригиналната статия. В нея не се отчита, че една дума има различни глаголни и граматически форми, а вместо това всяка форма бива разглеждана поотделно. Този вид класификация ще даде относително висока оценка за дума като "amazingly", защото е дълга, има четири срички и не е от най-често срещаните в английския език. В същото време, кореновата дума "amaze" би получила по-ниска оценка, а цялостната оценка би спаднала още, ако към атрибутите на думата добавим сбора от честотите на ползване на всички форми на думата: "amaze", "amazed", "amazes", "amazing", "amazement", "amazingly". В това има смисъл, защото човек е по-вероятно да сметне за лесна дума, която е срещал под различни форми и с чието кореново значение е добре запознат.

Целта на невронната мрежа в нашето решение е да обработи всички събрани данни и да предостави финалната оценка за сложността на съответните думи и изрази. Доколкото невронните мрежи са по-малко чувствителни към грешки в тренировачните данни, то се надяваме, че Бърт ще ни спомогне да намалим шума, който би останал в анотирания от класификатора ни датасет. Плюс, изглежда яко.

В заключение, нашата група счита, че Семевал 2021 таск 1 е проект със значителна трудност, който все пак не е непреодолим и имаме ясна представа как можем да изградим възможно решение. Подходът, който предлагаме се състои от разнообразни подзадачи, като всеки член на екипа ще може да се концентрира върху поне една от задачите по агрегиране на данни, класификация и анотация на примери, както и трениране на финалния модел.

## Допълнителни ресурси

### Дати в графика на Semeval

 - Training data ready: October 1, 2020
 - Test data ready: December 3, 2020
 - Evaluation start: January 10, 2021
 - Evaluation end: January 31, 2021
 - Paper submission due: February 23, 2021

### Линкове към ресурси на Task 1

 - Task 1
   - [website](https://sites.google.com/view/lcpsharedtask2021)
   - [Codalab competition page](https://competitions.codalab.org/competitions/27420)
   - [trial and training data, with password "YellowDolphin73!"](https://github.com/MMU-TDMLab/CompLex)
 - [Paper](https://arxiv.org/pdf/2003.07008.pdf)
 - Lexical corpora
   - [bible-corpus](https://github.com/christos-c/bible-corpus)
   - [CRAFT corpus](https://github.com/UCDenver-ccp/CRAFT)
   - [Europarl](https://www.statmt.org/europarl/)
   - [word frequency (used in task repo)](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus)
 - Linear regression lexical classifier
   - [GloVe](https://nlp.stanford.edu/projects/glove/)
   - [InferSent](https://github.com/facebookresearch/InferSent)
   - [Syllables package](https://pypi.org/project/syllables/)
 - BERT:
   - [Bert repo](https://github.com/google-research/bert)
   - [notes](/bert/notes)
 - Complex Word Identification (CWI) 2018:
   - [website](https://sites.google.com/view/cwisharedtask2018/)
   - [Paper - winners](https://www.aclweb.org/anthology/W18-0520.pdf)
   - [Datasets](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/complex-word-identification-dataset.html)
   - [Findings report](https://www.aclweb.org/anthology/W18-0507.pdf)
   - [Results](https://www.researchgate.net/publication/325591648_Complex_Word_Identification_Shared_Task_2018)
   - [Code](https://github.com/siangooding/cwi_2018)
 
### Разширен dataset и други файлове относно dataset-a
 - [Dataset](https://drive.google.com/drive/folders/1jyVOeiTkzEAxqKRhs61_rfoxC3W4fNHa?usp=sharing)
