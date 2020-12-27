# Bert: Research Notes

## Команди за създаване на средата
Създаване на среда в Анаконда:
```shell
conda create -n lcp python=3.8
conda activate lcp
```
Инсталиране на PyTorch
```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Инсталиране на хъгинг фейс трансформер.
```shell
conda install -c huggingface transformers
```

Тестване, че инсталациите са минали гладко.
```shell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"
```

## Фактоиди
Бърт изисква .тсв дата като има и специален формат, в който приема данните: ид, стринг, константа (буква), етикет. Може да не е универсално правило.

Има фиксирана дължина на инпут фийчърите като максималното е 512. Фийчърът е числов вектор генериран от токенезирания стринг.

## Възможни проблеми

Задачата изисква непрекъсната стойност 0...1. Възможно ли е Бърт да произведе подобен резултат или трябва да е точен етикет? Може ли да върне вектор с тежести, които да могат да се прехвърлят във единична стойност?

## Resources
* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [Simple guide](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04)
* [Bert with PyTorch](https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784?gi=703ac8fb9eb6)
* [Bert with Tensorflow](https://www.tensorflow.org/tutorials/text/classify_text_with_bert)
* [Some more Bert](https://towardsml.com/2019/09/17/bert-explained-a-complete-guide-with-theory-and-tutorial/)
* [Bert and GloVe](https://www.analyticsvidhya.com/blog/2019/09/demystifying-bert-groundbreaking-nlp-framework/)
* [Bert and Tensorflow 2.0](https://analyticsindiamag.com/bert-classifier-with-tensorflow-2-0/)
* [Fine-tuning Bert](https://nlp.gluon.ai/examples/sentence_embedding/bert.html)