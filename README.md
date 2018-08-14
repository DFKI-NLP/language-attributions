# language-attributions
by David Harbecke, Robert Schwarzenberg and Christoph Alt

Applying the PatternAttribution approach by

[PJ Kindermans, KT Schütt, M Alber, KR Müller, D Erhan, B Kim, S Dähne. Learning how to explain neural networks: PatternNet and PatternAttribution International Conference on Learning Representations (ICLR), 2018](https://arxiv.org/abs/1705.05598)

to language. 

Our implementation uses their [toolbox](https://github.com/albermax/innvestigate). 

We present the results of this project at the [2018 EMNLP Workshop on Analysing and Interpreting Neural Networks for NLP](https://blackboxnlp.github.io/). A preprint is available:

https://arxiv.org/abs/1808.04127


**Here are contributions to a negative sentiment classification that our method retrieved from a CNN sentiment classifier:**

![Negative Sentiment Contributions](https://github.com/rbtsbg/language-attributions-1/blob/master/images/negative_sentiment.png)

The review is taken from the Amazon Review Polarity dataset on which we also trained the sentiment model. 

# How to run experiments:

- install requirements.txt
- install english spacy model (python -m spacy download en)
- set values in config.INI (might not be necessary)
- run run_experiments.py
