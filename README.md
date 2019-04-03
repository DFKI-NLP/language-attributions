# language-attributions
by David Harbecke, Robert Schwarzenberg and Christoph Alt.

Code accompanying the paper

```
@inproceedings{harbecke_la_2018,
  title = {Learning Explanations from Language Data},
  booktitle = {Proceedings of the EMNLP 2018 Workshop on Analyzing and Interpreting Neural Networks for NLP},
  author = {Harbecke, David and Schwarzenberg, Robert and Alt, Christoph},
  location = {Brussels, Belgium},
  year = {2018}
  }

```

Applying the PatternAttribution approach by

[PJ Kindermans, KT Schütt, M Alber, KR Müller, D Erhan, B Kim, S Dähne. Learning how to explain neural networks: PatternNet and PatternAttribution International Conference on Learning Representations (ICLR), 2018](https://arxiv.org/abs/1705.05598)

to language. 

Our implementation uses their [toolbox](https://github.com/albermax/innvestigate). 


**Here are contributions to a negative sentiment classification that our method retrieved from a CNN sentiment classifier:**

![Negative Sentiment Contributions](https://github.com/rbtsbg/language-attributions-1/blob/master/images/negative_sentiment.png)

The review is taken from the Amazon Review Polarity dataset on which we also trained the sentiment model. 

# How to run experiments:

- install requirements.txt
- install english spacy model (python -m spacy download en)
- set values in config.INI (might not be necessary)
- run run_experiments.py
