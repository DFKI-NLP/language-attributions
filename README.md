# language-attributions
Applying the PatternAttribution approach by

[PJ Kindermans, KT Schütt, M Alber, KR Müller, D Erhan, B Kim, S Dähne. Learning how to explain neural networks: PatternNet and PatternAttribution International Conference on Learning Representations (ICLR), 2018](https://arxiv.org/abs/1705.05598)

to language. 

Our implementation uses their [toolbox](https://github.com/albermax/innvestigate).

![Negative Sentiment Contributions](https://github.com/rbtsbg/language-attributions-1/blob/master/images/negative_sentiment.png)

# How to run experiments:

- install requirements.txt
- install english spacy model (python -m spacy download en)
- set values in config.INI (might not be necessary)
- run run_experiments.py
