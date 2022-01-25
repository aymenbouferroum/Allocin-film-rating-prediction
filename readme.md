# Allociné film rating prediction

The objective of this challenge is to understand some of the problems of automatic natural language processing (NLP) through an automatic prediction task. This consists in processing texts corresponding to film reviews from the AlloCiné website, in order to automatically deduce the scores attributed to films by the authors of these reviews.
## Contributors
- Bouferroum Aymen Salah Eddine.
- Kracheni Zakaria.
## Supervisors 
- Vincent Labatut
- Mickaël Rouvier

## Requirements
```sh
$ pip install -r requirements.txt
```
## Preparing the dataset 
in ```/prepare_data/``` directory : 
1) Execute ```xml_parsing_train_dev.py```
2) Execute ```tokenization_train```
: **x_train.txt** and **y_train.txt** will be generated
3) Execute ```xml_parsing_test.py```
4) Execute ```tokenization_test```
: **x_test.txt** will be generated
