# coding: utf-8

from commonml.text import custom_dict_vectorizer
CustomDictVectorizer = custom_dict_vectorizer.CustomDictVectorizer

from commonml.text import analyzer
EsAnalyzer = analyzer.EsAnalyzer
EsTextAnalyzer = analyzer.EsTextAnalyzer
build_analyzer = analyzer.build_analyzer

from commonml.text import elasticsearch_corpus
ElasticsearchCorpus = elasticsearch_corpus.ElasticsearchCorpus
SentenceElasticsearchCorpus = elasticsearch_corpus.SentenceElasticsearchCorpus
