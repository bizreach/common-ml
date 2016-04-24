# coding: utf-8

from commonml.elasticsearch import es_reader
reader = es_reader.ElasticsearchReader
ElasticsearchReader = es_reader.ElasticsearchReader
sentence_reader = es_reader.SentenceElasticsearchReader
SentenceElasticsearchReader = es_reader.SentenceElasticsearchReader

from commonml.elasticsearch import reindex
reindex = reindex.reindex

from commonml.elasticsearch import es_analyzer
ElasticsearchAnalyzer = es_analyzer.ElasticsearchAnalyzer
analyzer = es_analyzer.ElasticsearchAnalyzer
ElasticsearchTextAnalyzer = es_analyzer.ElasticsearchTextAnalyzer
text_analyzer = es_analyzer.ElasticsearchTextAnalyzer
build_analyzer = es_analyzer.build_analyzer
