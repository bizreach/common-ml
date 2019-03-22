# coding: utf-8

from commonml.elasticsearch import es_reader
from commonml.elasticsearch import reindex
from commonml.elasticsearch import es_analyzer

reader = es_reader.ElasticsearchReader
ElasticsearchReader = es_reader.ElasticsearchReader

reindex = reindex.reindex

ElasticsearchAnalyzer = es_analyzer.ElasticsearchAnalyzer
analyzer = es_analyzer.ElasticsearchAnalyzer
ElasticsearchTextAnalyzer = es_analyzer.ElasticsearchTextAnalyzer
text_analyzer = es_analyzer.ElasticsearchTextAnalyzer
build_analyzer = es_analyzer.build_analyzer
