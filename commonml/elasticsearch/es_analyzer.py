# coding: utf-8

from logging import getLogger

from elasticsearch import Elasticsearch


logger = getLogger('commonml.elasticsearch.es_analyzer')


ANALYZER_CACHE = {}


def build_analyzer(analyzer, plain=True):
    if callable(analyzer):
        return analyzer

    cached = ANALYZER_CACHE.get(analyzer)
    if cached is not None:
        return cached

    if analyzer.startswith(u'es://') or analyzer.startswith(u'ess://'):
        values = analyzer.split('://')[1].split('/')
        if len(values) != 3 and len(values) != 4:
            raise ValueError(u'Invalid parameters: {0}'.format(analyzer))
        if analyzer.startswith(u'ess://'):
            host = u'https://{0}'.format(values[0])
        else:
            host = values[0]
        cached = ElasticsearchAnalyzer(host=host, index=values[1],
                                       analyzer=values[2])
        if plain:
            cached = ElasticsearchTextAnalyzer(cached)
        ANALYZER_CACHE[analyzer] = cached
        return cached

    raise ValueError('%s is not implemented.' % analyzer)


class ElasticsearchAnalyzer:

    def __init__(self, host, index, analyzer):
        self.es = Elasticsearch(hosts=host)
        self.host = host
        self.index = index
        self.analyzer = analyzer

    def __call__(self, text_data, request_params=None):
        if not text_data or len(text_data) == 0:
            return {}

        params = {"analyzer": self.analyzer,
                  "format": "json",
                  "request_timeout": 60}
        if request_params is not None:
            params.update(request_params)
        transport = self.es.indices.client.transport
        path = '/{}/_analyze_api'.format(self.index)
        http_status, data = transport.perform_request('GET',
                                                      path,
                                                      params=params,
                                                      body=text_data)

        if http_status != 200:
            raise IOError(u'Failed to parse text: {0} => {1}'
                          .format(http_status,
                                  text_data))

        return data

    def __getstate__(self):
        result = self.__dict__.copy()
        del result['es']
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        self.es = Elasticsearch(hosts=self.host)


class ElasticsearchTextAnalyzer:

    def __init__(self, es_analyzer):
        self.es_analyzer = es_analyzer

    def __call__(self, text, request_params=None):
        text_data = {'data': {'text': text}}
        data = self.es_analyzer(text_data)
        return [x['term'] for x in data['data']]
