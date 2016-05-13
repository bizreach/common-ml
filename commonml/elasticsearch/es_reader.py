# coding: utf-8

import json
from logging import getLogger

from elasticsearch import Elasticsearch

from elasticsearch.exceptions import NotFoundError


logger = getLogger('commonml.elasticsearch.es_reader')


class ElasticsearchReader(object):

    def __init__(self,
                 index,
                 hosts=None,
                 source='{"query":{"match_all":{}}}',
                 max_docs=0,
                 scroll_size=10,
                 scroll_time='5m',
                 request_timeout=600,
                 report=1000
                 ):
        if hosts is None:
            hosts = ['http://localhost:9200']
        self.index = index
        self.source = json.loads(source) if isinstance(source, str) else source
        self.max_docs = max_docs
        self.scroll_time = scroll_time
        self.scroll_size = scroll_size
        self.request_timeout = request_timeout
        self.es = Elasticsearch(hosts=hosts)
        self.report = report

    def __iter__(self):
        scroll_id = None
        counter = 0
        running = True
        while(running):
            try:
                if scroll_id is None:
                    response = self.es.search(index=self.index,
                                              body=self.source,
                                              params={"request_timeout": self.request_timeout,
                                                      "scroll": self.scroll_time,
                                                      "size": self.scroll_size})
                    logger.info(u'{0} docs exist.'.format(response['hits']['total']))
                else:
                    response = self.es.scroll(scroll_id=scroll_id,
                                              params={"request_timeout": self.request_timeout,
                                                      "scroll": self.scroll_time})
                if len(response['hits']['hits']) == 0:
                    running = False
                    break
                scroll_id = response['_scroll_id']
                for hit in response['hits']['hits']:
                    if '_source' in hit:
                        counter += 1
                        if self.max_docs > 0 and counter >= self.max_docs:
                            logger.info(u'%d docs are loaded, but it exceeded %d docs.',
                                        counter,
                                        self.max_docs)
                            running = False
                            break
                        if counter % self.report == 0:
                            logger.info(u'%d docs are loaded.', counter)
                        yield hit['_source']
            except NotFoundError:
                logger.exception(u'NotFoundError(Loaded %d docs)', counter)
                break
            except:
                logger.exception(u"Failed to load documents from Elasticsearch(Loaded %d docs).",
                                 counter)
                break

        logger.info('Loaded %d documents.', counter)
