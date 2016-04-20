# coding: utf-8
import json
from logging import getLogger

from elasticsearch import Elasticsearch

from commonml.utils import get_nested_value
from elasticsearch.exceptions import NotFoundError


logger = getLogger('commonml.text.elasticsearch_corpus')


def flatmap(f, items):
    return chain.from_iterable(imap(f, items))

class ElasticsearchCorpus(object):
    
    def __init__(self,
                 index,
                 hosts=['http://localhost:9200'],
                 source='{"query":{"match_all":{}},"sort":[{"_id":{"order":"asc"}}]}',
                 max_docs=0,
                 scroll_size=10,
                 scroll_time='5m',
                 request_timeout=600,
                 report=1000
                 ):
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
                                              scroll=self.scroll_time,
                                              size=self.scroll_size,
                                              body=self.source,
                                              params={"request_timeout":self.request_timeout})
                    logger.info(u'{0} docs exist.'.format(response['hits']['total']))
                else:
                    response = self.es.scroll(scroll_id=scroll_id,
                                              scroll=self.scroll_time,
                                              params={"request_timeout":self.request_timeout})
                if len(response['hits']['hits']) == 0:
                    running = False
                    break
                scroll_id = response['_scroll_id']
                for hit in response['hits']['hits']:
                    if '_source' in hit:
                        counter += 1
                        if self.max_docs > 0 and counter >= self.max_docs:
                            logger.info(u'{0} docs are loaded, but it exceeded {1} docs.'.format(counter, self.max_docs))
                            running = False
                            break
                        if counter % self.report == 0:
                            logger.info(u'{0} docs are loaded.'.format(counter))
                        yield hit['_source']
            except NotFoundError:
                logger.exception(u'NotFoundError: {0}'.format(hit))
                break
            except:
                logger.exception(u"Failed to load documents from Elasticsearch(Loaded {0} doc).".format(counter))
                break

        logger.info('Loaded {0} documents.'.format(counter))

class SentenceElasticsearchCorpus(object):

    def __init__(self, corpus, fields):
        self.corpus = corpus
        self.fields = fields

    def __iter__(self):
        for source in self.corpus:
            sentence_dict = {}
            for field in self.fields:
                text = get_nested_value(source, field)
                if text is not None:
                    count = 0
                    for sentence in self._get_sentences(text):
                        yield sentence

    def _get_sentences(self, text):
        # TODO remove Japanese characters
        return flatmap(lambda x1:x1.split('。'), flatmap(lambda x2:x2.split('. '), [text]))
