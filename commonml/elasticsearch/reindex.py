# coding: utf-8

from logging import getLogger
from elasticsearch.client import Elasticsearch
from elasticsearch.exceptions import NotFoundError

logger = getLogger('commonml.elasticsearch.reindex')


def reindex(from_hosts,
            from_index,
            to_hosts,
            to_index,
            to_type,
            source='{"query":{"match_all":{}}}',
            max_docs=0,
            page_size=10,
            logging_per_docs=1000,
            es_scroll='5m',
            request_timeout=60):

    if from_index is None:
        logger.warn('from_index is empty.')
        return

    from_es = Elasticsearch(hosts=from_hosts)
    to_es = Elasticsearch(hosts=to_hosts)

    scroll_id = None
    counter = 0
    running = True
    bulk_data = []
    while(running):
        try:
            if scroll_id is None:
                response = from_es.search(index=from_index,
                                          body=source,
                                          params={"request_timeout": request_timeout,
                                                  "scroll": es_scroll,
                                                  "size": page_size})
            else:
                response = from_es.scroll(scroll_id=scroll_id,
                                          params={"request_timeout": request_timeout,
                                                  "scroll": es_scroll})
            if len(response['hits']['hits']) == 0:
                running = False
                break
            scroll_id = response['_scroll_id']
            for hit in response['hits']['hits']:
                if '_source' in hit:
                    counter += 1
                    if counter % logging_per_docs == 0:
                        logger.info(u'Loaded {0} docs.'.format(counter))
                    if max_docs > 0 and counter >= max_docs:
                        logger.info(u'{0} docs are loaded, but it exceeded {1} docs.'.format(counter, max_docs))
                        running = False
                        break
                    op_index = to_index if to_index is not None else hit['_index']
                    op_type = to_type if to_type is not None else hit['_type']
                    bulk_data.append({"index": {
                                                "_index": op_index,
                                                "_type": op_type,
                                                "_id": hit['_id']
                                                }
                                      })
                    bulk_data.append(hit['_source'])
            if len(bulk_data) != 0:
                to_es.bulk(body=bulk_data, params={"request_timeout": request_timeout})
                bulk_data = []
        except NotFoundError:
            break
        except:
            logger.exception(u"Failed to load documents from Elasticsearch(Loaded {0} doc).".format(counter))
            break

    if len(bulk_data) != 0:
        to_es.bulk(body=bulk_data, params={"request_timeout": request_timeout})

    logger.info('Loaded {0} documents.'.format(counter))
