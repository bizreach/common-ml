Common Machine Learning Library for Python
====

## Overview

common-ml provides Python library for Machine Learning.

## Issues/Questions

Please file an [issue](https://github.com/bizreach/common-ml/issues "issue").

## Getting Started

### Install common-ml

    $ pip install common-ml

## Document

### CustomDictVectorizer

CustomDictVectorizer transforms nested dictionary, such as JSON, to vectors.
Unlike DictVectorizer, given properties are transformed with specified vectorizer or own function and then they are combined.

    from commonml.text import CustomDictVectorizer

    vect = CustomDictVectorizer(vect_rules=[
                {'name': 'title',
                 'vectorizer': CountVectorizer(tokenizer=analyzer,
                                               max_df=0.8,
                                               min_df=10,
                                               dtype=np.float32)},
                {'name': 'description',
                 'vectorizer': CountVectorizer(tokenizer=analyzer,
                                               max_df=0.8,
                                               min_df=10,
                                               dtype=np.float32)}
               ])
    X = vect.fit([
                  {'title':'Test 1','description':'Aaa'},
                  {'title':'Test 2','description':'Bbb'}
                 ])

### ElasticsearchAnalyzer/ElasticsearchTextAnalyzer

ElasticsearchAnalyzer nad ElasticsearchTextAnalyzer analyze texts with Elasticsearch analysis feature.
Therefore, in Python, you can use text analyzer you want.

First of all, you need to setup elasticsearch,

    $ wget https://download.elastic.co/elasticsearch/release/org/elasticsearch/distribution/zip/elasticsearch/2.3.1/elasticsearch-2.3.1.zip
    $ unzip elasticsearch-2.3.1.zip
    $ cd elasticsearch-2.3.1
    $ echo 'cluster.name: es-ml' >> config/elasticsearch.yml
    $ echo 'network.host: "0"' >> config/elasticsearch.yml
    $ ./bin/plugin install org.codelibs/elasticsearch-analyze-api/2.3.0

install analysis plugins you need,

    $ ./bin/plugin install analysis-kuromoji
    $ ./bin/plugin install analysis-icu
    $ ./bin/plugin install org.codelibs/elasticsearch-analysis-synonym/2.3.0 -b
    $ ./bin/plugin install org.codelibs/elasticsearch-analysis-ja/2.3.0 -b
    $ ./bin/plugin install org.codelibs/elasticsearch-analysis-kuromoji-neologd/2.3.0 -b

and then start elasticsearch.

    $ ./bin/elasticsearch &

To analyze texts, create elasticsearch's index with analyzers.

    $ curl -XPUT localhost:9200/.analyzer -d '
    {
      "settings": {
        "index": {
          "analysis": {
            "tokenizer": {
              "kuromoji_neologd_tokenizer": {
                "discard_punctuation": "false",
                "type": "kuromoji_neologd_tokenizer",
                "mode": "normal"
              }
            },
            "analyzer": {
              "kuromoji_neologd_analyzer": {
                "tokenizer": "kuromoji_neologd_tokenizer",
                "type": "custom"
              }
            }
          },
          "number_of_replicas": "0",
          "number_of_shards": "10",
          "refresh_interval": "60s"
        }
      }
    }'

To check \_analyze\_api request, send the following request:

    $ curl -XPOST "localhost:9200/.analyzer/_analyze_api?pretty&analyzer=kuromoji_neologd_analyzer&part_of_speech=true" -d'
    {
      "data":{
        "text":"今日の天気は晴れです。"
      }
    }'

If the above request is succeeded, you can analyze texts with ElasticsearchTextAnalyzer in Python.

    from commonml import es
    
    analyzer_url = 'es://localhost:9200/.analyzer/kuromoji_neologd_analyzer'
    es_analyzer = es.build_analyzer(analyzer_url)
    
    for term in es_analyzer('今日の天気は晴れです。'):
        print(term)

### ElasticsearchReader

ElasticsearchReader processes elasticsearch query and returns a list of dictionaries(JSON).

    from commonml import es

    list = es.reader(hosts=['localhost:9200'],
                     index='test_index',
                     source={"query":{"match_all":{}}})
    # list is a list of dict(JSON) for document

### ChainerEstimator

ChainerEstimator provides fit/predict interface of scikit-learn, and will make your code simple.
For example, [MNIST sample](https://github.com/pfnet/chainer/blob/master/examples/mnist/train_mnist.py) is abled to be replaced as below.

    from commonml.sklearn import ChainerEstimator, SoftmaxCrossEntropyClassifier
    ...
    model = net.MnistMLP(784, n_units, 10)
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
    xp = np if gpu < 0 else cuda.cupy

    clf = ChainerEstimator(model=SoftmaxCrossEntropyClassifier(model),
                           optimizer=optimizers.Adam(),
                           batch_size=batchsize,
                           gpu=gpu,
                           n_epoch=n_epoch)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test).argmax(axis=1) # [7, 2, 1, ..., 4, 5, 6]

