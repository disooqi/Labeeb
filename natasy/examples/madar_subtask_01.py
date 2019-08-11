import codecs
import datetime
import numpy as np
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from natasy.data_preparation import Dataset
from natasy.neural_network.network import NeuralNetwork
from natasy.neural_network import Activation
from natasy.optimization import Optimizer


class CORPUS_26_Dataset(Dataset):
    def __init__(self, *args, **kwargs):
        X_train, y_train = kwargs.get('train')
        X_dev, y_dev = kwargs.get('dev')
        X_test = kwargs.get('test')

        self.labels = ['ALE', 'ALG', 'ALX', 'AMM', 'ASW', 'BAG', 'BAS', 'BEI', 'BEN',
              'CAI', 'DAM', 'DOH', 'FES', 'JED', 'JER', 'KHA', 'MOS', 'MSA',
              'MUS', 'RAB', 'RIY', 'SAL', 'SAN', 'SFX', 'TRI', 'TUN']

        _y_train = [labels.index(tgt.replace(' ','')) for tgt in y_train]
        _y_dev = [labels.index(tgt.replace(' ','')) for tgt in y_dev]

        self.X_train = X_train.toarray().T
        self.y_train = np.reshape(_y_train, (len(_y_train), -1)).T
        self.X_dev = X_dev.toarray().T
        self.y_dev = np.reshape(_y_dev, (len(_y_dev), -1)).T
        self.X_test = X_test.toarray().T

        self.y_train, self.classes = self.prepare_target(self.y_train)
        self.y_dev, _ = self.prepare_target(self.y_dev)

        self.n, self.m = self.X_train.shape

        self.name = 'corpus-26'
        self.dev_size = None
        self.shuffle = None

    @staticmethod
    def prepare_target(y):
        # cs = unique_labels(y)
        classes = np.unique(y)
        incidence_y = np.zeros((classes.size, y.size))

        incidence_y[y.ravel(), np.arange(y.size)] = 1  # (5000, 10)
        return incidence_y, classes

    def accuracy(self, network, training_accuracy=False):
        # You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
        if training_accuracy:
            X, y = self.X_train, self.y_train
        else:
            X, y = self.X_dev, self.y_dev

        prediction = network.feedforward(X).argmax(axis=0)
        y = y.argmax(axis=0)
        res = np.equal(prediction, y)


        with open(f'../data/corpus-26/result/run_dev_np_{network.time_stamp}.G', mode='w') as pfo:
            for rs in y:
                pfo.write(f'{self.labels[rs]}\n')

        with open(f'../data/corpus-26/result/run_dev_np_{network.time_stamp}.P', mode='w') as pfo:
            for rs in prediction:
                pfo.write(f'{self.labels[rs]}\n')

        return 100 * np.sum(res) / y.size

    def test(self, network):
        # You only use dropout during training. Don't use dropout (randomly eliminate nodes) during test time.
        pred_test = network.feedforward(self.X_test).argmax(axis=0)
        with open(f'../data/corpus-26/result/run_test_np_{network.time_stamp}.P', mode='w') as pfo:
            for rs in pred_test:
                pfo.write(f'{self.labels[rs]}\n')

        return pred_test



def prepare_train_data_for_task_1(include_dev=False):
    X, y = [], []
    with codecs.open('../data/corpus-26/data/train.tsv') as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
    if include_dev:
        with codecs.open('../data/corpus-26/data/dev.tsv') as training:
            for i, line in enumerate(training):
                sentence_label = line.strip().split('\t')
                X.append(sentence_label[0])
                y.append(sentence_label[1])

    return train_test_split(X, y, test_size=0.0, shuffle=True)


def prepare_dev_data_for_task_1():
    X, y = [], []
    with codecs.open('../data/corpus-26/data/dev.tsv') as training:
        for i, line in enumerate(training):
            sentence_label = line.strip().split('\t')
            X.append(sentence_label[0])
            y.append(sentence_label[1])
    return X, y


def prepare_test_data_for_task_1():
    X = []
    with codecs.open('../data/corpus-26/data/test.tsv') as training:
        for i, line in enumerate(training):
            sentence = line.strip()
            X.append(sentence)
    return X


if __name__ == '__main__':
    _train_src, _, _train_tgt, _ = prepare_train_data_for_task_1()
    _dev_src, _dev_tgt = prepare_dev_data_for_task_1()
    _test_src = prepare_test_data_for_task_1()

    train_src = _train_src[:]
    dev_src = _dev_src[:]
    test_src = _test_src[:]

    labels = ['ALE', 'ALG', 'ALX', 'AMM', 'ASW', 'BAG', 'BAS', 'BEI', 'BEN',
              'CAI', 'DAM', 'DOH', 'FES', 'JED', 'JER', 'KHA', 'MOS', 'MSA',
              'MUS', 'RAB', 'RIY', 'SAL', 'SAN', 'SFX', 'TRI', 'TUN']

    # train_tgt = [labels.index(tgt) for tgt in _train_tgt]

    # count_vec = CountVectorizer(analyzer='word', min_df=1, max_df=0.95, ngram_range=(1, 1))
    # X_train_counts = count_vec.fit_transform(train_src)
    # X_dev_counts = count_vec.transform(dev_src)
    # X_test_counts = count_vec.transform(test_src)
    #
    # tf_vec = TfidfTransformer(use_idf=False)
    #
    # X_train_tf = tf_vec.fit_transform(X_train_counts)
    # X_dev_tf = tf_vec.transform(X_dev_counts)
    # X_test_tf = tf_vec.transform(X_test_counts)
    #
    # tfidf_vect = TfidfTransformer(use_idf=True, smooth_idf=False, sublinear_tf=True)
    # X_train_tfidf = tfidf_vect.fit_transform(X_train_counts)
    # X_dev_tfidf = tfidf_vect.transform(X_dev_counts)
    # X_test_tfidf = tfidf_vect.transform(X_test_counts)

    count_vec = TfidfVectorizer(analyzer='char', lowercase=False, max_df=0.95, ngram_range=(4, 4), smooth_idf=False,
                                 sublinear_tf=True)
    X_train_tfidf = count_vec.fit_transform(train_src)
    X_dev_tfidf = count_vec.transform(dev_src)
    X_test_tfidf = count_vec.transform(test_src)

    corpus26 = CORPUS_26_Dataset(train=(X_train_tfidf, _train_tgt), dev=(X_dev_tfidf, _dev_tgt), test=X_test_tfidf)

    nn01 = NeuralNetwork(n_features=len(count_vec.get_feature_names()), n_classes=len(labels))
    nn01.add_layer(512, activation=Activation.relu, dropout_keep_prob=0.8)

    nn01.add_layer(len(labels), activation=Activation.softmax_stable, output_layer=True)

    gd_optimizer = Optimizer(loss='multinomial_cross_entropy',
                             method='rmsprop')  # gd-with-momentum gradient-descent rmsprop adam
    # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    gd_optimizer.minimize(nn01, epochs=20, mini_batch_size=128, learning_rate=.0005, regularization_parameter=0,
                          dataset=corpus26)

    currentDT = datetime.datetime.now()
    nn01.time_stamp = currentDT.strftime("%Y%m%d%H%M%S")

    train_acc = corpus26.accuracy(nn01, training_accuracy=True)
    dev_acc = corpus26.accuracy(nn01)
    corpus26.test(nn01)
    print('train acc: {:.2f}%, Dev acc: {:.2f}%'.format(train_acc, dev_acc))



