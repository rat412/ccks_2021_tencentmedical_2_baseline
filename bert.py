#! -*- coding: utf-8 -*-
# CCKS 2021：面向中文医疗科普知识的内容理解（二）医疗科普知识答非所问识别    baseline

import os
import numpy as np
import pandas as pd
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm


# 基本信息
maxlen = 512
batch_size = 8

# bert配置
path_drive = 'chinese_L-12_H-768_A-12'

config_path = os.path.join(path_drive, 'bert_config.json')
checkpoint_path = os.path.join(path_drive, 'bert_model.ckpt')
dict_path = os.path.join(path_drive, 'vocab.txt')

# 读取数据
train_data, valid_data, test_data = [], [], []

train_df = pd.read_csv('raw_data/wa.train.fixtab.valid.tsv',sep='\t')
train_df = [train_df.iloc[j][['Label','Question','Description','Answer']] for j in range(len(train_df))]
for line in train_df:
    content1 = line['Question']+line['Description']
    content2 = line['Answer']
    train_data.append((content1, content2, line['Label']))

test_df = pd.read_csv('raw_data/wa.test.phase1.fixtab.valid.tsv',sep='\t')
test_df = [test_df.iloc[j][['Docid','Question','Description','Answer']] for j in range(len(test_df))]
for line in test_df:
    content1 = line['Question'] + line['Description']
    content2 = line['Answer']
    train_data.append((content1, content2, line['Docid']))

valid_data = train_data[:5000]
train_data = train_data[5000:]

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_labels = []
        for is_end, (content1, content2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content1, content2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [
                          batch_token_ids, batch_segment_ids
                      ], batch_labels
                batch_token_ids, batch_segment_ids = [], []
                batch_labels = []

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='bert',
    with_pool=True,
)

output = Dense(1, activation='sigmoid')(model.output)

model = Model(model.inputs, output)
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['accuracy']
)

#F1-score评价指标
def val_F1score(y_true,y_pred):
    TP=(y_true*y_pred).sum()
    TN=((1-y_true)*(1-y_pred)).sum()
    FP=((1-y_true)*y_pred).sum()
    FN=(y_true*(1-y_pred)).sum()
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1_score=2*precision*recall/(precision+recall)
    return precision,recall,F1_score

def evaluate(data):
    """评测函数
    """
    y_true_total = []
    y_pred_total = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, 0]
        y_true = y_true[:, 0]
        for i in y_true:
            y_true_total.append(i)
        for j in y_pred:
            if j > 0.5:
                y_pred_total.append(1)
            else:
                y_pred_total.append(0)
    precision, recall, f1_score = val_F1score(np.array(y_true_total), np.array(y_pred_total))
    print(u'precision: %.5f, recall: %.5f,F1_score: %.5f\n' %
          (precision, recall, f1_score))
    return f1_score

def predict_test(filename):
    """
    测试集预测到文件
    """
    with open(filename, 'w') as f:
        f.write('Docid,label\n')
        for x_true, y_true in tqdm(test_generator):
            y_pred_argmax = []
            y_pred = model.predict(x_true)
            for i in y_pred:
                if i > 0.5:
                    y_pred_argmax.append(1)
                else:
                    y_pred_argmax.append(0)
            y_true = y_true[:, 0]
            for id, y in zip(y_true, y_pred_argmax):
                f.write('%s,%s\n' % (id, y))
                
class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self, best_val_f1=0.):
        self.best_val_f1 = best_val_f1

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_generator)
        if val_f1 >= self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights('best_model.weights')

if __name__ == '__main__':
    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
    model.load_weights('best_model.weights')
    predict_test('valid_prediction.csv')
    
    test = pd.read_csv('raw_data/wa.test.phase1.fixtab.valid.tsv', sep='\t')
    prediction = pd.read_csv('valid_prediction.csv', sep=',')
    test_final = pd.merge(test, prediction, on='Docid')

    test_final[['Label','Docid','Question','Description','Answer']].to_csv('队伍名_valid_result.txt',sep='\t', index=False)
