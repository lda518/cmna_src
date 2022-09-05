import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf

class BERTClassifier(tf.keras.models.Model):
  def __init__(self):
    super(BERTClassifier, self).__init__()
    self.preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='preprocessing')
    self.encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3', trainable=True, name='BERT_encoder')

  def call(self, inputs):
    x = self.preprocessing_layer(inputs)
    x = self.encoder(x)
    x = x['pooled_output']
    return x

bert_coslf = BERTClassifier()
result = bert_coslf.call(tf.constant(["[CLS] Hello world [SEP]"]))
breakpoint()
