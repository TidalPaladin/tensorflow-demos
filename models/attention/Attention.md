---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Attention

Attention layers, as described in [this](https://arxiv.org/abs/1706.03762) paper,


# Attention

```python
import tensorflow as tf
import tensorflow.layers as layers

class Attention(tf.keras.Model):

    def __init__(self, heads=1, *args, **kwargs):

        self.heads = list()
        for i in range(heads):
            self.matmul_qk = layers.Dot(name='matmul_qk')
            self.matmul_v = layers.Dot(name='matmul_v')

            self.scale = layers.Multiply(name='scale')
            self.softmax = layers.Softmax()


    def __call__(self, inputs, *args, **kwargs):
        # TODO Precompute weight matrix product
        q, k, v = inputs
        _ = tf.linalg.transpose(q)
        _ = self.matmul_qk(self.query, _) 
        _ = self.softmax(_)
        _ = self.matmul_v(_, self.value)

```
