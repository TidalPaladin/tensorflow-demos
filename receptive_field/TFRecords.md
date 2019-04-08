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

# Receptive Field Calculation


```python
import tensorflow as tf
import tensorflow.keras.applications as models
import tensorflow.contrib.receptive_field as receptive_field

# Structure to hold constants
class Flags(object):

    def __init__(self):
        self.__dict__ = {
            'input_shape' : (254, 254, 3),
            'data_format' : 'channels_last'
        }
FLAGS = Flags()
```

```python
def get_model(func):
    model = func(
            weights=None, 
            include_top=False, 
            pooling=None,
            input_shape=FLAGS.input_shape
    )
    return model

def compute_field(model):
    inputs = tf.zeros(shape=FLAGS.input_shape, dtype=tf.float32)
    outputs = model(inputs, training=False)
    return receptive_field.compute_receptive_field_from_graph_def(
            tf.get_default_graph(),
            inputs,
            outputs
    )
```

```python
resnet = get_model(models.InceptionResNetV2)
resnet.summary()
```

```python
from tensorflow.contrib.receptive_field.python.util import parse_layer_parameters as temp
temp._UNCHANGED_RF_LAYER_OPS += ["BatchNormalization"]
print(temp._UNCHANGED_RF_LAYER_OPS)
field = compute_field(resnet)
```

## Building the Model

We can construct Resnet using a subclassed approach. This involves
creating modular blocks of layers that can be reused as needed, thus
increasing code reuseability and ease of maintainance. 

Specifically, we subclass `tf.keras.Model` and implement the methods
`__init__()` and `call()`. Our choice of `__init__()` method will define
the the types of layers in this block, but says nothing about how they
are connected. In the `call()` method we will define the connections
between layers. This method takes an input as a parameter and returns
an ouput that represents the feature maps after a forward pass through
all layers in the block.

The training state needed by layers like batch-norm is passed via
`**kwargs` in `call()`. Names are used for layers where possible to
simply debugging.

## Tail

We can begin by constructing the tail.

```python
class Tail(tf.keras.Model):

    def __init__(self, Ni, *args, **kwargs):
        super(Tail, self).__init__(*args, **kwargs)

        # Big convolution layer
        self.conv = layers.Conv2D(
                Ni,
                (7, 7),
                padding='same',
                data_format=FLAGS.data_format,
                use_bias=False,
                name='tail_conv')

        # Tail BN
        self.bn = layers.BatchNormalization(
                name='tail_bn')

        # Tail BN
        self.relu = layers.ReLU(name='tail_relu')

        # Max pooling layer
        self.pool = layers.MaxPool2D(
                Ni,
                (2, 2),
                padding='same',
                data_format=FLAGS.data_format,
                name='tail_pool')

    def call(self, inputs, **kwargs):

        # Residual forward pass
        _ = self.conv(inputs, **kwargs)
        _ = self.bn(_, **kwargs)
        _ = self.relu(_, **kwargs)
        return self.pool(_, **kwargs)

```

## Basic Block 

Next we define the fundamental CNN style 2D convolution block
of Resnet, ie batch-norm, relu, convolution.

Note that the number of filters and the kernel size are 
parameterized, and that parameter packs `*args, **kwargs`
are forwarded to the convolution layer. This is important
as it enables the reuse of this model for the various
types of convolutions that we will need.

```python
class ResnetBasic(tf.keras.Model):

    def __init__(self, filters, kernel_size, strides=(1,1), *args, **kwargs):
        super(ResnetBasic, self).__init__(*args, **kwargs)
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.conv2d = layers.Conv2D(
                filters,
                kernel_size,
                padding='same',
                data_format=FLAGS.data_format,
                activation=None,
                use_bias=False,
                strides=strides)

    def call(self, inputs, **kwargs):
        x = self.batch_norm(inputs, **kwargs)
        x = self.relu(x, **kwargs)
        return self.conv2d(x, **kwargs)
```

## Standard Bottleneck

From `ResnetBasic` we can build the bottleneck.

```python
class Bottleneck(tf.keras.Model):

    def __init__(self, Ni, *args, **kwargs):
        super(Bottleneck, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        kernels = [(1, 1), (3, 3), (1, 1)]
        feature_maps = [Ni // 4, Ni // 4, Ni]
        self.residual_filters = [
            ResnetBasic(N, K) 
            for N, K in zip(feature_maps, kernels) 
        ] 

        # Merge operation
        self.merge = layers.Add()

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res, **kwargs)

        # Combine residual pass with identity
        return self.merge([inputs, res], **kwargs)
```

## Special Bottleneck

We can define the special bottleneck layer by subclassing
the `Bottleneck` class as follows.

```python
class SpecialBottleneck(Bottleneck):

    def __init__(self, Ni, *args, **kwargs):

        # Layers that also appear in standard bottleneck
        super(SpecialBottleneck, self).__init__(Ni, *args, **kwargs)

        # Add convolution layer along main path
        self.main = layers.Conv2D(
                Ni,
                (1, 1),
                padding='same',
                data_format=FLAGS.data_format,
                activation=None,
                use_bias=False)

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res, **kwargs)

        # Convolution on main forward pass
        main = self.main(inputs, **kwargs)

        # Merge residual and main
        return self.merge([main, res])
```

## Downsampling

Next we need to define the downsampling layer.

```python
class Downsample(tf.keras.Model):

    def __init__(self, Ni, *args, **kwargs):
        super(Downsample, self).__init__(*args, **kwargs)

        # Three residual convolution blocks
        kernels = [(1, 1), (3, 3), (1, 1)]
        strides = [(2, 2), (1, 1), (1, 1)]
        feature_maps = [Ni // 2, Ni // 2, 2*Ni]

        self.residual_filters = [
            ResnetBasic(N, K, strides=S) 
            for N, K, S in zip(feature_maps, kernels, strides) 
        ] 

        # Convolution on main path
        self.main = ResnetBasic(2*Ni, (1,1), strides=(2,2))

        # Merge operation for residual and main
        self.merge = layers.Add()

    def call(self, inputs, **kwargs):

        # Residual forward pass
        res = inputs
        for res_layer in self.residual_filters:
            res = res_layer(res,**kwargs)

        # Main forward pass
        main = self.main(inputs, **kwargs)

        # Merge residual and main
        return self.merge([main, res])
```

## Final Model

Finally, we can assemble these blocks into the final model. 
Note that `Keras` provides a variety of simple ways to tweak
the model, such as adding regularization. In fact, one could
probably construct the model and override layers as member variables
to apply tweaks without altering the main class. Subclassing is
another option.

```python
class Resnet(tf.keras.Model):

    def __init__(self, classes, filters, levels, *args, **kwargs):
        super(Resnet, self).__init__(*args, **kwargs)


        # Lists to hold various layers
        self.blocks = list()

        # Tail
        self.tail = Tail(filters)

        # Special bottleneck layer with convolution on main path
        self.level_0_special = SpecialBottleneck(filters)

        # Loop through levels and their parameterized repeat counts
        for level, repeats in enumerate(levels):
            for block in range(repeats):
                # Append a bottleneck block for each repeat
                name = 'bottleneck_%i_%i' % (level, block)
                layer = Bottleneck(filters, name=name)
                self.blocks.append(layer)

            # Downsample and double feature maps at end of level
            name = 'downsample_%i' % (level)
            layer = Downsample(filters, name=name)
            self.blocks.append(layer) 
            filters *= 2

        self.level2_batch_norm = layers.BatchNormalization(name='final_bn')
        self.level2_relu = layers.ReLU(name='final_relu')

        # Decoder - global average pool and fully connected
        self.global_avg = layers.GlobalAveragePooling2D(
                data_format=FLAGS.data_format,
                name='GAP' 
                )

        # Dense with regularizer, just as a test
        self.dense = layers.Dense(
                classes, 
                name='dense',
              #  kernel_regularizer=tf.keras.regularizers.l2(0.01),
                use_bias=True)


    def call(self, inputs, **kwargs):
        x = self.tail(inputs, **kwargs)
        x = self.level_0_special(x)

        # Loop over layers by level
        for layer in self.blocks:
            x = layer(x, **kwargs)

        # Finish up specials in level 2
        x = self.level2_batch_norm(x, **kwargs)
        x = self.level2_relu(x)

        # Decoder
        x = self.global_avg(x)
        return self.dense(x, **kwargs)
```

## Using the Model

Now that we have defined a subclassed model, we need to
incorproate it into a training / testing environment. This is
where the beauty of the subclassed approach comes in. 
In our case
we want construct Resnet modified for Tiny Imagenet, where the
modifications are as follows:

 * Third level of residual blocks + downsampling
 * Full and half width versions

Our Resnet class accepts an interable of integers to define the
number of repeats at each level. As such, we need only add an
integer for the number of repeats at level 3 to our constructor call.
Similarly, we can scale the number of feature maps as needed to adjust
width.


```python
# As seen in CIFAR
standard_levels = [4, 6, 3]

# Add our new level
new_level_count = 2
modified_levels = standard_levels + [new_level_count]

model = Resnet(FLAGS.num_classes, FLAGS.width, modified_levels)
outputs = model(inputs)
```

Note that `model` returned by our class constructor is callable.
Thus our forward pass mapping inputs to outputs is invoked by
"calling" `model` on the inputs and storing the returned outputs.
The operation above defines this flow of information as part of
a computational graph but does not carry out operations yet.

Finally, we can get a summary of model

```python
model.summary()
```
