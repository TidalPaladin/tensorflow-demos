# Snippets

Here I keep various code snippets or demonstrations that are not
complex enough to be included as a standalone entity. The notebooks
here demonstrate some core idea or process, typically with the intent
of illustrating a technique that will optimize a ML workflow.

### Why Should I Care?

Most of these snippets make use of the high level APIs in Tensorflow
(ie Keras). These snippets, along with an understanding of
Tensorflow's high level APIs, will help by:

1. **Saving you time** - features that would take time to implement may already
be included in Keras.
2. **Improving code readability** - High level APIs can facilitate the writing of clean
and readable code, allowing you to focus on learning the core concepts.
3. **Improving compatibility** - Since high level APIs add a level of abstraction onto the
underlying operations, your code will be more resilient to changes made to these
underlying operations.
4. And probably some other things as well.

### Contents

* [Importing Images](./ImageDataGenerator.ipynb) - Using Keras to
	rapidly import an image dataset in a few lines of code.
* [Model Subclassing](./ModelSubclassing.ipynb) - How to build complex
	models efficiently through code reuse and modularity.
* [gpufix Subclassing](./gpufix) - Bash script to resolve
	`CUDA_ERROR_UNKNOWN` error after machine wakes from sleep.
