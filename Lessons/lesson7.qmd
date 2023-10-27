---
title: "7: Collaborative filtering"
---

::: {layout="[30,70]"}

![](../images/teddy_net.png)

You interact nearly every day with *recommendation systems*---algorithms which guess what products and services you might like, based on your past behavior. These systems largely rely on *collaborative-filtering*, an approach based on linear algebra that fills in the missing values in a matrix. Today we'll see two ways to do this: one based on a classic linear algebra formulation, and one based on deep learning.

:::

## Video

<iframe width="514" height="289" src="https://www.youtube-nocookie.com/embed/p4ZZq0736Po?modestbranding=1" title="fast.ai lesson 7" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

This lesson is based partly on [chapter 8](https://github.com/fastai/fastbook/blob/master/08_collab.ipynb) of the [book](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527).

## Resources

- Notebooks for this lesson:
  - Road to the top: [part 3](https://www.kaggle.com/code/jhoward/scaling-up-road-to-the-top-part-3) and [part 4](https://www.kaggle.com/code/jhoward/multi-target-road-to-the-top-part-4)
  - [Collaborative Filtering Deep Dive](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive/notebook)
- [Spreadsheets](https://github.com/fastai/course22/tree/master/xl) for this lesson:
  - [Softmax and cross-entropy](https://github.com/fastai/course22/blob/master/xl/entropy_example.xlsx)
  - [Collaborative filterings and embeddings](https://github.com/fastai/course22/blob/master/xl/collab_filter.xlsx)
- [Things that confused me about cross-entropy](https://chris-said.io/2020/12/26/two-things-that-confused-me-about-cross-entropy/) by Chris Said
- [Label Smoothing Explained using Microsoft Excel](https://amaarora.github.io/posts/2020-07-18-label-smoothing.html) by Aman Arora

