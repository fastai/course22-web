---
title: "10: Diving Deeper"
---

This lesson creates a complete Diffusers pipeline from the underlying components: the VAE, unet, scheduler, and tokeniser. By putting them together manually, this gives you the flexibility to fully customise every aspect of the inference process.

We also discuss three important new papers that have been released in the last week, which improve inference performance by over 10x, and allow any photo to be "edited" by just describing what the new picture should show.

In the second half of the lesson Jeremy begins the "from scratch" implementation of Stable Diffusion. He introduces the "miniai" library which will be created by students during the course, and discusses organising and simplifying code. The lesson discusses the Python data model, tensors, and random number generation. Jeremy introduces the Wickman-Hill random number generation algorithm and compares the performance of custom and Pytorch's built-in random number generators. The lesson concludes with creating a linear classifier using a tensor.

## Concepts discussed

- Papers:
  - Progressive Distillation for Fast Sampling of Diffusion Models
  - On Distillation of Guided Diffusion Models
  - Imagic
- Tokenizing input text
- CLIP encoder for embeddings
- Scheduler for noise determination
- Organizing and simplifying code
- Negative prompts and callbacks
- Iterators and generators in Python
- Custom class for matrices
- Dunder methods
- Python data model
- Tensors
- Pseudo-random number generation
  - Wickman-Hill algorithm
  - Random state in deep learning
- Linear classifier using a tensor


## Video

<iframe width="514" height="289" src="https://www.youtube-nocookie.com/embed/6StU6UtZEbU?modestbranding=1" title="fast.ai lesson 10" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
## Lesson resources

- [Discuss this lesson](https://forums.fast.ai/t/lesson-10-official-topic/101171)
- [Paper walkthrough video](https://www.youtube.com/watch?v=ZXuK6IRJlnk) by @johnowhitaker covering *Progressive Distillation for Fast Sampling of Diffusion Models*
- [diffusion-nbs repo](https://github.com/fastai/diffusion-nbs) (we continue walking through `stable_diffusion.ipynb` that we touched upon last time)
- [Fashion-MNIST reimplementation](https://mlops.systems/computervision/fastai/parttwo/2022/10/24/foundations-mnist-basics.html) of the lesson, with notes, by @strickvl 

## Links from the lesson

- [Course 2022p2 repo](https://github.com/fastai/course22p2) 
- [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)
- [Imagic paper](https://arxiv.org/abs/2210.09276). Within a few hours [stable diffusion versions](https://twitter.com/Buntworthy/status/1582307817884889088?s=20&t=BAiIP4MoZXt6ptq2kp9Xug) are appearing.
- APL: [Array programming - fast.ai Course Forums](https://forums.fast.ai/c/array-programming/56)

