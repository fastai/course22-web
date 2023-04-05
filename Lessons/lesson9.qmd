---
title: "9: Stable Diffusion"
image: /images/course2022p2l1.jpg
---

## What you need to know

Here's what you need to know to complete this course:

- The lesson is presented as a video, which you can jump directly to by clicking the table of contents on the right
- Each video goes through one or more Jupyter notebooks, which you'll need to run and experiment with to get the most out of the course
- All information needed to complete a lesson (including links to the repo with the notebooks) is in the "lesson resources" section of the lesson page
- Amongst the lesson resources you'll find a "discuss this lesson" link, which will take you to a Q&A page on our forums for that particular lesson
- The material covered in this course includes stuff that would normally only be included in post-graduate level programs. We try to present it in the clearest way possible, but you should expect to work hard and put in plenty of hours of study
- We assume familiarity with the material in part 1 of this course. If you find yourself unsure about some of the foundational deep learning ideas refered to in the lessons, we'd suggest going back to study the lessons in part 1 that cover those ideas
- If there's mathematical or coding concepts that we use that you're not comfortable with, don't be afraid to seek out other tutorials to help fill in your gaps
- On [forums.fast.ai](https://forums.fast.ai) there are many other students you can collaborate with, and many folks are looking for study groups or study buddies. Studying in groups has been shown to be more effective for most people than studying alone
- In many lessons we'll include a challenge for you to complete, some of which involve trying novel research directions where you'll be venturing into the academic unknown.

## Lesson overview

This lesson starts with a tutorial on how to use pipelines in the [Diffusers library](https://huggingface.co/docs/diffusers/index) to generate images. Diffusers is (in our opinion!) the best library available at the moment for image generation. It has many features and is very flexible. We explain how to use its many features, and discuss options for accessing the GPU resources needed to use the library.

We talk about some of the nifty tweaks available when using Stable Diffusion in Diffusers, and show how to use them: guidance scale (for varying the amount the prompt is used), negative prompts (for removing concepts from an image), image initialisation (for starting with an existing image), textual inversion (for adding your own concepts to generated images), Dreambooth (an alternative approach to textual inversion).

The second half of the lesson covers the key concepts involved in Stable Diffusion:

- CLIP embeddings
- The VAE (variational autoencoder)
- Predicting noise with the unet
- Removing noise with schedulers

Jeremy shows a theoretical foundation for how Stable Diffusion works, using a novel interpretation that shows an easily-understood intuition for the theory. He introduces the concept of finite differencing and analytic derivatives, using an example of training a neural network to identify pixel adjustments to make an image look more like a handwritten digit, and describes how the derivatives of such a model can provide the *score* needed to provide the basis of a diffusion process that generates handwritten digits.

The lesson also covers finite differencing, analytic derivatives, autoencoders, and U-Nets. Jeremy introduces the concept of creating a model that can take a sentence and return a vector of numbers representing the image, using two models: a text encoder and an image encoder. The lesson concludes with a discussion of the similarities between diffusion-based models and deep learning optimizers, suggesting new research directions.

## Concepts discussed

- Stable Diffusion
- Hugging Face's Diffusers library
- Pre-trained pipelines
- Guidance scale
- Negative prompts
- Image-to-image pipelines
- Finite differencing
- Analytic derivatives
- Autoencoders
- Textual inversion
- Dreambooth
- Latents
- U-Nets
- Text encoders and image encoders
- Contrastive loss function
- CLIP text encoder
- Deep learning optimizers
- Perceptual loss

## Video

<iframe width="514" height="289" src="https://www.youtube-nocookie.com/embed/_7rMfsA24Ls?modestbranding=1" title="fast.ai lesson 9" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Lesson resources

- Other Videos
  - [Lesson 9A video](https://www.youtube.com/watch?v=844LY0vYQhc)---Deep Dive---from @johnowhitaker (with [accompanying notebook](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb))
  - [Lesson 9B video](https://youtu.be/mYpjmM7O-30)---The Math of Diffusion---from @seem and @ilovescience 
- Jeremy's [lesson notes](https://forums.fast.ai/uploads/short-url/dI0cWOaQQEHFYPKVyM9BkSWY4rQ.pdf)
- The fastai book:
  - [Published version](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch-ebook-dp-B08C2KM7NR/dp/B08C2KM7NR)
  - [Free notebook version](https://github.com/fastai/fastbook)
  - [Sample full chapters](https://fastai.github.io/fastbook2e/)
- Student notes
  -- [Lesson Notes](https://rekil156.github.io/rekilblog/posts/lesson9_stableDissufion/Lesson9.html) h/t @barnacl 

## Links from the lesson

- [Discuss this lesson](https://forums.fast.ai/t/lesson-9-official-topic/100562)
- [Course repo](https://github.com/fastai/course22p2)
- [diffusion-nbs repo](https://github.com/fastai/diffusion-nbs)
- [HuggingFace Notebooks](https://github.com/huggingface/notebooks)
- GPU servers
  - [Lambda Labs](https://forums.fast.ai/t/lambda-gpu-cloud-for-deep-learning-a100s-at-1-10-gpu-hr-150-sign-up-credit/100942/2)
  - [Paperspace Gradient](https://www.paperspace.com/gradient)
  - [Jarvis Labs](https://jarvislabs.ai/)
  - [vast.ai - crowdsourced GPU service](https://vast.ai/)
- Prompt Engineering
  - [Lexica](https://lexica.art/)
  - [PromptHero](https://prompthero.com/)
  - [Hexo - 10M Images and Prompts](https://hexo.ai/)
- [Tools and Resources for AI Art](https://pharmapsychotic.com/tools.html)
- [fastai repo](https://github.com/fastai/fastai)

## Useful background on fast.ai courses

- [Homework](https://forums.fast.ai/t/did-you-do-the-homework/66034)
- [Summaries + Things Jeremy Says to do + Qs](/t/podcast-writeup-summaries-things-jeremy-says-to-do-qs/66194)
- Fastai: A Layered API for Deep Learning paper: [Information Journal](https://www.mdpi.com/2078-2489/11/2/108) or [arxiv](https://arxiv.org/abs/2002.04688) or [fast.ai](https://www.fast.ai/2020/02/13/fastai-A-Layered-API-for-Deep-Learning/)
- [Providing a Good Education in Deep Learning](https://www.fast.ai/2016/10/08/teaching-philosophy/): fast.ai teaching philosophy
- ["How not to do fastai” ](https://medium.com/@init_27/how-not-to-do-fast-ai-or-any-ml-mooc-3d34a7e0ab8c)
- ["FastAI Lesson Zero: video + notes"](https://www.alexstrick.com/blog/fastai-lesson-zero)

