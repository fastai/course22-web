---
title: "11: Matrix multiplication"
---

In this lesson, we discuss various techniques and experiments shared by students on the forum, such as interpolating between prompts for visually appealing transitions and improving the update process in text-to-image generation, and a novel approach to decreasing the guidance scale during image generation. We then dive into a new paper called DiffEdit, which focuses on semantic image editing using text-conditioned diffusion models. We walk through the process of reading and understanding the paper, emphasizing the importance of grasping the main idea and not getting bogged down in every detail.

We then embark on a deep exploration of matrix multiplication using Python, compare APL with PyTorch, and introduce the concept of Frobenius norm. We also discuss the powerful concept of broadcasting, which allows for operations between tensors of different shapes, and demonstrate its efficiency in speeding up matrix multiplication. The techniques introduced in this lesson allow us to speed up our initial Python implementation by a factor of around five million, including leveraging the GPU for massive parallelism!

## Concepts discussed

- Diffusion improvements
  - Interpolating between prompts for visually appealing transitions
  - Improving the update process in text-to-image generation
  - Decreasing the guidance scale during image generation
- Understanding research papers
- Matrix multiplication using Python and Numba
- Comparing APL with PyTorch
- Frobenius norm
- Broadcasting in deep learning and machine learning code

## Video

<iframe width="514" height="289" src="https://www.youtube-nocookie.com/embed/Tf-8F5q8Xww?modestbranding=1" title="fast.ai lesson 11" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Lesson resources

- [Discuss this lesson](https://forums.fast.ai/t/lesson-11-official-topic/101508)
- [DiffEdit: Diffusion-based semantic image editing with mask guidance](https://arxiv.org/abs/2210.11427)
- Math notation
  - [Greek letters](https://en.wikipedia.org/wiki/Greek_alphabet)
  - [All in one mathematics cheat sheet](https://ourway.keybase.pub/mathematics_cheat_sheet.pdf) (PDF)
  - [Glossary of mathematical symbols](https://en.wikipedia.org/wiki/Glossary_of_mathematical_symbols#Other_brackets) (wikipedia)
  - [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR) (open source) or [Mathpix](https://mathpix.com/) (commercial)
  - [Greek Letters for Deep Learning](https://ankiweb.net/shared/info/2118139507) - Anki deck containing fastai-related Greek letters
  - [Detexify](https://detexify.kirelabs.org/classify.html) Draw math symbols

