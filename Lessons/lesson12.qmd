---
title: "12: Mean shift clustering"
---

In this lesson, we start by discussing the CLIP Interrogator, a Hugging Face Spaces Gradio app that generates text prompts for creating CLIP embeddings. We then dive back into matrix multiplication, using Einstein summation notation and torch.einsum to simplify code and improve performance. We explore GPU acceleration with CUDA and Numba, demonstrating how to write a kernel function for matrix multiplication and launch it on the GPU.

Next up we exercise our tensor programming skills by implementing mean shift clustering, a technique for identifying clusters within a dataset. We create synthetic data, explain the mean shift algorithm, and introduce the Gaussian kernel for penalizing distant points. We implement the mean shift clustering algorithm using PyTorch and discuss the importance of tensor manipulation operations for efficient GPU programming.

Finally, we optimize the mean shift algorithm using PyTorch and GPUs, demonstrating how to calculate weights, multiply matrices, and sum up points to obtain new data points. We explore the impact of changing batch sizes on performance and encourage viewers to research other clustering algorithms.

The lesson concludes with an introduction to calculus, focusing on derivatives and the calculus of infinitesimals.

## Concepts discussed

- CLIP Interrogator
- Inverse problems
- Matrix multiplication
- Einstein summation notation and torch.einsum
- GPU acceleration and CUDA
- Numba
- Mean shift clustering
- Gaussian kernel
- Norms
- Euclidean distance
- Calculus
  - Derivatives and Infinitesimals

## Video

<iframe width="514" height="289" src="https://www.youtube-nocookie.com/embed/_xIzPbCgutY?modestbranding=1" title="fast.ai lesson 12" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Lesson resources

- [Discuss this lesson](https://forums.fast.ai/t/lesson-12-official-topic/101702)
- [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator)
- [Essence of calculus](https://www.youtube.com/watch?v=WUvTyaaNkzM) (3blue1brown)

