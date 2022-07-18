---
title: "Lesson 8"
---

## Building embeddings from scratch

- What will part 2 feel like? a lot deeper technically? Able to read and implement research papers? Models involve real life situations?
- Review build a neuralnet from scratch. How Pytorch create a neuralnet effortlessly? How Pytorch keep track of model weights through `Module`? How does `Module` store weights with `nn.Parameter`? How to check weights from the model using `parameters()`?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/c/ac1eeab37f915183a24b24161bc98b737375ed3a_2_690x284.png" alt="module-parameter" width="690" height="284" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/c/ac1eeab37f915183a24b24161bc98b737375ed3a_2_690x284.png, https://forums.fast.ai/uploads/default/optimized/3X/a/c/ac1eeab37f915183a24b24161bc98b737375ed3a_2_1035x426.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/a/c/ac1eeab37f915183a24b24161bc98b737375ed3a.png 2x">

- You can build a layer in Module with `nn.Linear` without `nn.Parameter` and Pytorch can read weights from it too.

<img src="https://forums.fast.ai/uploads/default/optimized/3X/6/1/6169abe36ea65492da4795d59819c0a7711926f3_2_690x252.png" alt="module-layer" width="690" height="252" srcset="https://forums.fast.ai/uploads/default/optimized/3X/6/1/6169abe36ea65492da4795d59819c0a7711926f3_2_690x252.png, https://forums.fast.ai/uploads/default/optimized/3X/6/1/6169abe36ea65492da4795d59819c0a7711926f3_2_1035x378.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/6/1/6169abe36ea65492da4795d59819c0a7711926f3.png 2x">

- How to create the `Embedding` function and the entire `DotProductBias` with pytorch using `create_params` from scratch? After it’s trained, the trained `movie_bias` can be checked. You can check the shape of the bias by `model.movie_bias.shape`

<img src="https://forums.fast.ai/uploads/default/optimized/3X/b/3/b3ad05b6e908f82d01ae5bb03833fd3d22cdd07a_2_690x216.png" alt="create-embedding" width="690" height="216" srcset="https://forums.fast.ai/uploads/default/optimized/3X/b/3/b3ad05b6e908f82d01ae5bb03833fd3d22cdd07a_2_690x216.png, https://forums.fast.ai/uploads/default/optimized/3X/b/3/b3ad05b6e908f82d01ae5bb03833fd3d22cdd07a_2_1035x324.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/b/3/b3ad05b6e908f82d01ae5bb03833fd3d22cdd07a_2_1380x432.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/d/4/d463560b5dac88fa6265df49f186581365b15cd7_2_690x266.png" alt="DotProductBias-pytorch" width="690" height="266" srcset="https://forums.fast.ai/uploads/default/optimized/3X/d/4/d463560b5dac88fa6265df49f186581365b15cd7_2_690x266.png, https://forums.fast.ai/uploads/default/optimized/3X/d/4/d463560b5dac88fa6265df49f186581365b15cd7_2_1035x399.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/d/4/d463560b5dac88fa6265df49f186581365b15cd7.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/b/9/b9c387939d608287db55c4fe3f2fde20579da859_2_690x352.jpeg" alt="movie-bias-trained" width="690" height="352" srcset="https://forums.fast.ai/uploads/default/optimized/3X/b/9/b9c387939d608287db55c4fe3f2fde20579da859_2_690x352.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/b/9/b9c387939d608287db55c4fe3f2fde20579da859_2_1035x528.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/b/9/b9c387939d608287db55c4fe3f2fde20579da859_2_1380x704.jpeg 2x">

- Questions: What does `Tensor.normal_` do?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/8/a8cd719c1c9b49df72a7b63cc140c558462bc2ca_2_690x173.png" alt="Tensor.Normal_" width="690" height="173" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/8/a8cd719c1c9b49df72a7b63cc140c558462bc2ca_2_690x173.png, https://forums.fast.ai/uploads/default/optimized/3X/a/8/a8cd719c1c9b49df72a7b63cc140c558462bc2ca_2_1035x259.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/a/8/a8cd719c1c9b49df72a7b63cc140c558462bc2ca.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/1/c/1c35d70ecfae6df4e054fecb879be854bbe5679a_2_690x74.png" alt="why-no-zeros" width="690" height="74" srcset="https://forums.fast.ai/uploads/default/optimized/3X/1/c/1c35d70ecfae6df4e054fecb879be854bbe5679a_2_690x74.png, https://forums.fast.ai/uploads/default/original/3X/1/c/1c35d70ecfae6df4e054fecb879be854bbe5679a.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/1/c/1c35d70ecfae6df4e054fecb879be854bbe5679a.png 2x">

## Interpretation of embeddings

- After training, what can the `movie_bias` tell us about each and all the movies? What does having a low bias mean for a movie? What does having a high bias mean for a movie? Can `user_bias` tell us which user just loves movies even the crapy ones? This is visualizing `movie_bias`

<img src="https://forums.fast.ai/uploads/default/optimized/3X/3/8/38eba26268dc461437ffe433be538e73f74d60bf_2_690x329.png" alt="interpreting-bias" width="690" height="329" srcset="https://forums.fast.ai/uploads/default/optimized/3X/3/8/38eba26268dc461437ffe433be538e73f74d60bf_2_690x329.png, https://forums.fast.ai/uploads/default/optimized/3X/3/8/38eba26268dc461437ffe433be538e73f74d60bf_2_1035x493.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/3/8/38eba26268dc461437ffe433be538e73f74d60bf.png 2x">

- What can we interpret or do about the huge matrix with shape `(num_users, 50)`? How to shrink the 50 latent factors into just 3 most important factors with `pca`?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/c/4/c456a152f3dd0f85b7d17e6eee93c0179bc942d6_2_690x227.png" alt="shrink-by-pca" width="690" height="227" srcset="https://forums.fast.ai/uploads/default/optimized/3X/c/4/c456a152f3dd0f85b7d17e6eee93c0179bc942d6_2_690x227.png, https://forums.fast.ai/uploads/default/optimized/3X/c/4/c456a152f3dd0f85b7d17e6eee93c0179bc942d6_2_1035x340.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/c/4/c456a152f3dd0f85b7d17e6eee93c0179bc942d6_2_1380x454.png 2x">

- How to interpret the PCA chart of movies rated with only just two PCA factors of out 3 compressed by 50 factors? How the taste or style of the movies are condensed into two factors and displayed and defined by the location of the two dimensional chart? This is visualizing movie_factors or embeddings.

<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/e/4eac3d75cab2db567ac45fba7fbf0ef17b2c514d_2_587x500.jpeg" alt="compress-movie-taste-by-2-factors" width="587" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/e/4eac3d75cab2db567ac45fba7fbf0ef17b2c514d_2_587x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/e/4eac3d75cab2db567ac45fba7fbf0ef17b2c514d_2_880x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/e/4eac3d75cab2db567ac45fba7fbf0ef17b2c514d_2_1174x1000.jpeg 2x">

- How fastai makes all the work above easier with just one line of code?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/e/d/ed57496a25b591290deb15ac0a87bc09276e6ee7_2_559x500.jpeg" alt="fastai-make-things-easier" width="559" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/e/d/ed57496a25b591290deb15ac0a87bc09276e6ee7_2_559x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/e/d/ed57496a25b591290deb15ac0a87bc09276e6ee7_2_838x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/e/d/ed57496a25b591290deb15ac0a87bc09276e6ee7_2_1118x1000.jpeg 2x">

- How fastai construct everything under the hood of `collab_learner`?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/a/4afaa7b7a3eb83460fb708c1038a4bb7636931ae_2_690x241.jpeg" alt="collab_learner" width="690" height="241" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/a/4afaa7b7a3eb83460fb708c1038a4bb7636931ae_2_690x241.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/a/4afaa7b7a3eb83460fb708c1038a4bb7636931ae_2_1035x361.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/a/4afaa7b7a3eb83460fb708c1038a4bb7636931ae_2_1380x482.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/3/6/36b95732e07160818fc282f97a1ea26feebde957_2_690x291.png" alt="EmbeddingDotBias" width="690" height="291" srcset="https://forums.fast.ai/uploads/default/optimized/3X/3/6/36b95732e07160818fc282f97a1ea26feebde957_2_690x291.png, https://forums.fast.ai/uploads/default/optimized/3X/3/6/36b95732e07160818fc282f97a1ea26feebde957_2_1035x436.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/3/6/36b95732e07160818fc282f97a1ea26feebde957_2_1380x582.png 2x">

- Questions: is PCA useful in other applications? Where to find more of PCA? Why should you take Rachel’s Computational Linear Algebra?
- How to use Embedding distance to find out movie similarities?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/3/5/3570ea9ea1d9e3d83fcb1fc79da041378c2ab6cb_2_690x205.jpeg" alt="embedding-distance-similarities" width="690" height="205" srcset="https://forums.fast.ai/uploads/default/optimized/3X/3/5/3570ea9ea1d9e3d83fcb1fc79da041378c2ab6cb_2_690x205.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/3/5/3570ea9ea1d9e3d83fcb1fc79da041378c2ab6cb_2_1035x307.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/3/5/3570ea9ea1d9e3d83fcb1fc79da041378c2ab6cb_2_1380x410.jpeg 2x">

- Go to read the fastbook for boostrapping a collaborative filtering model

## Deep learning for collaborative filtering

- How to do collaborative filtering with deep learning instead of matrix completion with dot product above? How to apply the easist neuralnet model architecture onto this collaborative filtering case?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/e/2e3885bd3fec97240d61b1114ada57a770dddcac_2_690x273.jpeg" alt="deep-learning-colab-filter" width="690" height="273" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/e/2e3885bd3fec97240d61b1114ada57a770dddcac_2_690x273.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/e/2e3885bd3fec97240d61b1114ada57a770dddcac_2_1035x409.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/e/2e3885bd3fec97240d61b1114ada57a770dddcac_2_1380x546.jpeg 2x">

- How does fastai use rules of thumb to recommend the number of latent factors for users and movies?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/0/70613b67c9727ff47186d2323466a5ddb589497a_2_690x154.png" alt="number-latent-factors" width="690" height="154" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/0/70613b67c9727ff47186d2323466a5ddb589497a_2_690x154.png, https://forums.fast.ai/uploads/default/original/3X/7/0/70613b67c9727ff47186d2323466a5ddb589497a.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/7/0/70613b67c9727ff47186d2323466a5ddb589497a.png 2x">

- How does fastai use deep learning to build collaborative filtering model in two ways?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/3/6/362b88b96fc4ec0055caf95a2be17003dee2e4fc_2_690x391.jpeg" alt="dl-way-2" width="690" height="391" srcset="https://forums.fast.ai/uploads/default/optimized/3X/3/6/362b88b96fc4ec0055caf95a2be17003dee2e4fc_2_690x391.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/3/6/362b88b96fc4ec0055caf95a2be17003dee2e4fc_2_1035x586.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/3/6/362b88b96fc4ec0055caf95a2be17003dee2e4fc_2_1380x782.jpeg 2x">

 Why the deep learning versions are not as good as DotProduct version? Is it because DotProduct is more tailored to the problem? How do companies combine both versions to do collaborative filtering? When you have lots of metadata, should you apply deep learning to it? How would you use metadata in the model?
- Questions: Can a smaller number of users and movies overwhelm everybody else? e.g., a small group of anime enthusiasts watch a lot of anime movies and give super high ratings. Details of how to deal with them won’t be discussed here
- How to apply embedding matrix into NLP model through a spreadsheet demo? What’s the essense of neuralnet?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/1/0/1052d5774f1d716e988c005abce8939662b6a2a3_2_561x500.jpeg" alt="embedding-nlp" width="561" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/1/0/1052d5774f1d716e988c005abce8939662b6a2a3_2_561x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/1/0/1052d5774f1d716e988c005abce8939662b6a2a3_2_841x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/1/0/1052d5774f1d716e988c005abce8939662b6a2a3_2_1122x1000.jpeg 2x">

- How to apply embeddings to tabular dataset and models? How to understand `TabularModel` and `tabular_learner` source?
- What’s going on inside a neuralnet through a shop sale prediction kaggle competition and a paper published based on it?

## Convolutions

- So far we have looked at what goes in as inputs and what goes out of a model as outputs. We have also looked at the middle as matrix multiplication. What are convolution (a particular kind of matrix multiplication in the middel)? How is it be very useful to CV? Why MNIST is one of the most famous CV dataset? How does Jeremy apply what Fergus and Zeiler’s paper onto MNIST using excel and convolution?
- How to understand convolution? What does a filter do and How does it help to detect horizontal and vertical edges? How to determine the size of the filter or kernel (3x3, or 5x5, or any)? conv1 means the first convolutional layer
- moving onto the second convolutional layer. Two filters give us two channels on the first convolutional layer. On the second convolutional layer, we create one 3D matrix filter which has two matrix filters to filter/process the two channels out of the first conv layer,  and condense the value. And we can also create a second channle for the 2nd conv layer using another 3D filter.
- How to determine the output and use SGD to train the model and optimize the filters?
- What is maxpooling? What’s the problem of maxpooling? How much data do we lose? Why it is a good thing? What is a dense layer and what does it do?
- How we do convolution slightly differently today? What is stride-two convolution and how does it work? (no more maxpooling) Then we do a lot of stride-two convolutions until the size shrinked to 7x7 and then do a `average_pooling` (no more dense layer). What does the 7x7 grid and take an average mean? What is the problem of such approach? When is the good time to use maxpool instead? How fastai made it easy for us to try both pooling by inventing a technique called `concat_pooling` to maxpool and `average_pool` and concat them together?
- How to understand convolution in terms of matrix multiplications?
- What is dropout and how to understand it using excel? What is droput mask? What’s its effect visually on excel? How to understand dropout as data augmentation for the activations? How does it help avoid overfitting? What’s the story of dropout and academia?
- Why Jeremy not spend much time on activation functions? We have seen many functions on metrics, loss and activations.
- What to do next before fastai part2? What Radek’s book meta learning is about? What are the things to do in Write, Help, Gather and Build?
- a fastai community member published mish activation used by many state of art models.

## Jeremy AMA

- How to keep up? To keep up by focusing in subfield of deep learning and focusing on things that don’t change much as the foundations of fastai have not changed much from 5 years ago. Everything else is just tweaks.
- Will huge dataset and GPU computation replace us with small dataset and one gpu? There is always smarter ways of doing things, eg. Fastai team trained on imagenet on standard GPU faster than all companies with huge amount of GPUs. Pick areas of different domains which smaller models can beat the state of the art.
- How Jeremy to teach kids math? all kids can learn algebra with dragonbox5+. Great, Jeremy promised to talk more about teaching kids some point later.
- Plans for walkthrus
- How to turn a model into business? Great news, Jeremy plans to build a course on this! What is the start of a business? What is the first step? How to gradually figure out whether your idea has a real need from people?
- How Jeremy stay so efficient at working? Finish something nicely, tenacity

