---
title: "Lesson 7"
---

## Gradient accumulation and GPU memory

- We have explored the simplest neural net with fully connected linear layers in earlier lectures. In this lecture we will focus on tweaking first and last layers, in the next few weeks on tweaking middle part of the neuralnet.
- Review of the notebook <em>Road to Top part 2</em> and congrats to fastai students beat Jeremy on 1st and 2nd
- What are the benefits of using larger models? What are the problems of larger models? (use up GPU memory as GPU is not as clever as CPU to find ways to free itself; so large model needs very expensive GPU) What can we do about it when GPU out of memory? first, to restart the notebook; then Jeremy is about to show us a trick to enable us to train extra large models on Kaggle, Wow!
- How big is Kaggle GPU? Do you have to run notebooks on kaggle sometimes for example code competitions? Why it is good and fair to use Kaggle notebook to win leaderboard?
- How did Jeremy use a 24G GPU to find out what can a 16G GPU do? How did Jeremy find out how much GPU memory will a model use? How did Jeremy choose the smallest subgroup of images as the training set? Will training the model longer take up more memory? (No) So, smallest training set + 1 epoch training can quickly tell us how much memory is needed for the model.
- Jeremy then trained different models to see how much memories they used up. How much memory does convnext-small model take? Which line of code does Jeremy use to find out the GPU memory used up by the model? Which two lines of code does Jeremy use to free unnecssarily occupied memories GPU so that you don’t need to restart the kernel to run the next model?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/1/8/1873321e4fc2c4572c625303180eec7843967dbb_2_690x210.png" alt="gc" width="690" height="210" srcset="https://forums.fast.ai/uploads/default/optimized/3X/1/8/1873321e4fc2c4572c625303180eec7843967dbb_2_690x210.png, https://forums.fast.ai/uploads/default/optimized/3X/1/8/1873321e4fc2c4572c625303180eec7843967dbb_2_1035x315.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/1/8/1873321e4fc2c4572c625303180eec7843967dbb.png 2x">

- What if a model causes a crash problem of cuda out of memory? What is GradientAccumulation? What is integer divide? (`//`).

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/6/26d7b272d6e0d5a3a2802981037cba31124bdd43_2_690x230.jpeg" alt="gradientaccumulation" width="690" height="230" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/6/26d7b272d6e0d5a3a2802981037cba31124bdd43_2_690x230.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/6/26d7b272d6e0d5a3a2802981037cba31124bdd43_2_1035x345.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/6/26d7b272d6e0d5a3a2802981037cba31124bdd43_2_1380x460.jpeg 2x">

- What is the problem of using smaller batch size? (smaller batch size, larger volatility of learning rate and weights) How can we make the model train in smaller batch size as if it is in large batch size? How to explain GradientAccumulation in code?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/5/f/5f5de3dc9a36ce6ef7bb0413df0816afad42742f_2_690x158.png" alt="training-loop-without-gradientaccumulation" width="690" height="158" srcset="https://forums.fast.ai/uploads/default/optimized/3X/5/f/5f5de3dc9a36ce6ef7bb0413df0816afad42742f_2_690x158.png, https://forums.fast.ai/uploads/default/optimized/3X/5/f/5f5de3dc9a36ce6ef7bb0413df0816afad42742f_2_1035x237.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/5/f/5f5de3dc9a36ce6ef7bb0413df0816afad42742f_2_1380x316.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/f/9/f9daf4e38da769cb3de7a0c8cdf20c74fd8d18b7_2_690x210.jpeg" alt="gradientaccumulation-explained" width="690" height="210" srcset="https://forums.fast.ai/uploads/default/optimized/3X/f/9/f9daf4e38da769cb3de7a0c8cdf20c74fd8d18b7_2_690x210.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/f/9/f9daf4e38da769cb3de7a0c8cdf20c74fd8d18b7_2_1035x315.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/f/9/f9daf4e38da769cb3de7a0c8cdf20c74fd8d18b7_2_1380x420.jpeg 2x">

- What is the implication of using GradientAccumulation? How much difference is the numeric result between using GradientAccumulation and not? What is the main cause for the difference?
- More questions: it should be `count &gt;= 64` in the code above when doing GradientAccumulation; `lr_find` uses batch size from the DataLoader;
- Why not just use a smaller batch size instead of GradientAccumulation? What is the rule of thumb for picking batch sizes? How about adjusting learning rate according to the batch size?
- How did Jeremy use GradientAccumulation to find out how many `accum` is needed to run those large models on Kaggle’s 16G GPUs? (`accum=1` always out of memory, but `accum=2` works for all large models).

<img src="https://forums.fast.ai/uploads/default/optimized/3X/c/c/cc4d36bc34d4f4d1c313b96cdf252b9e6c9ac33d_2_690x81.png" alt="gradientaccumulation-large-models" width="690" height="81" srcset="https://forums.fast.ai/uploads/default/optimized/3X/c/c/cc4d36bc34d4f4d1c313b96cdf252b9e6c9ac33d_2_690x81.png, https://forums.fast.ai/uploads/default/optimized/3X/c/c/cc4d36bc34d4f4d1c313b96cdf252b9e6c9ac33d_2_1035x121.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/c/c/cc4d36bc34d4f4d1c313b96cdf252b9e6c9ac33d_2_1380x162.png 2x">

## Creating an ensemble

- How did Jeremy put all the models and their settings together for experimenting later? Do we have to use the size of the model’s specification for now and how about in the future?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/b/2bac3e816fa5bb631ba278430031a9561b3fffbc_2_665x500.jpeg" alt="models-testing" width="665" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/b/2bac3e816fa5bb631ba278430031a9561b3fffbc_2_665x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/b/2bac3e816fa5bb631ba278430031a9561b3fffbc_2_997x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/original/3X/2/b/2bac3e816fa5bb631ba278430031a9561b3fffbc.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/a/2a249910c04e58ec006290f08688e7b0f8aeea81_2_690x56.png" alt="training-set-images" width="690" height="56" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/a/2a249910c04e58ec006290f08688e7b0f8aeea81_2_690x56.png, https://forums.fast.ai/uploads/default/optimized/3X/2/a/2a249910c04e58ec006290f08688e7b0f8aeea81_2_1035x84.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/a/2a249910c04e58ec006290f08688e7b0f8aeea81_2_1380x112.png 2x">

- How to run all the models with specifications without running out of memory

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/e/ae921c0da2bf64fe0e3d890f7cf3effe0e8369d5_2_690x304.jpeg" alt="run-all-models-without-out-memory" width="690" height="304" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/e/ae921c0da2bf64fe0e3d890f7cf3effe0e8369d5_2_690x304.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/a/e/ae921c0da2bf64fe0e3d890f7cf3effe0e8369d5_2_1035x456.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/a/e/ae921c0da2bf64fe0e3d890f7cf3effe0e8369d5_2_1380x608.jpeg 2x">

- Why does Jeremy don’t use `seed=42` here in training? What is the effect?
- What is ensemble or bagging of different good deep learning architectures? Why it is useful?
- How to do the ensemble of different deep learning models?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/e/2/e2960f76f91dcb5e888127e89ce267d08a6eb1bb_2_690x349.jpeg" alt="ensemble1" width="690" height="349" srcset="https://forums.fast.ai/uploads/default/optimized/3X/e/2/e2960f76f91dcb5e888127e89ce267d08a6eb1bb_2_690x349.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/e/2/e2960f76f91dcb5e888127e89ce267d08a6eb1bb_2_1035x523.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/e/2/e2960f76f91dcb5e888127e89ce267d08a6eb1bb_2_1380x698.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/3/43a6e2e47b5d49d201353fff1b15279f149b8d08_2_690x295.jpeg" alt="ensemble2" width="690" height="295" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/3/43a6e2e47b5d49d201353fff1b15279f149b8d08_2_690x295.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/3/43a6e2e47b5d49d201353fff1b15279f149b8d08_2_1035x442.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/3/43a6e2e47b5d49d201353fff1b15279f149b8d08_2_1380x590.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/5/45548cc497f861df5b5c4a1234bc17aee41d7cee_2_690x317.jpeg" alt="ensemble3" width="690" height="317" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/5/45548cc497f861df5b5c4a1234bc17aee41d7cee_2_690x317.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/5/45548cc497f861df5b5c4a1234bc17aee41d7cee_2_1035x475.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/5/45548cc497f861df5b5c4a1234bc17aee41d7cee_2_1380x634.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/5/1/51f580a36eab28daedbb68779840e1963c85f03b_2_690x183.png" alt="ensemble4" width="690" height="183" srcset="https://forums.fast.ai/uploads/default/optimized/3X/5/1/51f580a36eab28daedbb68779840e1963c85f03b_2_690x183.png, https://forums.fast.ai/uploads/default/optimized/3X/5/1/51f580a36eab28daedbb68779840e1963c85f03b_2_1035x274.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/5/1/51f580a36eab28daedbb68779840e1963c85f03b_2_1380x366.png 2x">

- Why should we improve and submit to Kaggle everyday? How the submission history can help trace your models developments and improvement?
- More questions: What is k-fold cross-validation and how can it be applied in this case? Why does Jeremy don’t use it?
- Are there any drawbacks of GradientAccumulation? Any GPU recommendations?
- In part 2 Jeremy may cover how to train a smaller model to do well as in large models for faster inference

## Multi-target model

- How to set the data split and item and batch transformations?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/6/7/6708adef38cb0f664442d2e9e3f846fda75a045d_2_690x235.png" alt="datablock-2-labels" width="690" height="235" srcset="https://forums.fast.ai/uploads/default/optimized/3X/6/7/6708adef38cb0f664442d2e9e3f846fda75a045d_2_690x235.png, https://forums.fast.ai/uploads/default/optimized/3X/6/7/6708adef38cb0f664442d2e9e3f846fda75a045d_2_1035x352.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/6/7/6708adef38cb0f664442d2e9e3f846fda75a045d_2_1380x470.png 2x">

- How to create a model to predict both disease and variety types? Can we see predicting both disease and variety in terms of predicting 20 things, 10 for disease, 10 for variety?
- What does the new model (and new dataloaders) need now to make predictions on disease?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/8/48c068723f1b2f8c4205d02c96d9f5d6e2d65189_2_690x254.jpeg" alt="new-disease-error-rate-loss" width="690" height="254" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/8/48c068723f1b2f8c4205d02c96d9f5d6e2d65189_2_690x254.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/8/48c068723f1b2f8c4205d02c96d9f5d6e2d65189_2_1035x381.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/8/48c068723f1b2f8c4205d02c96d9f5d6e2d65189_2_1380x508.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/e/4ee51a1545b8b3f3d8ea0a0b485d45dddc1a2a90_2_690x159.png" alt="new-model" width="690" height="159" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/e/4ee51a1545b8b3f3d8ea0a0b485d45dddc1a2a90_2_690x159.png, https://forums.fast.ai/uploads/default/optimized/3X/4/e/4ee51a1545b8b3f3d8ea0a0b485d45dddc1a2a90_2_1035x238.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/e/4ee51a1545b8b3f3d8ea0a0b485d45dddc1a2a90_2_1380x318.png 2x">

- When and how to provide our own loss function? fastai can detect appropriate loss for your datalaoders and use it by default in simple cases. In this special case, How do we create and use our custom loss for the new model?

## Cross-entropy and softmax

- What does `F.cross_entropy` do exactly? This function belong to the first and last layer, therefore we must understand them. What is the raw output of the model of predicting 5 things?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/5/6/563e70daa1930078e47617c7c2db3897446f3848_2_544x500.jpeg" alt="softmax1" width="544" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/5/6/563e70daa1930078e47617c7c2db3897446f3848_2_544x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/5/6/563e70daa1930078e47617c7c2db3897446f3848_2_816x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/5/6/563e70daa1930078e47617c7c2db3897446f3848_2_1088x1000.jpeg 2x">

- What is the formula of softmax and How to calculate it in the spreadsheet?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/0/b/0bee952ff080de7e7b050fa9e81dd483414da494_2_645x500.jpeg" alt="softmax-calculated" width="645" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/0/b/0bee952ff080de7e7b050fa9e81dd483414da494_2_645x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/0/b/0bee952ff080de7e7b050fa9e81dd483414da494_2_967x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/0/b/0bee952ff080de7e7b050fa9e81dd483414da494_2_1290x1000.jpeg 2x">

- What is the problem of softmax? How does it make the obvious wrong prediction when given a cat image to the bear classifier?
- What can we do about the problem of the softmax above? (all prediction probabilities not adding up to 1). When do you use softmax and when not to?
- What is the first part of the cross_entropy loss formula?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/6/1/613442943a389b25d30f73afe94a2172da533018_2_690x495.jpeg" alt="part1-crossentropy" width="690" height="495" srcset="https://forums.fast.ai/uploads/default/optimized/3X/6/1/613442943a389b25d30f73afe94a2172da533018_2_690x495.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/6/1/613442943a389b25d30f73afe94a2172da533018_2_1035x742.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/6/1/613442943a389b25d30f73afe94a2172da533018_2_1380x990.jpeg 2x">

- How to calculate cross-entropy from softmax?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/8/5/8546446e26005bcfb41f2dbea8f094ea6044c9cf_2_620x500.jpeg" alt="calc-cross-entropy" width="620" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/8/5/8546446e26005bcfb41f2dbea8f094ea6044c9cf_2_620x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/8/5/8546446e26005bcfb41f2dbea8f094ea6044c9cf_2_930x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/8/5/8546446e26005bcfb41f2dbea8f094ea6044c9cf_2_1240x1000.jpeg 2x">

- How to calculate binary-cross-entropy? How to understand its formula in predicting whether it is a cat or non-cat image? How to finally get the binary cross-entropy loss of a batch of 5 images?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/d/7/d7194a9f6353d643b0a9472756fa076dfd335597_2_690x473.jpeg" alt="binary-cross-entropy-calc" width="690" height="473" srcset="https://forums.fast.ai/uploads/default/optimized/3X/d/7/d7194a9f6353d643b0a9472756fa076dfd335597_2_690x473.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/d/7/d7194a9f6353d643b0a9472756fa076dfd335597_2_1035x709.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/d/7/d7194a9f6353d643b0a9472756fa076dfd335597_2_1380x946.jpeg 2x">

- What are two versions of cross-entropy in pytorch? and when to use each version? Which version do we use here?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/2/2245e4c140536aad1ef5b72a9543afda38ac65a5_2_690x264.jpeg" alt="cross-entropy-v1" width="690" height="264" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/2/2245e4c140536aad1ef5b72a9543afda38ac65a5_2_690x264.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/2/2245e4c140536aad1ef5b72a9543afda38ac65a5_2_1035x396.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/2/2245e4c140536aad1ef5b72a9543afda38ac65a5_2_1380x528.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/a/2ab6375dd73126e7635687a7fd3110b305eedded_2_690x114.png" alt="cross-entropy-v2" width="690" height="114" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/a/2ab6375dd73126e7635687a7fd3110b305eedded_2_690x114.png, https://forums.fast.ai/uploads/default/optimized/3X/2/a/2ab6375dd73126e7635687a7fd3110b305eedded_2_1035x171.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/a/2ab6375dd73126e7635687a7fd3110b305eedded_2_1380x228.png 2x">

## Multi-target activations

- With a dataloader having two targets, our new model needs to be informed what exactly is the loss func, metrics, and the size of output?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/2/222cd9dc5c842f4a39c8046b85ac640c65664f82_2_690x182.jpeg" alt="model-specifications" width="690" height="182" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/2/222cd9dc5c842f4a39c8046b85ac640c65664f82_2_690x182.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/2/222cd9dc5c842f4a39c8046b85ac640c65664f82_2_1035x273.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/2/222cd9dc5c842f4a39c8046b85ac640c65664f82_2_1380x364.jpeg 2x">

- How to create a learner for prediction two targets or 20 items? How does a learner use disease and variety losses to know which 10 items are disease predictions and which 10 are variety predictions? How to combine two loss functions together? How to understand the combined loss?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/8/a877afc66e453b40fd4194e9070c33505470975e_2_690x127.png" alt="model-output-20" width="690" height="127" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/8/a877afc66e453b40fd4194e9070c33505470975e_2_690x127.png, https://forums.fast.ai/uploads/default/optimized/3X/a/8/a877afc66e453b40fd4194e9070c33505470975e_2_1035x190.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/a/8/a877afc66e453b40fd4194e9070c33505470975e_2_1380x254.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/6/6/665c97da404c8608e3d198db8812c895e5c93660_2_690x151.png" alt="loss-disease" width="690" height="151" srcset="https://forums.fast.ai/uploads/default/optimized/3X/6/6/665c97da404c8608e3d198db8812c895e5c93660_2_690x151.png, https://forums.fast.ai/uploads/default/optimized/3X/6/6/665c97da404c8608e3d198db8812c895e5c93660_2_1035x226.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/6/6/665c97da404c8608e3d198db8812c895e5c93660_2_1380x302.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/b/7b91f907777aa2cb90cf68c1878c903c444694f1_2_690x139.png" alt="loss-variety" width="690" height="139" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/b/7b91f907777aa2cb90cf68c1878c903c444694f1_2_690x139.png, https://forums.fast.ai/uploads/default/optimized/3X/7/b/7b91f907777aa2cb90cf68c1878c903c444694f1_2_1035x208.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/7/b/7b91f907777aa2cb90cf68c1878c903c444694f1_2_1380x278.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/6/a61748d386d2c9bcbcd6f2555a88f3d058529bcf_2_690x132.png" alt="combine-loss" width="690" height="132" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/6/a61748d386d2c9bcbcd6f2555a88f3d058529bcf_2_690x132.png, https://forums.fast.ai/uploads/default/optimized/3X/a/6/a61748d386d2c9bcbcd6f2555a88f3d058529bcf_2_1035x198.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/a/6/a61748d386d2c9bcbcd6f2555a88f3d058529bcf_2_1380x264.png 2x">

- How to calc error rate for disease types and variety types? How to put them together and display them during training?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/9/0/90658b962d8ec66766ed27c63af27424cabe6613_2_690x279.jpeg" alt="metrics-all-together" width="690" height="279" srcset="https://forums.fast.ai/uploads/default/optimized/3X/9/0/90658b962d8ec66766ed27c63af27424cabe6613_2_690x279.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/9/0/90658b962d8ec66766ed27c63af27424cabe6613_2_1035x418.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/9/0/90658b962d8ec66766ed27c63af27424cabe6613_2_1380x558.jpeg 2x">

- How to make the new learner and how did it train? Why the multi-task model didn’t improve and even a little worse than the previous model? Why training the multi-task model longer could improve the accuracy on disease prediction? Why predicting a second thing together could help improve the prediction of the first thing? Using multi-task model did improve the result in a Kaggle fish prediction competition Jeremy did before. What are the reasons or benefits for building multi-task models?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/0/f/0f18c85770f6737745434ab2b4812fc73854baa1_2_690x138.png" alt="train-combined-loss-metrics-all" width="690" height="138" srcset="https://forums.fast.ai/uploads/default/optimized/3X/0/f/0f18c85770f6737745434ab2b4812fc73854baa1_2_690x138.png, https://forums.fast.ai/uploads/default/optimized/3X/0/f/0f18c85770f6737745434ab2b4812fc73854baa1_2_1035x207.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/0/f/0f18c85770f6737745434ab2b4812fc73854baa1_2_1380x276.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/0/a0ce75a163a66d0c5f00875347ea1abdb3379fb7_2_690x363.jpeg" alt="result-little-worse" width="690" height="363" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/0/a0ce75a163a66d0c5f00875347ea1abdb3379fb7_2_690x363.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/a/0/a0ce75a163a66d0c5f00875347ea1abdb3379fb7_2_1035x544.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/a/0/a0ce75a163a66d0c5f00875347ea1abdb3379fb7_2_1380x726.jpeg 2x">

 How to make multi-task modeling less confusing to you? (build a multi-task for Titanic dataset from scratch; explore and experiment this notebook)
by Chris Said of binary-cross-entropy?

## Collaborative filtering

- Collaborative filtering deep dive as chp 8 without change. What is the dataset used? Which version of the data we are using? How to read a tsv file using pandas? How to read/understand the dataset content/columns? recommendation system industry and Radek. How does Jeremy prefer to see the data? (cross tabulated) Why the image Jeremy talking about his preferred way of seeing the data has so few empty or missing data?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/e/c/ec8cf3b5f0af012bc4fca4cf1b82f55fab674aae_2_690x219.jpeg" alt="dataset" width="690" height="219" srcset="https://forums.fast.ai/uploads/default/optimized/3X/e/c/ec8cf3b5f0af012bc4fca4cf1b82f55fab674aae_2_690x219.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/e/c/ec8cf3b5f0af012bc4fca4cf1b82f55fab674aae_2_1035x328.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/e/c/ec8cf3b5f0af012bc4fca4cf1b82f55fab674aae_2_1380x438.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/9/a97bf7cf60d140f3b43748d771fd6a4fedb1cf3b_2_690x332.jpeg" alt="read-tsv" width="690" height="332" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/9/a97bf7cf60d140f3b43748d771fd6a4fedb1cf3b_2_690x332.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/a/9/a97bf7cf60d140f3b43748d771fd6a4fedb1cf3b_2_1035x498.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/a/9/a97bf7cf60d140f3b43748d771fd6a4fedb1cf3b_2_1380x664.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/6/4/64d971d762156b93315e3fd15e9f49cb0987f0e8_2_690x313.jpeg" alt="Jeremy-see-this-way" width="690" height="313" srcset="https://forums.fast.ai/uploads/default/optimized/3X/6/4/64d971d762156b93315e3fd15e9f49cb0987f0e8_2_690x313.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/6/4/64d971d762156b93315e3fd15e9f49cb0987f0e8_2_1035x469.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/6/4/64d971d762156b93315e3fd15e9f49cb0987f0e8_2_1380x626.jpeg 2x">

- How to fill in the missing data or gaps in the cross tabulated dataset? How to figure out whether a new user would like a particular movie which he/she has not watched before? Can we figure out what kind/genre of movie is the particular movie we are talking here? What does the type probabilities of a movie look like? What does a user’s preference probabilities look like? If we match the two sets of probabilities up, can we know how much does the user like the movie? How do we calculate that?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/5/9/592b2fdb4fb49e0dc932863c8035b5016cbf4425_2_690x188.jpeg" alt="types-movie" width="690" height="188" srcset="https://forums.fast.ai/uploads/default/optimized/3X/5/9/592b2fdb4fb49e0dc932863c8035b5016cbf4425_2_690x188.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/5/9/592b2fdb4fb49e0dc932863c8035b5016cbf4425_2_1035x282.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/5/9/592b2fdb4fb49e0dc932863c8035b5016cbf4425_2_1380x376.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/f/4f5531e5b7200f92d3a9af4247df3f504d1024af_2_690x292.jpeg" alt="user-preference-sum" width="690" height="292" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/f/4f5531e5b7200f92d3a9af4247df3f504d1024af_2_690x292.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/f/4f5531e5b7200f92d3a9af4247df3f504d1024af_2_1035x438.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/f/4f5531e5b7200f92d3a9af4247df3f504d1024af_2_1380x584.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/c/0/c00851f68905b8243d3b73658463e979ac2ee73b_2_690x304.png" alt="lower-value" width="690" height="304" srcset="https://forums.fast.ai/uploads/default/optimized/3X/c/0/c00851f68905b8243d3b73658463e979ac2ee73b_2_690x304.png, https://forums.fast.ai/uploads/default/optimized/3X/c/0/c00851f68905b8243d3b73658463e979ac2ee73b_2_1035x456.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/c/0/c00851f68905b8243d3b73658463e979ac2ee73b_2_1380x608.png 2x">

- So far so good, what is the problem of the approach of doing dot product between user preference probabilities and movie type probabilities to find out our new user’s rating of the movie? (we don’t know neither of the probabilities). How are we going to deal with this problem? Can we create such movie type probabilities without knowing even the types?
- What is the latent factors? If I don’t know anything about the movies, can we use SGD (stochastic gradient descent) to find them? Can we create a random 5 numbers as a movie’s 5 latent factors for describing the types of the movie, and figure them out later? Can we create latent factors for each user too? Now how to calc the probability of a user likes a movie? (mmult or dot product between two groups of latent factors).

<img src="https://forums.fast.ai/uploads/default/optimized/3X/3/2/329b0259a8e6b1d8e4bd225dbf378efe9e189be0_2_690x456.jpeg" alt="latent-factors-movie" width="690" height="456" srcset="https://forums.fast.ai/uploads/default/optimized/3X/3/2/329b0259a8e6b1d8e4bd225dbf378efe9e189be0_2_690x456.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/3/2/329b0259a8e6b1d8e4bd225dbf378efe9e189be0_2_1035x684.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/3/2/329b0259a8e6b1d8e4bd225dbf378efe9e189be0_2_1380x912.jpeg 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/d/c/dc1b0d2e8e766a0750c94881c77bad576fee0fdd_2_690x386.jpeg" alt="latent-factor-for-all-movies" width="690" height="386" srcset="https://forums.fast.ai/uploads/default/optimized/3X/d/c/dc1b0d2e8e766a0750c94881c77bad576fee0fdd_2_690x386.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/d/c/dc1b0d2e8e766a0750c94881c77bad576fee0fdd_2_1035x579.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/d/c/dc1b0d2e8e766a0750c94881c77bad576fee0fdd_2_1380x772.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/original/3X/0/4/043d85eca9ca0e2198ba5f1b6fa629ba470cebfe.jpeg" alt="latent-factors-user" width="554" height="500">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/6/7628a9b85d34d0383be3c904f3a7c0c49e9a2484_2_690x405.jpeg" alt="matrix-multiply-dot-product" width="690" height="405" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/6/7628a9b85d34d0383be3c904f3a7c0c49e9a2484_2_690x405.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/7/6/7628a9b85d34d0383be3c904f3a7c0c49e9a2484_2_1035x607.jpeg 1.5x, https://forums.fast.ai/uploads/default/original/3X/7/6/7628a9b85d34d0383be3c904f3a7c0c49e9a2484.jpeg 2x">

- Now the mmult or dot product can give us the prediction of how much a user likes a movie, so we can compare predictions with true label. What to do when there is a missing label or data? (we make the prediction empty or zero). Can we use SGD to improve the latent factors by comparing predictions with labels using a loss function? How to use excel solver to update latent factors using SGD and the loss?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/e/f/ef6ab1294f1093259e4c5df1c921703361fdd384_2_670x500.jpeg" alt="missing-data-empty" width="670" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/e/f/ef6ab1294f1093259e4c5df1c921703361fdd384_2_670x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/e/f/ef6ab1294f1093259e4c5df1c921703361fdd384_2_1005x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/original/3X/e/f/ef6ab1294f1093259e4c5df1c921703361fdd384.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/0/4/04fbaa36a4c360959c0a826caba9592b0b5514c9_2_523x500.jpeg" alt="make-missing-zero" width="523" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/0/4/04fbaa36a4c360959c0a826caba9592b0b5514c9_2_523x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/0/4/04fbaa36a4c360959c0a826caba9592b0b5514c9_2_784x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/0/4/04fbaa36a4c360959c0a826caba9592b0b5514c9_2_1046x1000.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/5/755438b324e3799f6ce766c713c6b5036ae8bf81_2_690x362.jpeg" alt="the-loss-pred-label" width="690" height="362" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/5/755438b324e3799f6ce766c713c6b5036ae8bf81_2_690x362.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/7/5/755438b324e3799f6ce766c713c6b5036ae8bf81_2_1035x543.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/7/5/755438b324e3799f6ce766c713c6b5036ae8bf81_2_1380x724.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/0/1/0159ecab4e53556ebdad0b406b13b73805ad6616_2_690x427.jpeg" alt="excel-solver" width="690" height="427" srcset="https://forums.fast.ai/uploads/default/optimized/3X/0/1/0159ecab4e53556ebdad0b406b13b73805ad6616_2_690x427.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/0/1/0159ecab4e53556ebdad0b406b13b73805ad6616_2_1035x640.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/0/1/0159ecab4e53556ebdad0b406b13b73805ad6616_2_1380x854.jpeg 2x">

 Why excel is so slow on calc gradients with even small dataset? What is the basis of collaborative filtering? (if we know A likes (a, b, c) and B likes (a, b, c), then if A likes (d, e), maybe B likes (d, e) too).
- Is the cosine of an angle between two vectors  is the same thing as the dot product?
- How do we do the things above in pytorch as they have different data format from excel? What does the dataset would look like in pytorch?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/b/a/ba1b5333c95a803967c9b6b3b5922f6200aec599_2_690x326.jpeg" alt="collab-dataset-pytorch" width="690" height="326" srcset="https://forums.fast.ai/uploads/default/optimized/3X/b/a/ba1b5333c95a803967c9b6b3b5922f6200aec599_2_690x326.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/b/a/ba1b5333c95a803967c9b6b3b5922f6200aec599_2_1035x489.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/b/a/ba1b5333c95a803967c9b6b3b5922f6200aec599_2_1380x652.jpeg 2x">

## Embeddings

- What is embedding? What are embedding matrix, user embeddings, and movie embeddings? (embeddings = look up something in an array).  The more intimidating words created in a field, the less intimidating the field actually is.

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/c/2c3081a4e2471d0dc86d6a4e76751662ebb27739_2_690x279.jpeg" alt="embeddings" width="690" height="279" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/c/2c3081a4e2471d0dc86d6a4e76751662ebb27739_2_690x279.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/c/2c3081a4e2471d0dc86d6a4e76751662ebb27739_2_1035x418.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/c/2c3081a4e2471d0dc86d6a4e76751662ebb27739_2_1380x558.jpeg 2x">

- What does our dataset look like before building a dataloaders on it? How to create a dataloaders for collaborative filtering using `CollabDataloaders.from_df`? What does its `show_batch` look like? How do we create the user and movie latent factors algetother?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/5/758eb7bc56f1562e7dbe2525c359cc80ae55cc23_2_610x500.jpeg" alt="dataset-movie-dataloader" width="610" height="500" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/5/758eb7bc56f1562e7dbe2525c359cc80ae55cc23_2_610x500.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/7/5/758eb7bc56f1562e7dbe2525c359cc80ae55cc23_2_915x750.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/7/5/758eb7bc56f1562e7dbe2525c359cc80ae55cc23_2_1220x1000.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/8/f/8f28b35c026c563c233282bbe7393cb9af0d4b67_2_690x172.jpeg" alt="build-dataloader-cf" width="690" height="172" srcset="https://forums.fast.ai/uploads/default/optimized/3X/8/f/8f28b35c026c563c233282bbe7393cb9af0d4b67_2_690x172.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/8/f/8f28b35c026c563c233282bbe7393cb9af0d4b67_2_1035x258.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/8/f/8f28b35c026c563c233282bbe7393cb9af0d4b67_2_1380x344.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/a/9/a9d0bdf31e8734a7c0ed40edd1ef3e36e53c2298_2_690x397.jpeg" alt="cf-show-batch" width="690" height="397" srcset="https://forums.fast.ai/uploads/default/optimized/3X/a/9/a9d0bdf31e8734a7c0ed40edd1ef3e36e53c2298_2_690x397.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/a/9/a9d0bdf31e8734a7c0ed40edd1ef3e36e53c2298_2_1035x595.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/a/9/a9d0bdf31e8734a7c0ed40edd1ef3e36e53c2298_2_1380x794.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/8/7838f8430473f724a2e3e5156a25f7d6bda54b72_2_690x211.png" alt="create-latent-factors" width="690" height="211" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/8/7838f8430473f724a2e3e5156a25f7d6bda54b72_2_690x211.png, https://forums.fast.ai/uploads/default/optimized/3X/7/8/7838f8430473f724a2e3e5156a25f7d6bda54b72_2_1035x316.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/7/8/7838f8430473f724a2e3e5156a25f7d6bda54b72.png 2x">

- How do you choose  the number of latent factors in fastai?
- How to understand looking up in excel for latent factors and doing dot product with one-hot embeddings are actually the same thing? Can we think of embeddings as a computational shortcut to multiply something by a one-hot-encoded vector? Can we think of embedding as a cook math trick of speeding up the matrix multiplication with dummy variables (without creating dummy variables nor one-hot encoded vector).

<img src="https://forums.fast.ai/uploads/default/optimized/3X/1/f/1f254bae3830f4f79cc07b6c16a0511ac63bbfaf_2_690x345.png" alt="look-up-dot-product" width="690" height="345" srcset="https://forums.fast.ai/uploads/default/optimized/3X/1/f/1f254bae3830f4f79cc07b6c16a0511ac63bbfaf_2_690x345.png, https://forums.fast.ai/uploads/default/optimized/3X/1/f/1f254bae3830f4f79cc07b6c16a0511ac63bbfaf_2_1035x517.png 1.5x, https://forums.fast.ai/uploads/default/original/3X/1/f/1f254bae3830f4f79cc07b6c16a0511ac63bbfaf.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/6/c/6c1dd2073754edda0a1bfe7acc38db80506396ba_2_690x243.jpeg" alt="one-hot-encoding-look-up" width="690" height="243" srcset="https://forums.fast.ai/uploads/default/optimized/3X/6/c/6c1dd2073754edda0a1bfe7acc38db80506396ba_2_690x243.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/6/c/6c1dd2073754edda0a1bfe7acc38db80506396ba_2_1035x364.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/6/c/6c1dd2073754edda0a1bfe7acc38db80506396ba_2_1380x486.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/b/0/b0b6205134341c68799a9739bf55c105edc870dd_2_690x373.jpeg" alt="mmultiplication" width="690" height="373" srcset="https://forums.fast.ai/uploads/default/optimized/3X/b/0/b0b6205134341c68799a9739bf55c105edc870dd_2_690x373.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/b/0/b0b6205134341c68799a9739bf55c105edc870dd_2_1035x559.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/b/0/b0b6205134341c68799a9739bf55c105edc870dd_2_1380x746.jpeg 2x">

## Object oriented programming

- How to build a collaborative filtering model from scratch? How do we create a class? (as a model is a class). How do we initiate a class object by `__init__`? Does `__init__` tell us what parameters to give in order to create a class instance? How does the class function `say` do? What is a super class? Where do we put it when creating a class? What does it give us? What is the super class (Module) for pytorch and fastai to use when creating a class? What does the `DotProduct` class look like?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/f/3/f39bb45a2a9aeec4c50c2230dbe546bbbd8ef903_2_690x244.jpeg" alt="learn-create-class" width="690" height="244" srcset="https://forums.fast.ai/uploads/default/optimized/3X/f/3/f39bb45a2a9aeec4c50c2230dbe546bbbd8ef903_2_690x244.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/f/3/f39bb45a2a9aeec4c50c2230dbe546bbbd8ef903_2_1035x366.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/f/3/f39bb45a2a9aeec4c50c2230dbe546bbbd8ef903_2_1380x488.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/5/2/52e85a3c36db128d42865dbb4714073ae54b1415_2_690x157.png" alt="class-example" width="690" height="157" srcset="https://forums.fast.ai/uploads/default/optimized/3X/5/2/52e85a3c36db128d42865dbb4714073ae54b1415_2_690x157.png, https://forums.fast.ai/uploads/default/optimized/3X/5/2/52e85a3c36db128d42865dbb4714073ae54b1415_2_1035x235.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/5/2/52e85a3c36db128d42865dbb4714073ae54b1415_2_1380x314.png 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/9/5/958dd964f8d2dc24382de7a0799e03c960a6e43a_2_690x302.jpeg" alt="super-class" width="690" height="302" srcset="https://forums.fast.ai/uploads/default/optimized/3X/9/5/958dd964f8d2dc24382de7a0799e03c960a6e43a_2_690x302.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/9/5/958dd964f8d2dc24382de7a0799e03c960a6e43a_2_1035x453.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/9/5/958dd964f8d2dc24382de7a0799e03c960a6e43a_2_1380x604.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/9/49d96bdbc7ea8dc9621aba7e72e9a901de712307_2_690x215.jpeg" alt="Dot-product-class" width="690" height="215" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/9/49d96bdbc7ea8dc9621aba7e72e9a901de712307_2_690x215.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/4/9/49d96bdbc7ea8dc9621aba7e72e9a901de712307_2_1035x322.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/9/49d96bdbc7ea8dc9621aba7e72e9a901de712307_2_1380x430.jpeg 2x">

- How to understand the `forward` function in the `DotProduct` class? What does `.sum(dim=1)` mean? (sum each row).

<img src="https://forums.fast.ai/uploads/default/optimized/3X/7/b/7befe31f450692edb4c008b80ea0b59b2d54b2b8_2_690x220.jpeg" alt="forward-understanding" width="690" height="220" srcset="https://forums.fast.ai/uploads/default/optimized/3X/7/b/7befe31f450692edb4c008b80ea0b59b2d54b2b8_2_690x220.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/7/b/7befe31f450692edb4c008b80ea0b59b2d54b2b8_2_1035x330.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/7/b/7befe31f450692edb4c008b80ea0b59b2d54b2b8_2_1380x440.jpeg 2x">

## Improving collaborative filtering

- How to create a collab learner and start training? The training is very fast even on CPU.

<img src="https://forums.fast.ai/uploads/default/optimized/3X/4/2/426c364888d12db84268e7fa5f3feac9babb2d55_2_690x175.png" alt="one-batch-look-like" width="690" height="175" srcset="https://forums.fast.ai/uploads/default/optimized/3X/4/2/426c364888d12db84268e7fa5f3feac9babb2d55_2_690x175.png, https://forums.fast.ai/uploads/default/optimized/3X/4/2/426c364888d12db84268e7fa5f3feac9babb2d55_2_1035x262.png 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/4/2/426c364888d12db84268e7fa5f3feac9babb2d55_2_1380x350.png 2x">


<img src="https://forums.fast.ai/uploads/default/optimized/3X/9/8/980f8b59b9e8339f4b22f4786d9eb3f8c1c6f627_2_690x262.jpeg" alt="collab-learner" width="690" height="262" srcset="https://forums.fast.ai/uploads/default/optimized/3X/9/8/980f8b59b9e8339f4b22f4786d9eb3f8c1c6f627_2_690x262.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/9/8/980f8b59b9e8339f4b22f4786d9eb3f8c1c6f627_2_1035x393.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/9/8/980f8b59b9e8339f4b22f4786d9eb3f8c1c6f627_2_1380x524.jpeg 2x">

- Why this collab model above is not great? (people who give ratings are people who love movies, they don’t rarely give 1, but many high ratings. Whereas the predictions have many occassions with ratings over 5). Review the sigmoid usage. How can we do sigmoid transformation to the predictions? How does this sigmoid work? Why do we use the up limit of the range `5.5` instead of `5`?  Does adding sigmoid always improve the result?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/e/a/ea31571a3891e95cfc3ce1b1b52007dff4b7a94b_2_690x245.jpeg" alt="sigmoid-transform" width="690" height="245" srcset="https://forums.fast.ai/uploads/default/optimized/3X/e/a/ea31571a3891e95cfc3ce1b1b52007dff4b7a94b_2_690x245.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/e/a/ea31571a3891e95cfc3ce1b1b52007dff4b7a94b_2_1035x367.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/e/a/ea31571a3891e95cfc3ce1b1b52007dff4b7a94b_2_1380x490.jpeg 2x">

- What interesting things did Jeremy observe from the dataset? (some users like to give high ratings to all movies, some tend to dislike all movies). Can we add one bias value to both user and movie latent factors to explain this interesting observation? How to use the bias factors inside the collab model?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/b/b/bbb126f8323ca2c266f1c1867b57c7acb46f4bcd_2_690x282.jpeg" alt="user-tendency" width="690" height="282" srcset="https://forums.fast.ai/uploads/default/optimized/3X/b/b/bbb126f8323ca2c266f1c1867b57c7acb46f4bcd_2_690x282.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/b/b/bbb126f8323ca2c266f1c1867b57c7acb46f4bcd_2_1035x423.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/b/b/bbb126f8323ca2c266f1c1867b57c7acb46f4bcd_2_1380x564.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/8/8/88b82221d76673d397e4f1e605f2d80134e344c4_2_690x448.jpeg" alt="bias-factor" width="690" height="448" srcset="https://forums.fast.ai/uploads/default/optimized/3X/8/8/88b82221d76673d397e4f1e605f2d80134e344c4_2_690x448.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/8/8/88b82221d76673d397e4f1e605f2d80134e344c4_2_1035x672.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/8/8/88b82221d76673d397e4f1e605f2d80134e344c4_2_1380x896.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/0/d/0d68c8c1c93606101295312c8c9e96aaf69f308d_2_690x319.jpeg" alt="adding-biases-to-prediction" width="690" height="319" srcset="https://forums.fast.ai/uploads/default/optimized/3X/0/d/0d68c8c1c93606101295312c8c9e96aaf69f308d_2_690x319.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/0/d/0d68c8c1c93606101295312c8c9e96aaf69f308d_2_1035x478.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/0/d/0d68c8c1c93606101295312c8c9e96aaf69f308d_2_1380x638.jpeg 2x">

- Why did the upgraded model with bias get worse? (overfitting).

<img src="https://forums.fast.ai/uploads/default/optimized/3X/1/5/15be1b4b512651c13de6f325e98ab67837020696_2_690x268.jpeg" alt="overfitting-collab" width="690" height="268" srcset="https://forums.fast.ai/uploads/default/optimized/3X/1/5/15be1b4b512651c13de6f325e98ab67837020696_2_690x268.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/1/5/15be1b4b512651c13de6f325e98ab67837020696_2_1035x402.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/1/5/15be1b4b512651c13de6f325e98ab67837020696_2_1380x536.jpeg 2x">

## Weight decay

- What is weight decay and How does it help? How to understand weight decay in solving the problem of overfitting?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/b/2ba175a248170dc42de2cb2a3e49a8a3c1afa777_2_690x290.jpeg" alt="weight-decay" width="690" height="290" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/b/2ba175a248170dc42de2cb2a3e49a8a3c1afa777_2_690x290.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/b/2ba175a248170dc42de2cb2a3e49a8a3c1afa777_2_1035x435.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/b/2ba175a248170dc42de2cb2a3e49a8a3c1afa777_2_1380x580.jpeg 2x">

- How to actually use weight decay in fastai code? Does fastai have a good default for collaborative filtering like CV? How does Jeremy suggest to find the appropriate `wd` value for your own dataset?

<img src="https://forums.fast.ai/uploads/default/optimized/3X/2/d/2d240b5fcf7b6b92643497067176a390d41778d7_2_690x268.jpeg" alt="weight-decay2" width="690" height="268" srcset="https://forums.fast.ai/uploads/default/optimized/3X/2/d/2d240b5fcf7b6b92643497067176a390d41778d7_2_690x268.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/2/d/2d240b5fcf7b6b92643497067176a390d41778d7_2_1035x402.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/2/d/2d240b5fcf7b6b92643497067176a390d41778d7_2_1380x536.jpeg 2x">

<img src="https://forums.fast.ai/uploads/default/optimized/3X/e/7/e728e7640b79a3ead582d8feefe518e1e8b999d5_2_690x260.jpeg" alt="weight-decay3" width="690" height="260" srcset="https://forums.fast.ai/uploads/default/optimized/3X/e/7/e728e7640b79a3ead582d8feefe518e1e8b999d5_2_690x260.jpeg, https://forums.fast.ai/uploads/default/optimized/3X/e/7/e728e7640b79a3ead582d8feefe518e1e8b999d5_2_1035x390.jpeg 1.5x, https://forums.fast.ai/uploads/default/optimized/3X/e/7/e728e7640b79a3ead582d8feefe518e1e8b999d5_2_1380x520.jpeg 2x">

- What is regularization? What’s wrong when the weights having high values or low values? How does weight decay help balance?
- More questions: any other rules other than Jeremy’s rule of thumb on number of latent factors, and recommendation on average rating is viable only when there are many metadata.

