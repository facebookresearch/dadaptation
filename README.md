# D-Adaptation

Learning rate free learning for SGD, AdaGrad and Adam! 

*by Aaron Defazio and Konstantin Mishchenko [(Arxiv)](https://arxiv.org/abs/2301.07733)*

``` pip install dadaptation ```

**NEW V3.0 release uses an improved algorithm that may give different results from past versions. The old version is still availiable under experimental/d_adapt_adam_preprint.**

## Details

The provided Pytorch Optimizer classes are drop-in replacements, either copy into your project or use via pip with dadaptation.DAdaptSGD,  dadaptation.DAdaptAdam or dadaptation.DAdaptAdaGrad.

 - **Set the LR parameter to 1.0**. This parameter is not ignored. Setting it larger to smaller will directly scale up or down the D-Adapted learning rate estimate.
 - **Use the same learning rate scheduler you would normally use on the problem.**
 - The Adam variant supports AdamW style weight decay, just set decouple=True. It is not turned on by default, so if you are replacing your adam implementation, make sure you use decoupled if necessary.
 - It may be necessary to use larger weight decay than you would normally use, try a factor of 2 or 4 bigger if you see overfitting. D-Adaptation uses larger learning rates than people typically hand-choose, in some cases that requires more decay.
 - Use the log_every setting to see the learning rate being used (d*lr) and the current D bound.
 - Only the AdaGrad version supports sparse gradients. It does not adapt as efficiently as the other variants and should be considered experimental.
 - Parameter-group level LR values are not fully supported. The optimizer only supports setting zero LR for some groups in order to do fine-tuning on parts of a model.
 
## Change Log

### Version 3.0
 - Major improvements to DAdaptAdam, improving the performance particularly on Transformer models. This variant may behave differently in practice. The old version is availiable under experimental/d_adapt_adam_preprint if you wish to continue to use it.
 - The IP variant is now the main variant of the method.
 - Added Lion. This is highly experimental. Feedback on it's performance is welcome.

### Version 2.0
 - Added Adan - should still be considered experimental.
 - Added support for PyTorch's Fully Sharded Data Parallel. 
 - Improved support of edge cases such as learning rate zero.
 - Improved logging - uses Python logging rather than print statements

 # Experimental results

![vision](figures/dadapt_cifar.png)
![vision](figures/dadapt_cifar100.png)
![vision](figures/dadapt_imagenet.png)
![vision](figures/dadapt_vit.png)
![vision](figures/dadapt_lstm.png)
![vision](figures/dadapt_roberta.png)
![vision](figures/dadapt_gpt.png)
![vision](figures/dadapt_fastmri.png)
![vision](figures/dadapt_detectron.png)
![vision](figures/dadapt_dlrm.png)

# License
See the [License file](/LICENSE).
