## 描述

对于论文[Convolutional neural network based multiple-rate compressive sensing for massive
MIMO CSI feedback: Design, simulation, and analysis](https://arxiv.org/abs/1906.06007)网络结构的实现
只实现了其中CsiNet+的部分，对于论文的重点SM-CsiNet+和PM-CsiNet+可能会在以后实现。

## 与CsiNet的对比

* 使用更大的卷积核，其实最主要的还是追求更大的感受野（尤其是在outdoor场景和高CR的情况下，需要更多的全局信息）
* 移除了decoder后面的卷积层，因为RefineNet的输出结果足够恢复CSI，加上一层卷积层反而会是结果更差（作者是这样解释的，并没有做消融实验）

## 参考文献

[1]C. Wen, W. Shih and S. Jin, “Deep Learning for Massive MIMO CSI
Feedback,” IEEE Wireless Communications Letters, vol. 7, no. 5, pp.
748-751, Oct. 2018
[2]J. Guo, C.-K. Wen, S. Jin, and G. Y. Li, “Convolutional neural network based multiple-rate compressive sensing for massive
MIMO CSI feedback: Design, simulation, and analysis,” arXiv preprint
arXiv:1906.06007, 2019