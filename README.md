# Robustness of Generalized Learning Vector Quantization Models against Adversarial Attacks
[Sascha Saralajew](https://scholar.google.com/citations?user=YTi93_0AAAAJ&hl=de), [Lars Holdijk](https://github.com/LarsHoldijk/), [Maike Rees](https://github.com/MaikeRees/), [Thomas Villmann](https://scholar.google.com/citations?user=K14cpD8AAAAJ&hl=de)

In our paper we evaluated the robustness of LVQ (Learning Vector Quantization) models against adversarial attacks on MNIST.
This repository contains all the models (except the Madry model) that we used for the evaluation. The models are constructed in tensorflow and saved as pb files. We provide methods to read these files and to convert them to foolbox models with the foolbox zoo.  

### Abstract
_Adversarial attacks and the development of (deep) neural networks robust against them are currently two widely researched topics. The robustness of Learning Vector Quantization (LVQ) models against adversarial attacks has however not yet been studied to the same extend. We therefore present an extensive evaluation of three LVQ models: Generalized LVQ, Generalized Matrix LVQ and Generalized Tangent LVQ. The evaluation suggests that both Generalized LVQ and Generalized Tangent LVQ have a high base robustness, on par with the current state-of-the-art in robust neural network methods. In contrast to this, Generalized Matrix LVQ shows a high susceptibility to adversarial attacks, scoring consistently behind all other models. Additionally, our numerical evaluation indicates that increasing the number of prototypes per class improves the robustness of the models._

<https://arxiv.org/abs/1902.00577>


## Using the models
Models can be retrieved either as standalone **tensorflow graph** or through the **[foolbox zoo](https://foolbox.readthedocs.io/en/latest/user/zoo.html)**. 

To retrieve the tensorflow graph the `tensorflow_session()` function can be called from the models respective module. In the same method call, the input and output tensors of the graph are returned as well.

To use the models through the foolbox zoo it is required to specify the correct module name (file name) when calling `zoo.get_model()`. This option is only available for foolbox versions `> 1.8.0`.

## About the models
All implementation details and evaluation results can be found in the paper. We give the # of prototypes and clean accuracy as well as the obtained worst-case-robustness scores for the models here briefly. We measure the robustness as the worst case median adversarial distance (MAD) and the threshold accuracy (TACC). For details on these metrics, see the paper. Higher scores mean higher robustness of the model. The measures here are included for the reader that does not want to jump back and forth between the code and the paper when using our implementation. 

| name | cnn | glvq | glvq_large | gmlvq | gmlvq_large | gtlvq | gtlvq_large |
| ---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| # prototypes| - | 1 | 128 | 1 | 49 | 1 | 10 |
| clean accuracy | 99% | 83%  | 95% | 88% | 93% | 95% | 97% |
| L2 MAD / TACC| 1.5 / 50% | 1.5 / 49% | 2.1 / 68% | 0.5 / 3% | 0.6 / 3% | 2.1 / 68% | **2.2 / 77%** |
| Linf MAD / TACC| 0.12 / 0% | 0.11 / 2% | 0.19 / 5% | 0.03 / 0% | 0.04 / 0% | 0.17 / 3% | 0.19 / 4% |
| L0 MAD / TACC| 19 / 73% | 22 / 64% | 32 / 79% | 3 / 6% | 6 / 18% | 34 / 80% |**35 / 85%** |



