# Robustness of Generalized Learning Vector Quantization Models against Adversarial Attacks
[Sascha Saralajew](https://scholar.google.com/citations?user=YTi93_0AAAAJ&hl=de), Lars Holdijk, Maike Rees, [Thomas Villmann](https://scholar.google.com/citations?user=K14cpD8AAAAJ&hl=de)

_Adversarial attacks and the development of (deep) neural networks robust against them are currently two widely researched topics. The robustness of Learning Vector Quantization (LVQ) models against adversarial attacks has however not yet been studied to the same extend. We therefore present an extensive evaluation of three LVQ models: Generalized LVQ, Generalized Matrix LVQ and Generalized Tangent LVQ. The evaluation suggests that both Generalized LVQ and Generalized Tangent LVQ have a high base robustness, on par with the current state-of-the-art in robust neural network methods. In contrast to this, Generalized Matrix LVQ shows a high susceptibility to adversarial attacks, scoring consistently behind all other models. Additionally, our numerical evaluation indicates that increasing the number of prototypes per class improves the robustness of the models._

<https://arxiv.org/abs/1902.00577>


###Using the models
Models can be retrieved either as standalone tensorflow sessions or through the [foolbox zoo](https://foolbox.readthedocs.io/en/latest/user/zoo.html). To retrieve the tensorflow session the `tensorflow_session()` function can be called from the models respective module.


