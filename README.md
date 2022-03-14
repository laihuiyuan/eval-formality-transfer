# Human Judgement as a Compass to Navigate Automatic Metrics for Formality Transfer

This respository mainly contains two parts: [Evaluation Method](#start) and [Papers](#paper) published over the last three years in the ACL Anthology.

## <span id="start">Quick Start</span>
### Style Strength

- Train Classifier/Regressor
```
python classifier/bert_train.py \
       -model bart \
       -dataset xformal \
       -task cls \
       -lr 1e-5
```
- Evaluation
```
python eval_style.py \
       -dataset xformal
       -task cls
```

### Content Preservation
```
python eval_content.py source.txt output.txt referece.txt
```

### Fluency
- Train Language Model
```
python train_lm.py \
       -dataset xformal \
       -style 0
```
- Evaluation
```
python eval_fluency.py \
       -model bart \
       
```

## <span id="paper">Text Style Transfer</span>
- [Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer](https://aclanthology.org/2021.acl-short.62/). ACL 2021.
- [Unsupervised Aspect-Level Sentiment Controllable Style Transfer](https://anthology.aclweb.org/2020.aacl-main.33/). ACL 2021.
- [Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization](https://aclanthology.org/2021.acl-long.8/). ACL 2021.
- [Improving Formality Style Transfer with Context-Aware Rule Injection](https://aclanthology.org/2021.acl-long.124/). ACL 2021.
- [Style is NOT a single variable: Case Studies for Cross-Stylistic Language Understanding](https://aclanthology.org/2021.acl-long.185/). ACL 2021.
- [TextSETTR: Few-Shot Text Style Extraction and Tunable Targeted Restyling](https://aclanthology.org/2021.acl-long.293/). ACL 2021.
- [Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer](https://aclanthology.org/2021.acl-short.7/). ACL 2021.
- [NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer](https://aclanthology.org/2021.findings-acl.138/). ACL Findings 2021.
- [LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer](https://aclanthology.org/2021.findings-acl.344/). ACL Findings 2021.
- [Text Style Transfer: Leveraging a Style Classifier on Entangled Latent Representations](https://aclanthology.org/2021.repl4nlp-1.9/). RepL4NLP-2021.
- [Multi-Pair Text Style Transfer for Unbalanced Data via Task-Adaptive Meta-Learning](https://aclanthology.org/2021.metanlp-1.4/). METANLP 2021.
- [Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen](https://aclanthology.org/2020.acl-main.100/). ACL 2020.
- [Parallel Data Augmentation for Formality Style Transfer](https://aclanthology.org/2020.acl-main.294/). ACL 2020.
- [Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer](https://aclanthology.org/2020.acl-main.639/). ACL 2020.
- [Politeness Transfer: A Tag and Generate Approach](https://aclanthology.org/2020.acl-main.169/). ACL 2020.
- [Learning to Generate Multiple Style Transfer Outputs for an Input Sentence](https://aclanthology.org/2020.ngt-1.2/). NGT 2020.
- [Challenges in Emotion Style Transfer: An Exploration with a Lexical Substitution Pipeline](https://aclanthology.org/2020.socialnlp-1.6/). SOCIALNLP 2020.
- [Disentangled Representation Learning for Non-Parallel Text Style Transfer](https://aclanthology.org/P19-1041/). ACL 2019.
- [A Hierarchical Reinforced Sequence Operation Method for Unsupervised Text Style Transfer](https://aclanthology.org/P19-1482/). ACL 2019.
- [Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation](https://aclanthology.org/P19-1601/). ACL 2019.
- [Evaluating the Evaluation Metrics for Style Transfer: A Case Study in Multilingual Formality Transfer](https://aclanthology.org/2021.emnlp-main.100/). EMNLP 2021.
- [Transductive Learning for Unsupervised Text Style Transfer](https://aclanthology.org/2021.emnlp-main.195/). EMNLP 2021.
- [Generic resources are what you need: Style transfer tasks without task-specific parallel training data](https://aclanthology.org/2021.emnlp-main.349/). EMNLP 2021.
- [Collaborative Learning of Bidirectional Decoders for Unsupervised Text Style Transfer](https://aclanthology.org/2021.emnlp-main.729/). EMNLP 2021.
- [Exploring Non-Autoregressive Text Style Transfer](https://aclanthology.org/2021.emnlp-main.730/). EMNLP 2021.
- [Rethinking Sentiment Style Transfer](https://aclanthology.org/2021.findings-emnlp.135/). EMNLP Findings 2021.
- [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://aclanthology.org/2020.emnlp-main.55/). EMNLP 2020.
- [Generating similes effortlessly like a Pro: A Style Transfer Approach for Simile Generation](https://aclanthology.org/2020.emnlp-main.524/). EMNLP 2020.
- [DGST: a Dual-Generator Network for Text Style Transfer](https://aclanthology.org/2020.emnlp-main.578/). EMNLP 2020.
- [Unsupervised Text Style Transfer with Padded Masked Language Models](https://aclanthology.org/2020.emnlp-main.699/). EMNLP 2020.
- [StyleDGPT: Stylized Response Generation with Pre-trained Language Models](https://aclanthology.org/2020.findings-emnlp.140/). EMNLP Findings 2020.
- [Semi-supervised Formality Style Transfer using Language Model Discriminator and Mutual Information Maximization](https://aclanthology.org/2020.findings-emnlp.212/). EMNLP Findings 2020.
- [Contextual Text Style Transfer](https://aclanthology.org/2020.findings-emnlp.263/). EMNLP Findings 2020.
- [“Transforming” Delete, Retrieve, Generate Approach for Controlled Text Style Transfer](https://aclanthology.org/D19-1322/). EMNLP 2019.
- [Domain Adaptive Text Style Transfer](https://aclanthology.org/D19-1325/). EMNLP 2019.
- [Harnessing Pre-Trained Neural Networks with Rules for Formality Style Transfer](https://aclanthology.org/D19-1365/). EMNLP 2019.
- [Multiple Text Style Transfer by using Word-level Conditional Generative Adversarial Network with Two-Phase Training](https://aclanthology.org/D19-1366/). EMNLP 2019.
- [Semi-supervised Text Style Transfer: Cross Projection in Latent Space](https://aclanthology.org/D19-1499/). EMNLP 2019.
- [StylePTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer](https://aclanthology.org/2021.naacl-main.171/). NAACL 2021.
- [Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual Formality Style Transfer](https://aclanthology.org/2021.naacl-main.256/). NAACL 2021.
- [Multi-Style Transfer with Discriminative Feedback on Disjoint Corpus](https://aclanthology.org/2021.naacl-main.275/). NAACL 2021.
- [On Learning Text Style Transfer with Direct Rewards](https://aclanthology.org/2021.naacl-main.337/). NAACL 2021.
- [Evaluating Style Transfer for Text](https://aclanthology.org/N19-1049/). NAACL 2019.
- [Reinforcement Learning Based Text Style Transfer without Parallel Training Corpus](https://aclanthology.org/N19-1320/). NAACL 2019.
