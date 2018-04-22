# Estimating the Information Gap between Textual and Graphical Representations
Investigating the discrepency of knowledge transfer conveyed by graphical resp. textual representations by using deep learning methods.

This repository comprises the implementations enclosed to the Master thesis of Christian Henning with the title:

**Estimating the Information Gap between Textual and Graphical Representations**
*(submitted on January 6th, 2017)*

Note, the labeling scheme has been changed for CMI (Cross-modal Mutual Information) since the submission of the thesis. See below.

More precisely, the repository comprises implementations for the *SimpleWiki* dataset retrieval, the *bbcAnnotator* and *wikiAnnotator* as well as the Tensorflow implementations of the designed deep neural networks and their evaluation scripts.

To retrieve a copy of the master thesis, please contact <henningc@ethz.ch>. The two publications that emerged from this thesis are cited below.

## Simplification of the annotation scheme

As described in section 4.2.3.1. of the thesis, the provided annotation scheme has been suffering from a complex, unintuitive labeling for CMI. The described annotation scheme in section 4.2.1. of the thesis had too many labels, leading to classes with only a small number of representatives (label sparsity). 

Therefore, the current implementation uses a simplified CMI annotation scheme. Label 1, 6 and 7 have been omitted. The remaining labels have been subject to a permutation, such that the MI classification problem can also be considered as a regression problem. Refer to the following table for details:

| *Originl Label*   | 0 | 5 | 4 | 3 | 2 |
| *New Label*       | 0 | 1 | 2 | 3 | 4 |

The made changes can easily be reverted, if one wishes to use the original annotation scheme. New input-data with the preferred labeling scheme would have to be provided for the neural nets and their config files would have to be adjusted.

## How to navigate through the repository

The following subsections will each give a short description about a folder contained in this repository, to clarify the overall structure.

Executable scripts usually allow the user to retrieve usage information by entering:

```
python3 SCRIPT_NAME.py --help
```

Additional to the information provided in this file and through the code documentation, major folders (containing not self-explanatory parts) will contain their own README file.

### annotations

The folder contains scripts to generate input samples for the *classifier*. Therefore, the annotations retrieved via the two annotation tools (*bbcAnnotator* and *wikiAnnotator*) has to be known. 
In addition, the folder contains a script to heuristically generate samples from the [MS Coco] dataset.

### autoencoder

The folder contains the complete implementation of the *autoencoder* model. The model is build on top of a modified version of the [Neural Image Captioner]. The folder comprises training and evaluation scripts as well as scripts to convert and preprocess samples into binary TFRecord files, used to generate input batches.

### bbcAnnotator

The tool to annotate articles from the [BBC News Corpora]. The tool is implemented in Java. An executable JAR should be generated from the source files using the included Eclipse project. Note, the JAR will comprise the dataset, which is currently contained in the folder as well.

### classifier

The folder contains the complete implementation of the *classifier* model. The model is build on top of a modified version of the [Neural Image Captioner]. Additionally, the encoding structure is taken from the autoencoder. The folder comprises training and evaluation scripts as well as scripts to convert and preprocess samples into binary TFRecord files, used to generate input batches. Furthermore, an extraction script is provided, that allows to extract hidden embeddings (i.e. article embeddings) for samples contained in provided TFRecord files. This extraction script has been used to retrieve feature vectors for comparison systems (i.e. a multiclass SVM) in the thesis.

### svm

Contains the implementation of a comparison system, namely a multiclass SVM. Feature vectors and results are contained as well.

### wikiAnnotator

The tool to annotate articles from the newly generated *SimpleWiki* dataset. The tool is implemented in Java. An executable JAR should be generated from the source files using the included Eclipse project. 

### wikiDataset

Contains a Web crawler for the newly generated *SimpleWiki* dataset. Additionally, the dataset (without images and image meta files) is contained by the folder as well. However, this should be deleted when publishing the repository due to unknown copyright status.

### word2vec

A Word2Vec implementation to generate valid initializazions for the *autoencoder* is provided by this folder. The model is build on top of a modified version of the [Neural Image Captioner]. Furthermore, the model is inspired by this [Word2Vec Tutorial].

## References

Christian Henning und Ralph Ewerth. [Estimating the Information Gap between Textual and Visual Representations](https://dl.acm.org/citation.cfm?doid=3078971.3078991). *Proceedings of the 2017 ACM on International Conference on Multimedia Retrieval*, 14 - 22, 2017 (Best Multimodal Paper Award)

Christian Henning und Ralph Ewerth. [Estimating the information gap between textual and visual representations](https://link.springer.com/article/10.1007\%2Fs13735-017-0142-y). *International Journal of Multimedia Information Retrieval*, Volume 7, Issue 1 (Special Issue: Top Papers of ACM ICMR 2017), 43 - 56, 2018

[MS Coco]: http://mscoco.org/dataset/#overview
[Neural Image Captioner]: https://github.com/tensorflow/models/tree/master/im2txt
[BBC News Corpora]: http://homepages.inf.ed.ac.uk/s0677528/data.html
[Word2Vec Tutorial]: https://www.tensorflow.org/tutorials/word2vec/
