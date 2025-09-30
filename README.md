# Latent Dirichlet Allocation (LDA)

An implementation of Latent Dirichlet Allocation, from scratch, demonstrated with two examples:  

- **BBC News Topic Identification** (`lda_bbc_news.py`)**:** Categorizes BBC News articles based on the topic.
- **Cell Clustering on SpatialLIBD Data** (lda_spatialLIBD.py)**:** Clusters cells based on gene expression from the human dorsolateral prefrontal cortex (DLPFC) using the SpatialLIBD dataset.

### What is (LDA)?
Latent Dirichlet Allocation is a generative probabilistic graphical model used for topic modelling. Every document in the dataset (spots, in the case of spatial transcriptomics), is represented as a mixture of multiple latent (hidden) topics. The latent topics are represented as distributions over words (genes, in the case of spatial transcriptomics). The model is considered generative since it learns distributions that allow one to generate documents of certain topics.

Firstly, $N$, the number of words in a document, is chosen from a Poisson distribution parameterized by $\xi$. Next, $\theta$, the distribution over topic mixtures, is chosen from a Dirichlet distribution parameterized by $\alpha$. For each word (of $N$), a topic ($z_n$) is chosen from a multinomial distribution parameterized by $\theta$, and a word ($w_n$) is chosen from a distibution, conditioned on $z_n$ and parameterized by $\beta$. $\beta$ is a matrix of size number of topics (k) by length of vocab (V), and $\beta_{ij}$ is the probability of the word $w_j$, given the topic $z_i$.

In most implementations, it is assumed that the number of words in a document, N, is predetermined and doesn't need to be chosen from a Poisson distribution parameterized by $\xi$. 

To carry out topic modeling, one must compute the probability of the topic mixture distribution ($\theta$) and the word topics (z) given the parameters $\alpha$ and $\beta$, and the words (w): $p(\theta, z \mid w, \alpha, \beta)$. This tells us the probabilty of a document having a certain topic mixture, given $\alpha$ and $\beta$. This is referred to as the posterior distribution. Computing this probability is intractable so a simplification is required. Using the free variational parameters, $\gamma$ and $\phi$, we define a variational distribution $q(\theta, z \mid \gamma, \phi)$ that acts as a proxy for the true posterior distribution. Variational inference is used to obtain the values of $\gamma$ and $\phi$ that minimize the KL divergence between the variational distribution, $q(\theta, z \mid \gamma, \phi)$, and the true posterior, $p(\theta, z \mid w, \alpha, \beta)$.

Similar to the Expectation-Maximization algorithm, Variational Inference can also be viewed as having an E-step and M-step. In the E-step, the variational parameters $\gamma$ and $\phi$ are optimized, and in the M-step, the parameters $\alpha$ and $\beta$ are optimized. The formulas for the computations are provided in the appendix of the LDA paper [1]. The E-step and M-step are repeated continuously, until the increase in the Evidence Lower Bound (ELBO) has converged. The formula for the ELBO computation is also provided in the appendix of the LDA paper [1].

## Results
### BBC News Dataset
The BBC News Dataset [2] consists of 2225 articles that fall under 5 different topics: business, entertainment, politics, sport and tech. We convert the corpus into a Bag-of-Words representation, before applying LDA. After applying the argmax operation to the $\gamma$ matrix, we get the most prevalent latent topic of each document. Tha matrix below shows the predicted latent topic (column) of each real topic (row), for all documents in the dataset:

```
 T0  T2  T3  T4  T5
[472   2   0  12  24] business
[  4 355   3  15   9] entertainment
[ 11   4   1  17 384] politics
[  0  35 427   0  49] sport
[  9  12   9 367   4] tech
```

Here are the values as percentages of documents that fall under each real topic:

```
  T0   T2   T3   T4   T5
[0.93 0.00 0.00 0.02 0.05] business
[0.01 0.92 0.01 0.04 0.02] entertainment
[0.03 0.01 0.00 0.04 0.92] politics
[0.00 0.07 0.84 0.00 0.10] sport
[0.02 0.03 0.02 0.92 0.01] tech
```

93% of business articles fall under latent topic 1, 92% of entertainment articles fall under latent topic 2, 84% of sport articles fall under latent topic 3, 92% of tech articles fall under latent topic 4 and 92% of politics articles fall under latent topic 5. This shows that our LDA implementation is able to capture the main themes in the BBC News dataset, with the majority of articles correctly clustered under their respective latent topics. Unfortunately, minor overlaps exists between the latent topics, which might indicate similarities between articles of different topics. For example, 7% of sports articles were predicted as entertainment (latent topic 2). Sport is a form of entertainment so this might be expected.

### Cell Clustering on SpatialLIBD Data
Here, I attempt to implement the STdeconvolve algorithm introduced in the paper "Reference-free cell type deconvolution of multi-cellular pixel-resolution spatially resolved transcriptomics data" (Miller et al.). I've chosen the Spatial LIBD dataset [3] which contains the gene expressions of multiple spots located in the human dorsolateral prefrontal cortex (DLPFC). Each spot contains multiple cells, and we use my LDA implementation to cluster the spots according to gene expression, to try and identify the dominant cell type in that spot. The DLPFC tissue is separated into 7 sections: layers 1 through 6 and white matter. I was intially under the impression that each layer was predominantly made of one cell but it turns out that this is not the case. Each spot in the dataset has a layer label but there are no labels for the actual cell types. For this reason, I was unable to carry out actual cell type deconvolution, but instead we attempt to cluster according to cortical layer.
<div align="center">
  <table>
    <tr>
      <img width="400" alt="dlpfc_ground_truth" src="https://github.com/user-attachments/assets/e1cb64c2-2a74-4a7b-b215-653f8e154bb2" />
      <img width="400" alt="dlpfc_7_topics_LDA" src="https://github.com/user-attachments/assets/6a6e65ec-04ec-45ef-a1d1-dbbd5a39a6fb" />
      <img width="400" alt="dlpfc_9_topics_lda" src="https://github.com/user-attachments/assets/e9e74ff6-6e01-481f-92a0-bae5fca66251" />
      <img width="400" alt="dlpfc_15_topics_lda" src="https://github.com/user-attachments/assets/9af6ae87-08e5-43eb-8884-c247897044e8" />
    </tr>
  </table>
</div>

<div align="center">
<i><b>Figure 1:</b> (Left to Right, Top to Bottom) Ground Truth, 7 Latent Topic LDA, 9 Latent Topic LDA, 15 Latent Topic LDA</i>
  <br>
<i>(Note: Colors are arbitrary and only indicate clusters.)</i>
</div>
<br>

I initially ran LDA with 7 latent topics and, as shown in the top right of figure 1, although it was able to capture a similar structure to the ground truth layer labels (top left), it is unable to accurately distinguish the different layers. For example, latent topic 0 extends over many layers, convering layer 6, 5, and 4 of the ground truth. As shown in the table below, 68% of Layer 6, 78% of Layer 5 and 20% of Layer 4 were all predicted to be latent topic 0.

```
7 Latent Topic LDA Predictions Table
  T0   T1   T2   T3   T4   T5   T6 
[0.01 0.16 0.65 0.16 0.00 0.01 0.01] Layer1
[0.00 0.75 0.21 0.02 0.00 0.00 0.00] Layer2
[0.01 0.24 0.29 0.00 0.03 0.00 0.43] Layer3
[0.20 0.00 0.21 0.00 0.16 0.00 0.43] Layer4
[0.78 0.00 0.09 0.00 0.11 0.00 0.02] Layer5
[0.68 0.00 0.14 0.04 0.01 0.13 0.00] Layer6
[0.02 0.00 0.01 0.01 0.00 0.96 0.00] WM
```

Here are a few top genes of latent topic 0 (PanglaoDB was used to identify potential cells): "SNAP25" (marker gene for Neurons), "TMSB10" (expressed in dendritic cells), TUBA1B, NRGN (expressed in smooth muscle cells). 

A few top genes from latent topic 5 (96% of White Matter was classified as topic 5): "MBP" (marker gene for oligodendrocytes), "PLP1" (oligodendrocytes), "GFAP" (marker gene for astrocytes).

I hypothesized that adding more latent topics would allow for better distinction between layers, but even with 9 latent topcs (bottom left of figure 1), the problem persisted: 54% of Layer 6 and 39% of Layer 5 are being predicted as latent topic 0, 69% of layer 4 and 46% of layer 3 are being predicted as topic 6, etc.

```
9 Latent Topic LDA Predictions Table
  T0   T1   T2   T3   T4   T5   T6   T7   T8
[0.03 0.19 0.01 0.00 0.00 0.08 0.05 0.45 0.19] Layer1
[0.00 0.06 0.00 0.00 0.00 0.01 0.03 0.10 0.80] Layer2
[0.06 0.01 0.00 0.00 0.00 0.00 0.46 0.22 0.25] Layer3
[0.08 0.00 0.00 0.10 0.00 0.00 0.69 0.12 0.00] Layer4
[0.39 0.00 0.00 0.39 0.00 0.00 0.13 0.09 0.00] Layer5
[0.54 0.02 0.18 0.09 0.00 0.04 0.00 0.12 0.00] Layer6
[0.00 0.00 0.98 0.00 0.00 0.01 0.00 0.00 0.00] WM
```

Finally, I tried 15 latent topics (bottom right of figure 1). The problem of overlapping latent topics persists. We are still not able to accurately separate the spots according to their cortical layers. For example, 22% of layer 6 is assigned to the same latent topic occupied by 98% of white matter. The predictions are also a lot noiser now, with quite a few latent topics not being assigned to very many spots.

```
15 Latent Topic LDA Predictions Table
  T0   T1   T2   T3   T4   T5   T6   T7   T8   T9   T10  T11  T12  T13  T14
[0.03 0.03 0.00 0.12 0.00 0.00 0.46 0.01 0.00 0.01 0.01 0.03 0.01 0.09 0.20] Layer1
[0.00 0.06 0.00 0.04 0.01 0.00 0.17 0.00 0.00 0.00 0.01 0.00 0.00 0.68 0.00] Layer2
[0.01 0.36 0.00 0.01 0.02 0.03 0.35 0.01 0.00 0.00 0.00 0.11 0.00 0.12 0.00] Layer3
[0.02 0.14 0.00 0.00 0.05 0.12 0.30 0.01 0.00 0.00 0.00 0.32 0.03 0.00 0.00] Layer4
[0.01 0.02 0.00 0.00 0.02 0.32 0.21 0.11 0.00 0.00 0.00 0.05 0.26 0.00 0.00] Layer5
[0.00 0.08 0.00 0.01 0.01 0.00 0.11 0.29 0.00 0.22 0.01 0.00 0.22 0.00 0.04] Layer6
[0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.98 0.00 0.00 0.01 0.00 0.01] WM
```

## A Few Implementational Details:
Let $k$ be the number of topics, $M$ the number of documents and $V$ the length of the vocab. $\alpha$ and $\beta$ are global level parameters, shared across all documents. They are of shape (k,) and (k, V), respectively. $\gamma$ and $\phi$ are document-level parameters. They are of shape (M,k) and (M, number of words in the document, k), respectively. As one can imagine, different documents are not guaranteed to have the same length so, in the unvectorized implementation, $\phi$ takes the form of a python list (of length M) filled with numpy arrays of shape (number of words in the document, k). Since documents can repeat a word multiple times, and the corresponding values in $\phi$ should be identical, the shape of the individual numpy arrays in $\phi$ could be set to (number of unique words in document, k). The appearance count of the unique words can be multipied to the value in $\phi$, whenever necessary. The unique words and associated counts for each document can easily be extracted since a Bag-of-Words is used to represent the whole corpus.

Evidently, the above formalization is not vectorized since $\phi$ is a python list of length M, requiring us to loop over documents. In our second LDA implementation in `lda.py`, we attempt to eliminate the loop over documents by coverting the Bag-of-Words representation of the corpus into 3 arrays: the row indices of non-zero values (selects documents), the column indices of non-zero values (selects words) and the corresponding values themselves (count of word in document). These three arrays allow us to eradicate loops over documents using vectorized numpy operations. Certain computations require the use of `np.bincount`. It allows us to sum variable lengths of the product of the column and value arrays, based on which document they are part of (according to the row array).

## References:

<b>[1]</b> Miller, B. F., Huang, F., Atta, L., Sahoo, A., & Fan, J. (2022). Reference-free cell type deconvolution of multi-cellular pixel-resolution spatially resolved transcriptomics data. Nature Communications, 13, 2339. https://doi.org/10.1038/s41467-022-30033-z

<b>[2]</b> Gültekin, H. (n.d.). BBC News Archive [Data set]. Kaggle. https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive

<b>[3]</b> Pardo, B., Spangler, A., Weber, L. M., Hicks, S. C., Jaffe, A. E., Martinowich, K., Maynard, K. R., & Collado-Torres, L. (2022). spatialLIBD: an R/Bioconductor package to visualize spatially-resolved transcriptomics data [Data package]. BMC Genomics. https://doi.org/10.1186/s12864-022-08601-w
