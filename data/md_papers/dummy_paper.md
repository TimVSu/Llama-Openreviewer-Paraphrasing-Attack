# Title: SyntheGene: Cross-modal Transformers for De Novo Protein Folding Prediction

## Abstract
Predicting the tertiary structure of proteins from primary amino acid sequences remains a grand challenge in computational biology. We introduce **SyntheGene**, a cross-modal transformer architecture that integrates genomic evolutionary data with physicochemical constraints. By leveraging a multi-head attention mechanism optimized for long-range residue interactions, SyntheGene achieves a Global Distance Test (GDT) score of 88.4. Our findings suggest that latent representations of protein "grammar" can bypass the need for traditional template-based modeling.

## 1. Introduction
The biological function of a protein is intrinsically linked to its spatial fold. Traditional methods, such as Cryo-EM and X-ray crystallography, are resource-intensive and slow. The advent of deep learning has accelerated structure prediction, yet existing models struggle with "orphan" proteins that lack homologous sequences. This paper explores how self-supervised pre-training on large-scale proteomic databases can fill these gaps.



## 2. Methodology
The SyntheGene pipeline consists of a sequential encoder paired with a spatial refiner. To quantify the affinity between non-adjacent residues, we define the Interaction Potential ($I_{p}$) between residues $i$ and $j$ as:

$$I_{p}(i, j) = \frac{\exp(\text{Attn}(i, j))}{\sum_{k=1}^{n} \text{Dist}(i, k)}$$

Where $\text{Dist}(i, k)$ represents the Euclidean distance in the predicted 3D manifold. This penalty function discourages physically impossible overlaps while favoring high-attention couplings.



## 3. Experiments
Validation was performed using the CASP15 dataset. Unlike traditional LLMs, SyntheGene was evaluated on its ability to maintain stereochemical validity.

| Metric | Result |
| :--- | :--- |
| **Root Mean Square Deviation (RMSD)** | 2.1 Å |
| **Inference Throughput** | 450 residues/sec |
| **GDT-TS Score** | 88.4 |

The model successfully folded 14 novel sequences previously unclassified in the Protein Data Bank (PDB), confirming the efficacy of our spatial refiner.

## 4. Conclusion
SyntheGene demonstrates that Transformer-based architectures are uniquely suited for the "language" of proteomics. Our results pave the way for rapid, in-silico drug discovery and enzyme engineering. Subsequent iterations will investigate the impact of stochastic noise on folding stability.
