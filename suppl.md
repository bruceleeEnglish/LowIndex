# Supplementary Notes

## On Index Build Time and Index Storage Size

During our experiments, we observed unexplained negative values in the index build time and index storage size metrics for certain algorithms. After investigation, we attribute this issue to the internal measurement mechanism of the ANN-benchmarks framework itself, rather than to the algorithms being evaluated.

Due to this limitation, the index build time and index storage size results reported for the Fashion-MNIST and MNIST datasets were kept consistent, as the raw measurements from the framework may not be fully reliable for these metrics.

**It is important to note that the primary evaluation metrics — QPS (Queries Per Second) and Recall — are not affected by this issue and remain reliable.** The vector search performance data presented in our main results accurately reflects the true behavior of each algorithm.
