# LorIndex

This repository contains the experiment code for the LorIndex algorithm, benchmarked on the [ANN-benchmarks](https://github.com/erikbern/ann-benchmarks/) framework for approximate nearest neighbor (ANN) search.

LorIndex combines low-rank matrix factorization with clustering strategies to improve the accuracy and efficiency of approximate nearest neighbor search.

This repository includes the algorithm implementation, benchmarking scripts, and experiment results.

-----------

**Requirements**:
- Python 3.10 or newer
- Docker
- For GPU experiments, an NVIDIA GPU is required

**Installation**:

`python3 -m pip install -r requirements.txt`

**Usage**:

To build an algorithm, run e.g. `python3 install.py --algorithm lorann`.

To build all algorithms, run

```sh
for algo in faiss faiss_gpu glass hnswlib lorann lorann_gpu mrpt pynndescent qsg_ngt raft scann; do
  python3 install.py --algorithm $algo
done
```

To run an algorithm for e.g. the data set _fashion-mnist-784-euclidean_, run

`python3 run.py --dataset fashion-mnist-784-euclidean --algorithm lorann --count 100 --parallelism 6`

To plot results for the data set, run

`python3 plot.py --dataset fashion-mnist-784-euclidean --count 100 --y-scale log`

For a list of all the data sets, refer to the end of the file [ann_benchmarks/datasets.py](ann_benchmarks/datasets.py).
