# Evaluation of nearest neighbor search algorithms

This repository contains a notebook for evaluating the difference in the number of Euclidean and splitting distance calculations performed during nearest neighbor searches using k-d trees.

Specifically, it compares the number of distance computations required when the Euclidean distance is calculated and the current best point is updated either during the downward traversal or during the upward traversal of the tree in a nearest neighbor search.

## Environment

To run the code, we will use [marimo](https://marimo.io/).

For a package manager, we will use [uv](https://docs.astral.sh/uv/) since it is tightly integrated with marimo.

**The remainder of these instructions assume a Unix-like shell environment (e.g., Linux, macOS, or WSL).**

To install uv, you can use the following command (taken from the uv homepage).

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Next, you will need to clone this repository.

```sh
git clone https://github.com/reidzemek/kd-tree-nn-benchmark.git
```

Finally, from the repository root, you can run the following command to install the dependencies using uv.

```sh
uv sync
```

## Adding the data

Due to the sensitivity of the data, is it not included here. You will need to copy the directory into the repository root from SharePoint.

## Running the code

To run the code, you need to start by activating the python virtual environment created for you by uv.

Again, in the repository root, you can run the following command.

```sh
source .venv/bin/activate
```

Now, still in the repository root, you can start marimo using the following command.

```sh
marimo edit notebook.py
```

Finally, opening the link in a browser will reveal the notebook where you can run each cell and observe the outputs in real time.
