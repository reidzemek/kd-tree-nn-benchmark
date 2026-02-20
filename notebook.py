import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import os
    from pypcd4 import PointCloud
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    import marimo as mo

    return Path, PointCloud, mo, np, os, plt


@app.cell
def _():
    import kdtree

    return (kdtree,)


@app.cell
def _(Path, PointCloud, kdtree, os):
    root = "data"

    # Source point cloud
    P = PointCloud.from_path(os.path.join(root, "source.pcd")).numpy(("x", "y", "z"))

    n_euclideans_ups = []
    n_euclideans_downs = []
    n_splits_ups = []
    n_splits_downs = []

    for target_path in Path(root).glob("target*.pcd"):

        # Target point cloud
        Q = PointCloud.from_path(target_path).numpy(("x", "y", "z"))

        # Construct the k-d tree
        Q_tree = kdtree.build(Q)

        # Search the tree and record the number of euclidean and split distance calculations
        # EUCLIDEAN DISTANCE CALCULATION ON THE DOWNWARD PASS
        _, [n_euclideans_down, n_splits_down] = kdtree.nn_search(Q_tree, P, down_eval=True)

        # Search the tree and record the number of euclidean and split distance calculations
        # EUCLIDEAN DISTANCE CALCULATION ON THE UPWARD PASS
        _, [n_euclideans_up, n_splits_up] = kdtree.nn_search(Q_tree, P, down_eval=False)

        n_euclideans_ups.append(sum(n_euclideans_up))
        n_euclideans_downs.append(sum(n_euclideans_down))

        n_splits_ups.append(sum(n_splits_up))
        n_splits_downs.append(sum(n_splits_down))

        n_euclideans_diffs = [a - b for a, b in zip(n_euclideans_ups,n_euclideans_downs)]
        n_splits_diffs = [a - b for a, b in zip(n_splits_ups, n_splits_downs)]
    return (
        n_euclideans_diffs,
        n_euclideans_downs,
        n_euclideans_ups,
        n_splits_diffs,
    )


@app.cell
def _(n_euclideans_downs, n_euclideans_ups, np, plt):
    x = np.arange(len(n_euclideans_ups))
    width = 0.4

    plt.bar(x - width/2, n_euclideans_ups, width, label="upward pass")
    plt.bar(x + width/2, n_euclideans_downs, width, label="downward pass")

    plt.legend()
    plt.ylabel("total # of euclidean distance calculations")
    plt.xlabel("target ID")
    plt.title("Evaluation of nearest neighbor search algorithms")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo, n_euclideans_diffs, n_splits_diffs):
    mo.md(rf"""
    ### Conclusions<br>
    **Difference in the total number of Euclidean distance calculations for nearest neighbor searches across 10 target point clouds:**<br>
    {n_euclideans_diffs}<br>
    **Difference in the total number of splitting distance calculations for nearest neighbor searches across 10 target point clouds:**<br>
    {n_splits_diffs}

    Though subtle, computing the Euclidean distance and updating the best point on the downward pass leads to additional
    pruning of the tree and a reduction in the total number of Euclidean and splitting distance calculations during tree traversal of the
    nearest neighbor search.
    """)
    return


if __name__ == "__main__":
    app.run()
