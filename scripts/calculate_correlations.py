import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def spearman_corr_with_pvalues(df, score_cols):
    cols = df.columns
    var_cols = [col for col in cols if col not in score_cols]
    corr = pd.DataFrame(index=var_cols, columns=score_cols, dtype=float)
    pvals = pd.DataFrame(index=var_cols, columns=score_cols, dtype=float)

    for i, c1 in enumerate(var_cols):
        for j, c2 in enumerate(score_cols):
            rho, p = spearmanr(df[c1], df[c2], nan_policy="omit")
            corr.loc[c1, c2] = rho
            pvals.loc[c1, c2] = p

    return corr, pvals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Spearman correlation matrix from a numerical dataframe"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input CSV file containing numerical values",
    )
    parser.add_argument(
        "--corr",
        "-c",
        required=True,
        help="Path to output CSV file to save Spearman correlation matrix",
    )
    parser.add_argument(
        "--pvals",
        "-p",
        required=True,
        help="Path to output CSV file to save Spearman correlation p-value matrix",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataframe
    df = pd.read_csv(args.input)

    # Compute Spearman correlation
    corr, pvals = spearman_corr_with_pvalues(
        df, score_cols=["edge_jaccard_distance", "graph_edit_distance"]
    )

    # Save to file
    corr.to_csv(args.corr)
    pvals.to_csv(args.pvals)


if __name__ == "__main__":
    main()
