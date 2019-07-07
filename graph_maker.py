from datetime import datetime as dt
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def load_data(path):
    return np.loadtxt(path, delimiter='\t', skiprows=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', type=str, required=True)

    # Graph options
    parser.add_argument('--xlabel', type=str, default='million steps')
    parser.add_argument('--ylabel', type=str, default='score')

    parser.add_argument('--filename', type=str, default='result.png')

    parser.add_argument('--xplim', type=int, default=100)
    parser.add_argument('--yplim', type=int, default=500)
    parser.add_argument('--xnlim', type=int, default=0)
    parser.add_argument('--ynlim', type=int, default=0)
    
    parser.add_argument('--legend-pos', type=str, default='upper left')

    args = parser.parse_args()

    results = load_data(args.file)

    avg_y = results[:, 1]
    med_y = results[:, 2]
    x = range(len(med_y))

    plt.figure(figsize=(5, 4), dpi=80)

    plt.plot(x, med_y, label='median', linewidth=1)
    plt.plot(x, avg_y, label='average', linewidth=1)

    plt.legend(loc=args.legend_pos, fontsize=8)
    plt.xlim(args.xnlim, args.xplim)
    plt.ylim(args.ynlim, args.yplim)

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)

    plt.savefig(args.filename)
    plt.show()
