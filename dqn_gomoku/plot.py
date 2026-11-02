"""把 train.csv 画成 2x2 的训练曲线图。"""
import argparse
import csv

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description='画出 DQN 五子棋训练曲线')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    rows = []
    with open(args.csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    steps = [int(r['step']) for r in rows]
    vs_random = [float(r['vs_random']) for r in rows]
    vs_heuristic = [float(r['vs_heuristic']) for r in rows]
    vs_prev = [float(r['vs_prev']) for r in rows]
    avg_ep_len = [float(r['avg_ep_len']) for r in rows]
    avg_loss = [float(r['avg_loss']) for r in rows]
    epsilon = [float(r['epsilon']) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.plot(steps, vs_random, label='vs_random')
    ax.plot(steps, vs_heuristic, label='vs_heuristic')
    ax.plot(steps, vs_prev, label='vs_prev')
    ax.set_title('Win rate')
    ax.set_xlabel('step')
    ax.set_ylabel('win rate')
    ax.legend()

    ax = axes[0, 1]
    ax.plot(steps, avg_ep_len)
    ax.set_title('Average episode length')
    ax.set_xlabel('step')
    ax.set_ylabel('length')

    ax = axes[1, 0]
    ax.plot(steps, avg_loss)
    ax.set_title('Average loss')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')

    ax = axes[1, 1]
    ax.plot(steps, epsilon)
    ax.set_title('Epsilon')
    ax.set_xlabel('step')
    ax.set_ylabel('epsilon')

    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == '__main__':
    main()
