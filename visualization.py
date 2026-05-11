import matplotlib.pyplot as plt
import os


def generate_grid_environment(n, m, trajectory, probabilities):
    fig, ax = plt.subplots()
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(range(m + 1))
    ax.set_yticks(range(n + 1))
    ax.grid(True)

    for i in range(m):
        ax.axvline(x=i, color='black')

    for i in range(n):
        ax.axhline(y=i, color='black')

    x, y = [], []
    for i, state in enumerate(trajectory):
        row = n - 1 - state // m
        col = state % m
        x.append(col + 0.5)
        y.append(row + 0.5)

        if i == 0:
            ax.plot(col + 0.5, row + 0.5, color='green', marker='s', markersize=8)
        elif i == len(trajectory) - 1:
            ax.plot(col + 0.5, row + 0.5, color='red', marker='o', markersize=8)

    ax.plot(x, y, color='blue')
    return ax


def save_plots_as_images(axes_list, output_dir, dpi=300):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, ax in enumerate(axes_list):
        output_filename = os.path.join(output_dir, f'plot_{i + 1}.png')
        ax.get_figure().savefig(output_filename, dpi=dpi)
        ax.clear()

    plt.close('all')
