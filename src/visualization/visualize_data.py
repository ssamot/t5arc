
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


# ......................................................................................................
cmap = colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
                               '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

# ......................................................................................................

def add_vhlines_to_plot(axis, data):
    data = np.array(data)
    num_y = data.shape[0]
    num_x = data.shape[1]
    axis.vlines(np.arange(-0.5, num_x - 0.5), ymin=-0.5, ymax=num_y - 0.5, colors='#101010', linewidths=0.5)
    axis.hlines(np.arange(-0.5, num_y - 0.5), xmin=-0.5, xmax=num_x - 0.5, colors='#101010', linewidths=0.5)


def plot_pic(x):
    x =  np.array(x)
    plt.imshow(x, cmap=cmap, norm=norm)
    num_x = x.shape[0]
    num_y = x.shape[1]
    plt.vlines(np.arange(-0.5, num_x - 0.5), ymin=-0.5, ymax=num_y - 0.5, colors='#101010', linewidths=0.5)
    plt.hlines(np.arange(-0.5, num_y - 0.5), xmin=-0.5, xmax=num_x - 0.5, colors='#101010', linewidths=0.5)
    # plt.show()


def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=( 4 *n ,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        t_in = np.array(t_in)
        add_vhlines_to_plot(axs[0][fig_num], t_in)

        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        t_out = np.array(t_out)
        add_vhlines_to_plot(axs[1][fig_num], t_out)

        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        t_in = np.array(t_in)
        add_vhlines_to_plot(axs[0][fig_num], t_in)

        fig_num += 1

    plt.tight_layout()
    # plt.show()