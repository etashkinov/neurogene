import matplotlib.pyplot as plt
import figures


def plot_images(figures_to_show, cls_pred=None):
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, int(len(figures_to_show) / 3))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        if i < len(figures_to_show):
            figure = figures_to_show[i]
            ax.imshow(figure.image)

            # Show true and predicted classes.
            if cls_pred is None:
                label = "{0}".format(figure.label)
            else:
                label = "{0} vs {1}".format(figure.label, figures.get_label(cls_pred[i]))

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(label)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
