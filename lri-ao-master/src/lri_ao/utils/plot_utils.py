import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def fmt(x, pos):
    return np.round(x, decimals=18)


def multidimensional_plot(
    function_to_plot,
    titles_list=None,
    plot_row_total=1,
    plot_row_current=1,
    row_title=None,
    coordinates=None,
    colorbar=True,
    **kwargs
):
    """Allows to plot the arrays as defined in Electric_field and
    Coordinate_system. Supports multiple titles and label without problem
    Probably need some cleaning work, but does the job well as is"""
    function_to_plot = np.array(function_to_plot)
    minimum = np.nanmin(function_to_plot)
    maximum = np.nanmax(function_to_plot)
    if minimum == maximum:
        maximum = maximum + 1e-21
        minimum = minimum - 1e-21
    if function_to_plot.shape[0] > 1:
        # formating titles
        try:
            titles_list[1]
            titles_list = np.array(titles_list)
        except TypeError or IndexError:
            titles_list = np.array(
                [titles_list for x in range(function_to_plot.shape[0])]
            )
        # formating extent
        if coordinates is not None:
            extent_list = []
            for w in range(function_to_plot.shape[0]):
                extent_list.append(
                    [
                        np.min(coordinates[w]),
                        np.max(coordinates[w]),
                        np.min(coordinates[w]),
                        np.max(coordinates[w]),
                    ]
                )
            extent_list = np.array(extent_list)
        else:
            extent_list = [None for x in range(function_to_plot.shape[0])]

        for plot_index in range(function_to_plot.shape[0]):
            plt.subplot(
                plot_row_total, function_to_plot.shape[0], plot_index + plot_row_current
            )
            if titles_list[plot_index] is not None:
                plt.title(titles_list[plot_index])
            if extent_list[plot_index] is not None:
                plt.imshow(
                    function_to_plot[plot_index],
                    vmin=minimum,
                    vmax=maximum,
                    interpolation="nearest",
                    extent=extent_list[plot_index],
                    **kwargs
                )
            else:
                plt.imshow(
                    function_to_plot[plot_index],
                    vmin=minimum,
                    vmax=maximum,
                    interpolation="nearest",
                    **kwargs
                )
            if None in extent_list:
                plt.axis("off")
            plt.xlabel("x")
            plt.ylabel("y")
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        if colorbar is True:
            cax = plt.axes([0.85, 0.35, 0.02, 0.3])
            plt.colorbar(cax=cax, format=ticker.FuncFormatter(fmt))

    else:
        if plot_row_total > 1:
            plt.subplot(plot_row_total, 1, plot_row_current)
        if coordinates is None:
            plt.imshow(function_to_plot[0], interpolation="nearest", **kwargs)
            plt.axis = None
        else:
            extent_list = np.array(
                [coordinates[:, 0][0][z] for axis in range(2) for z in (-1, 0)]
            )
            plt.imshow(
                function_to_plot[0],
                vmin=minimum,
                vmax=maximum,
                interpolation="nearest",
                extent=extent_list.T[0],
                **kwargs
            )
            plt.xlabel("x")
            plt.ylabel("y")
        if plot_row_total > 1:
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        if colorbar is True:
            cax = plt.axes([0.85, 0.2, 0.05, 0.6])
            plt.colorbar(cax=cax, format=ticker.FuncFormatter(fmt))
