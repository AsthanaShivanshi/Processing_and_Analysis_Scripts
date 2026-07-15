import matplotlib.pyplot as plt


MODEL_COLORS = {
    "Coarse": "#0072B2",    # blue
    "Bicubic": "#E69F00",   # orange
    "Bilinear": "#009E73",  # bluish green
    "UNet": "#CC79A7",      # reddish purple
    "DDIM": "#000000",      # black highlight
}


MODEL_LINESTYLES = {
    "Coarse": (0, (1, 1)),          # densely dotted
    "Bicubic": (0, (5, 2)),         # medium dashed
    "Bilinear": (0, (3, 1, 1, 1)),  # dash-dot-dot
    "UNet": (0, (7, 2, 1, 2)),      # long dash-dot
    "DDIM": "-",                    # solid highlight
}


VARIABLE_COLORS = {
    "precip": "#0072B2",        # blue
    "precipitation": "#0072B2",
    "temp": "#D55E00",          # vermillion
    "temperature": "#D55E00",
}


VARIABLE_CMAPS = {
    "precip": "YlGnBu",
    "precipitation": "YlGnBu",
    "temp": "RdBu_r",
    "temperature": "RdBu_r",
}

HIGHLIGHT_MODEL = "DDIM"


def apply_paper_style():
    plt.rcParams.update({
        "figure.figsize": (12, 8),
        "figure.dpi": 150,
        "savefig.dpi": 1500,
        "savefig.bbox": "tight",

        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,

        "axes.titleweight": "bold",
        "figure.titleweight": "bold",

        "lines.linewidth": 2.3,
        "lines.markersize": 6,

        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 1.0,
        "grid.alpha": 0.65,

        "legend.frameon": True,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def get_model_color(model):
    return MODEL_COLORS.get(model, "black")


def get_variable_color(variable):
    return VARIABLE_COLORS.get(str(variable).lower(), "black")


def style_model_line(model):
    return {
        "color": get_model_color(model),
        "linestyle": MODEL_LINESTYLES.get(model, "-"),
        "linewidth": 4.0 if model == HIGHLIGHT_MODEL else 2.3,
        "alpha": 1.0 if model == HIGHLIGHT_MODEL else 0.85,
        "zorder": 30 if model == HIGHLIGHT_MODEL else 10,
    }


def style_model_scatter(model):
    return {
        "color": get_model_color(model),
        "s": 90 if model == HIGHLIGHT_MODEL else 45,
        "alpha": 1.0 if model == HIGHLIGHT_MODEL else 0.75,
        "edgecolor": "white" if model == HIGHLIGHT_MODEL else "black",
        "linewidth": 0.9,
        "zorder": 30 if model == HIGHLIGHT_MODEL else 10,
    }


def style_model_fill(model):
    return {
        "color": get_model_color(model),
        "alpha": 0.08 if model == HIGHLIGHT_MODEL else 0.025,
        "zorder": 1,
    }



def style_axis(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, fontsize=18, fontweight="bold")

    ax.grid(grid, linestyle="--", linewidth=1.0, alpha=0.65)


def add_legend(ax, loc="best", ncol=1):
    ax.legend(
        loc=loc,
        ncol=ncol,
        frameon=True,
        fontsize=13,
    )



def add_bottom_legend(fig, handles, labels, ncol=None):
    if ncol is None:
        ncol = len(labels)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        frameon=True,
        fontsize=13,
        bbox_to_anchor=(0.5, -0.03),
    )


def style_highlight_scatter(model):
    return {
        "color": get_model_color(model),
        "edgecolor": "white",
        "linewidth": 0.9,
        "s": 65,
        "zorder": 40,
    }


def save_paper_figure(fig, path):
    save_figure(fig, path)


def save_figure(fig, path):
    fig.savefig(path, dpi=1500, bbox_inches="tight")




def get_variable_cmap(variable):
    """Return the paper colormap associated with a variable."""
    return VARIABLE_CMAPS.get(str(variable).lower(), "viridis")

