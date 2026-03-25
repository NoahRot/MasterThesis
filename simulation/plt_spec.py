"""
Global Matplotlib initialization script.

Import this module once at the start of a project to apply:
- LaTeX rendering
- Global font size
- Grid enabled by default
- Customization of background colors
- Seaborn color palette
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Palette setup
# -----------------------
def create_sns_palette(sns_name : str = "bright", palette_name : str = "seaborn", use_it : bool = True):
    mpl.color_sequences.register(palette_name, sns.color_palette(sns_name).as_hex())
    if use_it:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=mpl.color_sequences[palette_name])

def create_color_palette(palette, palette_name : str, use_it : bool = True):
    mpl.color_sequences.register(palette_name, palette)
    if use_it:
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=mpl.color_sequences[palette_name])

def use_color_palette(palette_name : str):
    if palette_name not in mpl.color_sequences:
        raise ValueError(f"Palette '{palette_name}' not found. Available palettes: {list(mpl.color_sequences.keys())}")
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=mpl.color_sequences[palette_name])

def init_plt(
    font_size=12,
    latex=True,
    grid=True,
    background_fig_color="white",
    background_axe_color="white",
):
    """
    Initialize matplotlib global settings.

    Parameters
    ----------
    font_size : int
        Base font size for all plot text.
    latex : bool
        Enable LaTeX text rendering.
    grid : bool
        Enable grid by default.
    """

    # -----------------------
    # Matplotlib rcParams
    # -----------------------
    mpl.rcParams.update({
        # Font sizes
        "font.size": font_size,
        "axes.titlesize": font_size * 1.2,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size * 0.9,
        "ytick.labelsize": font_size * 0.9,
        "legend.fontsize": font_size * 0.9,

        # Grid
        "axes.grid": grid,
        "grid.alpha": 0.7,
        "grid.linestyle": "--",

        # Figure
        "figure.figsize": (6.4, 4.8),
        "figure.dpi": 100,

        # Lines
        "lines.linewidth": 1.2,
        "lines.markersize": 6,

        # Location
        "legend.loc": "best",
        #"figure.autolayout": True,
        "figure.constrained_layout.use": True,

        # Background color
        "figure.facecolor": background_fig_color,
        "axes.facecolor": background_axe_color,
    })

    # -----------------------
    # LaTeX rendering
    # -----------------------
    if latex:
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        })
    else:
        mpl.rcParams["text.usetex"] = False
