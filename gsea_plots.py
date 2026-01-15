import operator
import sys
import warnings
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.category import UnitData
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import matplotlib.transforms as transforms

from gseapy.scipalette import SciPalette
from gseapy.plot import GSEAPlot


# ===================================================================
# Make lines thinner for GSEAPlot
# ===================================================================

class CustomGSEAPlot(GSEAPlot):
    def __init__(self, es_linewidth=1, spine_width=0.5, **kwargs):
        self.es_linewidth = es_linewidth 
        self.spine_width = spine_width
        super().__init__(**kwargs)

    def _adjust_spines(self, ax):
        """Helper to make all spines thinner."""
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

    def axes_stat(self, rect):
        """Override to adjust GSEA curve thickness."""
        ax4 = self.fig.add_axes(rect)
        ax4.plot(self._x, self.RES, linewidth=self.es_linewidth, color=self.color)  # Use custom linewidth
        self._adjust_spines(ax4)
        
        # Add statistics text (unchanged)
        ax4.text(0.1, 0.1, self._fdr_label, transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.2, self._pval_label, transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.3, self._nes_label, transform=ax4.transAxes, fontsize=14)

        # Add horizontal line at y=0 (unchanged)
        trans4 = transforms.blended_transform_factory(ax4.transAxes, ax4.transData)
        ax4.hlines(0, 0, 1, linewidth=1, transform=trans4, color="grey")
        ax4.set_ylabel("Enrichment Score", fontsize=16)
        ax4.tick_params(axis="both", which="both", bottom=False, top=False, right=False, labelbottom=False, labelsize=14)
        ax4.locator_params(axis="y", nbins=5)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        self.ax = ax4

    def axes_rank(self, rect):
        """Override for ranked metric plot + thin spines."""
        ax1 = self.fig.add_axes(rect)
        self._adjust_spines(ax1)  # Apply thin spines
        ax1.spines["top"].set_visible(False)  # Keep top spine hidden

    def axes_hits(self, rect, bottom=False):
        """Override to make gene rank lines thinner (optional)."""
        ax2 = self.fig.add_axes(rect)
        self._adjust_spines(ax2)  # Apply thin spines
        if bottom:
            ax2.spines["bottom"].set_visible(True)
        trans2 = transforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        
        # Gene hit lines (adjusted to linewidth=0.2, original: 0.5)
        ax2.vlines(self._hit_indices, 0, 1, linewidth=0.2, transform=trans2, color="black")
        
        # Hide ticks and labels (unchanged)
        ax2.tick_params(axis="both", which="both", bottom=bottom, top=False, right=False, 
                       left=False, labelbottom=bottom, labelleft=False)
        if bottom:
            ax2.set_xlabel("Gene Rank", fontsize=16)


def custom_gseaplot(term, pre_res, figsize=(6, 5.5), es_linewidth=2, spine_width=0.5, ofname=None, **kwargs):
    """Wrapper to simplify plotting with custom settings."""
    term_data = pre_res.results[term]
    g = CustomGSEAPlot(
        term=term,
        tag=term_data["hits"],  # Gene hit indices
        runes=term_data["RES"],  # Running enrichment scores
        nes=term_data["nes"],
        pval=term_data["pval"],
        fdr=term_data["fdr"],
        #rank_metric=term_data["rank_metric"],
        figsize=figsize,
        es_linewidth=es_linewidth,  # Pass custom linewidth
        spine_width=spine_width,
        ofname=ofname,
        **kwargs,
    )
    g.add_axes()
    if ofname is None:
        return g.fig.axes
    g.savefig()
    
    
# ===================================================================
# Set vmin at FDR=0.05 for DotPlot
# ===================================================================


class DotPlot(object):
    def __init__(
        self,
        df: pd.DataFrame,
        x: Optional[str] = None,
        y: str = "Term",
        hue: str = "Adjusted P-value",
        dot_scale: float = 5.0,
        x_order: Optional[List[str]] = None,
        y_order: Optional[List[str]] = None,
        thresh: float = 0.05,
        n_terms: int = 10,
        title: str = "",
        figsize: Tuple[float, float] = (6, 5.5),
        cmap: str = "viridis_r",
        ofname: Optional[str] = None,
        **kwargs,
    ):
        """Visualize GSEApy Results with categorical scatterplot
        When multiple datasets exist in the input dataframe, the `x` argument is your friend.

        :param df: GSEApy DataFrame results.
        :param x: Categorical variable in `df` that map the x-axis data. Default: None.
        :param y: Categorical variable in `df` that map the y-axis data. Default: Term.
        :param hue: Grouping variable that will produce points with different colors.
                    Can be either categorical or numeric

        :param x_order: bool, array-like list. Default: False.
                        If True, peformed hierarchical_clustering on X-axis.
                        or input a array-like list of `x` categorical levels.

        :param x_order: bool, array-like list. Default: False.
                        If True, peformed hierarchical_clustering on Y-axis.
                        or input a array-like list of `y` categorical levels.

        :param title: Figure title.
        :param thresh: Terms with `column` value < cut-off are shown. Work only for
                    ("Adjusted P-value", "P-value", "NOM p-val", "FDR q-val")
        :param n_terms: Number of enriched terms to show.
        :param dot_scale: float, scale the dot size to get proper visualization.
        :param figsize: tuple, matplotlib figure size.
        :param cmap: Matplotlib colormap for mapping the `column` semantic.
        :param ofname: Output file name. If None, don't save figure
        :param marker: The matplotlib.markers. See https://matplotlib.org/stable/api/markers_api.html
        """
        self.marker = "o"
        if "marker" in kwargs:
            self.marker = kwargs["marker"]
        self.y = y
        self.x = x
        self.x_order = x_order
        self.y_order = y_order
        self.hue = str(hue)
        self.colname = str(hue)
        self.figsize = figsize
        self.cmap = cmap
        self.ofname = ofname
        self.scale = dot_scale
        self.title = title
        self.n_terms = n_terms
        self.thresh = thresh
        self.data = self.process(df)
        plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    def isfloat(self, x):
        try:
            float(x)
        except:
            return False
        else:
            return True

    def process(self, df: pd.DataFrame):
        # check if any values in `df[colname]` can't be coerced to floats
        can_be_coerced = df[self.colname].map(self.isfloat).sum()
        if can_be_coerced < len(df):
            msg = "some value in %s could not be typecast to `float`" % self.colname
            raise ValueError(msg)
        # subset
        mask = df[self.colname] <= self.thresh
        if self.colname in ["Combined Score", "NES", "ES", "Odds Ratio"]:
            mask.loc[:] = True

        df = df.loc[mask]
        if len(df) < 1:
            msg = "Warning: No enrich terms when cutoff = %s" % self.thresh
            raise ValueError(msg)
        self.cbar_title = self.colname
        # clip GSEA lower bounds
        if self.colname in ["NOM p-val", "FDR q-val"]:
             df[self.colname] = df[self.colname].clip(1e-5, 1.0)
        # sorting the dataframe for better visualization
        if self.colname in ["Adjusted P-value", "P-value", "NOM p-val", "FDR q-val"]:
            # get top_terms
            #df = df.sort_values(by=self.colname)
            df[self.colname] = df[self.colname].replace(0, np.nan)
            # Use bfill() to fill the NaNs
            df[self.colname].astype(float).bfill()  ## asending order, use bfill
            df = df.assign(p_inv=np.log10(1 / df[self.colname].astype(float)))
            self.colname = "p_inv"
            self.cbar_title = r"$\log_{10} \frac{1}{FDR}$"

        # get top terms; sort ascending
        if (self.x is not None) and (self.x in df.columns):
            # get top term of each group
            df = (
                df.groupby(self.x)
                .apply(lambda _x: _x.sort_values(by=self.colname).tail(self.n_terms))
                .reset_index(drop=True)
            )
        else:
            df = df.sort_values(by=self.colname).tail(self.n_terms)  # acending
        # get scatter area
        ol = df.columns[df.columns.isin(["Overlap", "Tag %"])]
        temp = (
            df[ol].squeeze(axis=1).str.split("/", expand=True).astype(int)
        )  # axis=1, in case you have only 1 row
        df = df.assign(Hits_ratio=temp.iloc[:, 0] / temp.iloc[:, 1])
        return df

    def _hierarchical_clustering(self, mat, method, metric) -> List[int]:
        # mat.shape -> [n_sample, m_features]
        Y0 = sch.linkage(mat, method=method, metric=metric)
        Z0 = sch.dendrogram(
            Y0,
            orientation="left",
            # labels=mat.index,
            no_plot=True,
            distance_sort="descending",
        )
        idx = Z0["leaves"][::-1]  # reverse the order to make the view better
        return idx

    def get_x_order(
        self, method: str = "single", metric: str = "euclidean"
    ) -> List[str]:
        """See scipy.cluster.hierarchy.linkage()
        Perform hierarchical/agglomerative clustering.
        Return categorical order.
        """
        if isinstance(self.x_order, Iterable):
            return self.x_order
        mat = self.data.pivot(
            index=self.y,
            columns=self.x,
            values=self.colname,  # [self.colname, "Hits_ratio"],
        ).fillna(0)
        idx = self._hierarchical_clustering(mat.T, method, metric)
        return list(mat.columns[idx])

    def get_y_order(
        self, method: str = "single", metric: str = "euclidean"
    ) -> List[str]:
        """See scipy.cluster.hierarchy.linkage()
        Perform hierarchical/agglomerative clustering.
        Return categorical order.
        """
        if isinstance(self.y_order, Iterable):
            return self.y_order
        mat = self.data.pivot(
            index=self.y,
            columns=self.x,
            values=self.colname,  # [self.colname, "Hits_ratio"],
        ).fillna(0)
        idx = self._hierarchical_clustering(mat, method, metric)
        return list(mat.index[idx])

    def get_ax(self):
        """
        setup figure axes
        """
        # create fig
        if hasattr(sys, "ps1") and (self.ofname is None):
            # working inside python console, show figure
            fig = plt.figure(figsize=self.figsize)
        else:
            # If working on commandline, don't show figure
            fig = Figure(figsize=self.figsize)
            _canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        self.fig = fig
        return ax

    def set_x(self):
        """
        set x-axis's value
        """
        x = self.x
        xlabel = ""
        # set xaxis values, so you could get dotplot
        if (x is not None) and (x in self.data.columns):
            xlabel = x
        elif "Combined Score" in self.data.columns:
            xlabel = "Combined Score"
            x = xlabel
        elif "Odds Ratio" in self.data.columns:
            xlabel = "Odds Ratio"
            x = xlabel
        elif "NES" in self.data.columns:
            xlabel = "NES"
            x = xlabel
        else:
            # revert back to p_inv
            x = self.colname
            xlabel = self.cbar_title

        return x, xlabel

    def scatter(
        self,
        outer_ring: bool = False,
    ):
        """
        build scatter
        """
        # scatter colormap range
        # df = df.assign(colmap=self.data[self.colname].round().astype("int"))
        # make area bigger to better visualization
        # area = df["Hits_ratio"] * plt.rcParams["lines.linewidth"] * 100
        df = self.data.assign(
            area=(
                self.data["Hits_ratio"] * self.scale * plt.rcParams["lines.markersize"]
            ).pow(2)
        )
        colmap = df[self.colname].astype(int)
        # vmin = np.percentile(colmap.min(), 1)
        # vmax = np.percentile(colmap.max(), 99)
        
        vmin = np.log10(1/0.05)
        vmax = colmap.max()
        
        ax = self.get_ax()
        # if self.x is None:
        x, xlabel = self.set_x()
        y = self.y
        # set x, y order
        xunits = UnitData(self.get_x_order()) if self.x_order else None
        yunits = UnitData(self.get_y_order()) if self.y_order else None

        # outer ring
        if outer_ring:
            smax = df["area"].max()
            # TODO:
            # Matplotlib BUG: when setting edge colors,
            # there's the center of scatter could not aligned.
            # Must set backend to TKcario... to fix it
            # Instead, I just add more dots in the plot to get the ring
            blk_sc = ax.scatter(
                x=x,
                y=y,
                s=smax * 1.6,
                edgecolors="none",
                c="black",
                data=df,
                marker=self.marker,
                xunits=xunits,  # set x categorical order
                yunits=yunits,  # set y categorical order
                zorder=0,
            )
            wht_sc = ax.scatter(
                x=x,
                y=y,
                s=smax * 1.3,
                edgecolors="none",
                c="white",
                data=df,
                marker=self.marker,
                xunits=xunits,  # set x categorical order
                yunits=yunits,  # set y categorical order
                zorder=1,
            )
            # data = np.array(rg.get_offsets()) # get data coordinates
        # inner circle
        sc = ax.scatter(
            x=x,
            y=y,
            data=df,
            s="area",
            edgecolors="none",
            c=self.colname,
            cmap=self.cmap,
            vmin=vmin,
            vmax=vmax,
            marker=self.marker,
            xunits=xunits,  # set x categorical order
            yunits=yunits,  # set y categorical order
            zorder=2,
        )
        ax.set_xlabel(xlabel, fontsize=14)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_axisbelow(True)  # set grid blew other element
        ax.grid(axis="y", zorder=-1)  # zorder=-1.0
        ax.margins(x=0.25)

        # We change the fontsize of minor ticks label
        # ax.tick_params(axis='y', which='major', labelsize=16)
        # ax.tick_params(axis='both', which='minor', labelsize=14)

        # scatter size legend
        # we use the *func* argument to supply the inverse of the function
        # used to calculate the sizes from above. The *fmt* ensures to string you want
        handles, labels = sc.legend_elements(
            prop="sizes",
            num=3,  #
            fmt="{x:.2f}",
            color="gray",
            func=lambda s: np.sqrt(s) / plt.rcParams["lines.markersize"] / self.scale,
        )
        ax.legend(
            handles,
            labels,
            title="% Genes\nin set",
            bbox_to_anchor=(1.02, 0.9),
            loc="upper left",
            frameon=False,
            labelspacing=1.0,
        )
        ax.set_title(self.title, fontsize=16, fontweight="bold")
        self.add_colorbar(sc)

        return ax

    def add_colorbar(self, sc):
        """
        :param sc: matplotlib.Scatter
        """
        # colorbar
        # cax = fig.add_axes([1.0, 0.20, 0.03, 0.22])
        cbar = self.fig.colorbar(
            sc,
            shrink=0.25,
            aspect=10,
            anchor=(0.0, 0.2),  # (0.0, 0.2),
            location="right"
            # cax=cax,
        )
        # cbar.ax.tick_params(direction='in')
        cbar.ax.yaxis.set_tick_params(
            color="white", direction="in", left=True, right=True
        )
        cbar.ax.set_title(self.cbar_title, loc="left", fontweight="bold")
        for key, spine in cbar.ax.spines.items():
            spine.set_visible(False)

    def barh(self, color=None, group=None, ax=None):
        """
        Barplot
        """
        if ax is None:
            ax = self.get_ax()
        x, xlabel = self.set_x()
        bar = self.data.plot.barh(
            x=self.y, y=self.colname, alpha=0.75, fontsize=16, ax=ax
        )
        if self.hue in ["Adjusted P-value", "P-value", "FDR q-val", "NOM p-val"]:
            xlabel = r"$- \log_{10}$ (%s)" % self.hue
        else:
            xlabel = self.hue
        bar.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        bar.set_ylabel("")
        bar.set_title(self.title, fontsize=24, fontweight="bold")
        bar.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

        # get default color cycle
        if (not isinstance(color, str)) and isinstance(color, Iterable):
            _colors = list(color)
        else:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            _colors = prop_cycle.by_key()["color"]
        colors = _colors
        # remove old legend first
        bar.legend_.remove()
        if (group is not None) and (group in self.data.columns):
            num_grp = self.data[group].value_counts(sort=False)
            # set colors for each bar (groupby hue)
            colors = []
            legend_elements = []
            for i, n in enumerate(num_grp):
                # cycle _colors if num_grp > len(_colors)
                c = _colors[i % len(_colors)]
                colors += [c] * n
                ele = Line2D(
                    xdata=[0],
                    ydata=[0],
                    marker="o",
                    color="w",
                    label=num_grp.index[i],
                    markerfacecolor=c,
                    markersize=8,
                )
                legend_elements.append(ele)
            # add custom legend
            ax.legend(
                handles=legend_elements,
                loc="upper left",
                title=group,
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
        # update color of bars
        for j, b in enumerate(ax.patches):
            c = colors[j % len(colors)]
            b.set_facecolor(c)

        # self.adjust_spines(ax, spines=["left", "bottom"])
        for side in ["right", "top"]:
            ax.spines[side].set_visible(False)
        # set ticks
        ax.tick_params(axis="both", which="both", top=False, right=False)
        return ax

    def to_edgelist(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        return two dataframe of nodes, and edges
        """
        num_nodes = len(self.data)
        # build graph
        # G = nx.Graph()
        group_loc = None
        if self.x is not None:
            group_loc = self.data.columns.get_loc(self.x)
        term_loc = self.data.columns.get_loc(self.y)  # "Terms"
        if "Genes" in self.data.columns:
            gene_loc = self.data.columns.get_loc("Genes")
        elif "Lead_genes":
            gene_loc = self.data.columns.get_loc("Lead_genes")
        else:
            raise KeyError("Sorry, could not locate enriched gene list")
        # build graph
        # nodes = []
        nodes = self.data.reset_index(drop=True)
        nodes.index.name = "node_idx"
        genes = self.data.iloc[:, gene_loc].str.split(";")
        # ns_loc = self.data.columns.get_loc("Hits_ratio")
        edge_list = []
        for i in range(num_nodes):
            # nodes.append([i, self.data.iloc[i, term_loc], self.data.iloc[i, ns_loc]])
            # if group_loc is not None:
            #     nodes[-1].append(self.data.iloc[i, group_loc])
            for j in range(i + 1, num_nodes):
                set_i = set(genes.iloc[i])
                set_j = set(genes.iloc[j])
                ov = set_i.intersection(set_j)
                if len(ov) < 1:
                    continue
                jaccard_coefficient = len(ov) / len(set_i.union(set_j))
                overlap_coefficient = len(ov) / min(len(set_i), len(set_j))
                edge = [
                    i,
                    j,
                    self.data.iloc[i, term_loc],
                    self.data.iloc[j, term_loc],
                    jaccard_coefficient,
                    overlap_coefficient,
                    ",".join(ov),
                ]
                edge_list.append(edge)
                # G.add_edge(src,
                # targ,
                # jaccard= jaccard_coefficient,
                # overlap = overlap_coefficient,
                # genes = list(ov))
        edges = pd.DataFrame(
            edge_list,
            columns=[
                "src_idx",
                "targ_idx",
                "src_name",
                "targ_name",
                "jaccard_coef",
                "overlap_coef",
                "overlap_genes",
            ],
        )
        # node_c = ["node_idx", "node_name", "node_size"]
        # if group_loc is not None:
        #    node_c += ["node_group"]
        # nodes = pd.DataFrame(nodes, columns=node_c)
        return nodes, edges
    

def dotplot(
    df: pd.DataFrame,
    column: str = "Adjusted P-value",
    x: Optional[str] = None,
    y: str = "Term",
    x_order: Union[List[str], bool] = False,
    y_order: Union[List[str], bool] = False,
    title: str = "",
    cutoff: float = 0.05,
    top_term: int = 10,
    size: float = 5,
    figsize: Tuple[float, float] = (4, 6),
    cmap: str = "viridis_r",
    ofname: Optional[str] = None,
    xticklabels_rot: Optional[float] = None,
    yticklabels_rot: Optional[float] = None,
    marker: str = "o",
    show_ring: bool = False,
    **kwargs,
):
    """Visualize GSEApy Results with categorical scatterplot
    When multiple datasets exist in the input dataframe, the `x` argument is your friend.

    :param df: GSEApy DataFrame results.
    :param column: column name in `df` that map the dot colors. Default: Adjusted P-value.
    :param x: Categorical variable in `df` that map the x-axis data. Default: None.
    :param y: Categorical variable in `df` that map the y-axis data. Default: Term.

    :param x_order: bool, array-like list. Default: False.
                    If True, peformed hierarchical_clustering on X-axis.
                    or input a array-like list of `x` categorical levels.

    :param x_order: bool, array-like list. Default: False.
                    If True, peformed hierarchical_clustering on Y-axis.
                    or input a array-like list of `y` categorical levels.

    :param title: Figure title.
    :param cutoff: Terms with `column` value < cut-off are shown. Work only for
                   ("Adjusted P-value", "P-value", "NOM p-val", "FDR q-val")
    :param top_term: Number of enriched terms to show.
    :param size: float, scale the dot size to get proper visualization.
    :param figsize: tuple, matplotlib figure size.
    :param cmap: Matplotlib colormap for mapping the `column` semantic.
    :param ofname: Output file name. If None, don't save figure
    :param marker: The matplotlib.markers. See https://matplotlib.org/stable/api/markers_api.html
    :param show_ring bool: Whether to draw outer ring.

    :return: matplotlib.Axes. return None if given ofname.
             Only terms with `column` <= `cut-off` are plotted.
    """
    if "group" in kwargs:
        warnings.warn("group is deprecated; use x instead", DeprecationWarning, 2)
        return

    dot = DotPlot(
        df=df,
        x=x,
        y=y,
        x_order=x_order,
        y_order=y_order,
        hue=column,
        title=title,
        thresh=cutoff,
        n_terms=int(top_term),
        dot_scale=size,
        figsize=figsize,
        cmap=cmap,
        ofname=ofname,
        marker=marker,
    )
    ax = dot.scatter(outer_ring=show_ring)

    if xticklabels_rot:
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(xticklabels_rot)

    if yticklabels_rot:
        for label in ax.get_yticklabels():
            label.set_ha("right")
            label.set_rotation(yticklabels_rot)

    if ofname is None:
        return ax
    dot.fig.savefig(ofname, bbox_inches="tight", dpi=300)