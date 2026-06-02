import os
import yaml
import numpy as np
import pandas as pd
import glob
import pprint
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

matplotlib.use("Agg")
from datetime import datetime

# from pyomo.environ import units as pyunits, value
# from idaes.core.base.costing_base import register_idaes_currency_units
# from watertap.core import Database

"""
This file will create a multi-page PDF with cost curves for 
all zero-order unit models that have a results csv in the 
"results" directory, organized by category.
"""

here = os.path.dirname(__file__)
res_path = os.path.join(here, "results")

pretty_subtypes = {
    "default": "Default",
    "anion_exchange": "Anion Exchange",
    "cation_exchange": "Cation Exchange",
    "alum": "Alum",
    "ammonia": "Ammonia",
    "anhydrous_ammonia": "Anhydrous Ammonia",
    "anti-scalant": "Anti-Scalant",
    "caustic_soda": "Caustic Soda",
    "ferric_chloride": "Ferric Chloride",
    "hydrochloric_acid": "Hydrochloric Acid",
    "lime": "Lime",
    "sodium_bisulfite": "Sodium Bisulfite",
    "sulfuric_acid": "Sulfuric Acid",
    "chlorine": "Chlorine",
    "uv": "UV",
    "uv_aop": "UV AOP",
    "ozone": "Ozone",
    "ozone_aop": "Ozone AOP",
    "pressure_vessel": "GAC - Pressure Vessels",
    "gravity": "GAC - Gravity Basins",
    "coag_and_floc": "Coagulation & Flocculation",
    "sw_onshore_intake": "Seawater Intake",
    "ultra_filtration": "Ultrafiltration",
}

# def create_chem_addition_page(df):

categories = {
    "intake": {
        "pretty_name": "Intake & Conveyance",
        "units": [
            "water_pumping_station",
            "sw_onshore_intake",
            "screen",
            "well_field",
        ],
        "hue_col": "unit",
        "mask": {"flow_mgd": (0.1, 20)},
        "unit_mask": {
            "water_pumping_station": {"subtype": "raw"},
            "screen": {"subtype": ["micro"]},
            "well_field": {"subtype": ["default"], "pipe_distance": 1},
        },
    },
    "pretreatment": {
        "pretty_name": "Pretreatment",
        "units": [
            "coag_and_floc",
            "electrocoagulation",
            # "sedimentation",
            # "filtration",
        ],
        "hue_col": "unit",
        "unit_mask": {
            "electrocoagulation": {
                "tds": 1000,
                "metal_dose": 10,
                "elec_material": "aluminum",
            }
        },
    },
    "chemical_addition": {
        "pretty_name": "Chemical Addition",
        # "units": ["chemical_addition","co2_addition", "decarbonator", "static_mixer"],
        "hue_col": "subtype",
        "mask": {"dose": 100},
    },
    "clarification": {
        "pretty_name": "Clarification & Settling",
        "units": [
            "clarifier",
            "primary_separator",
            "dissolved_air_flotation",
            "air_flotation",
            "sedimentation",
            "settling_pond",
        ],
        "hue_col": "unit",
        "mask": {"subtype": ["default"], "flow_mgd": (0.1, 20)},
    },
    "adsorption": {
        "pretty_name": "GAC & Ion Exchange",
        "units": ["gac", "ion_exchange"],
        "mask": {"flow_mgd": (0.1, 20)},
        "unit_mask": {
            "ion_exchange": {
                "subtype": ["anion_exchange", "cation_exchange"],
                "tds": [50, 200],
            },
            "gac": {
                "subtype": [
                    "pressure_vessel",
                    "gravity",
                    # "default"
                ]
            },
        },
        "hue_col": "subtype",
        "skip_default": True,
    },
    "membrane_processes": {
        "pretty_name": "Membrane Processes",
        "units": [
            "nanofiltration",
            "ultra_filtration",
            "microfiltration",
            "reverse_osmosis",
            # "electrodialysis_reversal",
        ],
        "hue_col": "unit",
        "mask": {"flow_mgd": (0.1, 20), "subtype": ["default"]},
    },
    "media_filtration_technologies": {
        "pretty_name": "Media Filtration",
        "units": [
            "dual_media_filtration",
            "media_filtration",
            "bio_active_filtration",
            "cartridge_filtration",
            "tri_media_filtration",
            "walnut_shell_filter",
        ],
        "hue_col": "unit",
        "mask": {"flow_mgd": (0.1, 20)},
    },
    "uv_ozone_aop": {
        "pretty_name": "UV & Ozone AOP",
        "units": [
            "uv",
            "uv_aop",
            "ozone",
            "ozone_aop",
        ],
        # "mask": {"oxidant_dose": 10},
        "unit_mask": {
            "uv": {"sec": 0.1},
            "uv_aop": {"sec": 0.1, "oxidant_dose": 10},
            "ozone": {"dose": 9},
            "ozone_aop": {"dose": 9, "oxidant_dose": 10},
        },
        "hue_col": "unit",
    },
    "bio_treatment": {
        "pretty_name": "Biological Treatment",
        "units": [
            "anaerobic_digestion_oxidation",
            "anaerobic_mbr_mec",
            "bio_active_filtration",
            "conventional_activated_sludge",
            "fixed_bed",
        ],
        "mask": {"flow_mgd": (0.1, 20)},
        "hue_col": "unit",
        "unit_mask": {
            "fixed_bed": {"subtype": ["default"]},
            "conventional_activated_sludge": {"subtype": ["default"]},
        },
    },
    "brine_management": {
        "pretty_name": "Brine Management",
        "units": [
            "brine_concentrator",
            "crystallizer",
            "evaporation_pond",
            "deep_well_injection",
        ],
        "hue_col": "unit",
        "mask": {"flow_mgd": (0.01, 1)},
        "unit_mask": {
            "brine_concentrator": {"tds": 60000},
            "crystallizer": {"tds": 145000},
            "evaporation_pond": {"solar_radiation": 25},
            "deep_well_injection": {"pipe_distance": 25},
        },
        # "ycols": ["fs.costing.LCOW", "fs.costing.total_capital_cost", "fs.costing.SEC"],
    },
}


def chem_add_footer(lines):
    line = "Chemical addition costs are run with a fixed dose of 100 mg/L for each chemical"
    lines.append(line)
    return lines


def brine_management_footer(lines):
    line = "Brine concentrator and crystallizer CAPEX and SEC are a function inlet flow rate and salinity."
    lines.append(line)
    line = "These results are for a feed salinity of 60,000 mg/L TDS for the concentrator and 145,000 mg/L TDS for the crystallizer."
    lines.append(line)
    line = "Evaporation pond costs are a function of inlet flow rate and solar radiation, with results shown for 25 MJ/m²/day."
    lines.append(line)
    line = "Deep well injection costs are a function of inlet flow rate and pipe distance, with results shown for a pipe distance of 25 miles."
    lines.append(line)
    return lines


def uv_ozone_footer(lines):
    line = "UV and ozone AOP costs are run with a fixed oxidant dose of 10 mg/L and UV SEC of 0.1 kWh/m³."
    lines.append(line)
    line = "Ozone costs are for a fixed ozone dose of 9 mg/L."
    lines.append(line)
    return lines


footer_dict = {
    "chemical_addition": chem_add_footer,
    "brine_management": brine_management_footer,
    "uv_ozone_aop": uv_ozone_footer,
}


def add_watertap_logo(fig, loc=None, alpha=0.7):
    if loc is None:
        loc = [0.925, 0.03, 0.075, 0.075]
    inset_ax = fig.add_axes(loc, zorder=1)
    # img_path = os.path.join(here, "cost_curve_schematic.png")
    img_path = f"{here}/watertap-logo.png"
    img = mpimg.imread(img_path)
    inset_ax.imshow(img, alpha=alpha)
    inset_ax.axis("off")
    return fig


def add_pg_number(fig, pg):
    # fig.text(0.925 + 0.075 / 2, 0.01, f"Page {pg}", ha="center", va="bottom", fontsize=8, color="#555555")
    fig.text(
        0.5, 0.01, f"Page {pg}", ha="center", va="bottom", fontsize=8, color="#555555"
    )


def add_date_created(fig):

    date_str = datetime.now().strftime("%Y-%m-%d")
    fig.text(
        0.5,
        0.02,
        f"Created {date_str}",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )


def create_title_page():
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5,
        0.6,
        "**DRAFT**\nWaterTAP Zero-Order Model\nLCOW & CAPEX Curves",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="#1a5276",
    )
    fig.text(
        0.5,
        0.42,
        # r"$C_{cap} = A \times \left(\frac{Q_{in}}{Q_{basis}}\right)^B$" "\n\n"
        "\n" "All costs in 2023 USD",
        ha="center",
        va="center",
        fontsize=14,
        color="#555555",
    )
    fig.text(
        0.5,
        0.2,
        "Generated from model runs\n",
        ha="center",
        va="center",
        fontsize=11,
        color="#888888",
    )
    loc = [0.4, 0.75, 0.2, 0.2]
    fig = add_watertap_logo(fig, loc=loc, alpha=1)
    add_date_created(fig)
    return fig


def create_watertap_cost_curve_doc(save_as, xcol="flow_mgd", colormap="tab20"):
    """Produce the multi-page cost-curve PDF."""

    default_ycols = ["fs.costing.LCOW", "fs.costing.total_capital_cost"]

    with PdfPages(save_as) as pdf:

        # ── Title page ────────────────────────────────────────────────────
        fig = create_title_page()
        pdf.savefig(fig)
        plt.close()

        # ── Per-category pages ────────────────────────────────────────────

        for pg, (cat, d) in enumerate(categories.items(), start=1):
            cat_name = d["pretty_name"]
            if "ycols" in d:
                ycols = d["ycols"]
            else:
                ycols = default_ycols

            fig, axes = plt.subplots(
                nrows=len(ycols), ncols=2, figsize=(11, 4.25 * len(ycols))
            )
            fig.suptitle(
                cat_name,
                fontsize=16,
                fontweight="bold",
                color="#1a5276",
                # y=0.97,
            )

            mask = list()
            df = pd.DataFrame()

            if cat in all_res["unit"].unique():
                # category is a unit model name
                # the whole category is one unit model with subtypes
                df = all_res[all_res["unit"] == cat].copy()
            elif "units" in d:
                df = all_res[all_res["unit"].isin(d["units"])].copy()

            if len(df) == 0:
                # This should not happen
                raise ValueError(f"No data found for unit '{cat}' in results.")

            df.dropna(axis=1, inplace=True, how="all")

            if "mask" in d:
                mask = list()
                for col, filt in d["mask"].items():
                    if col not in df.columns:
                        raise ValueError(
                            f"Mask key '{col}' not found in DataFrame columns."
                        )
                    if isinstance(filt, int) or isinstance(filt, float):
                        mask.append(df[col] == filt)
                    elif isinstance(filt, list):
                        mask.append(df[col].isin(filt))
                    elif isinstance(filt, tuple) and len(filt) == 2:
                        mask.append((df[col] >= filt[0]) & (df[col] <= filt[1]))
                    else:
                        raise ValueError(f"Invalid mask value for key '{col}': {filt}")

                mask = np.logical_and.reduce(mask)
                df = df[mask].copy()

            if "unit_mask" in d:
                df2 = pd.DataFrame()
                for u in d["units"]:
                    if u not in d["unit_mask"].keys():
                        tmp = df[df["unit"] == u].copy()
                        df2 = pd.concat([df2, tmp], ignore_index=True)

                for u, conditions in d["unit_mask"].items():
                    umask = df["unit"] == u
                    for col, filt in conditions.items():
                        if col not in df.columns:
                            raise ValueError(
                                f"Unit mask key '{col}' not found in DataFrame columns."
                            )
                        if (
                            isinstance(filt, int)
                            or isinstance(filt, float)
                            or isinstance(filt, str)
                        ):
                            print(f"Filtering unit '{u}' with {col} == {filt}")
                            if filt not in df[col].values and not isinstance(filt, str):
                                # print(df[col].unique())
                                filt2 = df[col].max()

                                print("!" * 20)
                                print(
                                    f"Warning: value {filt} for column '{col}' not found in DataFrame for unit '{u}'. Using max in {col} of {filt2}."
                                )
                                print("!" * 20)
                                print(f"For unit = {u}, col = {col}, options are:")
                                for option in df[col].unique():
                                    print(f"  - {option:2f}")
                                filt = filt2
                            umask = umask & (df[col] == filt)
                        elif isinstance(filt, list):
                            print(f"Filtering unit '{u}' with {col} in {filt}")
                            umask = umask & df[col].isin(filt)
                        elif isinstance(filt, tuple) and len(filt) == 2:
                            print(
                                f"Filtering unit '{u}' with {col} between {filt[0]} and {filt[1]}"
                            )
                            umask = umask & (df[col] >= filt[0]) & (df[col] <= filt[1])
                        else:
                            raise ValueError(
                                f"Invalid unit mask value for key '{col}': {filt}"
                            )
                    tmp = df[umask].copy()
                    df2 = pd.concat([df2, tmp], ignore_index=True)

                df = df2.copy()

            plt.tight_layout(rect=[0, 0.25, 1, 0.95])
            fig, axes = plot_page(
                df,
                fig=fig,
                axes=axes,
                xcol=xcol,
                ycols=ycols,
                colormap=colormap,
                hue_col=d.get("hue_col", None),
                skip_default=d.get("skip_default", False),
            )
            plt.tight_layout(rect=[0, 0.25, 1, 0.95])
            footer_func = footer_dict.get(cat, None)
            if footer_func is not None:
                lines = ["Details:"]
                lines = footer_func(lines)

                fig.text(
                    0.05,
                    0.05,
                    "\n".join(lines),
                    fontsize=8.5,
                    fontfamily="monospace",
                    verticalalignment="bottom",
                    color="#555555",
                )
            fig = add_watertap_logo(fig)
            fig = add_pg_number(fig, pg)
            pdf.savefig(fig)
            plt.close()


def plot_page(
    rd,
    fig=None,
    axes=None,
    xcol="flow_mgd",
    ycols=None,
    hue_col="subtype",
    set_kwargs=dict(),
    plot_kwargs=dict(),
    colormap="tab20",
    skip_default=False,
):

    def _dollar_formatter(x, _pos):
        if x >= 1e6:
            return f"${x / 1e6:.1f}M"
        elif x >= 1e3:
            return f"${x / 1e3:.0f}k"
        else:
            return f"${x:.0f}"

    def _LCOW_formatter(x, _pos):
        if x < 1e-2:
            return f"{x:.0e}"
        elif 1e-2 <= x < 1:
            return f"{x:.2f}"
        else:
            return f"{x:.1f}"

    def plot_the_row(row):
        handles = []
        labels = []
        y = ycols[row]
        r = rd.sort_values(by=[xcol, y])
        if hue_col is None:
            line = ax_lin.plot(r[xcol], r[y], linewidth=1.8, label="Default")
            ax_log.loglog(
                r[xcol],
                r[y],
                linewidth=1.8,
            )
            handles.append(line)
        else:
            hues = rd[hue_col].unique()

            for i, hue in enumerate(hues):
                if hue == "default" and skip_default:
                    # sometimes default is the same as another subtype
                    continue
                color = color_list[i]
                subset = rd[rd[hue_col] == hue].copy()
                subset = subset.sort_values(by=xcol)
                h = hue.replace("_", " ").title() if isinstance(hue, str) else hue
                line = ax_lin.plot(
                    subset[xcol],
                    subset[y],
                    linewidth=1.8,
                    label=pretty_subtypes.get(hue, h),
                    color=color,
                )
                ax_log.loglog(
                    subset[xcol],
                    subset[y],
                    linewidth=1.8,
                    label=pretty_subtypes.get(hue, h),
                    color=color,
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linewidth=1.8,
                    )
                )
                labels.append(pretty_subtypes.get(hue, h))
        return handles, labels

    formatter_dict = {
        # "fs.costing.LCOW": _LCOW_formatter,
        "fs.costing.total_capital_cost": _dollar_formatter,
    }

    ylabel_dict = {
        "fs.costing.LCOW": "LCOW (USD/m³)",
        "fs.costing.total_capital_cost": "CAPEX (USD)",
        "fs.costing.SEC": "SEC (kWh/m³)",
        "fs.costing.total_operating_cost": "OPEX (USD/m³)",
    }

    if ycols is None:
        ycols = ("fs.costing.LCOW", "fs.costing.total_capital_cost")

    if (fig, axes) == (None, None):
        fig, axes = plt.subplots(
            nrows=len(ycols), ncols=2, figsize=(8, 4.25 * len(ycols))
        )

    if len(rd[hue_col].unique()) > 5:
        colormap = "tab20"
        n = 20
    else:
        colormap = "tab10"
        n = 10

    cmap = getattr(plt.cm, colormap)
    color_list = [cmap(i) for i in np.linspace(0, 1, n)]

    for row, ax in enumerate(axes):
        if len(ycols) == 1:
            ax_lin, ax_log = [*axes]
            ax = [ax_lin, ax_log]
        else:
            ax_lin, ax_log = [*ax]
        handles, labels = plot_the_row(row)
        for col, a in enumerate(ax):
            a.set(**set_kwargs)
            a.grid(True, alpha=0.3, which="both")
            frmtr = formatter_dict.get(ycols[row], None)
            if frmtr is not None:
                a.yaxis.set_major_formatter(plt.FuncFormatter(frmtr))
            if row == len(ycols) - 1:
                # bottom row gets the x-axis label
                a.set_xlabel("Plant Capacity (MGD)")
            if col == 0:
                a.set_ylabel(ylabel_dict.get(ycols[row], ycols[row]))
            if (row, col) == (0, 0):
                a.set_title("Linear Scale", fontsize=11)
            elif (row, col) == (0, 1):
                a.set_title("Log-Log Scale", fontsize=11)
        if row == len(ycols) - 1:
            break

    bot_left_pos = ax[0].get_position()
    bot_right_pos = ax[1].get_position()
    # print(f" bottom left position\n\t{bot_left_pos}")
    # print(f" bottom right position\n\t{bot_right_pos}")

    width = bot_right_pos.x1 - bot_left_pos.x0
    x0, y0, w, h = bot_left_pos.x0, bot_left_pos.y0, width, 0
    y0 = 0.245
    fig.legend(
        handles,
        labels,
        loc="upper center",
        # mode="expand",
        bbox_to_anchor=(x0, y0, w, h),
        bbox_transform=fig.transFigure,
        ncol=4,
        #    frameon=False
    )

    fig.tight_layout()
    return fig, axes


def combine_results():

    print(f"Combining results from {res_path}...")
    csvs = glob.glob(os.path.join(res_path, "*.csv"))
    all_res = pd.DataFrame()
    for csv in csvs:
        df = pd.read_csv(csv)
        all_res = pd.concat([all_res, df], ignore_index=True)
    all_res.set_index(
        ["unit_class", "unit", "subtype", "flow_mgd"], inplace=True, drop=True
    )
    # all_res.to_csv(os.path.join(res_path, "all_results.csv"), index=True)
    all_res.to_csv(f"{here}/all_costing_results.csv", index=True)


if __name__ == "__main__":
    combine_results()
    all_res = pd.read_csv(f"{here}/all_costing_results.csv")
    save_as = f"{here}/DRAFT_watertap_cost_curves_doc.pdf"
    create_watertap_cost_curve_doc(save_as)
