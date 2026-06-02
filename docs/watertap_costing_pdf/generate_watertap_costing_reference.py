"""
Generate WaterTAP Costing Reference Guide PDF.

Scrapes default costing parameters from the WaterTAP repository
(both detailed unit models and zero-order YAML files) and produces
a two-page reference PDF.

Requirements:
- watertap-dev environment + reportlab (pip) + pyyaml (pip) + pypdf (pip)
"""

import pypdf
import os
import pprint
import re
import yaml
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.platypus import Image
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.dirname(here)
repo_path = os.path.dirname(docs_path)

# for when auto-formatting doesn't look good
u_name_dict = {
    "ultra_filtration": "Ultrafiltration",
    "co2_addition": "CO<sub>2</sub> Addition",
    "sw_onshore_intake": "Seawater Intake",
    "mbr": "Membrane Bioreactor",
    "bio_active_filtration": "Bio-Active Filtration",
    "fixed_bed": "Fixed Bed Bioreactor",
    "waiv": "WAIV",
    "smp": "SMP",
    "secondary_treatment_wwtp": "Secondary Treatment WWTP",
}

# formatting pyunits
pretty_dims = {
    "Mgallons/day": "MGD",
    "m^3/hr": "m³/hr",
    "m^3/day": "m³/d",
    "liter/second": "L/s",
}


# def scrape_detailed_params(repo_path):
#     """
#     Parse watertap/costing/unit_models/*.py for Var/Param definitions
#     with initialize= defaults.  Returns dict[module_name] -> list[dict].
#     """
#     costing_dir = os.path.join(repo_path, "watertap", "costing", "unit_models")
#     results = {}

#     for fname in sorted(os.listdir(costing_dir)):
#         if not fname.endswith(".py") or fname.startswith("__"):
#             continue

#         module_name = fname.replace(".py", "")
#         with open(os.path.join(costing_dir, fname)) as f:
#             content = f.read()

#         params = []
#         pattern = (
#             r"(?:blk|parameter_blk|cost_blk)\.([\w]+)\s*=\s*"
#             r"pyo\.(?:Var|Param)\(\s*([^)]+)\)"
#         )
#         for match in re.finditer(pattern, content):
#             name = match.group(1)
#             body = match.group(2)

#             init_match = re.search(r"(?:initialize|default)\s*=\s*([0-9eE.\-+]+)", body)
#             units_match = re.search(r"units\s*=\s*([^,\n]+?)(?:\s*,|\s*\))", body)
#             doc_match = re.search(r'doc\s*=\s*["\']([^"\']+)["\']', body)

#             if init_match:
#                 params.append(
#                     {
#                         "name": name,
#                         "default": init_match.group(1),
#                         "units": units_match.group(1).strip() if units_match else "",
#                         "doc": doc_match.group(1) if doc_match else "",
#                     }
#                 )

#         if params:
#             results[module_name] = params

#     return results


def scrape_zo_params(repo_path=None):
    """
    Parse watertap/data/techno_economic/*.yaml for zero-order costing
    parameters (A, B, energy, recovery).  Returns list[dict].
    """
    # if repo_path is None:
    #     repo_path = repo_path_default
    yaml_dir = os.path.join(repo_path, "watertap", "data", "techno_economic")

    # these units don't follow C = A*x**b for capex
    other_skips = [
        "hrcs_case_1575",
        "case_1617",
        "component_list",
        "global_costing",
        "magprex_case_1575",
        "amo_1595",
        "amo_1690",
        "default_case_study",
        "groundwater_treatment_case_study",
        "peracetic_acid_case_study",
    ]
    unit_skips = [
        "anaerobic_mbr_mec",
        "autothermal_hydrothermal_liquefaction",
        # "bioreactor",
        "brine_concentrator",
        "cando_p",
        "centrifuge",
        "chlorination",
        "chemical_addition",
        "cloth_media_filtration",
        "cofermentation",
        "constructed_wetlands",
        "coag_and_floc",
        "crystallizer",
        "deep_well_injection",
        "dmbr",
        "electrochemical_nutrient_removal",
        "electrocoagulation",
        "evaporation_pond",
        "gac",
        "gas_sparged_membrane",
        "hrcs",
        "hydrothermal_gasification",
        "iron_and_manganese_removal",
        "mabr",
        "magprex",
        "filter_press",
        "ion_exchange",
        "membrane_evaporator",
        "metab",
        "microbial_battery",
        # "municipal_wwtp",
        "ozone",
        "ozone_aop",
        "peracetic_acid_disinfection",
        "photothermal_membrane",
        "pump_electricity",
        "sedimentation",
        # "secondary_treatment_wwtp",
        "storage_tank",
        "struvite_classifier",
        "suboxic_anaerobic_sludge_process",
        # "smp",
        "suboxic_activated_sludge_process",
        "supercritical_salt_precipitation",
        "uv",
        "uv_aop",
        "vfa_recovery",
        "well_field",
    ]

    # NOTE: below are some general notes on ZO units
    # dissolved_air_flotation - same as clarifier, create new cost relationship
    # settling_pond - same as clarifier, remove
    # these units are the same costing as bio_active_filtration
    # - aeration_basin
    # - cartridge_filtration
    # - conventional_activated_sludge
    # - dual_media_filtration
    # - media_filtration
    # - mbr
    # - tri_media_filtration
    # - walnut_shell_filter
    # these units are based on nothing / have no references / have poor costing relationships:
    # - blending_reservoir
    # - buffer_tank
    # - feed_water_tank
    # - landfill
    # - sludge_tank
    # - tramp_oil_tank
    # - secondary_treatment_wwtp
    # - municipal_wwtp
    # these units are just pumps and are redundant
    # - municipal_drinking
    # - surface_discharge
    # - cooling_tower
    # - cooling_supply
    # - air_flotation
    # - pump_electricity
    # these units are based on a single value
    # - electrodialysis_reversal
    # - decarbonator

    results = []

    for fname in sorted(os.listdir(yaml_dir)):

        if not fname.endswith(".yaml"):
            continue
        if any(u in fname.lower() for u in unit_skips + other_skips):
            continue

        with open(os.path.join(yaml_dir, fname)) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            continue

        module_name = fname.replace(".yaml", "")
        # default subtype
        section = data.get("default", data)
        if not isinstance(section, dict):
            continue

        cc = section.get("capital_cost", None)
        if cc is None:
            continue

        info = {"module_name": module_name}
        sname = fname.replace(".yaml", "")

        # print(sname)

        if sname in u_name_dict:
            info["name"] = u_name_dict[sname]
        else:
            info["name"] = fname.replace(".yaml", "").replace("_", " ").title()

        print(f"Processing {info['name']}...")

        # Extract A and B
        # for key in ("capital_a", "capital_a_parameter"):
        key = "capital_a_parameter"
        if "capital_a_parameter" in cc:
            v = cc[key]
            if isinstance(v, dict) and "value" in v:
                try:
                    info["A"] = float(v["value"])
                    units = v.get("units", "")
                    if "USD_" in units:
                        info["year"] = units.split("USD_")[1][:4]
                except (ValueError, TypeError):
                    pass

        # for key in ("capital_b", "capital_b_parameter"):
        key = "capital_b_parameter"
        if key in cc:
            v = cc[key]
            if isinstance(v, dict) and "value" in v:
                try:
                    b = float(v["value"])
                    units = v.get("units", "")
                    if "dimensionless" in units.lower() or units == "":
                        info["B"] = b
                except (ValueError, TypeError):
                    pass

        # Energy intensity
        ei = section.get(
            "energy_electric_flow_vol_inlet",
            section.get("electricity_intensity_parameter", {}),
        )
        ei_fq = [  # electricity is a function of flow rate (pumping electricity)
            "backwash_solids_handling",
            "anaerobic_mbr_mec",
            "cofermentation",
            "deep_well_injection",
            "gas_sparged_membrane",
            "ion_exchange",
            "municipal_drinking",
            "surface_discharge",
            "sw_onshore_intake",
            "well_field",
            "mbr",
        ]
        ei_fx = [  # electricity is a function of other variables
            "electrodialysis_reversal"
        ]
        if "value" in ei:
            if ei["value"] == 0:
                # don't want to report zero
                info["energy"] = "None"
            else:
                info["energy"] = f"{ei['value']:.3}"
        elif module_name in ei_fq:
            info["energy"] = "f(Q)"
        elif module_name in ei_fx:
            info["energy"] = "f(x)"
        else:
            info["energy"] = "None"

        # Recovery
        rec = section.get("recovery_frac_mass_H2O", {})
        if "value" in rec:
            if rec["value"] == 1:
                info["recovery"] = "N/A"
            else:
                info["recovery"] = rec["value"]
        else:
            info["recovery"] = "N/A"

        # Reference
        ref = cc.get("reference", "Unknown")
        info["reference"] = ref

        rs = cc.get("reference_state", {})
        if "value" in rs:
            info["reference_state"] = f"{rs['value']:.1f}"
            info["reference_state_units"] = rs["units"]

        results.append(info)

    return results


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "SmallBody",
            parent=styles["Normal"],
            fontSize=7,
            leading=8.5,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallBold",
            parent=styles["Normal"],
            fontSize=7,
            leading=8.5,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallBoldCentered",
            parent=styles["Normal"],
            fontSize=7,
            leading=8.5,
            fontName="Helvetica-Bold",
            alignment=1,  # Center alignment
        )
    )
    styles.add(
        ParagraphStyle(
            "TinyBody",
            parent=styles["Normal"],
            fontSize=6.5,
            leading=7.5,
        )
    )
    styles.add(
        ParagraphStyle(
            "TinyBodyCentered",
            parent=styles["Normal"],
            fontSize=6.5,
            leading=7.5,
            alignment=1,  # Center alignment
        )
    )
    styles.add(
        ParagraphStyle(
            "SectionHead",
            parent=styles["Heading2"],
            fontSize=11,
            spaceAfter=4,
            spaceBefore=8,
            textColor=colors.HexColor("#1a5276"),
        )
    )
    styles.add(
        ParagraphStyle(
            "DocTitle",
            parent=styles["Title"],
            fontSize=16,
            textColor=colors.HexColor("#1a5276"),
        )
    )
    styles.add(
        ParagraphStyle(
            "FootNote",
            parent=styles["Normal"],
            fontSize=6,
            leading=7,
            textColor=colors.grey,
        )
    )
    return styles


def _table_style():
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dbdb")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [colors.white, colors.HexColor("#f8f9f9")],
            ),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]
    )


def create_detailed_costing_story(story):
    styles = build_styles()

    P = Paragraph

    story.append(P("WaterTAP Costing Reference Guide", styles["DocTitle"]))
    # story.append(
    #     P(
    #         "Default Parameters Across All Unit Models — "
    #         "For Technology Screening &amp; Proposal Support",
    #         styles["Normal"],
    #     )
    # )
    story.append(Spacer(1, 6))

    story.append(
        P(
            "Detailed Unit Model Costing Parameters",
            styles["SectionHead"],
        )
    )
    story.append(
        P(
            "These models compute costs from physics-derived sizing variables. "
            "Capital cost equations vary by unit type. All parameters are tunable.",
            styles["SmallBody"],
        )
    )
    story.append(Spacer(1, 4))

    # ── BUILD detailed_rows from scraped data ──────────────────────
    # detailed_rows = [
    #     ["Unit Model", "Capital Cost", "Key Default Parameters", "Reference"],
    #     ["-", "—", "—", "—"],
    # ]
    membrane_cost = "C<sub>mem</sub>"
    membrane_cost_hp = "C<sub>mem,HP</sub>"
    electrode_cost = "C<sub>electrode</sub>"
    pump_cost = "C<sub>pump</sub>"
    px_cost = "C<sub>PX</sub>"
    evap_cost = "C<sub>evap</sub>"
    hx_cost = "C<sub>HX</sub>"
    reactor_cost = "C<sub>reactor</sub>"
    compressor_cost = "C<sub>comp</sub>"
    area_mem = "A<sub>mem</sub>"
    area_surf = "A<sub>surf</sub>"
    erd_cost = "C<sub>ERD</sub>"
    # detailed_data = scrape_detailed_params(repo_path)
    detailed_header = [
        "Unit Model",
        "Capital Cost Eq.",
        "Key Default Parameters",
        "Reference",
    ]
    # rows for detailed costing method are hardcoded
    detailed_rows = [
        [
            "Reverse Osmosis",
            f" {membrane_cost} × {area_mem}",
            f"{membrane_cost} = $30/m<super>2</super><br/><br/>{membrane_cost_hp} = $75/m<super>2</super><br/><br/>replacement = 20 %/yr",
            "Bartholomew et al. 2018",
        ],
        [
            "OARO",
            f" {membrane_cost} × {area_mem}",
            f"{membrane_cost} = $30/m<super>2</super><br/><br/>{membrane_cost_hp} = $50/m<super>2</super><br/><br/>replacement = 15 %/yr",
            "Bartholomew et al. 2017",
        ],
        [
            "Nanofiltration",
            f" {membrane_cost} × {area_mem}",
            f"{membrane_cost} = $15/m<super>2</super><br/><br/>replacement = 20 %/yr",
            "",
        ],
        [
            "Membrane Distillation",
            f" {membrane_cost} × {area_mem}",
            f"{membrane_cost} = $56/m<super>2</super><br/><br/>replacement = 20 %/yr",
            "",
        ],
        [
            "Electrodialysis",
            f" {membrane_cost} × {area_mem} + {electrode_cost}",
            f"{membrane_cost} = $160/m<super>2</super><br/><br/>{electrode_cost} = $2100/m<super>2</super><br/><br/>replacement = 20 %/yr (mem + electrode)",
            "",
        ],
        [
            "Electrolyzer",
            "C<sub>mem</sub> + C<sub>anode</sub> + C<sub>cathode</sub>",
            "C<sub>mem</sub> = $25/m<super>2</super><br/><br/>C<sub>anode</sub> = $300/m<super>2</super><br/><br/>C<sub>cathode</sub> = $600/m<super>2</super><br/><br/>mat fraction = 65 %",
            "Desalination 452 (2019) 265–278",
        ],
        [
            "Ion Exchange (Cation / Anion)",
            "C<sub>resin</sub> + C<sub>vessels</sub> + C<sub>tanks</sub>",
            "C<sub>res,AX</sub> = $205/ft<super>3</super><br/><br/>C<sub>res,CX</sub> = $153/ft<super>3</super><br/><br/> vessel A=1596.5, b=0.46<br/><br/>resin replacement = 5 %/yr",
            "EPA-WBS 2021",
        ],
        [
            "GAC",
            "C<sub>contactor</sub> + C<sub>gac</sub> + C<sub>other</sub>",
            "C<sub>contactor</sub> = polynomial f(vol)<br/><br/>C<sub>gac</sub> = exponential f(mass)<br/><br/>C<sub>other</sub> = power f(vol)<br/><br/>C<sub>regen</sub> = $4.28/kg<br/><br/>C<sub>makeup</sub> = $4.58/kg",
            "EPA-WBS 2021",
        ],
        [
            "Pump (high pressure)",
            f" {pump_cost} × W<sub>mech</sub>",
            f"{pump_cost} = $53/W",
            "Malek et al. 1996",
        ],
        [
            "Pump (low pressure)",
            f" {pump_cost} × Q",
            f"{pump_cost} = $889/(m<super>3</super>/s)",
            "",
        ],
        [
            "Pressure Exchanger",
            f" {px_cost} × Q",
            f"{px_cost} = $535/(m<super>3</super>/s)",
            "",
        ],
        [
            "Energy Recovery Device",
            f" {erd_cost} × Q",
            f"{erd_cost} = $535/(m<super>3</super>/s)",
            "",
        ],
        [
            "Evaporator",
            f" {evap_cost} × A<sub>evap</sub>",
            f"{evap_cost} = $1000/m<super>2</super><br/><br/>material factor = 1.0",
            "",
        ],
        [
            "Heat Exchanger",
            f" {hx_cost} × A<sub>hx</sub>",
            f"{hx_cost} = $300/m<super>2</super><br/><br/>steam = $0.008/kg",
            "",
        ],
        [
            "Crystallizer",
            " A × (Q/Q<sub>ref</sub>)<super>B</super> × IEC",
            "A = $675,000<br/><br/>Q<sub>ref</sub> = 1 m<super>3</super>/hr<br/><br/>B = 0.53<br/><br/>IEC = 1.43<br/><br/>steam = $0.004/kg",
            "Woods, 2007; Diab and Gerogiorgis, 2017; Yusuf et al., 2019; Panagopoulos (2019)",
        ],
        [
            "Compressor",
            f" {compressor_cost} × W<sub>mech</sub><super>B</super>",
            f"{compressor_cost} = $7364<br/><br/>B = 0.7",
            "El-Sayed et al., 2001",
        ],
        [
            "UV+AOP",
            f"{reactor_cost} + C<sub>lamp</sub>",
            f"{reactor_cost} = $202.35/kW<br/><br/>C<sub>lamp</sub> = $235.50/kW<br/><br/>lamp replacement = 33.3 %/yr",
            "",
        ],
        [
            "Clarifier (Primary)",
            "A × Q<sub>MGD</sub><super>B</super>",
            "A = $538,746<br/><br/>B = 0.7",
            "Byun et al. 2022",
        ],
        [
            "Clarifier (Circular)",
            f"A × {area_surf}² + B × {area_surf} + C",
            "A = -6 × 10<super>-4</super><br/><br/>B = 98.952<br/><br/>C = 191,806",
            "Sharma et al. 2013",
        ],
        [
            "Clarifier (Rectangular)",
            f" A × {area_surf}² + B × {area_surf} + C",
            "A = -2.9 × 10<super>-3</super><br/><br/>B = 169.19<br/><br/>C = $94,365",
            "Sharma et al. 2013",
        ],
        [
            "CSTR",
            " A × V<super>B</super>",
            "A = $1,246.1/m<super>3</super><br/><br/>B = 0.71",
            "C.C. Tang, 1984",
        ],
        [
            "Anaerobic Digester",
            " A × (Q/Q<sub>ref</sub>)<super>B</super>",
            "A = $19.36M<br/><br/>B = 0.6<br/><br/>Q<sub>ref</sub> = 911 m<super>3</super>/day",
            "NREL Waste-to-Energy Model",
        ],
        [
            "Dewatering (Filter Belt Press)",
            " A × Q + B",
            "A = $146.29/(gal/hr) <br/><br/>B = $433,972",
            "McGivney & Kawamura, 2008",
        ],
        [
            "Dewatering (Filter Plate Press)",
            " A × Q<super>B</super>",
            "A = $102,784/(gal/hr)<br/><br/>B = 0.4216",
            "McGivney & Kawamura, 2008",
        ],
        [
            "Dewatering (Centrifuge)",
            " A × Q + B",
            "A = $328.03/(gal/hr)<br/><br/>B = $751,295",
            "McGivney & Kawamura, 2008",
        ],
        [
            "Thickener",
            f" A × {area_surf} + B",
            "A = 4729.8/ft²<br/><br/>B = $37,068",
            "McGivney & Kawamura, 2008",
        ],
        [
            "Steam Ejector",
            " A × (M)<super>B</super>",
            "M = motive steam + entrained vapor; kg/hr<br/><br/>A = $1949<br/><br/>B = 0.3<br/><br/>steam = $0.008/kg",
            "Gabriel 2015, Desalination",
        ],
        [
            "Mixer",
            "C<sub>mix</sub> × Q",
            "Generic = $361/(L/s)<br/><br/>NaOCl mixer = $5.08/(m<super>3</super>/day)<br/><br/>CaOH<sub>2</sub> = 873.9/(kg/day)",
            "",
        ],
        [
            "Chiller",
            "C<sub>chill</sub> × (W<sub>duty</sub>/COP)",
            "C<sub>chill</sub> = $200/kW<br/><br/>COP = 7",
            "",
        ],
        [
            "Heater (Electric)",
            "C<sub>heat</sub> × W<sub>duty</sub>/eff",
            "C<sub>heat</sub> = $66/kW<br/><br/>eff = 0.99",
            "",
        ],
    ]

    # for automatically scraping parameters; doesn't work very well
    # for module_name, params in sorted(detailed_data.items()):
    #     # Join parameters into "name = default units — doc"
    #     param_strs = []
    #     for p in params:
    #         parts = []
    #         parts.append(f"{p['name']} = {p['default']}")
    #         if p["units"]:
    #             parts[-1] += f" {p['units']}"
    #         if p["doc"]:
    #             parts.append(p["doc"])
    #         param_strs.append(" — ".join(parts))
    #     param_text = "; ".join(param_strs)
    #     # Capital cost basis and reference currently unknown in scraped output
    #     detailed_rows.append([module_name, "—", param_text, "—"])

    detailed_header = [
        "Unit Model",
        "Capital Cost Eq.",
        "Key Default Parameters",
        "Reference",
    ]
    table_data = [[P(c, styles["SmallBoldCentered"]) for c in detailed_header]]
    print(f"\nCreating Detailed Costing Table with {len(detailed_rows)} units...\n")

    detailed_rows = sorted(detailed_rows, key=lambda r: r[0])  # sort by unit model name
    for i, row in enumerate(detailed_rows):
        print(f"Adding {row[0]} to table")
        style1 = styles["TinyBody"]
        style2 = styles["TinyBodyCentered"]
        table_data.append(
            [P(c, style1 if j in [1, 2] else style2) for j, c in enumerate(row)]
        )

    t = Table(table_data, colWidths=[1.1 * inch, 1.4 * inch, 2 * inch, 1.2 * inch])
    t.setStyle(_table_style())
    story.append(t)

    return story


def create_zo_costing_story(story):
    styles = build_styles()

    P = Paragraph

    # ── PAGE 2: zero-order models ─────────────────────────────────-
    story.append(PageBreak())
    story.append(P("WaterTAP Costing Reference Guide (continued)", styles["DocTitle"]))
    story.append(Spacer(1, 4))

    story.append(
        P(
            "Zero-Order Unit Model Costing — Default Parameters",
            styles["SectionHead"],
        )
    )
    story.append(
        P(
            "Zero-order models use: C<sub>cap</sub> = A × "
            "(Q<sub>in</sub>/Q<sub>basis</sub>)<super>B</super>. "
            "A is in USD for the reference year shown. Costs are CPI-adjusted to "
            "the study year.",
            styles["SmallBody"],
        )
    )
    story.append(Spacer(1, 4))

    # ── BUILD zo_rows from scraped data ───────────────────────────
    print()
    zo_data = scrape_zo_params(repo_path)  # list of dicts
    zo_rows = [
        [
            "Unit Model",
            "A",
            "B",
            "Q<sub>basis</sub>",
            "Cost Year",
            "Energy (kWh/m³)",
            "Recovery",
            "Reference",
        ]
    ]
    print(f"\nCreating ZO Table with {len(zo_data)} units...\n")
    zo_sort = "name"
    for z in sorted(zo_data, key=lambda d: d[zo_sort]):
        print(f"Adding {z['name']} to table")
        A_str = f"{z.get('A','—'):,.0f}" if "A" in z else "—"
        B_str = f"{z.get('B','—'):.3f}" if "B" in z else "—"
        if B_str != "—":
            if float(B_str) == 1:
                B_str = "1"
        rs = f"{z.get('reference_state','—')}" if "reference_state" in z else "—"
        rs = f"{float(rs):,.0f}"
        rsu = (
            f" {z.get('reference_state_units','')}"
            if "reference_state_units" in z
            else ""
        )
        rsu = pretty_dims.get(rsu.strip(), rsu)
        ref_state = rs + " " + rsu
        year_str = f"{z.get('year','—')}" if "year" in z else "—"
        energy_str = f"{z.get('energy','—')}" if "energy" in z else "—"
        rec_str = f"{z.get('recovery','—')}" if "recovery" in z else "—"
        ref_str = f"{z.get('reference','—')}" if "reference" in z else "—"
        zo_rows.append(
            [z["name"], A_str, B_str, ref_state, year_str, energy_str, rec_str, ref_str]
        )

    table_data2 = []
    for i, row in enumerate(zo_rows):
        style = styles["SmallBoldCentered"] if i == 0 else styles["TinyBodyCentered"]
        table_data2.append([P(c, style) for c in row])

    t2 = Table(
        table_data2,
        colWidths=[
            1.2 * inch,
            0.8 * inch,
            0.6 * inch,
            0.6 * inch,
            0.7 * inch,
            0.7 * inch,
            0.7 * inch,
            1.2 * inch,
        ],
    )
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dbdb")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f8f9f9")],
                ),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    story.append(t2)

    story.append(Spacer(1, 6))
    story.append(
        P(
            "All parameters are defaults and fully tunable by the user. "
            "Source: watertap-org/watertap GitHub repository.",
            styles["FootNote"],
        )
    )

    return story


def create_watertap_costing_reference(save_as):
    """Build the two-page costing reference PDF."""

    doc = SimpleDocTemplate(
        save_as,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )
    story = []

    story = create_detailed_costing_story(story)
    story = create_zo_costing_story(story)

    doc.build(story)

    print(f"\nWaterTAP costing reference saved here:\n{save_as}")


def generate_combined_pdf(cost_curves_path, costing_ref_path, save_as):
    """Combine costing reference with cost curves doc."""

    merger = pypdf.PdfWriter()
    merger.append(cost_curves_path)
    merger.append(costing_ref_path)
    with open(save_as, "wb") as f_out:
        merger.write(f_out)

    merger.close()


if __name__ == "__main__":

    save_as = f"{here}/DRAFT_watertap_costing_reference.pdf"

    create_watertap_costing_reference(save_as)

    # cost_curves_path = f"{here}/DRAFT_watertap_cost_curves_doc.pdf"
    # costing_ref_path = f"{here}/DRAFT_watertap_costing_reference.pdf"
    # save_as = f"{here}/DRAFT_watertap_cost_curves_and_reference.pdf"

    # generate_combined_pdf(cost_curves_path, costing_ref_path, save_as)
