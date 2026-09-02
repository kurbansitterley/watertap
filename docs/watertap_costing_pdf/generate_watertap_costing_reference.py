"""
Generate WaterTAP Costing Reference Guide PDF.

Scrapes default costing parameters from the WaterTAP repository
(both detailed unit models and zero-order YAML files) and produces
a two-page reference PDF.

Requirements:
- watertap-dev environment + reportlab (pip) + pyyaml (pip) + pypdf (pip)

e.g.,
conda create --name watertap-dev-pdf-docs python=3.12
conda activate watertap-dev-pdf-docs
pip install reportlab pyyaml pypdf
pip install -r requirements-dev.txt
"""

import pypdf
import os
import pprint
import re
import yaml
from datetime import datetime
from reportlab.lib.pagesizes import letter, landscape
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
    "municipal_wwtp": "Municipal WWTP",
    "anaerobic_mbr_mec": "Anaerobic MBR-MEC",
    "hrcs": "High-Rate Solids Clarifier",
    "gac": "GAC",
    "iron_and_manganese_removal": "Fe/Mn Removal",
    "mabr": "MABR",
    "magprex": "MagPrex",
    "uv": "UV",
    "uv_aop": "UV+AOP",
    "vfa_recovery": "VFA Recovery",
    "ozone_aop": "Ozone+AOP",
    "dmbr": "DMBR",
    "CANDO_P": "CANDO-P",
    "coag_and_floc": "Coagulation & Flocculation",
}

# formatting pyunits
pretty_dims = {
    "Mgallons/day": "MGD",
    "m^3/hr": "m³/hr",
    "m^3/day": "m³/d",
    "liter/second": "L/s",
    "Mgallons": "× 10<sup>6</sup> gal",
    "gallons": "gal",
    "gallons/day": "gal/d",
    "gallon/minute": "gpm",
    "ft^2": "ft²",
    "m^2": "m²",
    "mg/liter": "mg/L",
    "dimensionless": "",
    "lb/hour": "lb/hr",
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


def get_subtype_info(data, subtype="default"):

    section = data[subtype]

    assert isinstance(
        section, dict
    ), f"Expected dict for module {module_name} subtype {subtype}, got {type(section)}"
    # print(module_name, subtype)
    if module_name in ["ozone", "ozone_aop"]:
        # pprint.pprint(section)
        # assert False
        cc = section["capital_cost"]["ozone_capital_cost"]
    else:
        cc = section.get("capital_cost", None)
    if cc is None:
        return None

    info = {"module_name": module_name, "subtype": subtype}
    sname = fname.replace(".yaml", "")

    # print(sname)

    if sname in u_name_dict:
        info["name"] = u_name_dict[sname]
    else:
        info["name"] = fname.replace(".yaml", "").replace("_", " ").title()

    print(f"Processing {info['name']}...")

    # Extract A and B
    # if module_name != "ion_exchange":
    key = "capital_a_parameter"
    if "capital_a_parameter" in cc and "capital_c_parameter" not in cc:
        v = cc[key]
        if isinstance(v, dict) and "value" in v:
            try:
                info["A"] = float(v["value"])
                units = v.get("units", "")
                if "USD_" in units:
                    info["year"] = units.split("USD_")[1][:4]
                else:
                    USDs = list(extract_this_value(v, this_value="USD"))
                    if USDs:
                        info["year"] = USDs[0][:4]
            except (ValueError, TypeError):
                pass

    key = "capital_b_parameter"
    if key in cc and "capital_c_parameter" not in cc:
        v = cc[key]
        if isinstance(v, dict) and "value" in v:
            try:
                b = float(v["value"])
                units = v.get("units", "")
                if "dimensionless" in units.lower() or units == "":
                    info["B"] = b
            except (ValueError, TypeError):
                pass

    if "year" not in info:
        USDs = list(set(list(extract_this_value(cc, this_value="USD"))))
        if USDs:
            years = list(USD.split("_")[1][:4] for USD in USDs if "_" in USD)
            if years:
                info["year"] = "; ".join(sorted(set(years)))
        # if USDs:
        #     USD = USDs[0]
        #     USD = USD.split("/")[0]  # remove any trailing units, assume it is numerator
        #     if USD:
        #         info["year"] = USD.split("_")[1][:4]
        # print(f"Year not found in capital_a_parameter, extracted from cc: {info['year']}")
        # print(USDs)
        # print(USD)
        # assert False

    key = "validity_range"
    if key in cc:
        vr = cc[key]
        # pprint.pprint(vr)
        if isinstance(vr, dict) and all(
            k in vr for k in ["lower_bound", "upper_bound"]
        ):
            for k, v in vr.items():
                valid_range = float(v["value"])
                valid_range_units = v.get("units", None)
                info[f"valid_range_{k}"] = valid_range
                info[f"valid_range_{k}_units"] = valid_range_units
        elif isinstance(vr, dict):
            info["valid_range_vars"] = list(vr.keys())
            for k, v in vr.items():
                if k not in ["lower_bound", "upper_bound"]:
                    # there are nested validity ranges
                    for subk, subv in v.items():
                        info[f"valid_range_{k}_{subk}"] = float(subv["value"])
                        info[f"valid_range_{k}_{subk}_units"] = subv.get("units", None)
        else:
            raise ValueError(
                f"Unexpected validity range format for module {module_name}"
            )

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
        "water_pumping_station",
        "vfa_recovery",
    ]
    ei_fx = [
        "electrodialysis_reversal",
        "chemical_addition",
        "electrocoagulation",
        "filter_press",
        "ozone",
        "ozone_aop",
        "coag_and_floc",
        "brine_concentrator",
        "crystallizer",
    ]  # electricity is a function of other variables
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
    all_refs = list(set(list(extract_this_key(section, this_key="reference"))))
    all_refs = [
        ref for ref in all_refs if not any(x in ref for x in ["https://", "http://"])
    ]
    if len(all_refs) == 0:
        info["reference"] = "Unknown"
    else:
        info["reference"] = "; <br/>".join(all_refs)
    # ref = cc.get("reference", "Unknown")
    # info["reference"] = ref

    rs = cc.get("reference_state", {})
    if "value" in rs:
        info["reference_state"] = f"{rs['value']:.1f}"
        info["reference_state_units"] = rs["units"]

    # validity_ranges = list(set(list(extract_this_key(section, this_key="validity_range"))))
    # pprint.pprint(validity_ranges)
    # assert False

    return info


def extract_this_value(data, this_value="USD"):
    """Recursively yields all string values that contain 'USD' anywhere in the data."""
    if isinstance(data, dict):
        for val in data.values():
            # If the value itself is a string, check it
            if isinstance(val, str) and this_value.upper() in val.upper():
                yield val
            else:
                # Otherwise, keep digging deeper
                yield from extract_this_value(val, this_value=this_value)

    elif isinstance(data, list):
        for item in data:
            if isinstance(item, str) and this_value.upper() in item.upper():
                yield item
            else:
                yield from extract_this_value(item, this_value=this_value)
    # If the data is neither a dict nor a list, do nothing (base case)


def extract_this_key(data, this_key="reference"):
    """Find all values associated with the this_key."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == this_key:
                # Convert value to string in case it's an int, float, etc.
                yield str(value)
            else:
                yield from extract_this_key(value, this_key=this_key)
    elif isinstance(data, list):
        for item in data:
            yield from extract_this_key(item, this_key=this_key)


def scrape_zo_params(
    repo_path=None,
    get_by_subtype=[
        "ion_exchange",
        "chemical_addition",
        "screen",
        "water_pumping_station",
    ],
):
    """
    Parse watertap/data/techno_economic/*.yaml for zero-order costing
    parameters (A, B, energy, recovery, validity_range).  Returns list[dict].
    """
    global module_name, fname

    yaml_dir = os.path.join(repo_path, "watertap", "data", "techno_economic")

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
        "water_sources",
    ]
    unit_skips = [
        # "anaerobic_mbr_mec",
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
    unit_skips = ["secondary_treatment_wwtp", "municipal_wwtp"]

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
        subtypes = list(data.keys())

        if "default" not in subtypes:
            continue  # its not a unit model
        if len(subtypes) > 1:
            units_with_subtypes.append((module_name, subtypes))

        if module_name in get_by_subtype:
            for subtype in subtypes:
                if subtype == "default":
                    continue
                info = get_subtype_info(data, subtype=subtype)
                if info is not None:
                    info["name"] = (
                        info["name"] + " (" + subtype.replace("_", " ").title() + ")"
                    )
                    results.append(info)
        else:
            subtype = "default"
            info = get_subtype_info(data, subtype=subtype)
            if info is not None:
                results.append(info)
        if info is None:
            print(f"Skipping {module_name} due to missing capital_cost")
            continue
    return results


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "SmallBody",
            parent=styles["Normal"],
            fontSize=7,
            leading=10.5,
        )
    )
    styles.add(
        ParagraphStyle(
            "SmallBold",
            parent=styles["Normal"],
            fontSize=7,
            leading=10.5,
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
            "TableColumnHeader",
            parent=styles["Normal"],
            fontSize=8.5,
            leading=14,
            fontName="Helvetica-Bold",
            alignment=1,  # Center alignment
            textColor=colors.white,
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
            leading=9.5,
            alignment=1,  # Center alignment
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell1",
            parent=styles["Normal"],
            fontSize=6.5,
            leading=7.5,
            alignment=1,  # Center alignment
        )
    )
    styles.add(
        ParagraphStyle(
            "TableCell2",
            parent=styles["Normal"],
            fontSize=6.5,
            leading=12.5,
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

    # -- BUILD detailed_rows from scraped data --

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
            "Shamlou, Vidic, & Khanna, 2022",
        ],
        [
            "Electrodialysis",
            f" {membrane_cost} × {area_mem} + {electrode_cost}",
            f"{membrane_cost} = $160/m<super>2</super><br/><br/>{electrode_cost} = $2100/m<super>2</super><br/><br/>replacement = 20 %/yr (mem + electrode)",
            "Bian et al., 2018",
        ],
        [
            "Electrolyzer",
            "C<sub>mem</sub> + C<sub>anode</sub> + C<sub>cathode</sub>",
            "C<sub>mem</sub> = $25/m<super>2</super><br/><br/>C<sub>anode</sub> = $300/m<super>2</super><br/><br/>C<sub>cathode</sub> = $600/m<super>2</super><br/><br/>mat fraction = 65 %",
            "Bommaraju & O’Brien, 2015; Kent (Ed.), 2007; O’Brien, Bommaraju, & Hine, 2005; Yee, 2012",
        ],
        [
            "Ion Exchange (Cation / Anion)",
            "C<sub>resin</sub> + C<sub>vessels</sub> + C<sub>tanks</sub><br/><br/>C<sub>i</sub> = A<super>b</super>",
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
            "Bartholomew et al. 2018",
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
            "Woods, 2007; Diab and Gerogiorgis, 2017; Yusuf et al., 2019; Panagopoulos, 2019",
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
            "UV System Cost Analysis Tool (UVCAT); Wright, Gaithuma, Greene, Aieta, 2006",
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
            "Gabriel, 2015",
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
        [
            "Electrocoagulation",
            "C<sub>electrodes</sub> + C<sub>powersupply</sub> + C<sub>reactor</sub>",
            "C<sub>electrodes</sub> = $2.23/kg (aluminum), $3.41/kg (iron)<br/><br/>C<sub>powsup</sub> = linear f(watt)<br/><br/>C<sub>reactor</sub> = power f(vol)",
            "McGivney & Kawamura, 2008; Smith, 2005; Anuf et al., 2022; magna-power.com",
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
    table_data = [[P(c, styles["TableColumnHeader"]) for c in detailed_header]]
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
    global units_with_subtypes
    units_with_subtypes = list()
    styles = build_styles()

    P = Paragraph

    # -- TABLE 2: ZO models --
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
            # "This table is for WaterTAP Zero Order Models "
            "Units with A and B values follow the equation: C<sub>cap</sub> = A × "
            "(Q<sub>in</sub>/Q<sub>basis</sub>)<super>B</super>. "
            "A is in USD for the reference year shown. ",
            # "Costs are CPI-adjusted to "
            # "the study year.",
            styles["SmallBody"],
        )
    )
    story.append(Spacer(1, 4))

    # -- BUILD zo_rows from scraped data --
    print()
    zo_data = scrape_zo_params(repo_path)  # list of dicts

    zo_rows = [
        [
            "Unit Model",
            "A",
            "B",
            "Q<sub>basis</sub>",
            "Cost Year",
            # "Energy (kWh/m³)",
            "Energy",
            "Recovery",
            "Validity Range",
            "Reference",
        ]
    ]
    print(f"\nCreating ZO Table with {len(zo_data)} units...\n")
    zo_sort = "name"
    missing_AB_symbol = "†"
    year_symbol = "‡"
    added_year_symbol = False
    energy_symbol = "§"
    added_energy_symbol1 = False
    added_energy_symbol2 = False

    for i, z in enumerate(sorted(zo_data, key=lambda d: d[zo_sort]), 1):
        if z["reference"] in ["Unknown", None, "None"]:
            continue
        # name =
        print(f"Adding {z['name']} to table")
        # if z["module_name"] == "ozone":
        #     pprint.pprint(z)
        #     assert False
        # if i == 1:
        #     A_str = f"{z.get('A','—'):,.0f}" if "A" in z else "—<sup>†</sup>"
        # else:
        A_str = (
            f"{z.get('A','—'):,.1f}" if "A" in z else f"<sup>{missing_AB_symbol}</sup>"
        )
        B_str = (
            f"{z.get('B','—'):.3f}" if "B" in z else f"<sup>{missing_AB_symbol}</sup>"
        )
        if B_str != f"<sup>{missing_AB_symbol}</sup>":
            if float(B_str) == 1:
                B_str = "1"
        rs = f"{z.get('reference_state','—')}" if "reference_state" in z else "—"
        if rs != "—":
            rs = f"{float(rs):,.0f}"
        rsu = (
            f" {z.get('reference_state_units','')}"
            if "reference_state_units" in z
            else ""
        )
        rsu = pretty_dims.get(rsu.strip(), rsu)
        ref_state = rs + " " + rsu
        year_str = f"{z.get('year','—')}" if "year" in z else "—"

        if len(year_str) > 4 and not added_year_symbol:
            year_str += f"<sup>{year_symbol}</sup>"
            added_year_symbol = True

        energy_str = f"{z.get('energy','—')}" if "energy" in z else "—"
        if energy_str == "f(Q)" and not added_energy_symbol1:
            energy_str += f"<sup>{energy_symbol}</sup>"
            added_energy_symbol1 = True
        if energy_str == "f(x)" and not added_energy_symbol2:
            energy_str += f"<sup>{energy_symbol}</sup>"
            added_energy_symbol2 = True

        rec_str = f"{z.get('recovery','—')}" if "recovery" in z else "—"
        ref_str = f"{z.get('reference','—')}" if "reference" in z else "—"

        if "valid_range_lower_bound" in z and "valid_range_upper_bound" in z:
            print(f"Processing validity range for {z['name']}")
            vr_low = f"{float(z['valid_range_lower_bound']):,.0f}"
            vr_high = f"{float(z['valid_range_upper_bound']):,.0f}"
            vrul = z.get("valid_range_lower_bound_units", None)
            vruh = z.get("valid_range_upper_bound_units", None)
            assert (
                vrul == vruh
            ), f"Units mismatch for {z['name']} upper vs lower validity range"
            if vrul is None:
                raise ValueError(f"Should have units for {z['name']} validity range")
            vrul = pretty_dims.get(vrul.strip(), vrul)
            validity_range = f"{vr_low}-{vr_high} {vrul}"
        elif "valid_range_vars" in z:
            val_range_var_dict = {
                # brine concentrator
                "flow_vol": "Q<sub>in</sub>",
                "recovery_vol": "RR",
                "tds": "TDS",
                # crystallizer
                "purge_fraction": "f<sub>purge</sub>",
                # chlorination
                "chlorine_dose": "[Cl<sub>2</sub>]",
                # clarifier
                "basin_surface_area": "A<sub>basin</sub>",
                # coag_and_floc
                "rapid_mix": "V<sub>mix</sub>",
                "alum_addition": "m<sub>Alum</sub>",
                "coagulant_addition": "m<sub>Coag</sub>",
                "polymer_addition": "m<sub>Poly</sub>",
                "flocculator": "V<sub>floc</sub>",
                # ion exchange
                "sulfate_influent": "[SO<sub>4</sub>]",
                "tds_influent": "TDS",
                # primary separator
                "basin_area": "A<sub>basin</sub>",
                # ozone / ozone_aop
                "ozone_dose": "[O<sub>3</sub>]",
            }
            validity_range = []
            for var in z["valid_range_vars"]:
                if var not in val_range_var_dict:
                    raise ValueError(
                        f"Unknown validity range variable {var} for {z['name']}"
                    )
                vr_pretty = val_range_var_dict[var]
                if var in ["purge_fraction", "recovery_vol"]:
                    vr_v_low = f"{float(z[f'valid_range_{var}_lower_bound']):.0%}"
                    vr_v_high = f"{float(z[f'valid_range_{var}_upper_bound']):.0%}"
                else:
                    vr_v_low = f"{float(z[f'valid_range_{var}_lower_bound']):,.0f}"
                    vr_v_high = f"{float(z[f'valid_range_{var}_upper_bound']):,.0f}"
                vr_vul = z.get(f"valid_range_{var}_lower_bound_units", None)
                vr_vuh = z.get(f"valid_range_{var}_upper_bound_units", None)
                assert (
                    vr_vul == vr_vuh
                ), f"Units mismatch for {z['name']} upper vs lower validity range for {var}"
                if vr_vul is None:
                    raise ValueError(
                        f"Should have units for {z['name']} validity range for {var}"
                    )
                vr_vul = pretty_dims.get(vr_vul.strip(), vr_vul)
                validity_range.append(f"{vr_pretty}: {vr_v_low}-{vr_v_high} {vr_vul}")
            validity_range = "<br/>".join(validity_range)
        else:
            validity_range = "—"

        zo_rows.append(
            [
                z["name"],
                A_str,
                B_str,
                ref_state,
                year_str,
                energy_str,
                rec_str,
                validity_range,
                ref_str,
            ]
        )

    table_data2 = []
    for i, row in enumerate(zo_rows):
        style_header = styles["TableColumnHeader"]
        style1 = styles["TableCell1"]
        style2 = styles["TableCell2"]
        if i == 0:
            table_data2.append([P(c, style_header) for c in row])
        else:
            table_data2.append(
                [
                    P(c, style1 if j in [1, 2, 3, 4, 5, 6, 7, 9] else style2)
                    for j, c in enumerate(row, 1)
                ]
            )
    # for i, row in enumerate(detailed_rows):
    #     print(f"Adding {row[0]} to table")
    #     style1 = styles["TinyBody"]
    #     style2 = styles["TinyBodyCentered"]
    #     table_data.append(
    #         [P(c, style1 if j in [1, 2] else style2) for j, c in enumerate(row)]
    #     )
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
            1.4 * inch,
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
            "MGD = million gallons per day; m³ = cubic meters; ft² = square feet; RR = water recovery ratio; f<sub>purge</sub> = purge fraction; m<sub>x</sub> = mass flow rate of x; TDS = total dissolved solids; Q = volumetric flow rate",
            styles["FootNote"],
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        P(
            f"{missing_AB_symbol}: units use an alternate costing method",
            styles["FootNote"],
        )
    )

    story.append(Spacer(1, 2))
    story.append(
        P(
            f"{energy_symbol}: f(x) indicates energy calculated from other variables, f(Q) indicates energy calculated with alternative relationship as function of influent flow rate.",
            styles["FootNote"],
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        P(
            f"{year_symbol}: If multiple Cost Year values, costing method includes parameters from several years.",
            styles["FootNote"],
        )
    )
    story.append(Spacer(1, 2))
    story.append(
        P(
            "Chemical addition units costed with the volumetric or mass-based flow of the chemical.",
            styles["FootNote"],
        )
    )
    # story.append(
    #     P(
    #         "All parameters are defaults and fully tunable by the user. "
    #         "Source: watertap-org/watertap GitHub repository.",
    #         styles["FootNote"],
    #     )
    # )

    return story


def create_watertap_costing_reference(save_as):
    """Build the two-page costing reference PDF."""

    doc = SimpleDocTemplate(
        save_as,
        pagesize=letter,
        # pagesize=landscape,
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

    date_str = datetime.now().strftime("%Y-%m-%d")
    save_as = f"{here}/watertap_costing_reference-{date_str}.pdf"

    create_watertap_costing_reference(save_as)

    # To combine in single doc:
    cost_curves_path = f"{here}/watertap_cost_curves_doc-{date_str}.pdf"
    costing_ref_path = f"{here}/watertap_costing_reference-{date_str}.pdf"
    save_as = f"{here}/watertap_cost_curves_and_reference-{date_str}.pdf"

    generate_combined_pdf(cost_curves_path, costing_ref_path, save_as)
