#################################################################################
# WaterTAP Copyright (c) 2020-2026, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Laboratory of the Rockies, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import os
import pprint
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    value,
    Reference,
    assert_optimal_termination,
    check_optimal_termination,
    units as pyunits,
)
from pyomo.opt.results import SolverStatus, TerminationCondition
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_unfixed_variables,
)
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
import idaes.core.util.scaling as iscale

from watertap.core.wt_database import Database
from watertap.core.zero_order_properties import WaterParameterBlock
import watertap.unit_models.zero_order as zo
from watertap.core import (
    build_pt,
    build_sido,
    build_siso,
    build_sido_reactive,
    pump_electricity,
    constant_intensity,
)
from watertap.core.util.model_diagnostics.infeasible import (
    print_infeasible_constraints,
    print_infeasible_bounds,
    print_variables_close_to_bounds,
    print_constraints_close_to_bounds,
    print_close_to_bounds,
)
from watertap.costing import ZeroOrderCosting
from watertap.core.solvers import get_solver
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

here = os.path.dirname(os.path.abspath(__file__))

solver = get_solver()

sidor_db_path = os.path.dirname(os.path.abspath(__file__))

rho = 1000 * pyunits.kg / pyunits.m**3  # density of water

f = "/Users/ksitterl/Documents/Python/watertap/watertap/docs/kurby/WT3_unit_classification_for_doc.xlsx"
df = pd.read_excel(f)

unit_name_list = [i.title() for i in df["Name"]]
model_type_list = df["model type long"]
model_type_short_list = df["type"]

model_type_ref_list = df["model type doc ref"]
elect_func_list = df["energy"]
cost_func_list = df["Cost function"]
zo_name_list = df["zo_unit"]
class_name_list = df["class_name"]
energy_helper_list = df["energy_helper_func"]
class_list = df["class"]

p_subtype_exceptions = {"MetabZO": "hydrogen"}
has_subtype = {}

default_comps = [
    "toc",
    "tkn",
    "tss",
    "cod",
    "tds",
    "nitrogen",
    "phosphates",
    "phosphorus",
    "struvite",
    "nonbiodegradable_cod",
    "hydrogen",
    "ammonium_as_nitrogen",
    "nitrate",
    "bod",
    "organic_solid",
    "organic_liquid",
    "iron",
    "filtration_media",
    "peracetic_acid",
    "total_coliforms_fecal_ecoli",
]


def build_zo_model(unit_class, comps=None, subtype="default", flow_mgd=0.1, conc=100):

    m = ConcreteModel()
    m.db = Database()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.costing = ZeroOrderCosting()
    m.fs.costing.base_currency = pyunits.USD_2023

    if comps is None:
        comps = default_comps

    flows_to_register = [
        "catalyst_ATHTL",
        "polymer",
        "magnesium_chloride",
        "catalyst_HTG",
        "hydrogen_product",
        "filtration_media",
        "filtration_media_disposal",
        "disinfection_solution",
        "struvite_product",
    ]
    for flow in flows_to_register:
        m.fs.costing.register_flow_type(flow, 0)

    # print(m.fs.costing.flow_types)
    # m.fs.costing.flow_types.display()
    # assert False

    # comps = [
    #     "toc",
    #     "tkn",
    #     "tss",
    #     "cod",
    #     "tds",
    #     "nitrogen",
    #     "phosphates",
    #     "phosphorus",
    #     "struvite",
    #     "nonbiodegradable_cod",
    #     "hydrogen",
    #     "ammonium_as_nitrogen",
    #     "nitrate",
    #     "bod",
    #     "organic_solid",
    #     "organic_liquid",
    #     "iron",
    #     "filtration_media",
    #     "peracetic_acid",
    #     "total_coliforms_fecal_ecoli",
    # ]

    m.fs.properties = WaterParameterBlock(
        solute_list=comps,
    )
    unit = getattr(zo, unit_class)

    m.fs.unit = unit(
        property_package=m.fs.properties,
        database=m.db,
        process_subtype=subtype,
    )
    m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    flow_mgd = flow_mgd * pyunits.Mgallons / pyunits.day
    conc = conc * pyunits.mg / pyunits.L

    flow_mass_water = pyunits.convert(flow_mgd * rho, to_units=pyunits.kg / pyunits.s)
    flow_mass_water = flow_mgd * rho

    comp_flow_dict = dict()

    for c in comps:
        # conc is the same for all comps
        comp_flow_dict[c] = flow_mgd * conc

    prop_in = (
        m.fs.unit.properties_in[0]
        if hasattr(m.fs.unit, "properties_in")
        else m.fs.unit.properties[0]
    )
    prop_out = (
        m.fs.unit.properties_out[0] if hasattr(m.fs.unit, "properties_out") else prop_in
    )

    prop_in.flow_mass_comp["H2O"].fix(flow_mass_water)

    for c in comps:
        prop_in.flow_mass_comp[c].fix(comp_flow_dict[c])
        m.fs.properties.set_default_scaling(
            "flow_mass_comp", 1 / value(comp_flow_dict[c]), index=(c)
        )

    data = m.db.get_unit_operation_parameters(m.fs.unit._tech_type, subtype=subtype)
    # pprint.pprint(data)
    # assert False
    m.fs.unit.load_parameters_from_database(use_default_removal=True)
    if m.fs.unit.find_component("removal_frac_mass_comp") is not None:
        for c in comps:
            if c in data["removal_frac_mass_comp"].keys():
                m.fs.unit.removal_frac_mass_comp[0, c].fix(
                    data["removal_frac_mass_comp"][c]["value"]
                )
            else:
                # default removal_fracs of 0 can cause solving problems
                m.fs.unit.removal_frac_mass_comp[0, c].fix(0.1)

    m.fs.costing.cost_process()
    m.fs.costing.add_LCOW(prop_out.flow_vol)
    m.fs.costing.add_specific_energy_consumption(prop_out.flow_vol, name="SEC")

    iscale.calculate_scaling_factors(m)
    return m


skips = [
    "ATHTLZO",
    "CentrifugeZO",
    "ChemicalAdditionZO",
    "ElectroNPZO",
    "EvaporationPondZO",
    "FeedZO",
    "FilterPressZO",
    "HTGZO",
    "MABRZO",
    "MagprexZO",
    "MetabZO",
    "MicrobialBatteryZO",
    "OzoneAOPZO",
    "PeraceticAcidDisinfectionZO",
    "StruviteClassifierZO",
]
flow_problems = [
    "ATHTLZO",  # Error in convert: units not compatible.: pound / hour not compatible with USD_2023 / year.
    "CentrifugeZO",  # Error in convert: units not compatible.: kilogram / hour not compatible with USD_2023 / year.
    "ElectroNPZO",  # Error in convert: units not compatible.: kilogram / hour not compatible with USD_2023 / year.
    "HTGZO",  # Error in convert: units not compatible.: pound / hour not compatible with USD_2023 / year.
    "MagprexZO",  # Error in convert: units not compatible.: kilogram / hour not compatible with USD_2023 / year.
    "MetabZO",  # Error in convert: units not compatible.: kilogram / second not compatible with USD_2023 / year.
    "MicrobialBatteryZO",  # Error in convert: units not compatible.: kilogram / second not compatible with USD_2023 / year.
    "PeraceticAcidDisinfectionZO",  # Error in convert: units not compatible.: liter / second not compatible with USD_2023 / year.
    "StruviteClassifierZO",  # Error in convert: units not compatible.: kilogram / second not compatible with USD_2023 / year.
]
yaml_problems = [
    "ChemicalAdditionZO",  # KeyError: 'Error when trying to retrieve costing parameter for capital_a_parameter. Please check the YAML file for this technology for errors.'
]
costing_problems = [
    "EvaporationPondZO",  # WaterTAP models with a capital_cost must also supply a direct_capital_cost. Found unit fs.unit with `capital_cost` but no `direct_capital_cost`.
    "FilterPressZO",  # WaterTAP models with a capital_cost must also supply a direct_capital_cost. Found unit fs.unit with `capital_cost` but no `direct_capital_cost`.
    "OzoneAOPZO",  # WaterTAP models with a capital_cost must also supply a direct_capital_cost. Found unit fs.unit with `capital_cost` but no `direct_capital_cost`.
]
initialize_problems = [
    "MABRZO",  # fs.unit failed to initialize successfully. Please check the output logs for more information.
]

from watertap.kurby import *
def find_key(d, target_key):
    if isinstance(d, dict):
        for key, value in d.items():
            if key == target_key:
                return value
            result = find_key(value, target_key)
            if result is not None:
                return result
    elif isinstance(d, list):
        # Handle case where values are lists of dicts
        for item in d:
            result = find_key(item, target_key)
            if result is not None:
                return result
    return None

def parse_units(s):

    if "/" in s:
        num, denom = s.split("/")
        num = num.strip()
        num = getattr(pyunits, num)
        denom = denom.strip()
        denom = getattr(pyunits, denom)
        return num / denom

def run_zo_sweep(
    u=None,
    comps=None,
    subtype="default",
    flow_mgd=0.1,
    # flow_mgd_bou=0.1,
    conc=100,
    x=None,
    y=None,
    n_flow_pts=10,
):
    # global units_list 
    # units_list = list()
    # if x is None:
    #     x = np.linspace()
    m = build_zo_model(u, comps=comps, subtype=subtype, flow_mgd=flow_mgd, conc=conc)
    print(f"dof = {degrees_of_freedom(m)}")
    # m.fs.costing.base_currency = pyunits.USD_2014
    # m.fs.unit.initialize()
    # results = solver.solve(m, tee=True)
    # assert_optimal_termination(results)
    # m.fs.unit.report()
    # m.fs.unit.ozone_consumption.display()
    rd = build_results_dict(m, skips=["fs.unit.properties"])
    rd["flow_mgd"] = list()
    data = m.db.get_unit_operation_parameters(m.fs.unit._tech_type, subtype=subtype)
    # pprint.pprint(data)
    # pprint.pprint(rd)
    vr = find_key(data, "validity_range")
    if vr is None:
        return m
    if "lower_bound" in vr.keys():
        # vr = vr["lower_bound"]

        # lb = vr["lower_bound"]["value"]
        # lb_u = vr["lower_bound"]["units"]
        pass
    elif "flow_vol" in vr.keys():
        vr = vr["flow_vol"]
        lb = vr["lower_bound"]["value"]
        lb_u = vr["lower_bound"]["units"]
    elif "flow_mass" in vr.keys():
        vr = vr["flow_mass"]
        lb = vr["lower_bound"]["value"]
        lb_u = vr["lower_bound"]["units"]
    
    else:
        # return m
        lb = find_key(vr, "lower_bound")["value"]
        # lb_u = find_key(vr, "lower_bound_units")
        lb = find_key(vr, "lower_bound")["units"]
        pass
    pprint.pprint(vr)
    lb = vr["lower_bound"]["value"]
    lb_u = vr["lower_bound"]["units"]
    # u = vr["lower_bound"]["units"]
    units_list.append(lb_u)
    # print(pyunits.get_units(lb_u))
    print(lb_u)
    lb_u = parse_units(lb_u)
    # lb = vr["lower_bound"]["value"]
    # pprint.pprint(vr)
    return m


if __name__ == "__main__":
    units_list = list()
    for u in class_name_list:
        print(f"\n\nRunning {u}\n\n")
        if u in skips:
            print(f"skipping {u}")
            continue
        try:
            m = run_zo_sweep(u=u, comps=None, flow_mgd=1, conc=0.1)
        except Exception as e:
            print(f"Error running {u}: {e}")
            # continue
            raise
    pprint.pprint(set(units_list))
    # m = run_zo_sweep(u="TriMediaFiltrationZO", comps=["toc"], flow_mgd=1, conc=0.1)
    # assert False
    # u = "TriMediaFiltrationZO"
    # m = build_zo_model(u)
    # m.fs.unit.display()
    # from watertap.kurby import *

    # res = dict()
    # subtype = "default"
    # u = "OzoneZO"
    # m = build_zo_model(u, comps=["toc"], flow_mgd=1, conc=0.1)
    # print(f"dof = {degrees_of_freedom(m)}")
    # # assert False, "stop"
    # m.fs.costing.base_currency = pyunits.USD_2014
    # # m.fs.unit.initialize()
    # # m.fs.unit.display()
    # m.fs.unit.ozone_consumption.fix(1 * pyunits.mg / pyunits.L)
    # m.fs.unit.concentration_time.unfix()
    # m.fs.unit.concentration_time.setlb(0)
    # m.fs.unit.concentration_time.setub(None)
    # # m.fs.unit.concentration_
    # m.fs.unit.initialize()
    # results = solver.solve(m, tee=True)
    # assert_optimal_termination(results)
    # m.fs.unit.report()
    # m.fs.unit.ozone_consumption.display()
    # m.fs.unit.costing.direct_capital_cost.display()
    # ozone_cost = pd.read_csv(f"{here}/ozone_cost_data.csv")

    # rd = build_results_dict(m, skips=["fs.unit.properties"])
    # rd["flow_mgd"] = []
    # fig, ax = plt.subplots()

    # res = pd.DataFrame()
    # handles = list()
    # colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    # cmap = plt.get_cmap("tab10")
    # import itertools
    # n = 6
    # colors = [cmap(i) for i in range(n)]
    # colors = itertools.cycle(colors)
    # flows_data = ozone_cost["flow_mgd"].unique()
    # flows = np.linspace(1, 25, 50)
    # # flows = np.concatenate((flows_data, flows))
    # err = list()
    # doses = [1, 5, 10, 15, 20, 25]
    # for dose in doses:
    #     rd = build_results_dict(m, skips=["fs.unit.properties"])
    #     rd["flow_mgd"] = []
    #     rd["dose"] = []
    #     c = next(colors)
    #     for flow in flows:
    #         print(f"\n\nflow = {flow:.4f} MGD\n\ndose = {dose}\n\n")
    #         try:
    #             m = build_zo_model(u, comps=["toc"], flow_mgd=flow, conc=0.01)
    #             m.fs.costing.base_currency = pyunits.USD_2014
    #             # m.fs.unit.initialize()
    #             m.fs.unit.ozone_consumption.fix(dose * pyunits.mg / pyunits.L)
    #             m.fs.unit.concentration_time.unfix()
    #             m.fs.unit.concentration_time.setlb(0)
    #             m.fs.unit.concentration_time.setub(None)
    #             results = solver.solve(m, tee=False)
    #             assert_optimal_termination(results)
    #             rd = results_dict_append(m, rd)
    #             rd["flow_mgd"].append(flow)
    #             rd["dose"].append(dose)
    #         except:
    #             print(f"Model did not solve optimally for flow = {flow:.4f} MGD and dose = {dose} mg/L")

    #             print_infeasible_constraints(m)
    #             # print_infeasible_bounds(m)
    #             print_variables_close_to_bounds(m)
    #             continue
    #     df = pd.DataFrame(rd)
    #     res = pd.concat([res, df], ignore_index=True)
    #     o = ozone_cost[ozone_cost.dose == dose].copy()
    #     handles.append(Line2D([0], [0], color=c, label=f"{dose} mg/L", markerfacecolor=c, marker=".", markersize=10))

    #     # df.to_csv(f"{here}/ozone_results.csv", index=False)

    # # fig, ax = plt.subplots()

    #     ax.plot(df["flow_mgd"], df["fs.unit.costing.direct_capital_cost"]*1e-3, marker=".", color=c, label=f"{dose} mg/L")
    #     ax.scatter(o["flow_mgd"], o["capex"], marker="x", color=c)
    # handles.append(Line2D([0], [0], color="w", marker="x",markerfacecolor="k", markeredgecolor="k", markersize=10, label="data"))
    # ax.legend(handles=handles)

    # res.to_csv(f"{here}/ozone_results.csv", index=False)
    # ax.set(xlabel="Flow (MGD)", ylabel="Capital Cost (USD2014)", title="Capital Cost vs Flow for OzoneZO")
    # ax.set_axisbelow(True)
    # ax.grid(visible=True)
    # fig.tight_layout()

    # fig.savefig(f"{here}/ozone_cost_line.png", dpi=300, bbox_inches="tight")

    # set_dict = dict(
    #     xlabel="Flow (MGD)",
    #     ylabel="Ozone Dose (mg/L)",
    #     title="CAPEX ($MM)",)
    # # fig2, ax2, = plt.subplots()
    # fig2, ax2 = plot_contour(res, x="flow_mgd", y="dose", z="fs.unit.costing.direct_capital_cost", z_adj=1e-6, cmap="viridis", set_dict=set_dict)
    # fig2.tight_layout()
    # fig2.savefig(f"{here}/ozone_cost_contour.png", dpi=300, bbox_inches="tight")

    # set_dict['title'] = "LCOW"

    # fig2, ax2 = plot_contour(res, x="flow_mgd", y="dose", z="fs.costing.LCOW", cmap="plasma", set_dict=set_dict)
    # fig2.tight_layout()
    # fig2.savefig(f"{here}/ozone_lcow_contour.png", dpi=300, bbox_inches="tight")

    # set_dict["title"] = "SEC (kWh/m3)"

    # fig2, ax2 = plot_contour(res, x="flow_mgd", y="dose", z="fs.costing.SEC", cmap="cividis", set_dict=set_dict)
    # fig2.tight_layout()
    # fig2.savefig(f"{here}/ozone_sec_contour.png", dpi=300, bbox_inches="tight")

    # plt.show()
    # # # print(flows)
    # # for flow in flows:
    # #     print(f"\n\nflow = {flow:.4f} MGD\n\n")
    # #     m = build_zo_model(u, flow_mgd=flow)

    # #     try:
    # #         m.fs.unit.initialize()

    # #         # solver = get_solver()
    # #         results = solver.solve(m, tee=False)
    # #         assert_optimal_termination(results)
    # #         rd = results_dict_append(m, rd)
    #     #     except:

    #     #         print_infeasible_constraints(m)
    #     #         # print_infeasible_bounds(m)
    #     #         print_variables_close_to_bounds(m)
    #     #         continue
    # #         # results = None
    # #     # if not check_optimal_termination(results):
    # #     #     print(f"Model did not solve optimally for flow = {flow:.4f} MGD")
    # #     #     # continue
    # #     #     break

    # # for k, v in rd.items():

    # # m.fs.unit.display()
    # # for u in class_name_list:
    # #     print(u)
    # #     if u  not in skips:
    # #         print(f"skipping {u}")
    # #         continue
    # #     m = build_zo_model(u, subtype=subtype)
    # #     res[u] = value(m.fs.costing.LCOW)
    # #     # m.fs.display()
    # #     print(f"unit = {u}\ndof = {degrees_of_freedom(m)}")
    # #     print(f"tech_type = {m.fs.unit._tech_type}\n")
    # #     # m.fs.unit.display()
    # #     # break
