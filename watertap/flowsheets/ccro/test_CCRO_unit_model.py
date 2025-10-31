import os
import json

from copy import deepcopy
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import *
from pyomo.environ import units as pyunits

from idaes.core.util.initialization import propagate_state
from watertap.flowsheets.ccro.multiperiod import CCRO_acc_volume_flushing as CCRO
from watertap.flowsheets.ccro.CCRO import *
from watertap.flowsheets.ccro.utils import *
from IPython.display import clear_output
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    set_scaling_factor,
    constraint_scaling_transform,
)
# __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
import os
from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    value,
    Var,
    NonNegativeReals,
    assert_optimal_termination,
    Objective,
    units as pyunits,
)
from pyomo.environ import TransformationFactory
from pyomo.network import Arc
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core.util.misc import add_object_reference
import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.scaling import calculate_scaling_factors
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Product, Feed
from idaes.models.unit_models.mixer import (
    Mixer,
    MomentumMixingType,
    MaterialBalanceType,
)
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

from watertap.unit_models.pressure_changer import Pump
from watertap.property_models.NaCl_T_dep_prop_pack import NaClParameterBlock
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.reverse_osmosis_0D import (
    ConcentrationPolarizationType,
    MassTransferCoefficient,
)
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.core.util.initialization import *
from watertap.core.solvers import get_solver

from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ReverseOsmosis1DData,
)
from watertap.unit_models.pseudo_steady_state import CCRO1D, DeadVolume0D
from watertap.unit_models.pseudo_steady_state.flushing import FlushingSurrogate

from watertap.costing import (
    WaterTAPCosting,
    PumpType,
    MixerType,
    ROType,
)

from watertap.flowsheets.ccro.utils import *

atmospheric_pressure = 101325 * pyunits.Pa


def filtration_build():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    # m.fs.configuration = configuration

    m.fs.properties = NaClParameterBlock()

    m.fs.raw_feed = Feed(property_package=m.fs.properties)
    m.fs.CCRO = CCRO1D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
        transformation_scheme="BACKWARD",
        transformation_method="dae.finite_difference",
        finite_elements=4,
        module_type="spiral_wound",
        has_full_reporting=True,
        cycle_phase="filtration",
    )

    # Feed pump
    m.fs.P1 = Pump(property_package=m.fs.properties)
    # Recirculation pump
    m.fs.P2 = Pump(property_package=m.fs.properties)

    m.fs.M1 = Mixer(
        property_package=m.fs.properties,
        has_holdup=False,
        num_inlets=2,
        momentum_mixing_type=MomentumMixingType.equality,
    )

    m.fs.product = Product(property_package=m.fs.properties)
    # Add connections
    m.fs.raw_feed_to_P1 = Arc(
        source=m.fs.raw_feed.outlet, destination=m.fs.P1.inlet
    )

    m.fs.P1_to_M1 = Arc(source=m.fs.P1.outlet, destination=m.fs.M1.inlet_1)
    m.fs.P2_to_M1 = Arc(source=m.fs.P2.outlet, destination=m.fs.M1.inlet_2)

    m.fs.M1_to_RO = Arc(source=m.fs.M1.outlet, destination=m.fs.CCRO.inlet)

    m.fs.RO_permeate_to_product = Arc(
        source=m.fs.CCRO.permeate, destination=m.fs.product.inlet
    )

    # m.fs.RO_retentate_to_dead_volume = Arc(
    #     source=m.fs.CCRO.retentate, destination=m.fs.CCRO.dead_volume_inlet
    # )
    m.fs.dead_volume_to_P2 = Arc(
        source=m.fs.CCRO.outlet, destination=m.fs.P2.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m

def scale_filtration_system(m):
    """
    Scale filtration model configuration
    """

    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "NaCl")
    )

    set_scaling_factor(m.fs.P1.control_volume.work, 1e-3)
    set_scaling_factor(m.fs.P2.control_volume.work, 1e-3)
    set_scaling_factor(m.fs.CCRO.area, 1e-2)

    # set_scaling_factor(
    #     m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1
    # )
    # set_scaling_factor(
    #     m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"], 1
    # )

    calculate_scaling_factors(m)



def filtration_set_operating_conditions(m, op_dict):
        
    # Feed block operating conditions
    m.fs.raw_feed.properties[0].pressure.fix(atmospheric_pressure)
    m.fs.raw_feed.properties[0].temperature.fix(op_dict["temperature"])

    # Pump 1 operating conditions
    m.fs.P1.efficiency_pump.fix(op_dict["p1_eff"])
    m.fs.P1.control_volume.properties_out[0].pressure.fix(
        op_dict["p1_pressure_start"]
    )

    # Pump 2 operating conditions
    m.fs.P2.efficiency_pump.fix(op_dict["p2_eff"])

    # Set RO configuration parameters
    m.fs.CCRO.A_comp.fix(op_dict["A_comp"])
    m.fs.CCRO.B_comp.fix(op_dict["B_comp"])
    m.fs.CCRO.area.fix(op_dict["membrane_area"])
    m.fs.CCRO.length.fix(op_dict["membrane_length"])
    m.fs.CCRO.feed_side.channel_height.fix(op_dict["channel_height"])
    m.fs.CCRO.feed_side.spacer_porosity.fix(op_dict["spacer_porosity"])

    m.fs.CCRO.permeate.pressure[0].fix(atmospheric_pressure)

    # m.fs.CCRO.feed_side.K.setlb(1e-6)
    m.fs.CCRO.feed_side.friction_factor_darcy.setub(200)
    # m.fs.CCRO.flux_mass_phase_comp.setub(1)
    # m.fs.CCRO.feed_side.cp_modulus.setub(50)
    # m.fs.CCRO.feed_side.cp_modulus.setlb(0.1)
    m.fs.CCRO.deltaP.setlb(None)

    # Dead Volume operating conditions
    # Fixed volume
    m.fs.CCRO.accumulation_volume_ratio.fix(1e-3)
    m.fs.CCRO.accumulation_volume_block.volume[0].set_value(op_dict["dead_volume"])
    m.fs.CCRO.previous_state.volume[0, "Liq"].set_value(op_dict["dead_volume"])

    m.fs.CCRO.accumulation_time.fix(op_dict["accumulation_time"])

    # Fixing the flow rate of the dead volume delta state
    # Using the feed to calculate the mass fraction and density

    m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"]
    m.fs.raw_feed.properties[0].flow_vol_phase["Liq"]
    m.fs.raw_feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"]

    m.fs.raw_feed.properties[0].flow_vol_phase["Liq"].fix(
        op_dict["recycle_flowrate"]
    )
    m.fs.raw_feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].fix(
        op_dict["recycle_conc_start"]
    )
    solver = get_solver()
    results = solver.solve(m.fs.raw_feed)
    assert_optimal_termination(results)
    print(f"raw_feed solved")

    m.fs.raw_feed.properties[0].flow_vol_phase["Liq"].unfix()
    m.fs.raw_feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].unfix()

    # I found fixing mass fraction and density is easiest way to get initial state
    # we will also use these as connection points between current and future state.

    m.fs.CCRO.previous_state.mass_frac_phase_comp[0, "Liq", "NaCl"].fix(
        m.fs.raw_feed.properties[0].mass_frac_phase_comp["Liq", "NaCl"].value
    )
    m.fs.CCRO.previous_state.dens_mass_phase[0, "Liq"].fix(
        m.fs.raw_feed.properties[0].dens_mass_phase["Liq"].value
    )

    # Reassign the raw feed flowrate and concentration
    m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(
        op_dict["raw_feed_flow_mass_water"]
    )
    m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"].fix(
        op_dict["raw_feed_flow_mass_salt"]
    )

    scale_filtration_system(m)
    print(f"dof = {degrees_of_freedom(m.fs.raw_feed)}")
    solver = get_solver()
    results = solver.solve(m.fs.raw_feed)
    assert_optimal_termination(results)
    print(f"dof = {degrees_of_freedom(m)}")

def filtration_initialize(m):
    # m.fs.raw_feed.initialize()
    propagate_state(m.fs.raw_feed_to_P1)

    m.fs.P1.outlet.pressure[0].fix(
        m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5
    )
    m.fs.P2.outlet.pressure[0].fix(
        m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"].value * 2 + 2e5
    )
    m.fs.P1.initialize()

    propagate_state(m.fs.P1_to_M1)
    copy_inlet_state_for_mixer(m)

    m.fs.M1.initialize()

    propagate_state(m.fs.M1_to_RO)
    m.fs.CCRO.initialize()

    propagate_state(m.fs.RO_permeate_to_product)
    # propagate_state(m.fs.RO_retentate_to_dead_volume)

    # m.fs.dead_volume.initialize()

    propagate_state(m.fs.dead_volume_to_P2)
    m.fs.P2.initialize()
    m.fs.P2.outlet.pressure[0].unfix()

    propagate_state(m.fs.P2_to_M1)

    m.fs.product.properties[0].flow_vol_phase["Liq"]
    m.fs.product.initialize()



def flushing_build():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    # m.fs.configuration = configuration

    m.fs.properties = NaClParameterBlock()

    m.fs.raw_feed = Feed(property_package=m.fs.properties)
    m.fs.CCRO = CCRO1D(
        property_package=m.fs.properties,
        cycle_phase="flushing",
        surrogate_model_file="/Users/ksitterl/Documents/Python/watertap/watertap/watertap/flowsheets/ccro/data/flushing_surrogate_multiple_tau_n_2.json"
    )

    # Feed pump
    m.fs.P1 = Pump(property_package=m.fs.properties)
    # Recirculation pump
    m.fs.P2 = Pump(property_package=m.fs.properties)

    # Add connections
    m.fs.raw_feed_to_P1 = Arc(
        source=m.fs.raw_feed.outlet, destination=m.fs.P1.inlet
    )
    m.fs.P1_to_CCRO = Arc(
        source=m.fs.P1.outlet, destination=m.fs.CCRO.inlet
    )

    m.fs.CCRO_to_P2 = Arc(
        source=m.fs.CCRO.outlet, destination=m.fs.P2.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def set_flushing_operating_conditions(m, op_dict):

    m.fs.raw_feed.properties[0].pressure.fix(atmospheric_pressure)
    m.fs.raw_feed.properties[0].temperature.fix(op_dict["temperature"])

    m.fs.raw_feed.properties.calculate_state(
        var_args={("flow_vol_phase", "Liq"): op_dict["flushing_flowrate"], ("conc_mass_phase_comp", ("Liq", "NaCl")): op_dict["raw_feed_conc"]},
        hold_state=True,    
    )
    # m.fs.raw_feed.properties[0].flow_vol_phase["Liq"].fix(
    #     op_dict["recycle_flowrate"]
    # )
    # m.fs.raw_feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"].fix(
    #     op_dict["recycle_conc_start"]
    # )

    m.fs.raw_feed.properties[0].pressure_osm_phase["Liq"]
    m.fs.raw_feed.properties[0].flow_vol_phase["Liq"]
    m.fs.raw_feed.properties[0].conc_mass_phase_comp["Liq", "NaCl"]

    m.fs.P1.efficiency_pump.fix(op_dict["p1_eff"])
    m.fs.P1.control_volume.properties_out[0].pressure.fix(
        op_dict["p1_pressure_start"]
    )

    # Pump 2 operating conditions - Add only for costing function. No work is done by this pump
    m.fs.P2.efficiency_pump.fix(op_dict["p2_eff"])

    m.fs.CCRO.flushing_efficiency.fix( 0.5)
    m.fs.CCRO.flushing_time.fix(20)

    solve(model=m.fs.raw_feed)

def scale_flushing_system(m):
    """
    Scale flushing model configuration
    """

    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "NaCl")
    )

    set_scaling_factor(m.fs.P1.control_volume.work, 1e-3)
    set_scaling_factor(m.fs.P2.control_volume.work, 1e-3)

    # set_scaling_factor(
    #     m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "H2O"], 1
    # )
    # set_scaling_factor(
    #     m.fs.raw_feed.properties[0].flow_mass_phase_comp["Liq", "NaCl"], 1
    # )

    calculate_scaling_factors(m)

def copy_time_period_links(m_old, m_new):
    """
    Copy linking variables between time periods.
    """
    m_new.fs.CCRO.previous_state.mass_frac_phase_comp[0, "Liq", "NaCl"].fix(
        m_old.fs.CCRO.accumulation_volume_block.properties_out[0]
        .mass_frac_phase_comp["Liq", "NaCl"]
        .value
    )
    m_new.fs.CCRO.previous_state.dens_mass_phase[0, "Liq"].fix(
        m_old.fs.CCRO.accumulation_volume_block.properties_out[0].dens_mass_phase["Liq"].value
    )


if __name__ == "__main__": 

    op_dict = dict(
        rho=1000,
        raw_feed_conc=5,  # g/L
        raw_feed_flowrate=1.8,  # L/min
        recycle_flowrate=49.1,  # L/min
        recycle_conc_start=11.7,
        temperature=298,  # K
        p1_pressure_start=306,  # psi
        p2_pressure_start=306,
        p1_eff=0.8,
        p2_eff=0.8,
        A_comp=5.96e-12,
        B_comp=3.08e-08,
        membrane_area=7.9,  # m2
        membrane_length=1,  # m
        channel_height=0.0008636,
        spacer_porosity=0.9,
        dead_volume=0.035564,
        accumulation_time=60,
        single_pass_water_recovery=0.063,
        include_costing=True,
        flushing_efficiency=0.9,
    )
    op_dict = config_op_dict(op_dict)


    ## FILTRATION PHASE DEMO 
    m = filtration_build()
    filtration_set_operating_conditions(m, op_dict)
    filtration_initialize(m)
    print(f"dof after init = {degrees_of_freedom(m)}")

    m.fs.P1.control_volume.properties_out[0].pressure.unfix()
    m.fs.P2.control_volume.properties_out[0].pressure.unfix()

    m.fs.P2.control_volume.properties_out[0].flow_vol_phase["Liq"].fix(
        op_dict["recycle_flowrate"]
    )
    print(f"dof after init = {degrees_of_freedom(m)}")
    results = solve(m)
    assert_optimal_termination(results)

    m_old = m

    m = flushing_build()
    set_flushing_operating_conditions(m, op_dict)
    scale_flushing_system(m)
    copy_time_period_links(m_old, m)
    print(f"dof = {degrees_of_freedom(m)}")

    propagate_state(m.fs.raw_feed_to_P1)
    m.fs.P1.initialize()

    propagate_state(m.fs.P1_to_CCRO)
    m.fs.CCRO.initialize()

    propagate_state(m.fs.CCRO_to_P2)
    m.fs.P2.initialize()
    m.fs.P2.outlet.pressure[0].fix(atmospheric_pressure)
