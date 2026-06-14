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

import pytest

from pyomo.util.check_units import assert_units_consistent
import pyomo.environ as pyo
import idaes.core as idc

from watertap.costing.watertap_costing_package import WaterTAPCosting
import watertap.flowsheets.lsrro.lsrro as lsrro


@pytest.mark.component
def test_watertap_costing_package():
    m = pyo.ConcreteModel()
    m.fs = idc.FlowsheetBlock(dynamic=False)

    m.fs.costing = WaterTAPCosting()

    m.fs.electricity = pyo.Var(units=pyo.units.kW, initialize=1)

    m.fs.costing.cost_flow(m.fs.electricity, "electricity")

    assert "foo" not in m.fs.costing.flow_types
    with pytest.raises(
        ValueError,
        match="foo is not a recognized flow type. Please check "
        "your spelling and that the flow type has been registered with"
        " the FlowsheetCostingBlock.",
    ):
        m.fs.costing.cost_flow(m.fs.electricity, "foo")

    m.fs.costing.foo_cost = foo_cost = pyo.Var(
        initialize=42,
        doc="foo",
        units=pyo.units.USD_2020 / pyo.units.m / pyo.units.second,
    )

    m.fs.costing.register_flow_type("foo", m.fs.costing.foo_cost)

    # make sure the component was not replaced
    # by register_defined_flow
    assert foo_cost is m.fs.costing.foo_cost

    m.fs.foo = pyo.Var(units=pyo.units.m, initialize=10)

    m.fs.costing.cost_flow(m.fs.foo, "foo")

    m.fs.costing.bar_base_cost = pyo.Var(
        initialize=0.42,
        doc="bar",
        units=pyo.units.USD_2020 / pyo.units.g / pyo.units.hour,
    )
    m.fs.costing.bar_purity = pyo.Param(
        initialize=0.50, doc="bar purity", units=pyo.units.dimensionless
    )
    # Add breakdown of NaCl usage per unit product flow to costing package with name "regenerant_usage"
    m.fs.costing.add_flow_component_breakdown(
        "NaCl",
        m.fs.product.properties[0].flow_vol_phase["Liq"],
        name="regenerant_usage",
    )

    m.fs.costing.register_flow_type(
        "bar", m.fs.costing.bar_base_cost * m.fs.costing.bar_purity
    )

    bar_cost = m.fs.costing.bar_cost
    assert isinstance(bar_cost, pyo.Expression)
    assert pyo.value(bar_cost) == 0.21

    m.fs.costing.bar_base_cost.value = 1.5
    assert pyo.value(bar_cost) == 0.75

    m.fs.costing.baz_cost = pyo.Var(initialize=5)

    with pytest.raises(
        RuntimeError,
        match="Component baz_cost already exists on fs.costing but is not 42",
    ):
        m.fs.costing.register_flow_type(
            "baz", 42 * pyo.units.USD_2020 / pyo.units.m**2 / pyo.units.day
        )

    m.fs.costing.flow_types.remove("baz")

    m.fs.costing.register_flow_type(
        "ham", 42 * pyo.units.USD_2021 / pyo.units.kg / pyo.units.minute
    )

    assert isinstance(m.fs.costing.ham_cost, pyo.Var)

    m.fs.costing.cost_process()
    # no error, wacc, plant_lifetime fixed
    m.fs.costing.initialize()

    m.fs.costing.capital_recovery_factor.fix()
    with pytest.raises(
        RuntimeError,
        match="Exactly two of the variables fs.costing.plant_lifetime, "
        "fs.costing.wacc, fs.costing.capital_recovery_factor should be "
        "fixed and the other unfixed.",
    ):
        # error, capital_recovery_factor,  wacc, plant_lifetime all fixed
        m.fs.costing.initialize()
    m.fs.costing.wacc.unfix()

    # no error
    m.fs.costing.initialize()

def optimize_system(m):
    # Example of optimizing number of IX columns based on desired effluent equivalent concentration

    # Adding an objective to model.
    # In this case, we want to optimze the model to minimize the LCOW.
    m.fs.obj = Objective(expr=m.fs.costing.LCOW)
    ix = m.fs.ion_exchange
    target_ion = m.fs.ion_exchange.config.target_ion

    # For this demo, we are optimizing the model to have an effluent concentration of 25 mg/L.
    # Our initial model resulted in an effluent concentration of 0.21 mg/L.
    # By increasing the effluent concentration, we will have a longer breakthrough time, which will lead to less regeneration solution used,
    # and (hopefully) a lower cost.
    ix.process_flow.properties_out[0].conc_mass_phase_comp["Liq", target_ion].fix(0.025)

    # With the new effluent conditions for our ion exchange model, this will have implications for our downstream models (the Product and Regen blocks)
    # Thus, we must re-propagate the new effluent state to these models...
    propagate_state(m.fs.ix_to_product)
    propagate_state(m.fs.ix_to_regen)
    # ...and re-initialize them to our new conditions.
    m.fs.product.initialize()
    m.fs.regen.initialize()

    # To adjust solution to fixed-pattern to achieve desired effluent, must unfix dimensionless_time.
    ix.dimensionless_time.unfix()
    # Can optimize around different design variables, e.g., bed_depth, service_flow_rate (or combinations of these)
    # Here demonstrates optimization around column design
    ix.number_columns.unfix()
    ix.bed_depth.unfix()
    optimized_results = solver.solve(m)
    assert_optimal_termination(optimized_results)


def get_ion_config(ions):

    if not isinstance(ions, (list, tuple)):
        ions = [ions]
    diff_data = {
        "Na_+": 1.33e-9,
        "Ca_2+": 9.2e-10,
        "Cl_-": 2.03e-9,
        "Mg_2+": 0.706e-9,
        "SO4_2-": 1.06e-9,
    }
    mw_data = {
        "Na_+": 23e-3,
        "Ca_2+": 40e-3,
        "Cl_-": 35e-3,
        "Mg_2+": 24e-3,
        "SO4_2-": 96e-3,
    }
    charge_data = {"Na_+": 1, "Ca_2+": 2, "Cl_-": -1, "Mg_2+": 2, "SO4_2-": -2}
    ion_config = {
        "solute_list": [],
        "diffusivity_data": {},
        "mw_data": {"H2O": 18e-3},
        "charge": {},
    }
    for ion in ions:
        ion_config["solute_list"].append(ion)
        ion_config["diffusivity_data"][("Liq", ion)] = diff_data[ion]
        ion_config["mw_data"][ion] = mw_data[ion]
        ion_config["charge"][ion] = charge_data[ion]
    return ion_config


def display_results(m):

    ix = m.fs.ion_exchange
    liq = "Liq"
    header = f'{"PARAM":<40s}{"VALUE":<40s}{"UNITS":<40s}\n'

    prop_in = ix.process_flow.properties_in[0]
    prop_out = ix.process_flow.properties_out[0]

    recovery = prop_out.flow_vol_phase["Liq"]() / prop_in.flow_vol_phase["Liq"]()
    target_ion = ix.config.target_ion
    ion_set = ix.config.property_package.ion_set
    bv_to_regen = (ix.vel_bed() * ix.t_breakthru()) / ix.bed_depth()

    title = f'\n{"=======> SUMMARY <=======":^80}\n'
    print(title)
    print(header)
    print(f'{"LCOW":<40s}{f"{m.fs.costing.LCOW():<40.4f}"}{"$/m3":<40s}')
    print(
        f'{"TOTAL Capital Cost":<40s}{f"${ix.costing.capital_cost():<39,.2f}"}{"$":<40s}'
    )
    print(
        f'{"Specific Energy Consumption":<40s}{f"{m.fs.costing.specific_energy_consumption():<39,.5f}"}{"kWh/m3":<40s}'
    )
    regen_usage = m.fs.costing.regenerant_usage_component["fs.ion_exchange"]
    print(
        f'{f"Specific {ix.config.regenerant} Consumption":<40s}{f"{regen_usage():<39,.5f}"}{"kg/m3":<40s}'
    )
    print(
        f'{f"Annual Regenerant cost ({ix.config.regenerant})":<40s}{f"${m.fs.costing.aggregate_flow_costs[ix.config.regenerant]():<39,.2f}"}{"$/yr":<40s}'
    )
    print(f'{"BV Until Regen":<40s}{bv_to_regen:<40.3f}{"Bed Volumes":<40s}')
    print(
        f'{f"Breakthrough/Initial Conc. [{target_ion}]":<40s}{ix.c_norm[target_ion]():<40.3%}'
    )

    assert_units_consistent(m)

    m.fs.costing.initialize()

    total_LCOW = pyo.value(m.fs.costing.LCOW)

    summed_aggregates = pyo.value(
        sum(m.fs.costing.LCOW_aggregate_direct_capex.values())
        + sum(m.fs.costing.LCOW_aggregate_indirect_capex.values())
        + sum(m.fs.costing.LCOW_aggregate_fixed_opex.values())
        + sum(m.fs.costing.LCOW_aggregate_variable_opex.values())
        # electricity is counted both in the aggregate variable opex
        # per unit and per flow, so it is double-counted in this sum
        - m.fs.costing.LCOW_aggregate_variable_opex["electricity"]
    )
    assert pytest.approx(total_LCOW) == summed_aggregates

    summed_components = pyo.value(
        sum(m.fs.costing.LCOW_component_direct_capex.values())
        + sum(m.fs.costing.LCOW_component_indirect_capex.values())
        + sum(m.fs.costing.LCOW_component_fixed_opex.values())
        + sum(m.fs.costing.LCOW_component_variable_opex.values())
    )
    assert pytest.approx(total_LCOW) == summed_components

    sec = pyo.value(m.fs.costing.specific_energy_consumption)
    summed_sec = pyo.value(
        sum(m.fs.costing.specific_energy_consumption_component.values())
    )
    assert pytest.approx(sec) == summed_sec

    seci = pyo.value(m.fs.costing.specific_electrical_carbon_intensity)
    summed_seci = pyo.value(
        sum(m.fs.costing.specific_electrical_carbon_intensity_component.values())
    )
    assert pytest.approx(seci) == summed_seci
