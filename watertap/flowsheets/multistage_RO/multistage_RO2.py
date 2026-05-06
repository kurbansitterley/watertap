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
import logging
from pyomo.environ import (
    RangeSet,
    ConcreteModel,
    value,
    Constraint,
    Objective,
    Expression,
    Var,
    Param,
    TransformationFactory,
    assert_optimal_termination,
    units as pyunits,
)
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import (
    Mixer,
    Product,
    Feed,
    StateJunction,
    Mixer,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale

from watertap.property_models.NaCl_T_dep_prop_pack import NaClParameterBlock
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump, EnergyRecoveryDevice
from watertap.costing import WaterTAPCosting
from watertap.unit_models.reverse_osmosis_1D import ReverseOsmosis1D
from watertap.core import (  # noqa # pylint: disable=unused-import
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.core.solvers import get_solver
import watertap.flowsheets.multistage_RO.utils as utils

__author__ = "Alexander V. Dudchenko, Kurban A. Sitterley"

"""
Build and solve N-stage RO system with optional ERD and booster pumps.
"""

_logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "MWMWMWMWMW %(asctime)s [%(levelname)s] %(module)s: %(message)s",
    "%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
_logger.addHandler(handler)
_logger.setLevel(logging.DEBUG)

solver = get_solver()

default_bwro_op_dict = {
    "A_comp": 3.5 * pyunits.liter / pyunits.m**2 / pyunits.hour / pyunits.bar,
    "B_comp": 0.2 * pyunits.liter / pyunits.m**2 / pyunits.hour,
}
default_swro_op_dict = {
    "A_comp": 1.5 * pyunits.liter / pyunits.m**2 / pyunits.hour / pyunits.bar,
    "B_comp": 0.1 * pyunits.liter / pyunits.m**2 / pyunits.hour,
}


def set_stage_op_conditions(stage, m=None, max_pressure=400e5, ro_op_dict={}):
    """
    Set operating conditions for a single stage
    """

    if m is None:
        m = stage.model()

    stage_num = stage.index()

    # Fix default RO values
    stage.RO.A_comp.fix(4.2e-12)
    stage.RO.B_comp.fix(3.5e-8)
    stage.RO.feed_side.channel_height.fix(1e-3)
    stage.RO.feed_side.spacer_porosity.fix(0.9)
    stage.RO.permeate.pressure[0].fix(101325)
    # Scale area and width to flow vol
    stage.RO.width.setub(5000)
    stage.RO.width.fix(5 * value(m.flow_vol))
    stage.RO.area.fix(30 * value(m.flow_vol))

    # Fix user-specified RO params
    for p, val in ro_op_dict.items():
        v = stage.RO.find_component(p)
        if v is not None:
            v.fix(val)
        else:
            msg = f"Component {p} not found in RO unit model"
            raise ValueError(msg)


def scale_stage(stage, m=None, **kwargs):
    """
    Scale a single stage
    """
    if m is None:
        m = stage.model()

    full_scaling = kwargs.get("full_scaling", True)
    utils.overscale_ro(stage.RO, m.fs.properties, full_scaling=full_scaling)

    if stage.has_pump:
        iscale.set_scaling_factor(stage.pump.control_volume.work, 1e-3)
        iscale.set_scaling_factor(
            stage.pump.control_volume.properties_out[0].flow_vol_phase["Liq"], 1
        )
        iscale.set_scaling_factor(stage.pump.work_fluid[0], 1)


def set_stage_bounds(stage):
    """
    Set bounds for a single stage
    """

    for t, x, p, c in stage.RO.flux_mass_phase_comp:
        if p == "H2O":
            stage.RO.flux_mass_phase_comp[t, x, p, c].setlb(
                0 * pyunits.kg / (pyunits.m**2 * pyunits.hr)
            )
            stage.RO.flux_mass_phase_comp[t, x, p, c].setub(
                200 * pyunits.kg / (pyunits.m**2 * pyunits.hr)
            )

    stage.RO.length.setlb(1 * pyunits.meter)
    stage.RO.width.setlb(0.1 * pyunits.meter)
    stage.RO.feed_side.velocity[0, 0].setub(0.3)
    stage.RO.feed_side.velocity[0, 0].setlb(0.1)
    stage.RO.feed_side.cp_modulus.setub(20)
    stage.RO.feed_side.friction_factor_darcy.setub(20)


def build_costing(m, hpro_costing=False):
    """
    Build costing block on model
    """

    cma_pump = {"pump_type": "high_pressure"}
    if hpro_costing:
        cma_ro = {"ro_type": "high_pressure"}

    else:
        cma_ro = {"ro_type": "standard"}

    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyunits.USD_2023

    for n, stage in m.fs.stage.items():
        if stage.has_pump:
            stage.pump.costing = UnitModelCostingBlock(
                flowsheet_costing_block=m.fs.costing, costing_method_arguments=cma_pump
            )
        stage.RO.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing, costing_method_arguments=cma_ro
        )
    if m.fs.find_component("ERD") is not None:
        m.fs.ERD.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    m.fs.costing.cost_process()

    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol_phase["Liq"])
    m.fs.costing.add_specific_energy_consumption(
        m.fs.product.properties[0].flow_vol_phase["Liq"], name="SEC"
    )


def build_n_stage_system(
    prop_pack="seawater",
    n_stages=2,
    flow_vol=1,  # L/s
    salinity=35,  # g/L
    temperature=25,  # degC
    water_recovery=0.5,  # only for initialization
    over_pressure=0.1,
    max_perm_conc=0.5,
    pump_dict={1: True},  # first stage always has pump
    hpro_costing=False,
    add_erd=True,
    ro_op_dict={},
    add_costing=True,
    max_pressure=400e5,
    *args,
    **kwargs,
):
    """
    Build N-stage system
    """

    _logger.info(
        f"Running {n_stages} stage system with {sum(pump_dict.values())} pumps."
    )

    m = ConcreteModel()
    m.flow_vol = flow_vol * pyunits.liter / pyunits.s
    m.salinity = salinity * pyunits.gram / pyunits.liter
    m.over_pressure = over_pressure
    m.water_recovery = water_recovery

    m.fs = FlowsheetBlock(dynamic=False)

    if prop_pack.lower() == "nacl":
        m.fs.properties = NaClParameterBlock()
    elif prop_pack.lower() == "seawater":
        m.fs.properties = SeawaterParameterBlock()
    else:
        msg = f"Only NaCl or Seawater property models can be used but {prop_pack} was passed."
        raise ValueError(msg)

    comp = m.fs.properties.solute_set.at(1)

    m.fs.n_stages = Param(
        initialize=n_stages, mutable=False, doc="Number of RO stages in system"
    )
    m.fs.add_erd = add_erd
    m.fs.stages_set = RangeSet(m.fs.n_stages)
    m.fs.stage = FlowsheetBlock(m.fs.stages_set, dynamic=False)

    m.fs.feed = Feed(property_package=m.fs.properties)

    tot_area_expr = 0
    tot_power_expr = 0

    for n, stage in m.fs.stage.items():
        # build_stage(stage, m=m, has_pump=pump_dict.get(n, False))
        stage.has_pump = pump_dict.get(n, False)
        if stage.has_pump:
            stage.pump = Pump(property_package=m.fs.properties)
            tot_power_expr += stage.pump.work_mechanical[0]
        stage.RO = ReverseOsmosis1D(
            property_package=m.fs.properties,
            has_pressure_change=True,
            pressure_change_type=PressureChangeType.calculated,
            mass_transfer_coefficient=MassTransferCoefficient.calculated,
            concentration_polarization_type=ConcentrationPolarizationType.calculated,
            transformation_scheme="BACKWARD",
            transformation_method="dae.finite_difference",
            module_type="spiral_wound",
            finite_elements=10,
            has_full_reporting=True,
        )
        tot_area_expr += stage.RO.area

    if add_erd:
        m.fs.ERD = EnergyRecoveryDevice(property_package=m.fs.properties)

    m.fs.product_mixer = Mixer(
        property_package=m.fs.properties,
        momentum_mixing_type=MomentumMixingType.minimize,
        inlet_list=[f"stage{i}" for i in m.fs.stages_set],
    )

    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.disposal = Product(property_package=m.fs.properties)

    m.fs.system_recovery = Var(
        initialize=0.5,
        bounds=(0.0001, 0.9999),
        units=pyunits.dimensionless,
        doc="System recovery",
    )

    m.fs.system_recovery_constraint = Constraint(
        expr=m.fs.system_recovery
        == m.fs.product.properties[0].flow_vol_phase["Liq"]
        / m.fs.feed.properties[0].flow_vol_phase["Liq"]
    )

    m.fs.perm_conc_lb = Constraint(
        expr=m.fs.product.properties[0].conc_mass_phase_comp["Liq", comp]
        <= max_perm_conc
    )

    m.fs.total_area = Expression(
        expr=tot_area_expr, doc="Total membrane area in system"
    )
    m.fs.total_power = Expression(
        expr=pyunits.convert(tot_power_expr, to_units=pyunits.kW),
        doc="Total pumping power required",
    )
    m.fs.system_flux = Expression(
        expr=pyunits.convert(
            m.fs.product.properties[0].flow_vol_phase["Liq"] / m.fs.total_area,
            to_units=pyunits.liter / pyunits.m**2 / pyunits.hour,
        ),
        doc="System flux (based on total membrane area)",
    )

    for b in [m.fs.feed, m.fs.product, m.fs.disposal]:
        b.properties[0].flow_vol_phase["Liq"]
        b.properties[0].conc_mass_phase_comp["Liq", comp]

    if add_costing:
        build_costing(m, hpro_costing=hpro_costing)
        m.fs.obj = Objective(expr=m.fs.costing.LCOW, sense="minimize")
    else:
        m.fs.SEC = Var(
            initialize=0.5,
            bounds=(0, None),
            units=pyunits.kWh / pyunits.m**3,
            doc="Specific Energy Consumption",
        )
        m.fs.SEC_constraint = Constraint(
            expr=m.fs.SEC
            == pyunits.convert(
                m.fs.total_power / m.fs.product.properties[0].flow_vol_phase["Liq"],
                to_units=pyunits.kWh / pyunits.m**3,
            )
        )
        m.fs.obj = Objective(expr=m.fs.SEC, sense="minimize")

    ### CONNECTIONS
    m.arc_order = list()
    m.initialization_order = [m.fs.feed]

    m.fs.feed_to_stage1 = Arc(
        source=m.fs.feed.outlet, destination=m.fs.stage[1].pump.inlet
    )
    m.arc_order.append(m.fs.feed_to_stage1)

    for n, stage in m.fs.stage.items():
        if stage.has_pump:
            # make pump to RO connection
            a = Arc(source=stage.pump.outlet, destination=stage.RO.inlet)
            stage.add_component(f"pump_to_RO", a)
            m.initialization_order.append(stage.pump)
            m.arc_order.append(a)

        m.initialization_order.append(stage.RO)

        if n != m.fs.stages_set.last():
            # connect this stage to next stage
            next_stage = m.fs.stage[n + 1]
            if next_stage.has_pump:
                a = Arc(
                    source=stage.RO.retentate,
                    destination=next_stage.pump.inlet,
                )
            else:
                a = Arc(
                    source=stage.RO.retentate,
                    destination=next_stage.RO.inlet,
                )
            m.fs.add_component(f"stage{n}_to_stage{n+1}", a)
            m.arc_order.append(a)

        # connect this stage permeate to the mixer
        a = Arc(
            source=stage.RO.permeate,
            destination=m.fs.product_mixer.find_component(f"stage{n}"),
        )
        m.fs.add_component(f"stage{n}_product_to_mixer", a)

        if n == m.fs.stages_set.last() and add_erd:
            # connect this stage disposal to the ERD
            a = Arc(source=stage.RO.retentate, destination=m.fs.ERD.inlet)
            m.fs.add_component(f"stage{n}_to_ERD", a)
            m.arc_order.append(a)

    if add_erd:
        a = Arc(
            source=m.fs.ERD.outlet,
            destination=m.fs.disposal.inlet,
        )
        m.fs.add_component(f"ERD_to_disposal", a)
        m.initialization_order.append(m.fs.ERD)
        m.arc_order.append(a)
    else:
        a = Arc(source=stage.RO.retentate, destination=m.fs.disposal.inlet)
        m.fs.add_component(f"stage{n}_to_disposal", a)
        m.arc_order.append(a)

    m.initialization_order.append(m.fs.disposal)

    m.mixer_arcs = [
        m.fs.find_component(f"stage{_n}_product_to_mixer") for _n in m.fs.stages_set
    ]
    m.fs.product_mixer_to_product = Arc(
        source=m.fs.product_mixer.outlet,
        destination=m.fs.product.inlet,
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    #### SCALING
    rho = 1000 * pyunits.kg / pyunits.m**3  # approximate for scaling
    mass_flow_water = pyunits.convert(m.flow_vol * rho, to_units=pyunits.kg / pyunits.s)
    mass_flow_salt = pyunits.convert(
        m.flow_vol * m.salinity, to_units=pyunits.kg / pyunits.s
    )

    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1 / value(mass_flow_water), index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        1 / value(mass_flow_salt),
        index=("Liq", comp),
    )

    for n, stage in m.fs.stage.items():
        scale_stage(stage, m=m, **kwargs)

    if add_erd:
        iscale.set_scaling_factor(m.fs.ERD.control_volume.work, 1e-3)

    iscale.calculate_scaling_factors(m)

    #### SET OPERATING CONDITIONS
    m.fs.feed.properties.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): m.flow_vol,
            ("conc_mass_phase_comp", ("Liq", comp)): m.salinity,
            ("pressure", None): 101325,
            ("temperature", None): temperature + 273.15,
        },
        hold_state=True,
    )

    m.fs.feed.initialize()
    propagate_state(m.fs.feed_to_stage1)

    fsb = m.fs.feed.properties[0]
    last_pump = m.fs.stage[1].pump

    for n, stage in m.fs.stage.items():
        set_stage_bounds(stage)
        set_stage_op_conditions(
            stage, m=m, max_pressure=max_pressure, ro_op_dict=ro_op_dict
        )

        operating_pressure = utils.calculate_operating_pressure(
            feed_state_block=fsb,
            over_pressure=m.over_pressure,
            water_recovery=m.water_recovery,
            salt_passage=0.01,
            solver=solver,
        )

        if stage.has_pump:

            if operating_pressure >= max_pressure:
                operating_pressure = max_pressure

            if n > 1:
                c = Constraint(
                    expr=stage.pump.control_volume.properties_out[0].pressure
                    - last_pump.control_volume.properties_out[0].pressure
                    >= 1 * pyunits.bar
                )
                stage.add_component(f"pressure_increase_constr_stage{n}", c)

            _logger.info(
                f"Stage {stage.index()} estimated operating pressure = {operating_pressure*1e-5:.2f} bar"
            )

            stage.pump.control_volume.properties_out[0].pressure.setub(max_pressure)
            # stage.pump.control_volume.properties_out[0].pressure.fix(operating_pressure)
            stage.pump.efficiency_pump.fix(0.8)

            stage.pump.control_volume.properties_in[0].pressure_osm_phase
            stage.pump.control_volume.properties_in[0].conc_mass_phase_comp

            init_flags = stage.pump.control_volume.initialize()
            stage.pump.control_volume.properties_in.display()
            # assert False
            stage.pump.control_volume.release_state(init_flags)
            target_pressure = stage.pump.control_volume.properties_in[
                0
            ].pressure_osm_phase["Liq"]() * (1 + m.over_pressure)

            stage.pump.outlet.pressure[0].fix(target_pressure)
            _logger.info(
                f"Stage {stage.index()} estimated operating pressure = {target_pressure * 1e-5:.2f} bar"
            )

            stage.pump.initialize()
            propagate_state(stage.pump_to_RO)
            last_pump = stage.pump
            # report_pump(stage)
            # assert False
        else:
            stage.RO.inlet.pressure[0].fix(operating_pressure)

        stage.RO.initialize()
        stage.RO.inlet.pressure[0].unfix()
        fsb = stage.RO.feed_side.properties[0, 1]

        a = m.fs.find_component(f"stage{n}_product_to_mixer")
        propagate_state(a)

        if n != m.fs.stages_set.last():
            a = m.fs.find_component(f"stage{n}_to_stage{n+1}")
            propagate_state(a)

        if n == m.fs.stages_set.last():
            if add_erd:
                a = m.fs.find_component(f"stage{n}_to_ERD")
            else:
                a = m.fs.find_component(f"stage{n}_to_disposal")
            propagate_state(a)

    if add_erd:
        m.fs.ERD.efficiency_pump.fix(0.95)
        m.fs.ERD.control_volume.properties_out[0].pressure.fix(101325)
        m.fs.ERD.initialize()
        propagate_state(m.fs.ERD_to_disposal)

    m.fs.disposal.initialize()

    m.fs.product_mixer.initialize()
    propagate_state(m.fs.product_mixer_to_product)

    m.fs.product.initialize()

    # Unfix area, width, and pump pressure; optimize recovery
    for n, stage in m.fs.stage.items():
        stage.RO.area.unfix()
        stage.RO.width.unfix()
        if stage.has_pump:
            stage.pump.control_volume.properties_out[0].pressure.unfix()

    _logger.info(f"DOF before optimization = {degrees_of_freedom(m)}")

    results = utils.solve(model=m, tee=True)
    assert_optimal_termination(results)

    # Fix area, with, and pump pressure
    for n, stage in m.fs.stage.items():
        stage.RO.area.fix()
        stage.RO.width.fix()
        if stage.has_pump:
            stage.pump.control_volume.properties_out[0].pressure.fix()

    _logger.info(f"Final DOF = {degrees_of_freedom(m)}")
    assert degrees_of_freedom(m) == 0
    results = utils.solve(model=m, tee=True)
    assert_optimal_termination(results)

    return m


def set_system_recovery(m, recovery):
    """
    Fix system recovery and unfix membrane area, width, and pump pressure in each stage
    """

    m.fs.system_recovery.fix(recovery)

    for n, stage in m.fs.stage.items():
        stage.RO.area.unfix()
        stage.RO.width.unfix()
        if stage.has_pump:
            stage.pump.control_volume.properties_out[0].pressure.unfix()

    return m


def run_n_stage_system(*args, **kwargs):

    m = build_n_stage_system(*args, **kwargs)
    # initialize_n_stage_system(m, *args, **kwargs)

    return m


def report_pump(blk, w=30):
    title = "Pump Report"
    side = int(((3 * w) - len(title)) / 2) - 1
    header = "=" * side + f" {title} " + "=" * side
    print(f"\n{header}\n")
    print(f'{"Parameter":<{w}s}{"Value":<{w}s}{"Units":<{w}s}')
    print(f"{'-' * (3 * w)}")

    pin = blk.pump.control_volume.properties_in[0].pressure
    deltaP = blk.pump.deltaP[0]
    pout = blk.pump.control_volume.properties_out[0].pressure
    work = blk.pump.work_mechanical[0]

    print(
        f'{f"Inlet Pressure (bar)":<{w}s}{value(pyunits.convert(pin, to_units=pyunits.bar)):<{w}.3f}{"bar"}'
    )
    print(
        f'{f"Pressure Change (bar)":<{w}s}{value(pyunits.convert(deltaP, to_units=pyunits.bar)):<{w}.3f}{"bar"}'
    )
    print(
        f'{f"Outlet Pressure (bar)":<{w}s}{value(pyunits.convert(pout, to_units=pyunits.bar)):<{w}.3f}{"bar"}'
    )
    print(
        f'{f"Power (kW)":<{w}s}{value(pyunits.convert(work, to_units=pyunits.kW)):<{w}.3f}{"kW"}'
    )
    print(f'{f"Efficiency (-)":<{w}s}{value(blk.pump.efficiency_pump[0]):<{w}.3f}{"-"}')


def report_ro(blk, w=30):
    comp = blk.RO.config.property_package.solute_set.at(1)
    title = "RO Report"
    side = int(((3 * w) - len(title)) / 2) - 1
    header = "=" * side + f" {title} " + "=" * side
    print(f"\n{header}\n")
    print(f'{"Parameter":<{w}s}{"Value":<{w}s}{"Units":<{w}s}')
    print(f"{'-' * (3 * w)}")

    print(
        f'{f"Inlet Pressure":<{w}s}{value(pyunits.convert(blk.RO.feed_side.properties[0, 0].pressure, to_units=pyunits.bar)):<{w}.3f}{f"bar"}'
    )
    # print(
    #     f'{f"Inlet Flow":<{w}s}{value(blk.feed.properties[0].flow_vol_phase["Liq"])*1000:<{w}.3f}{"L/s"}'
    # )
    print(
        f'{f"Inlet Flow":<{w}s}{value(blk.RO.feed_side.properties[0, 0].flow_vol_phase["Liq"])*1000:<{w}.3f}{"L/s"}'
    )
    print(
        f'{f"Brine Flow":<{w}s}{value(blk.RO.feed_side.properties[0, 1].flow_vol_phase["Liq"])*1000:<{w}.3f}{"L/s"}'
    )
    print(
        f'{f"Product Flow":<{w}s}{value(blk.RO.mixed_permeate[0].flow_vol_phase["Liq"])*1000:<{w}.3f}{"L/s"}'
    )
    # print(f'{f"Recovery":<{w}s}{value(blk.recovery)*100:<{w}.3f}{"%"}')
    print(
        f'{f"Inlet Conc.":<{w}s}{value(blk.RO.feed_side.properties[0, 0].conc_mass_phase_comp["Liq", comp]):<{w}.3f}{"g/L"}'
    )
    print(
        f'{f"Perm Conc.":<{w}s}{value(blk.RO.mixed_permeate[0].conc_mass_phase_comp["Liq", comp]):<{w}.3f}{"g/L"}'
    )
    print(
        f'{f"Brine Conc.":<{w}s}{value(blk.RO.feed_side.properties[0, 1].conc_mass_phase_comp["Liq", comp]):<{w}.3f}{"g/L"}'
    )
    print(
        f'{f"Rejection":<{w}s}{value(blk.RO.rejection_phase_comp[0, "Liq", comp])*100:<{w}.3f}{"%"}'
    )
    print(
        f'{f"deltaP":<{w}s}{value(pyunits.convert(blk.RO.deltaP[0], to_units=pyunits.bar)):<{w}.3f}{f"bar"}'
    )
    acomp = blk.RO.A_comp[0, "H2O"]
    print(
        f'{f"Water Perm (A)":<{w}s}{value(pyunits.convert(acomp, to_units=pyunits.liter / pyunits.m**2 / pyunits.hour / pyunits.bar)):<{w}.3f}{f"LMH/bar"}'
    )
    print(f'{f"Water Perm (A)":<{w}s}{value(acomp):<{w}.3e}{f"m/s/Pa"}')
    bcomp = blk.RO.B_comp[0, comp]
    print(
        f'{f"Salt Perm (B)":<{w}s}{value(pyunits.convert(bcomp, to_units=pyunits.liter / pyunits.m**2 / pyunits.hour)):<{w}.3f}{f"LMH"}'
    )
    print(f'{f"Salt Perm (B)":<{w}s}{value(bcomp):<{w}.3e}{f"m/s"}')
    # print(f'{f"Average Flux (LMH)":<{w}s}{value(blk.average_flux_LMH):<{w}.3f}{f"LMH"}')

    print(
        f'{f"Membrane Area":<{w}s}{value(blk.RO.area):<{w}.3f}{f"{pyunits.get_units(blk.RO.area)}"}'
    )
    print(
        f'{f"Membrane Width":<{w}s}{value(blk.RO.width):<{w}.3f}{f"{pyunits.get_units(blk.RO.width)}"}'
    )
    print(
        f'{f"Membrane Length":<{w}s}{value(blk.RO.length):<{w}.3f}{f"{pyunits.get_units(blk.RO.length)}"}'
    )
    print(
        f'{f"Inlet Crossflow Vel.":<{w}s}{value(blk.RO.feed_side.velocity[0, 0])*100:<{w}.3f}{f"cm/s"}'
    )
    print(
        f'{f"Outlet Crossflow Vel.":<{w}s}{value(blk.RO.feed_side.velocity[0, 1])*100:<{w}.3f}{f"cm/s"}'
    )
    print(
        f'{f"Perm Backpressure":<{w}s}{value(pyunits.convert(blk.RO.mixed_permeate[0].pressure, to_units=pyunits.bar)):<{w}.3f}{f"bar"}'
    )


def report_costing(blk, w=30):
    title = "Costing Report"
    side = int(((3 * w) - len(title)) / 2) - 1
    header = "=" * side + f" {title} " + "=" * side
    print(f"\n{header}\n")
    print(f'{"Parameter":<{w}s}{"Value":<{w}s}{"Units":<{w}s}')
    print(f"{'-' * (3 * w)}")
    print(f'{f"LCOW":<{w}s}{value(blk.LCOW):<{w}.3f}{f"{pyunits.get_units(blk.LCOW)}"}')
    print(f'{f"SEC":<{w}s}{value(blk.SEC):<{w}.3f}{f"{pyunits.get_units(blk.SEC)}"}')
    print(
        f'{f"CAPEX":<{w}s}{value(blk.total_capital_cost):<{w}.3f}{f"{pyunits.get_units(blk.total_capital_cost)}"}'
    )
    print(
        f'{f"OPEX":<{w}s}{value(blk.total_operating_cost):<{w}.3f}{f"{pyunits.get_units(blk.total_operating_cost)}"}'
    )


def report_n_stage_system(m, w=30):
    comp = m.fs.properties.solute_set.at(1)
    title = f"N-Stage System Report (N={len(m.fs.stage)})"
    side = int(((3 * w) - len(title)) / 2) - 1
    header = "=" * side + f" {title} " + "=" * side
    print(f"\n{header}\n")
    perm_conc = value(m.fs.product.properties[0].conc_mass_phase_comp["Liq", comp])
    feed_conc = value(m.fs.feed.properties[0].conc_mass_phase_comp["Liq", comp])
    brine_conc = value(m.fs.disposal.properties[0].conc_mass_phase_comp["Liq", comp])
    print(f'{f"System Recovery":<{w}s}{value(m.fs.system_recovery)*100:<{w}.3f}{"%"}')
    print(f'{f"Feed {comp} Conc.":<{w}s}{feed_conc:<{w}.3f}{"g/L"}')
    print(f'{f"Perm {comp} Conc.":<{w}s}{perm_conc:<{w}.3f}{"g/L"}')
    print(f'{f"Brine {comp} Conc.":<{w}s}{brine_conc:<{w}.3f}{"g/L"}')

    for n, stage in m.fs.stage.items():

        title = f"STAGE {n}"
        side = int(((3 * w) - len(title)) / 2) - 1
        header = "(" * side + f" {title} " + ")" * side
        print(f"\n{header}\n")
        if stage.has_pump:
            report_pump(stage, w=w)
        report_ro(stage, w=w)

    if hasattr(m.fs, "costing"):
        report_costing(m.fs.costing, w=w)


if __name__ == "__main__":

    pump_dict = {1: True, 2: False, 3: True, 4: False, 5: True}
    m = build_n_stage_system(
        n_stages=3,
        salinity=95,
        flow_vol=5,
        add_erd=True,
        pump_dict=pump_dict,
        max_pressure=400e5,
    )
    # print(f"dof = {degrees_of_freedom(m)}")

    # initialize_n_stage_system(m)

    m = set_system_recovery(m, 0.4)
    res = utils.solve(model=m, tee=False)
    report_n_stage_system(m)
