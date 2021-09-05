###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################

"""Desalination flowsheet components"""

from pyomo.environ import ConcreteModel, TransformationFactory
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import Mixer
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor
from idaes.core.util.initialization import propagate_state
from proteuslib.unit_models.pump_isothermal import Pump
from proteuslib.flowsheets.full_treatment_train.example_flowsheets import feed_block
from proteuslib.flowsheets.full_treatment_train.example_models import unit_separator, unit_0DRO, property_models
from proteuslib.flowsheets.full_treatment_train.util import solve_with_user_scaling, check_dof


def build_desalination(m, has_desal_feed=False, is_twostage=False,
                       RO_type='0D', RO_base='TDS', RO_level='simple'):
    """
    Builds RO desalination including specified feed and auxiliary equipment.
    Arguments:
        has_desal_feed: True or False, default = False,
            if True a feed block is created and specified to the standard feed
        RO_type: 'Sep' or '0D', default = '0D'
        RO_level: 'simple' or 'detailed', default = 'simple'
        RO_base: 'TDS' only, default = 'ion'
    """
    desal_port = {}
    prop = property_models.get_prop(m, base=RO_base)

    if has_desal_feed:
        # build feed
        feed_block.build_feed(m, base=RO_base)

    # build RO
    if RO_type == 'Sep':
        unit_separator.build_SepRO(m, base=RO_base)
    elif RO_type == '0D':
        unit_0DRO.build_RO(m, base='TDS', level=RO_level)
    else:
        raise ValueError('Unexpected model type {RO_type} provided to build_desalination'
                         ''.format(RO_type=RO_type))
    if is_twostage:
        if RO_type == '0D':
            unit_0DRO.build_RO(m, base='TDS', level=RO_level, name_str='RO2')
        else:
            raise ValueError('Unexpected model type {RO_type} provided to build_desalination when is_twostage is True'
                             ''.format(RO_type=RO_type))

    # auxiliary units
    if RO_type == 'Sep':
        # build auxiliary units (none)

        # connect models
        if has_desal_feed:
            m.fs.s_desal_feed_RO = Arc(source=m.fs.feed.outlet, destination=m.fs.RO.inlet)

        # specify (RO already specified)

        # inlet/outlet ports for pretreatment
        if not has_desal_feed:
            desal_port['in'] = m.fs.RO.inlet

    elif RO_type == '0D':
        # build auxiliary units
        m.fs.pump_RO = Pump(default={'property_package': prop})
        if is_twostage:
            m.fs.pump_RO2 = Pump(default={'property_package': prop})
            m.fs.mixer_permeate = Mixer(default={
                "property_package": prop,
                "inlet_list": ['RO', 'RO2']})

        # connect models
        if has_desal_feed:
            m.fs.s_desal_feed_pumpRO = Arc(source=m.fs.feed.outlet, destination=m.fs.pump_RO.inlet)

        m.fs.s_desal_pumpRO_RO = Arc(source=m.fs.pump_RO.outlet, destination=m.fs.RO.inlet)

        if is_twostage:
            m.fs.s_desal_RO_pumpRO2 = Arc(source=m.fs.RO.retentate, destination=m.fs.pump_RO2.inlet)
            m.fs.s_desal_pumpRO2_RO2 = Arc(source=m.fs.pump_RO2.outlet, destination=m.fs.RO2.inlet)
            m.fs.s_desal_permeateRO_mixer = Arc(source=m.fs.RO.permeate, destination=m.fs.mixer_permeate.RO)
            m.fs.s_desal_permeateRO2_mixer = Arc(source=m.fs.RO2.permeate, destination=m.fs.mixer_permeate.RO2)

        # specify (RO already specified, Pump 2 DOF)
        m.fs.pump_RO.efficiency_pump.fix(0.80)
        m.fs.pump_RO.control_volume.properties_out[0].pressure.fix(50e5)
        if is_twostage:
            m.fs.pump_RO2.efficiency_pump.fix(0.80)
            m.fs.pump_RO2.control_volume.properties_out[0].pressure.fix(55e5)

        # inlet/outlet ports for pretreatment
        if not has_desal_feed:
            desal_port['in'] = m.fs.pump_RO.inlet

    if is_twostage:
        desal_port['out'] = m.fs.mixer_permeate.outlet
        desal_port['waste'] = m.fs.RO2.retentate
    else:
        desal_port['out'] = m.fs.RO.permeate
        desal_port['waste'] = m.fs.RO.retentate

    return desal_port


def scale_desalination(m, **kwargs):
    """
    Scales the model created by build_desalination.
    Arguments:
        m: pyomo concrete model with a built desalination train
        **kwargs: same keywords as provided to the build_desalination function
    """

    if kwargs['has_desal_feed']:
        calculate_scaling_factors(m.fs.feed)

    calculate_scaling_factors(m.fs.RO)

    if kwargs['is_twostage']:
        calculate_scaling_factors(m.fs.RO2)

    if kwargs['RO_type'] == '0D':
        set_scaling_factor(m.fs.pump_RO.control_volume.work, 1e-3)
        calculate_scaling_factors(m.fs.pump_RO)

        if kwargs['is_twostage']:
            set_scaling_factor(m.fs.pump_RO2.control_volume.work, 1e-3)
            calculate_scaling_factors(m.fs.pump_RO2)
            calculate_scaling_factors(m.fs.mixer_permeate)


def initialize_desalination(m, **kwargs):
    """
    Initialized the model created by build_desalination.
    Arguments:
        m: pyomo concrete model with a built desalination train
        **kwargs: same keywords as provided to the build_desalination function
    """
    optarg = {'nlp_scaling_method': 'user-scaling'}

    if kwargs['has_desal_feed']:
        m.fs.feed.initialize(optarg=optarg)

    if kwargs['RO_type'] == 'Sep':
        if kwargs['has_desal_feed']:
            propagate_state(m.fs.s_desal_feed_RO)
        m.fs.RO.mixed_state[0].mass_frac_phase_comp  # touch properties to have a constraint on stateblock
        m.fs.RO.permeate_state[0].mass_frac_phase_comp
        m.fs.RO.retentate_state[0].mass_frac_phase_comp
        # m.fs.RO.initialize(optarg=optarg)  # IDAES error on initializing separators, simple enough to not need it

    elif kwargs['RO_type'] == '0D':
        if kwargs['has_desal_feed']:
            propagate_state(m.fs.s_desal_feed_pumpRO)
        m.fs.pump_RO.initialize(optarg=optarg)
        propagate_state(m.fs.s_desal_pumpRO_RO)
        m.fs.RO.initialize(optarg=optarg)

        if kwargs['is_twostage']:
            propagate_state(m.fs.s_desal_RO_pumpRO2)
            m.fs.pump_RO2.initialize(optarg=optarg)
            propagate_state(m.fs.s_desal_pumpRO2_RO2)
            m.fs.RO2.initialize(optarg=optarg)
            propagate_state(m.fs.s_desal_permeateRO_mixer)
            propagate_state(m.fs.s_desal_permeateRO2_mixer)
            m.fs.mixer_permeate.initialize(optarg=optarg)


def solve_build_desalination(**kwargs):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    property_models.build_prop(m, base='TDS')
    build_desalination(m, **kwargs)
    TransformationFactory("network.expand_arcs").apply_to(m)

    scale_desalination(m, **kwargs)

    initialize_desalination(m, **kwargs)

    check_dof(m)
    solve_with_user_scaling(m)

    return m

if __name__ == "__main__":
    solve_build_desalination(has_desal_feed=True, is_twostage=True,
                             RO_type='0D', RO_base='TDS', RO_level='detailed')