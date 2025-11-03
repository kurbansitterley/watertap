#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
import pandas as pd

import numpy as np
from scipy.stats import gamma

# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Block,
    Constraint,
    Param,
    NonNegativeReals,
    NegativeReals,
    value,
    check_optimal_termination,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    useDefault,
)
from idaes.core.util import scaling as iscale
from idaes.core.util.exceptions import InitializationError, ConfigurationError
from idaes.core.util.misc import add_object_reference, StrEnum
import idaes.logger as idaeslog

from watertap.core import (  # noqa # pylint: disable=unused-import
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    MembraneChannel1DBlock,
    PressureChangeType,
)
from watertap.core.membrane_channel1d import CONFIG_Template
from watertap.unit_models.pressure_changer import Pump
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    ReverseOsmosis1DData,
)
from watertap.unit_models.reverse_osmosis_base import _add_has_full_reporting
from watertap.costing.unit_models.ccro import cost_ccro
from watertap.costing.unit_models.reverse_osmosis import cost_reverse_osmosis
from watertap.core.util.initialization import interval_initializer
from watertap.core.solvers import get_solver
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock

from idaes.core import UnitModelBlockData

__author__ = "Adam Atia, Bernard Knueven"

# Set up logger
_log = idaeslog.getLogger(__name__)


class CyclePhase(StrEnum):
    filtration = "filtration"
    flushing = "flushing"


@declare_process_block_class("CCRO1D")
class CCRO1DData(ReverseOsmosis1DData):
    """
    CCRO 1D Unit Model Class.
    """

    # this model is meant to be identical to RO1D but with different costing

    CONFIG = ReverseOsmosis1DData.CONFIG()

    CONFIG.declare(
        "cycle_phase",
        ConfigValue(
            default=CyclePhase.filtration,
            domain=In(CyclePhase),
            description="Cycle phase - must be either 'filtration' or 'flushing'",
            doc="""Indicates the current cycle phase of the CCRO process.
    **default** - 'filtration'.""",
        ),
    )

    CONFIG.declare(
        "surrogate_model_file",
        ConfigValue(
            default=None,
            domain=str,
            description="Path to surrogate model file",
            doc="""User provided surrogate model .json file.""",
        ),
    )
    CONFIG.declare(
        "surrogate_filename_save",
        ConfigValue(
            default=None,
            domain=str,
            description="Filename used to save surrogate model to .json",
            doc="""Filename used to save surrogate model file to .json""",
        ),
    )
    CONFIG.declare(
        "dataset_filename",
        ConfigValue(
            default=None,
            domain=str,
            description="Path to data file",
            doc="""Path to data file. Must be a .csv""",
        ),
    )
    # _add_has_full_reporting(CONFIG)

    def build(self):

        # super(UnitModelBlockData, self).build()
        UnitModelBlockData.build(self)

        if self.config.cycle_phase == CyclePhase.filtration:
            # use ReverseOsmosis1D build method
            super().build()
            self.del_component(self.retentate)

        if self.config.cycle_phase == CyclePhase.flushing:
            # UnitModelBlockData.build(self)
            # Parameters
            self.number_tanks_in_series = Param(
                initialize=5,
                units=pyunits.dimensionless,
                doc="Number of tanks in series to represent the a plug flow reactor with mixing",
            )

            self.accumulation_volume = Param(
                initialize=0.0,
                units=pyunits.m**3,
                doc="Accumulation volume is being flushed",
            )

            self.flushing_flow_rate = Param(
                initialize=0.0,
                units=pyunits.m**3 / pyunits.s,
                doc="Flow rate of the flushing water",
            )

            self.mean_residence_time = Var(
                initialize=30.0,
                bounds=(0, None),
                units=pyunits.s,
                doc="Mean residence time in the system",
            )

            self.pre_flushing_concentration = Var(
                initialize=10,
                bounds=(0, None),
                units=pyunits.kg / pyunits.m**3,
                doc="Concentration of the accumulation volume prior to flushing at the end of the concentration cycle",
            )

            self.post_flushing_concentration = Var(
                initialize=10,
                bounds=(0, None),
                units=pyunits.kg / pyunits.m**3,
                doc="Concentration of the accumulation volume after flushing at the start of the concentration cycle",
            )

            # Variables - Not sure if these are needed or if their created with the surrogate model
            self.flushing_efficiency = Var(
                initialize=0.1,
                units=pyunits.dimensionless,
                bounds=(0, 1),
                doc="Flushing efficiency of the system",
            )

            self.flushing_time = Var(
                initialize=20.0,
                units=pyunits.s,
                bounds=(0, None),
                doc="Duration of flushing",
            )

            # If a surrogate model is passed, it is used to calculate the concentration
            if self.config.surrogate_model_file is not None:
                try:
                    self.load_surrogate()
                except Exception as e:
                    err_msg = f"Error loading surrogate model: {e}"
                    raise ConfigurationError(err_msg)
            else:
                self.create_surrogate_model()

        self.accumulation_time = Var(
            initialize=1,
            units=pyunits.s,
            bounds=(0, None),
            doc="Time for accumulation",
        )

        self.accumulation_volume_block = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,  # will create our own volume and mass balance handling for this
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )

        self.accumulation_volume_ratio = Var(
            initialize=1e-3,
            bounds=(1e-4, 1),
            units=pyunits.m**3
            / pyunits.meter**2,  # m3 dead volume per m2 membrane area
            doc="Required accumulation volume to membrane area ratio",
        )

        # Add volume to the accumulation volume block
        self.accumulation_volume_block.add_geometry()
        if self.config.cycle_phase == CyclePhase.filtration:

            @self.Constraint(
                self.flowsheet().config.time,
            )
            def eq_acc_volume(b, t):
                return b.accumulation_volume_block.volume[t] == pyunits.convert(
                    b.accumulation_volume_ratio * b.area, to_units=pyunits.m**3
                )

        self.accumulation_volume_block.add_state_blocks(has_phase_equilibrium=False)

        if self.config.cycle_phase == CyclePhase.flushing:
            # Need an inlet if flushing
            self.add_inlet_port(
                name="inlet", block=self.accumulation_volume_block.properties_in
            )

        self.add_outlet_port(
            name="outlet", block=self.accumulation_volume_block.properties_out
        )

        self.accumulation_volume_block.mass_frac_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="Accumulated mass fractions in dead volume",
        )
        self.accumulation_volume_block.mass_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=1,
            bounds=(0, None),
            units=pyunits.kg,
            doc="Accumulated mass in dead volume",
        )

        # Add previous state block
        self.previous_state = Block()

        self.previous_state.mass_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=1,
            units=pyunits.kg,
            doc="Prior mass in dead volume",
        )
        self.previous_state.volume = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            initialize=1,
            units=pyunits.m**3,
            doc="Prior volume in dead volume",
        )
        self.previous_state.mass_frac_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=1,
            units=pyunits.dimensionless,
            doc="Prior mass fraction in dead volume",
        )
        self.previous_state.dens_mass_phase = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            initialize=1,
            units=pyunits.kg / pyunits.m**3,
            doc="Prior density in dead volume",
        )

        # @self.Constraint(
        #     self.flowsheet().config.time,
        # )
        # def eq_acc_volume(b, t):
        #     return b.accumulation_volume_block.volume[t] == pyunits.convert(
        #         b.accumulation_volume_ratio * b.area, to_units=pyunits.m**3
        #     )

        @self.Constraint(
            self.flowsheet().config.time,
        )
        def eq_acc_vol_prev_state_vol(b, t):
            return (
                b.accumulation_volume_block.volume[t]
                == b.previous_state.volume[t, "Liq"]
            )

        @self.accumulation_volume_block.Constraint(
            self.flowsheet().config.time,
            doc="Isothermal energy balance for accumulation volume",
        )
        def eq_isothermal(b, t):
            return b.properties_in[t].temperature == b.properties_out[t].temperature

        @self.accumulation_volume_block.Constraint(
            self.flowsheet().config.time,
            doc="Isobaric pressure balance for accumulation volume",
        )
        def eq_isobaric(b, t):
            return b.properties_in[t].pressure == b.properties_out[t].pressure

        # Accumulation volume constraints
        @self.accumulation_volume_block.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass after accumulation",
        )
        def eq_mass_phase_comp(b, t, p, j):
            return b.mass_phase_comp[t, p, j] == (
                (
                    b.properties_in[t].flow_mass_phase_comp[p, j]
                    - b.properties_out[t].flow_mass_phase_comp[p, j]
                )
                * self.accumulation_time
                + self.previous_state.mass_phase_comp[t, p, j]
            )

        @self.accumulation_volume_block.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass fractions after accumulation",
        )
        def eq_mass_frac_phase_comp(b, t, p, j):
            mass_sum = []
            for comp in self.config.property_package.component_list:
                mass_sum.append(b.mass_phase_comp[t, p, comp])
            return b.mass_frac_phase_comp[t, p, j] == (
                b.mass_phase_comp[t, p, j] / sum(mass_sum)
            )

        # Previous state constraints
        @self.previous_state.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Prior mass fractions",
        )
        def eq_mass_frac_phase_comp(b, t, p, j):
            mass_sum = sum(
                b.mass_phase_comp[t, p, jj]
                for jj in self.config.property_package.component_list
            )
            return b.mass_frac_phase_comp[t, p, j] == (
                b.mass_phase_comp[t, p, j] / mass_sum
            )

        @self.previous_state.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            doc="Prior dead volume constraint",
        )
        def eq_volume(b, t, p):
            mass_sum = sum(
                b.mass_phase_comp[t, p, jj]
                for jj in self.config.property_package.component_list
            )
            return b.volume[t, p] == pyunits.convert(
                (mass_sum / b.dens_mass_phase[t, p]), to_units=pyunits.m**3
            )

        @self.accumulation_volume_block.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            doc="Current volume constraint",
        )
        def eq_volume(b, t, p):
            mass_sum = sum(
                b.mass_phase_comp[t, p, jj]
                for jj in self.config.property_package.component_list
            )
            return b.volume[t] == pyunits.convert(
                (mass_sum / b.properties_out[t].dens_mass_phase[p]),
                to_units=pyunits.m**3,
            )

        @self.accumulation_volume_block.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass fractions after accumulation",
        )
        def eq_flow_mass_phase_comp_out(b, t, p, j):
            if j == "H2O":
                return Constraint.Skip
            else:
                return (
                    b.properties_out[t].mass_frac_phase_comp[p, j]
                    == b.mass_frac_phase_comp[t, p, j]
                )

        if self.config.cycle_phase == CyclePhase.filtration:

            @self.Constraint(
                self.flowsheet().config.time,
                self.config.property_package.phase_list,
                self.config.property_package.component_list,
                doc="Mass flow from RO brine to accumulation volume",
            )
            def eq_acc_vol_mass_flow_in(b, t, p, j):
                return (
                    b.accumulation_volume_block.properties_in[t].flow_mass_phase_comp[
                        p, j
                    ]
                    == b.feed_side.properties[t, 1].flow_mass_phase_comp[p, j]
                )

            @self.Constraint(
                self.flowsheet().config.time,
                doc="Temperature at accumulation volume inlet",
            )
            def eq_acc_vol_temperature_in(b, t):
                return (
                    b.accumulation_volume_block.properties_in[t].temperature
                    == b.feed_side.properties[t, 1].temperature
                )

            @self.Constraint(
                self.flowsheet().config.time,
                doc="Pressure at accumulation volume inlet",
            )
            def eq_acc_vol_pressure_in(b, t):
                return (
                    b.accumulation_volume_block.properties_in[t].pressure
                    == b.feed_side.properties[t, 1].pressure
                )

        if self.config.cycle_phase == CyclePhase.flushing:

            @self.Constraint()
            def eq_pre_flushing_concentration(b):
                return (
                    b.pre_flushing_concentration
                    == b.previous_state.mass_frac_phase_comp[0, "Liq", "NaCl"]
                    * b.previous_state.dens_mass_phase[0, "Liq"]
                )

            # Concentration after flushing should be the dead volume properties out concentration
            @self.Constraint()
            def eq_post_flushing_concentration(b):
                return (
                    b.post_flushing_concentration
                    * b.accumulation_volume_block.volume[0]
                    == b.accumulation_volume_block.mass_phase_comp[0, "Liq", "NaCl"]
                )

            @self.Constraint()
            def eq_post_flushing_concentration_2(b):
                return (
                    b.post_flushing_concentration
                    == (1 - b.flushing_efficiency) * b.pre_flushing_concentration
                    + b.flushing_efficiency
                    * b.accumulation_volume_block.properties_in[0].conc_mass_phase_comp[
                        "Liq", "NaCl"
                    ]
                )

            # Constraint to calculate the concentration of the accumulation volume at the end of flushing
            # self.flushing_concentration_constraint = Constraint(
            #     expr=self.post_flushing_concentration
            #     == (1 - self.flushing_efficiency) * self.pre_flushing_concentration
            #     + self.flushing_efficiency * self.accumulation_volume_block.properties_in[0].conc_mass_phase_comp["Liq", "NaCl"]
            # )

    def initialize_build(
        self, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None
    ):
        """
        General wrapper for pressure changer initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """

        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        if self.config.cycle_phase == CyclePhase.filtration:
            # If filtration we want to inherit the RO1D initialization
            super().initialize_build(
                state_args=state_args, outlvl=outlvl, solver=solver, optarg=optarg
            )

        # pre-solve using interval arithmetic
        interval_initializer(self)

        opt = get_solver(solver, optarg)
        acc_vol = self.accumulation_volume_block.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )

        init_log.info("Dead Volume Step 1 Initialized.")

        if self.config.cycle_phase == CyclePhase.flushing:

            # Initialize surrogate
            pre_con_fixed = self.pre_flushing_concentration.fixed
            post_con_fixed = self.post_flushing_concentration.fixed
            self.pre_flushing_concentration.fix()
            self.post_flushing_concentration.unfix()
            self.init_data = pd.DataFrame(
                {
                    "time": [value(self.flushing_time)],
                    "mean_residence_time": [value(self.mean_residence_time)],
                }
            )

            self.init_output = self.surrogate.evaluate_surrogate(self.init_data)

            # Set initial values for model variables
            self.flushing_efficiency.set_value(self.init_output["F_t"].values[0])

            # Solve unit
            # opt = get_solver(solver, optarg)

            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(self, tee=slc.tee)

            if pre_con_fixed == False:
                self.pre_flushing_concentration.unfix()
            if post_con_fixed == False:
                self.post_flushing_concentration.unfix()

            init_log.info_high(f"Initialization Step 2 {idaeslog.condition(res)}")

            if not check_optimal_termination(res):
                raise InitializationError(
                    f"Unit model {self.name} failed to initialize"
                )

            init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)

        init_log.info_high("Dead Volume Step 2 Initialized.")
        self.accumulation_volume_block.release_state(acc_vol, outlvl)

    def calculate_scaling_factors(self):

        if self.config.cycle_phase == CyclePhase.filtration:
            super().calculate_scaling_factors()
        if self.config.cycle_phase == CyclePhase.flushing:
            
            UnitModelBlockData.calculate_scaling_factors(self)
            iscale.constraint_scaling_transform(
                self.eq_pre_flushing_concentration, 1e-1
            )
            iscale.constraint_scaling_transform(
                self.eq_post_flushing_concentration, 1e-1
            )
            iscale.set_scaling_factor(self.pre_flushing_concentration, 1e-1)
            iscale.set_scaling_factor(self.post_flushing_concentration, 1e-1)

        sf = iscale.get_scaling_factor(self.accumulation_time)
        if sf is None:
            sf = 1 / self.accumulation_time.value
            iscale.set_scaling_factor(
                self.accumulation_time,
                1,
            )
        for p in self.config.property_package.phase_list:
            for c in self.config.property_package.component_list:
                sf = 1
                iscale.set_scaling_factor(
                    self.accumulation_volume_block.mass_frac_phase_comp, sf
                )

                iscale.set_scaling_factor(self.previous_state.mass_frac_phase_comp, sf)

                iscale.constraint_scaling_transform(
                    self.accumulation_volume_block.eq_mass_frac_phase_comp[0, p, c], sf
                )
                iscale.constraint_scaling_transform(
                    self.previous_state.eq_mass_frac_phase_comp[0, p, c], sf
                )

                sf = iscale.get_scaling_factor(
                    self.accumulation_volume_block.properties_in[
                        0
                    ].flow_mass_phase_comp[p, c]
                )
                iscale.set_scaling_factor(
                    self.accumulation_volume_block.mass_phase_comp[0, p, c], sf
                )
                iscale.set_scaling_factor(
                    self.previous_state.mass_phase_comp[0, p, c], sf
                )
                iscale.constraint_scaling_transform(
                    self.accumulation_volume_block.eq_mass_phase_comp[0, p, c], sf
                )

            sf = iscale.get_scaling_factor(self.accumulation_volume_block.volume[0])
            if sf is None:
                sf = 1 / self.volume.value
                iscale.set_scaling_factor(
                    self.volume,
                    1,
                )
            iscale.constraint_scaling_transform(
                self.accumulation_volume_block.eq_volume[0, p], sf
            )
            sf = iscale.get_scaling_factor(self.previous_state.volume[0, p])

            if iscale.get_scaling_factor(self.previous_state.volume[0, p]) is None:
                sf = 1 / self.previous_state.volume[0, p].value
                iscale.set_scaling_factor(
                    self.previous_state.volume[0, p],
                    sf,
                )
            iscale.constraint_scaling_transform(self.previous_state.eq_volume[0, p], sf)
        iscale.constraint_scaling_transform(
            self.accumulation_volume_block.eq_isobaric[0], 1e-5
        )
        iscale.constraint_scaling_transform(
            self.accumulation_volume_block.eq_isothermal[0], 1 / 273
        )

    def generate_rtd_profile(self):
        # Generate the RTD profile for the flushing process
        rtd_profile = pd.DataFrame()
        time = np.linspace(0, 3 * 35, 100)
        N= self.number_tanks_in_series

        for t_m in np.linspace(0, 2*35, 5):
            scale = t_m / N
            F_t = gamma.cdf(time, a=N, scale=scale)
            df = pd.DataFrame({"time": time, "F_t": F_t})
            df["mean_residence_time"] = t_m
            df["N"] = N
            rtd_profile = pd.concat([rtd_profile, df], ignore_index=True)

        self.rtd_profile = rtd_profile

    def create_surrogate_model(self):

        # Create the surrogate model for the flushing process
        # Check is a dataset file was passed

        if self.config.dataset_filename is not None:
            # TODO: Create surrogate using experimental data
            pass

        else:
            # Create a default surrogate model with generated rtd profile
            self.generate_rtd_profile()

            # Create a surrogate using the default rtd profile
            # TODO: Surrogate creation and fit check - Update the

    def load_surrogate(self):
        self.surrogate_blk = SurrogateBlock(concrete=True)
        self.surrogate = PysmoSurrogate.load_from_file(self.config.surrogate_model_file)
        self.surrogate_blk.build_model(
            self.surrogate,
            input_vars=[self.flushing_time, self.mean_residence_time],  
            output_vars=self.flushing_efficiency,
        )

    @property
    def default_costing_method(self):
        return cost_ccro
        # return cost_reverse_osmosis
