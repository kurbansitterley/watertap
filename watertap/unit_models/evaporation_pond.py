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

from copy import deepcopy

# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    check_optimal_termination,
    Param,
    Suffix,
    exp,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from watertap.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.constants import Constants
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import StrEnum
from idaes.core.util.exceptions import InitializationError, ConfigurationError

import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.core import ControlVolume0DBlock, InitializationMixin

# from watertap.costing.unit_models.ion_exchange import cost_ion_exchange

__author__ = "Kurban Sitterley"


"""
REFERENCES



"""


_log = idaeslog.getLogger(__name__)

area_correction_factor_param_dict = {
    12: (2.6429, -0.202),
    8: (2.0512, -0.152),
    4: (1.5357, -0.092),
}


@declare_process_block_class("EvaporationPond")
class IonExchangeODData(InitializationMixin, UnitModelBlockData):
    """
    Zero order evaporation pond model
    """

    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False.""",
        ),
    )

    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False.""",
        ),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
    **default** - MaterialBalanceType.useDefault.
    **Valid values:** {
    **MaterialBalanceType.useDefault - refer to property package for default
    balance type
    **MaterialBalanceType.none** - exclude material balances,
    **MaterialBalanceType.componentPhase** - use phase component balances,
    **MaterialBalanceType.componentTotal** - use total component balances,
    **MaterialBalanceType.elementTotal** - use total element balances,
    **MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.none,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.none.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
        ),
    )
    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
        **default** - MomentumBalanceType.pressureTotal.
        **Valid values:** {
        **MomentumBalanceType.none** - exclude momentum balances,
        **MomentumBalanceType.pressureTotal** - single pressure balance for material,
        **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
        **MomentumBalanceType.momentumTotal** - single momentum balance for material,
        **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )

    def build(self):
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True  # inlet block is an inlet
        self.properties_in = self.config.property_package.state_block_class(
            self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        )

        # Add outlet and waste block
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet

        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of liquid outlet",
            **tmp_dict,
        )
        self.area_correction_factor_base = Param(
            initialize=area_correction_factor_param_dict[self.dike_height][0],
            mutable=True,
            units=pyunits.dimensionless,
            doc="Area correction factor base",
        )

        self.area_correction_factor_exp = Param(
            initialize=area_correction_factor_param_dict[self.dike_height][1],
            mutable=True,
            units=pyunits.dimensionless,
            doc="Area correction factor exponent",
        )

        self.daily_change_water_temperature = Param(
            initialize=1,
            mutable=True,
            units=pyunits.degK / pyunits.day,
            doc="Daily change in water temperature",
        )

        self.mw_ratio = Param(
            initialize=0.622,
            units=pyunits.dimensionless,
            doc="Ratio of molecular weight of water vapor to air",
        )

        self.arden_buck_coeff_a = Param(
            initialize=6.1121,
            units=pyunits.millibar,
            doc="Arden Buck equation for saturation vapor pressure, a coefficient",  # https://en.wikipedia.org/wiki/Arden_Buck_equation
        )

        self.arden_buck_coeff_b = Param(
            initialize=18.678,
            units=pyunits.dimensionless,
            doc="Arden Buck equation for saturation vapor pressure, b coefficient",  # https://en.wikipedia.org/wiki/Arden_Buck_equation
        )

        self.arden_buck_coeff_c = Param(
            initialize=257.14,
            units=pyunits.dimensionless,
            doc="Arden Buck equation for saturation vapor pressure, c coefficient",  # https://en.wikipedia.org/wiki/Arden_Buck_equation
        )

        self.arden_buck_coeff_d = Param(
            initialize=234.5,
            units=pyunits.dimensionless,
            doc="Arden Buck equation for saturation vapor pressure, d coefficient",  # https://en.wikipedia.org/wiki/Arden_Buck_equation
        )

        self.latent_heat_of_vaporization_intercept = Param(
            initialize=2500.78,
            units=pyunits.kilojoule / pyunits.kg,
            doc="Intercept of latent heat of vaporization equation",  # doi:10.1029/2008JD010174
        )

        self.latent_heat_of_vaporization_slope = Param(
            initialize=2.3601,
            units=pyunits.kilojoule / (pyunits.kg * pyunits.degK),
            doc="Slope of latent heat of vaporization equation",  # doi:10.1029/2008JD010174
        )

        self.heat_capacity_air = Param(
            initialize=1013,
            mutable=True,
            units=pyunits.joule / (pyunits.kg * pyunits.degK),
            doc="Specific heat capacity of dry air",  # doi:10.1029/2008JD010174
        )

        self.heat_capacity_water = Param(
            initialize=4186,
            mutable=True,
            units=pyunits.joule / (pyunits.kg * pyunits.degK),
            doc="Specific heat capacity of water",  # doi:10.1029/2008JD010174
        )

        ## solids precipitation rate is a function of TDS concentration [g / L]
        ## solids_precipitation_rate [ft / yr] = a1 * C**2 + a2 * C + intercept
        ## solids_precipitation_rate [ft / yr] = 4.12e-6 * C**2 + 1.92e-4 * C + 1.15e-3

        self.solids_precipitation_rate_a1 = Param(
            initialize=4.12e-6,
            mutable=True,
            units=pyunits.feet * pyunits.year**-1,
            doc="Solids precipitation rate a1 coefficient",
        )

        self.solids_precipitation_rate_a2 = Param(
            initialize=1.92e-4,
            mutable=True,
            units=pyunits.feet * pyunits.year**-1,
            doc="Solids precipitation rate a2 coefficient",
        )

        self.solids_precipitation_rate_intercept = Param(
            initialize=1.15e-3,
            mutable=True,
            units=pyunits.feet * pyunits.year**-1,
            doc="Solids precipitation rate intercept",
        )

        self.dens_solids = Param(
            initialize=2.16,
            mutable=True,
            units=pyunits.g / pyunits.cm**3,
            doc="Density of precipitated solids",
        )

        self.pressure_atm = Param(
            initialize=101325,
            mutable=True,
            units=pyunits.Pa,
            doc="Atmospheric pressure",
        )

        self.dens_vap = Param(
            initialize=1.293,
            mutable=True,
            units=pyunits.kg / pyunits.m**3,
            doc="Density of air",
        )

        self.evaporation_pond_depth = Param(
            initialize=18,
            mutable=True,
            units=pyunits.inches,
            doc="Depth of evaporation pond",
        )

        self.evaporation_rate_salinity_adjustment_factor = Param(
            initialize=0.7,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Factor to reduce evaporation rate for higher salinity",
        )

        self.evaporation_rate_enhancement_adjustment_factor = Param(
            initialize=1,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Factor to increase evaporation rate due to enhancement",
        )

        self.air_temperature = Param(
            initialize=300, mutable=True, units=pyunits.degK, doc="Air temperature"
        )

        self.water_temperature_calc_slope = Param(
            initialize=1.04,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Slope of equation to calculate water temperature based on air temperature",
        )

        self.water_temperature_calc_intercept = Param(
            initialize=0.22,
            mutable=True,
            units=pyunits.degK,
            doc="Intercept of equation to calculate water temperature based on air temperature",
        )

        self.air_temperature_C = self.air_temperature - 273.15 * pyunits.degK

        self.relative_humidity = Param(
            initialize=0.5,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Relative humidity",
        )

        self.net_solar_radiation = Param(
            initialize=150,
            mutable=True,
            units=pyunits.watt / pyunits.m**2,
            doc="Net incident solar radiation",  # net shortwave radiation - net longwave radiation
        )

        self.net_heat_flux = Var(
            initialize=100,
            bounds=(0, None),
            units=pyunits.watt / pyunits.m**2,
            doc="Net heat flux out of system (water, soil)",
        )

        self.area_correction_factor = Var(
            initialize=1,
            bounds=(0.99, 3.2),
            units=pyunits.dimensionless,
            doc="Area correction factor",
        )

        self.saturation_vapor_pressure = Var(
            initialize=6,
            bounds=(0, None),
            units=pyunits.kPa,
            doc="Saturation vapor pressure at air temperature",
        )

        self.vapor_pressure = Var(
            initialize=4,
            bounds=(0, None),
            units=pyunits.kPa,
            doc="Vapor pressure at air temperature",
        )

        self.latent_heat_of_vaporization = Var(
            initialize=2500,
            bounds=(0, None),
            units=pyunits.kilojoule / pyunits.kg,
            doc="Latent heat of vaporization of water",
        )

        self.bowen_ratio = Var(
            initialize=0.3,
            bounds=(0, 2),
            units=pyunits.dimensionless,
            doc="Bowen ratio for BREB calculation of evaporation rate",
        )

        self.psychrometric_constant = Var(
            initialize=0.06,
            bounds=(0, None),
            units=pyunits.kPa * pyunits.degK**-1,
            doc="Psychrometric constant",
        )

        self.mass_flux_water_vapor = Var(
            initialize=1e-5,
            bounds=(0, 1e-3),
            units=pyunits.kg / (pyunits.m**2 * pyunits.s),
            doc="Mass flux of water vapor evaporated according using BREB method",
        )

        self.evaporation_rate = Var(
            initialize=0.03,
            bounds=(0, None),
            units=pyunits.m / pyunits.s,
            doc="Evaporation rate",
        )

        self.solids_precipitation_rate = Var(
            initialize=0.01,
            bounds=(0, None),
            units=pyunits.feet / pyunits.year,
            doc="Rate at which solids precipitate ",
        )

        self.total_evaporative_area_required = Var(
            initialize=100000,
            bounds=(0, None),
            units=pyunits.m**2,
            doc="Total evaporative area required",
        )

        self.evaporative_area_per_pond = Var(
            initialize=10000,
            bounds=(0, 405000),
            units=pyunits.m**2,
            doc="Evaporative area required per pond",
        )

        self.evaporation_pond_area = Var(
            initialize=10000,
            bounds=(0, None),
            units=pyunits.m**2,
            doc="Area of single evaporation pond",
        )

        self.number_evaporation_ponds = Var(
            initialize=1,
            bounds=(1, 25),
            units=pyunits.dimensionless,
            doc="Number of evaporation ponds",
        )

        @self.Expression()
        def evap_rate_mm_d(b):
            return pyunits.convert(
                b.evaporation_rate, to_units=pyunits.mm / pyunits.day
            )

        @self.Expression(doc="Total pond area in acres")
        def total_pond_area_acre(b):
            return pyunits.convert(
                b.evaporation_pond_area * b.number_evaporation_ponds,
                to_units=pyunits.acre,
            )

        @self.Expression(doc="Evaporative area per pond in acres")
        def evaporative_area_acre(b):
            return pyunits.convert(b.evaporative_area_per_pond, to_units=pyunits.acre)

        @self.Expression(doc="Individual pond area in acres")
        def pond_area_acre(b):
            return pyunits.convert(b.evaporation_pond_area, to_units=pyunits.acre)

        @self.Expression(doc="Net radiation for evaporation")
        def net_radiation(b):
            return b.net_solar_radiation - b.net_heat_flux

        @self.Expression(doc="Water temperature in deg C")
        def water_temperature_C(b):
            return b.properties_in.temperature - 273.15 * pyunits.degK

        @self.Expression(doc="Mass flow of precipitated solids")
        def mass_flow_precipitate(b):
            return pyunits.convert(
                b.number_evaporation_ponds
                * b.evaporation_pond_area
                * b.solids_precipitation_rate
                * b.dens_solids,
                to_units=pyunits.kg / pyunits.year,
            )

        @self.Expression(doc="Arden Buck fraction component")
        def arden_buck_exponential_term(b):
            temp_degC_dim = pyunits.convert(
                b.air_temperature_C * pyunits.degK**-1,
                to_units=pyunits.dimensionless,
            )  # dimensionless temperature in degC
            return (b.arden_buck_coeff_b - temp_degC_dim / b.arden_buck_coeff_d) * (
                temp_degC_dim / (b.arden_buck_coeff_c + temp_degC_dim)
            )

        @self.Constraint(doc="Solids precipitation rate")
        def solids_precipitation_rate_constraint(b):
            tds_in_dim = pyunits.convert(
                b.properties_in.conc_mass_comp["TDS"] * pyunits.g**-1 * pyunits.L,
                to_units=pyunits.dimensionless,
            )
            return (
                b.solids_precipitation_rate
                == b.solids_precipitation_rate_a1 * tds_in_dim**2
                + b.solids_precipitation_rate_a2 * tds_in_dim
                + b.solids_precipitation_rate_intercept
            )

        @self.Constraint(doc="Area correction factor calculation")
        def area_correction_factor_constraint(b):
            evap_per_pond_acre_dim = pyunits.convert(
                b.evaporative_area_per_pond * pyunits.acre**-1,
                to_units=pyunits.dimensionless,
            )
            return (
                b.area_correction_factor
                == b.area_correction_factor_base
                * evap_per_pond_acre_dim**b.area_correction_factor_exp
            )

        @self.Constraint(doc="Water temperature as function of air temperature")
        def water_temperature_constraint(b):
            return (
                b.properties_in.temperature
                == (
                    b.water_temperature_calc_slope * b.air_temperature_C
                    + b.water_temperature_calc_intercept
                )
                + 273.15 * pyunits.degK
            )

        @self.Constraint(doc="Net heat flux out of surroundings/ecosystem")
        def net_flux_heat_constraint(b):
            return b.net_heat_flux == pyunits.convert(
                b.properties_in.dens_mass
                * b.heat_capacity_water
                * b.evaporation_pond_depth
                * (b.properties_in.temperature - b.air_temperature),
                to_units=pyunits.watt / pyunits.m**2,
            )

        @self.Constraint(doc="Arden Buck equation for saturation vapor pressure")
        def arden_buck_constraint(b):
            return b.saturation_vapor_pressure == pyunits.convert(
                b.arden_buck_coeff_a * exp(b.arden_buck_exponential_term),
                to_units=pyunits.kPa,
            )

        @self.Constraint(doc="Latent heat of vaporization equation")
        def latent_heat_of_vaporization_constraint(b):
            return (
                b.latent_heat_of_vaporization
                == b.latent_heat_of_vaporization_intercept
                + b.latent_heat_of_vaporization_slope * b.air_temperature_C
            )

        @self.Constraint(doc="Psychrometric constant equation")
        def psychrometric_constant_constraint(b):
            return b.psychrometric_constant == pyunits.convert(
                (b.heat_capacity_air * b.pressure_atm)
                / (b.mw_ratio * b.latent_heat_of_vaporization),
                to_units=pyunits.kPa * pyunits.degK**-1,
            )

        @self.Constraint(doc="Vapor pressure equation")
        def vapor_pressure_constraint(b):
            return b.vapor_pressure == b.saturation_vapor_pressure * b.relative_humidity

        @self.Constraint(doc="Bowen ratio calculation")
        def bowen_ratio_constraint(b):
            return b.bowen_ratio == pyunits.convert(
                b.psychrometric_constant
                * (
                    (b.water_temperature_C - b.air_temperature_C)
                    / (b.saturation_vapor_pressure - b.vapor_pressure)
                ),
                to_units=pyunits.dimensionless,
            )

        @self.Constraint(doc="Mass flux water vapor using BREB method")
        def mass_flux_water_vapor_constraint(b):
            return (
                b.mass_flux_water_vapor
                == b.evaporation_rate_salinity_adjustment_factor
                * b.evaporation_rate_enhancement_adjustment_factor
                * pyunits.convert(
                    b.net_radiation
                    / (b.latent_heat_of_vaporization * (1 + b.bowen_ratio)),
                    to_units=pyunits.kg / (pyunits.m**2 * pyunits.s),
                )
            )

        @self.Constraint(doc="Evaporation rate")
        def evaporation_rate_constraint(b):
            return b.evaporation_rate == pyunits.convert(
                b.mass_flux_water_vapor / b.properties_in.dens_mass,
                to_units=pyunits.m / pyunits.s,
            )

        @self.Constraint(doc="Mass balance")
        def total_evaporative_area_required_constraint(b):
            return (
                b.mass_flux_water_vapor * b.total_evaporative_area_required
                == b.properties_in.flow_mass_phase_comp["Liq", "H2O"]
            )

        @self.Constraint(doc="Total evaporation pond area")
        def evaporative_area_per_pond_constraint(b):
            return (
                b.evaporative_area_per_pond * b.number_evaporation_ponds
                == b.total_evaporative_area_required
            )

        @self.Constraint(doc="Evaporation pond area")
        def evaporation_pond_area_constraint(b):
            return (
                b.evaporation_pond_area
                == b.evaporative_area_per_pond * b.area_correction_factor
            )

    def initialize(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for initialization routines

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

        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        flags = self.process_flow.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info("Initialization Step 1a Complete.")

        # ---------------------------------------------------------------------
        # Initialize other state blocks
        # Set state_args from inlet state
        if state_args is None:
            self.state_args = state_args = {}
            state_dict = self.process_flow.properties_in[
                self.flowsheet().config.time.first()
            ].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        state_args_out = deepcopy(state_args)

        for p, j in self.process_flow.properties_out.phase_component_set:
            if j in self.target_ion_set:
                state_args_out["flow_mol_phase_comp"][(p, j)] = (
                    state_args["flow_mol_phase_comp"][(p, j)] * 1e-3
                )

        # self.process_flow.properties_out.initialize(
        #     outlvl=outlvl,
        #     optarg=optarg,
        #     solver=solver,
        #     state_args=state_args_out,
        # )
        # init_log.info("Initialization Step 1b Complete.")

        # state_args_regen = deepcopy(state_args)

        # self.regeneration_stream.initialize(
        #     outlvl=outlvl,
        #     optarg=optarg,
        #     solver=solver,
        #     state_args=state_args_regen,
        # )

        # init_log.info("Initialization Step 1c Complete.")

        # # Solve unit
        # with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
        #     res = opt.solve(self, tee=slc.tee)
        #     if not check_optimal_termination(res):
        #         init_log.warning(
        #             f"Trouble solving unit model {self.name}, trying one more time"
        #         )
        #         res = opt.solve(self, tee=slc.tee)

        # init_log.info("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        # # Release Inlet state
        # self.process_flow.properties_in.release_state(flags, outlvl=outlvl)
        # init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        # if not check_optimal_termination(res):
        #     raise InitializationError(f"Unit model {self.name} failed to initialize.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

    # def _get_stream_table_contents(self, time_point=0):

    #     return create_stream_table_dataframe(
    #         {
    #             "Feed Inlet": self.inlet,
    #             "Liquid Outlet": self.outlet,
    #             "Regen Outlet": self.regen,
    #         },
    #         time_point=time_point,
    #     )

    def _get_performance_contents(self, time_point=0):
        var_dict = dict()
        return {"vars": var_dict}

    # @property
    # def default_costing_method(self):
    #     return cost_ion_exchange
