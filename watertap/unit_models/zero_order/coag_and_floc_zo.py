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
"""
This module contains a zero-order representation of a coagulation/flocculation unit
operation.
"""

import pyomo.environ as pyo

# from pyomo.environ import Constraint, units as pyo.units, Var
from idaes.core import declare_process_block_class

from watertap.core import build_pt, ZeroOrderBaseData
from watertap.core.zero_order_electricity import pump_electricity
from watertap.custom_exceptions import FrozenPipes

__author__ = "Adam Atia"


@declare_process_block_class("CoagulationFlocculationZO")
class CoagulationFlocculationZOData(ZeroOrderBaseData):
    """
    Zero-Order model for a Coagulation/Flocculation unit operation.
    """

    CONFIG = ZeroOrderBaseData.CONFIG()

    def build(self):
        super().build()

        self._tech_type = "coag_and_floc"

        build_pt(self)

        self.alum_dose = pyo.Var(
            units=pyo.units.mg / pyo.units.L,
            bounds=(0, None),
            doc="Dosing rate of alum",
        )

        self.alum_ratio_in_solution = pyo.Var(
            units=pyo.units.dimensionless,
            bounds=(0, 1),
            doc="Mass ratio of alum in feed solution",
        )

        self.alum_solution_density = pyo.Var(
            units=pyo.units.kg / pyo.units.m**3,
            bounds=(0, None),
            doc="Mass density of alum feed solution",
        )

        self.alum_flow_mass = pyo.Var(
            units=pyo.units.kg / pyo.units.s,
            bounds=(0, None),
            doc="Mass flow rate of alum",
        )

        self.alum_flow_vol = pyo.Var(
            self.flowsheet().time,
            units=pyo.units.m**3 / pyo.units.s,
            bounds=(0, None),
            doc="Volumetric flow rate of alum solution",
        )

        self.polymer_dose = pyo.Var(
            units=pyo.units.mg / pyo.units.L,
            bounds=(0, None),
            doc="Dosing rate of polymer",
        )

        self.polymer_ratio_in_solution = pyo.Var(
            units=pyo.units.dimensionless,
            bounds=(0, 1),
            doc="Mass ratio of polymer in feed solution",
        )

        self.polymer_solution_density = pyo.Var(
            units=pyo.units.kg / pyo.units.m**3,
            bounds=(0, None),
            doc="Mass density of polymer feed solution",
        )

        self.polymer_flow_mass = pyo.Var(
            units=pyo.units.kg / pyo.units.s,
            bounds=(0, None),
            doc="Mass flow rate of polymer",
        )

        self.polymer_flow_vol = pyo.Var(
            self.flowsheet().time,
            units=pyo.units.m**3 / pyo.units.s,
            bounds=(0, None),
            doc="Volumetric flow rate of polymer solution",
        )

        self.rapid_mix_retention_time = pyo.Var(
            units=pyo.units.seconds,
            doc="Rapid Mix Retention Time",
        )

        self.floc_retention_time = pyo.Var(
            units=pyo.units.minutes,
            doc="Floc Retention Time",
        )

        self.rapid_mix_basin_vol = pyo.Var(
            units=pyo.units.m**3, doc="Rapid Mix Basin Volume"
        )

        self.floc_basin_vol = pyo.Var(units=pyo.units.m**3, doc="Floc Basin Volume")

        self.num_rapid_mixers = pyo.Var(
            units=pyo.units.dimensionless, doc="Number of Rapid Mixers"
        )

        self.num_floc_mixers = pyo.Var(
            units=pyo.units.dimensionless, doc="Number of Floc Mixers"
        )

        self.num_rapid_mix_processes = pyo.Var(
            units=pyo.units.dimensionless, doc="Number of Rapid Mix Processes"
        )

        self.num_floc_processes = pyo.Var(
            units=pyo.units.dimensionless, doc="Number of Floc Processes"
        )

        self.num_coag_processes = pyo.Var(
            units=pyo.units.dimensionless, doc="Number of Coagulation Processes"
        )

        self.num_floc_injection_processes = pyo.Var(
            units=pyo.units.dimensionless, doc="Number of Floc Injection Processes"
        )

        self.velocity_gradient_rapid_mix = pyo.Var(
            self.flowsheet().config.time,
            units=pyo.units.s**-1,
            doc="Rapid Mix Velocity Gradient",
        )

        self.velocity_gradient_floc = pyo.Var(
            self.flowsheet().config.time,
            units=pyo.units.s**-1,
            doc="Floc Velocity Gradient",
        )

        self.power_rapid_mix = pyo.Var(
            self.flowsheet().config.time,
            units=pyo.units.kW,
            doc="Rapid Mix Power Consumption",
        )

        self.power_floc = pyo.Var(
            self.flowsheet().config.time,
            units=pyo.units.kW,
            doc="Floc Power Consumption",
        )

        self.electricity = pyo.Var(
            self.flowsheet().config.time,
            units=pyo.units.kW,
            doc="Total Power Consumption",
        )

        self._fixed_perf_vars.append(self.alum_dose)
        self._fixed_perf_vars.append(self.polymer_dose)
        self._fixed_perf_vars.append(self.rapid_mix_retention_time)
        self._fixed_perf_vars.append(self.floc_retention_time)
        self._fixed_perf_vars.append(self.num_floc_injection_processes)
        self._fixed_perf_vars.append(self.num_floc_processes)
        self._fixed_perf_vars.append(self.num_rapid_mixers)
        self._fixed_perf_vars.append(self.num_coag_processes)
        self._fixed_perf_vars.append(self.num_floc_mixers)
        self._fixed_perf_vars.append(self.num_rapid_mix_processes)
        self._fixed_perf_vars.append(self.velocity_gradient_rapid_mix)
        self._fixed_perf_vars.append(self.velocity_gradient_floc)

        self._perf_var_dict["Alum Dosage (mg/L)"] = self.alum_dose
        self._perf_var_dict["Polymer Dosage (mg/L)"] = self.polymer_dose
        self._perf_var_dict["Alum Flow (kg/s)"] = self.alum_flow_mass
        self._perf_var_dict["Polymer Flow (kg/s)"] = self.polymer_flow_mass
        self._perf_var_dict["Rapid Mix Basin Volume (m^3)"] = self.rapid_mix_basin_vol
        self._perf_var_dict["Floc Basin Volume (m^3)"] = self.floc_basin_vol
        self._perf_var_dict["Rapid Mix Retention Time (s)"] = (
            self.rapid_mix_retention_time
        )
        self._perf_var_dict["Floc Retention Time (min)"] = self.floc_retention_time
        self._perf_var_dict["Rapid Mix Velocity Gradient (1/s)"] = (
            self.velocity_gradient_rapid_mix
        )
        self._perf_var_dict["Floc Velocity Gradient (1/s)"] = (
            self.velocity_gradient_floc
        )
        self._perf_var_dict["Rapid Mix Power (kW)"] = self.power_rapid_mix
        self._perf_var_dict["Floc Power (kW)"] = self.power_floc
        self._perf_var_dict["Total Power Consumption (kW)"] = self.electricity

        def rule_rapid_mix_basin_vol(blk):
            return (
                blk.rapid_mix_basin_vol
                == blk.properties[0].flow_vol * blk.rapid_mix_retention_time
            )

        self.rapid_mix_basin_vol_constraint = pyo.Constraint(
            rule=rule_rapid_mix_basin_vol
        )

        def rule_floc_basin_vol(blk):
            return blk.floc_basin_vol == pyo.units.convert(
                blk.properties[0].flow_vol * blk.floc_retention_time,
                to_units=pyo.units.m**3,
            )

        self.floc_basin_vol_constraint = pyo.Constraint(rule=rule_floc_basin_vol)

        def rule_alum_flow_mass(blk):
            return blk.alum_flow_mass == pyo.units.convert(
                (blk.alum_dose * blk.properties[0].flow_vol)
                / blk.alum_ratio_in_solution,
                to_units=pyo.units.kg / pyo.units.s,
            )

        self.alum_flow_mass_constraint = pyo.Constraint(rule=rule_alum_flow_mass)

        def rule_alum_flow_vol(blk):
            return blk.alum_flow_vol[0] == pyo.units.convert(
                blk.alum_flow_mass / blk.alum_solution_density,
                to_units=pyo.units.m**3 / pyo.units.s,
            )

        self.alum_flow_vol_constraint = pyo.Constraint(rule=rule_alum_flow_vol)
        pump_electricity(self, self.alum_flow_vol)

        # def rule_polymer_flow_mass(blk):
        #     return blk.polymer_flow_mass == pyo.units.convert(
        #         (blk.polymer_dose * blk.properties[0].flow_vol)
        #         / blk.polymer_ratio_in_solution,
        #         to_units=pyo.units.m**3 / pyo.units.s,
        #     )

        # self.polymer_flow_mass_constraint = pyo.Constraint(rule=rule_polymer_flow_mass)

        # def rule_polymer_flow_vol(blk):
        #     return blk.polymer_flow_vol[0] == pyo.units.convert(
        #         blk.polymer_flow_mass / blk.polymer_solution_density,
        #         to_units=pyo.units.m**3 / pyo.units.s,
        #     )
    
        # self.polymer_flow_vol_constraint = pyo.Constraint(rule=rule_polymer_flow_vol)

        # pump_electricity(self, self.polymer_flow_vol)

        # def rule_chem_flow(blk, t, j):
        #     if j == "alum":
        #         chemical_dosage = blk.alum_dose[t]
        #     elif j == "polymer":
        #         chemical_dosage = blk.polymer_dose[t]
        #     else:
        #         raise FrozenPipes(
        #             f"Unexpected chemical {j} in coagulation/flocculation unit"
        #         )
        #     return blk.chemical_flow_mass[t, j] == pyo.units.convert(
        #         chemical_dosage * blk.properties[t].flow_vol,
        #         to_units=pyo.units.kg / pyo.units.s,
        #     )

        # self.chemical_flow_constraint = pyo.Constraint(
        #     self.flowsheet().time, ["alum", "polymer"], rule=rule_chem_flow
        # )


        # @self.pyo.Constraint(
        #     self.flowsheet().time, doc="Constraint for rapid mix power consumption"
        # )
        def rule_power_rapid_mix(b):
            return b.power_rapid_mix == pyo.units.convert(
                b.num_rapid_mixers
                * b.properties[0].visc_d
                * b.rapid_mix_basin_vol
                * b.velocity_gradient_rapid_mix ** 2,
                to_units=pyo.units.kW,
            )
        
        self.rapid_mix_power_constraint = pyo.Constraint(rule=rule_power_rapid_mix
        )

        # @self.pyo.Constraint(
        #     self.flowsheet().time, doc="Constraint for floc power consumption"
        # )
        def rule_power_floc(b):
            return b.power_floc == pyo.units.convert(
                b.num_floc_mixers
                * b.properties[0].visc_d
                * b.floc_basin_vol
                * b.velocity_gradient_floc ** 2,
                to_units=pyo.units.kW,
            )

        self.floc_power_constraint = pyo.Constraint(
            self.flowsheet().time, rule=rule_power_floc
        )
        # @self.pyo.Constraint(self.flowsheet().time, doc="Total power consumption")
        # def electricity_constraint(b, t):
        #     return b.electricity[t] == b.power_floc[t] + b.power_rapid_mix[t]

        def rule_electricity(b):
            return b.electricity[0] == b.power_floc + b.power_rapid_mix

    @property
    def default_costing_method(self):
        return self.cost_coag_and_floc

    @staticmethod
    def cost_coag_and_floc(blk):
        """
        General method for costing coagulation/flocculation processes. Capital cost
        is based on the alum flowrate and the polymer flowrate of the incoming stream.
        This method also registers the electricity demand as a costed flow.
        """
        t0 = blk.flowsheet().time.first()

        # Get parameter dict from database
        parameter_dict = blk.unit_model.config.database.get_unit_operation_parameters(
            blk.unit_model._tech_type, subtype=blk.unit_model.config.process_subtype
        )

        # Get costing parameter sub-block for this technology
        A, B, C, D, E, F, G, H = blk.unit_model._get_tech_parameters(
            blk,
            parameter_dict,
            blk.unit_model.config.process_subtype,
            [
                "capital_mix_a_parameter",
                "capital_mix_b_parameter",
                "capital_floc_a_parameter",
                "capital_floc_b_parameter",
                "capital_coag_inj_a_parameter",
                "capital_coag_inj_b_parameter",
                "capital_floc_inj_a_parameter",
                "capital_floc_inj_b_parameter",
            ],
        )

        # Add cost variable and constraint
        blk.capital_cost = pyo.pyo.Var(
            initialize=1,
            units=blk.config.flowsheet_costing_block.base_currency,
            bounds=(0, None),
            doc="Capital cost of unit operation",
        )

        blk.cost_rapid_mix = pyo.pyo.Expression(
            expr=(
                A
                * pyo.units.convert(
                    blk.unit_model.rapid_mix_basin_vol, to_units=pyo.units.gallons
                )
                + B
            )
            * blk.unit_model.num_rapid_mix_processes
        )

        blk.cost_floc = pyo.pyo.Expression(
            expr=(
                C
                * pyo.units.convert(
                    blk.unit_model.floc_basin_vol, to_units=pyo.units.Mgallons
                )
                + D
            )
            * blk.unit_model.num_floc_processes
        )

        blk.cost_coag_inj = pyo.pyo.Expression(
            expr=(
                E
                * pyo.units.convert(
                    blk.unit_model.alum_flow_mass,
                    to_units=(pyo.units.lb / pyo.units.hour),
                )
                + F
            )
            * blk.unit_model.num_coag_processes
        )

        blk.cost_floc_inj = pyo.pyo.Expression(
            expr=(
                G
                * pyo.units.convert(
                    blk.unit_model.polymer_flow_mass,
                    to_units=(pyo.units.lb / pyo.units.day),
                )
                + H
            )
            * blk.unit_model.num_floc_injection_processes
        )

        expr = (
            pyo.units.convert(
                blk.cost_rapid_mix,
                to_units=blk.config.flowsheet_costing_block.base_currency,
            )
            + pyo.units.convert(
                blk.cost_floc, to_units=blk.config.flowsheet_costing_block.base_currency
            )
            + pyo.units.convert(
                blk.cost_coag_inj,
                to_units=blk.config.flowsheet_costing_block.base_currency,
            )
            + pyo.units.convert(
                blk.cost_floc_inj,
                to_units=blk.config.flowsheet_costing_block.base_currency,
            )
        )

        blk.costing_package.add_cost_factor(
            blk, parameter_dict["capital_cost"]["cost_factor"]
        )

        blk.capital_cost_constraint = pyo.pyo.Constraint(
            expr=blk.capital_cost == blk.cost_factor * expr
        )

        # Register flows
        blk.config.flowsheet_costing_block.cost_flow(
            blk.unit_model.electricity[t0], "electricity"
        )
