#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import pytest
from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    value,
    Var,
    log10,
    units,
)
from pyomo.network import Port
from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
)
from watertap.unit_models.electroNP_ZO import ElectroNPZO
from watertap.property_models.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
)
from watertap.property_models.activated_sludge.modified_asm2d_reactions import (
    ModifiedASM2dReactionParameterBlock,
    ModifiedASM2dReactionBlock,
)
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
)
from pyomo.util.check_units import assert_units_consistent
from idaes.core import UnitModelCostingBlock
from watertap.costing import WaterTAPCosting

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


class TestElectroNP:
    @pytest.fixture(scope="class")
    def ElectroNP_frame(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = ModifiedASM2dParameterBlock(
            additional_solute_list=["S_K", "S_Mg"]
        )

        m.fs.unit = ElectroNPZO(property_package=m.fs.properties)

        EPS = 1e-8

        m.fs.unit.inlet.temperature.fix(298.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(10 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(16 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(3.6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(30 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(20 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(30 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(25 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(125 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(30 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_MeOH"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_MeP"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_TSS"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(EPS * units.mg / units.liter)

        # Alkalinity was givien in mg/L based on C
        m.fs.unit.inlet.alkalinity[0].fix(61 / 12 * units.mmol / units.liter)

        # Unit option
        m.fs.unit.energy_electric_flow_mass.fix(0.044 * units.kWh / units.kg)
        m.fs.unit.magnesium_chloride_dosage.fix(0.388)

        return m

    @pytest.mark.unit
    def test_dof(self, ElectroNP_frame):
        m = ElectroNP_frame
        assert degrees_of_freedom(m) == 0

    @pytest.mark.unit
    def test_units(self, ElectroNP_frame):
        assert_units_consistent(ElectroNP_frame)

    @pytest.mark.unit
    def test_calculate_scaling(self, ElectroNP_frame):
        m = ElectroNP_frame
        calculate_scaling_factors(m)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, ElectroNP_frame):
        initialization_tester(ElectroNP_frame)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, ElectroNP_frame):
        m = ElectroNP_frame
        results = solver.solve(m)

        # Check for optimal solution
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, ElectroNP_frame):
        m = ElectroNP_frame
        assert (
            abs(
                value(
                    m.fs.unit.inlet.flow_vol[0] * m.fs.properties.dens_mass
                    - m.fs.unit.treated.flow_vol[0] * m.fs.properties.dens_mass
                    - m.fs.unit.byproduct.flow_vol[0] * m.fs.properties.dens_mass
                )
            )
            <= 1e-6
        )
        for j in m.fs.properties.solute_set:
            assert 1e-6 >= abs(
                value(
                    m.fs.unit.inlet.flow_vol[0] * m.fs.unit.inlet.conc_mass_comp[0, j]
                    - m.fs.unit.treated.flow_vol[0]
                    * m.fs.unit.treated.conc_mass_comp[0, j]
                    - m.fs.unit.byproduct.flow_vol[0]
                    * m.fs.unit.byproduct.conc_mass_comp[0, j]
                )
            )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, ElectroNP_frame):
        m = ElectroNP_frame
        assert value(m.fs.unit.treated.flow_vol[0]) == pytest.approx(0.213495, rel=1e-3)

        assert value(m.fs.unit.treated.temperature[0]) == pytest.approx(
            298.15, rel=1e-4
        )
        assert value(m.fs.unit.treated.pressure[0]) == pytest.approx(101325, rel=1e-4)
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_A"]) == pytest.approx(
            0.02, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_F"]) == pytest.approx(
            0.03, rel=1e-2
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_I"]) == pytest.approx(
            0.03, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_N2"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_NH4"]) == pytest.approx(
            0.0112, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_NO3"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_O2"]) == pytest.approx(
            0.01, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_PO4"]) == pytest.approx(
            7.2e-5, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_AUT"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_H"]) == pytest.approx(
            0.03, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_I"]) == pytest.approx(
            0.025, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_MeOH"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_MeP"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_PAO"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_PHA"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_PP"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_S"]) == pytest.approx(
            0.125, rel=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "X_TSS"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_Mg"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.treated.conc_mass_comp[0, "S_K"]) == pytest.approx(
            0, abs=1e-4
        )
        assert value(m.fs.unit.byproduct.conc_mass_comp[0, "S_NH4"]) == pytest.approx(
            4.5289e14, rel=1e-4
        )
        assert value(m.fs.unit.byproduct.conc_mass_comp[0, "S_PO4"]) == pytest.approx(
            3.328756e14, rel=1e-4
        )
        assert value(m.fs.unit.treated.alkalinity[0]) == pytest.approx(
            0.005083, rel=1e-4
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_costing(self, ElectroNP_frame):
        m = ElectroNP_frame

        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.costing.cost_process()
        m.fs.costing.add_LCOW(m.fs.unit.properties_treated[0].flow_vol)
        results = solver.solve(m)

        assert_optimal_termination(results)

        # Check solutions
        assert pytest.approx(1295.765, rel=1e-5) == value(
            m.fs.unit.costing.capital_cost
        )
        assert pytest.approx(5.800325e-5, rel=1e-5) == value(m.fs.costing.LCOW)

    @pytest.mark.unit
    def test_report(self, ElectroNP_frame):
        m = ElectroNP_frame
        m.fs.unit.report()