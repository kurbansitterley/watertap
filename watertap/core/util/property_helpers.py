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

from pyomo.environ import Constraint, Var, Reals, units as pyunits

__author__ = "Adam Atia"


def get_property_metadata(prop_pkg):
    """Return metadata for every property a package marks as supported.

    Args:
        prop_pkg: a property model ParameterBlock (e.g. m.fs.properties)

    Returns:
        list of dicts with keys "Description", "Name", "Units", sorted alphabetically by Description
    """
    pset = prop_pkg.get_metadata().properties
    rows = []
    for prop in pset:
        # TODO: switch prop._doc to prop.doc once doc property added in IDAES
        doc = getattr(prop, "doc", None) or prop._doc
        # indices can include None, phase_comp, etc.
        for idx in prop._indices:
            sub = prop[idx]
            if sub.supported:
                rows.append(
                    {"Description": doc, "Name": sub.name, "Units": str(sub.units)}
                )
        sorted_rows = sorted(rows, key=lambda r: r["Description"].lower())

    return sorted_rows


def print_property_metadata(prop_pkg):
    """Pretty-print supported properties as a fixed-width table."""
    rows = get_property_metadata(prop_pkg)
    if not rows:
        print("No supported properties found.")
        return
    cols = ["Description", "Name", "Units"]
    widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(r[c].ljust(widths[c]) for c in cols))


def add_dens_mass_params(blk):

    # mass density parameters, 0-180 C, 0-150 g/kg, 0-12 MPa
    # eq. 8 in Sharqawy et al. (2010)
    dens_units = pyunits.kg / pyunits.m**3
    t_inv_units = pyunits.K**-1
    s_inv_units = pyunits.kg / pyunits.g

    blk.dens_mass_param_A1 = Var(
        within=Reals,
        initialize=9.999e2,
        units=dens_units,
        doc="Mass density parameter A1",
    )
    blk.dens_mass_param_A2 = Var(
        within=Reals,
        initialize=2.034e-2,
        units=dens_units * t_inv_units,
        doc="Mass density parameter A2",
    )
    blk.dens_mass_param_A3 = Var(
        within=Reals,
        initialize=-6.162e-3,
        units=dens_units * t_inv_units**2,
        doc="Mass density parameter A3",
    )
    blk.dens_mass_param_A4 = Var(
        within=Reals,
        initialize=2.261e-5,
        units=dens_units * t_inv_units**3,
        doc="Mass density parameter A4",
    )
    blk.dens_mass_param_A5 = Var(
        within=Reals,
        initialize=-4.657e-8,
        units=dens_units * t_inv_units**4,
        doc="Mass density parameter A5",
    )
    blk.dens_mass_param_B1 = Var(
        within=Reals,
        initialize=8.020e2,
        units=dens_units,
        doc="Mass density parameter B1",
    )
    blk.dens_mass_param_B2 = Var(
        within=Reals,
        initialize=-2.001,
        units=dens_units * t_inv_units,
        doc="Mass density parameter B2",
    )
    blk.dens_mass_param_B3 = Var(
        within=Reals,
        initialize=1.677e-2,
        units=dens_units * t_inv_units**2,
        doc="Mass density parameter B3",
    )
    blk.dens_mass_param_B4 = Var(
        within=Reals,
        initialize=-3.060e-5,
        units=dens_units * t_inv_units**3,
        doc="Mass density parameter B4",
    )
    blk.dens_mass_param_B5 = Var(
        within=Reals,
        initialize=-1.613e-5,
        units=dens_units * t_inv_units**2,
        doc="Mass density parameter B5",
    )


def add_dens_mass_phase_method(blk):

    if blk.params.find_component("ion_set") is not None:
        # blk is MCASStateBlockData, so mass fraction of TDS is sum of all solutes
        temperature = blk.temperature
        dens_mass_solvent = blk.dens_mass_solvent
        mass_frac = sum(
            blk.mass_frac_phase_comp["Liq", j] for j in blk.params.solute_set
        )
    
    elif blk.params.find_component("liq_comp_set") is not None:
        # blk is AirWaterEqStateBlockData, "TDS" is required component
        mass_frac = blk.mass_frac_phase_comp["Liq", "TDS"]
        temperature = blk.temperature["Liq"]
        dens_mass_solvent = blk.dens_mass_solvent["H2O"]

    else:
        # blk is SeawaterStateBlockData, "TDS" is only component
        temperature = blk.temperature
        dens_mass_solvent = blk.dens_mass_solvent
        mass_frac = blk.mass_frac_phase_comp["Liq", "TDS"]

    blk.dens_mass_phase = Var(
        blk.params.phase_list,
        initialize=1e3,
        bounds=(1, 1e6),
        units=pyunits.kg * pyunits.m**-3,
        doc="Mass density",
    )

    # Sharqawy et al. (2010), eq. 8, 0-180 C, 0-150 g/kg, 0-12 MPa
    def rule_dens_mass_phase(b, p):
        t = temperature - 273.15 * pyunits.K
        s = mass_frac
        dens_mass = (
            dens_mass_solvent
            + b.params.dens_mass_param_B1 * s
            + b.params.dens_mass_param_B2 * s * t
            + b.params.dens_mass_param_B3 * s * t**2
            + b.params.dens_mass_param_B4 * s * t**3
            + b.params.dens_mass_param_B5 * s**2 * t**2
        )
        return b.dens_mass_phase[p] == dens_mass

    blk.eq_dens_mass_phase = Constraint(
        ["Liq"], rule=rule_dens_mass_phase
    )

def add_dens_mass_solvent_method(blk):
    
    blk.dens_mass_solvent = Var(
        initialize=1e3,
        bounds=(1, 1e6),
        units=pyunits.kg * pyunits.m**-3,
        doc="Mass density of pure water",
    )
    if blk.params.find_component("ion_set") is not None:
        # blk is MCASStateBlockData, so mass fraction of TDS is sum of all solutes
        temperature = blk.temperature
        dens_mass_solvent = blk.dens_mass_solvent

    elif blk.params.find_component("liq_comp_set") is not None:
        # blk is AirWaterEqStateBlockData, "TDS" is required component
        temperature = blk.temperature["Liq"]
        dens_mass_solvent = blk.dens_mass_solvent["H2O"]

    else:
        # blk is SeawaterStateBlockData, "TDS" is only component
        temperature = blk.temperature
        dens_mass_solvent = blk.dens_mass_solvent

    # Sharqawy et al. (2010), eq. 7, 0-180 C, 0-150 g/kg, 0-12 MPa
    def rule_dens_mass_solvent(b):
        t = temperature - 273.15 * pyunits.K
        return b.dens_mass_solvent == (
            b.params.dens_mass_param_A1
            + b.params.dens_mass_param_A2 * t
            + b.params.dens_mass_param_A3 * t**2
            + b.params.dens_mass_param_A4 * t**3
            + b.params.dens_mass_param_A5 * t**4
        )

    blk.eq_dens_mass_solvent = Constraint(rule=rule_dens_mass_solvent)


def add_visc_d_params(blk):

    visc_d_units = pyunits.Pa * pyunits.s
    t_inv_units = pyunits.K**-1
    # dynamic viscosity parameters, 0-180 C, 0-150 g/kg
    # eq. 22 and 23 in Sharqawy et al. (2010)
    blk.visc_d_param_muw_A = Var(
        within=Reals,
        initialize=4.2844e-5,
        units=visc_d_units,
        doc="Dynamic viscosity parameter A for pure water",
    )
    blk.visc_d_param_muw_B = Var(
        within=Reals,
        initialize=0.157,
        units=t_inv_units**2 * visc_d_units**-1,
        doc="Dynamic viscosity parameter B for pure water",
    )
    blk.visc_d_param_muw_C = Var(
        within=Reals,
        initialize=64.993,
        units=pyunits.K,
        doc="Dynamic viscosity parameter C for pure water",
    )
    blk.visc_d_param_muw_D = Var(
        within=Reals,
        initialize=91.296,
        units=visc_d_units**-1,
        doc="Dynamic viscosity parameter D for pure water",
    )
    blk.visc_d_param_A_1 = Var(
        within=Reals,
        initialize=1.541,
        units=pyunits.dimensionless,
        doc="Dynamic viscosity parameter 1 for term A",
    )
    blk.visc_d_param_A_2 = Var(
        within=Reals,
        initialize=1.998e-2,
        units=t_inv_units,
        doc="Dynamic viscosity parameter 2 for term A",
    )
    blk.visc_d_param_A_3 = Var(
        within=Reals,
        initialize=-9.52e-5,
        units=t_inv_units**2,
        doc="Dynamic viscosity parameter 3 for term A",
    )
    blk.visc_d_param_B_1 = Var(
        within=Reals,
        initialize=7.974,
        units=pyunits.dimensionless,
        doc="Dynamic viscosity parameter 1 for term B",
    )
    blk.visc_d_param_B_2 = Var(
        within=Reals,
        initialize=-7.561e-2,
        units=t_inv_units,
        doc="Dynamic viscosity parameter 2 for term B",
    )
    blk.visc_d_param_B_3 = Var(
        within=Reals,
        initialize=4.724e-4,
        units=t_inv_units**2,
        doc="Dynamic viscosity parameter 3 for term B",
    )


def add_visc_d_method(blk):

    if blk.params.find_component("ion_set") is not None:
        # blk is MCASStateBlockData, so mass fraction of TDS is sum of all solutes
        mass_frac = sum(
            blk.mass_frac_phase_comp["Liq", j] for j in blk.params.solute_set
        )
    else:
        mass_frac = blk.mass_frac_phase_comp["Liq", "TDS"]

    blk.visc_d_phase = Var(
        blk.params.phase_list,
        initialize=1e-3,
        bounds=(0.0, 1),
        units=pyunits.Pa * pyunits.s,
        doc="Viscosity",
    )

    # Sharqawy et al. (2010), eq. 22 and 23, 0-180 C, 0-150 g/kg
    def rule_visc_d_phase(b, p):
        # temperature in degC, but pyunits are K
        t = b.temperature - 273.15 * pyunits.K
        s = mass_frac
        mu_w = (
            b.params.visc_d_param_muw_A
            + (
                b.params.visc_d_param_muw_B * (t + b.params.visc_d_param_muw_C) ** 2
                - b.params.visc_d_param_muw_D
            )
            ** -1
        )
        A = (
            b.params.visc_d_param_A_1
            + b.params.visc_d_param_A_2 * t
            + b.params.visc_d_param_A_3 * t**2
        )
        B = (
            b.params.visc_d_param_B_1
            + b.params.visc_d_param_B_2 * t
            + b.params.visc_d_param_B_3 * t**2
        )
        return b.visc_d_phase[p] == mu_w * (1 + A * s + B * s**2)

    blk.eq_visc_d_phase = Constraint(blk.params.phase_list, rule=rule_visc_d_phase)
