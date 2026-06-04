.. _how_to_create_custom_costing_method:

How to create a custom costing method
=======================================

Overview 
*********

This guide provides an example of how to create a custom costing method for a new or existing unit model. 
This how-to guide uses a generic chemical addition unit model as an example, but the same approach can be used to create custom costing methods for any unit model.
The custom unit model costing method presented here is also used in :ref:`how to use WaterTAP costing<how_to_use_watertap_costing>`. 

How to
******


The code below shows an example of how to build a custom costing method for a new chemical addition unit model :sup:`1` that adds the chemical called "bazchem". 
This method will create the parameters and constraints needed to calculate capital and operating costs for the chemical addition unit, and also register the "bazchem" flow type with the costing package to calculate variable operating costs based on the mass flow of bazchem :sup:`2`.
This is the general structure of all :ref:`costing methods for existing WaterTAP unit models<detailed_unit_model_costing>` that are in the *watertap/costing/unit_models* directory.

Consider you have a flowsheet with a new chemical addition unit model without a defined costing method:

.. code-block:: python
    
    # Create model, flowsheet, and add new chemical addition unit model without a costing method
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.unit = NewChemAddition()

Prior to adding the unit model costing block, we will create a custom costing method for the chemical addition unit model.
We need to define the parameter block for the new chemical addition unit model and the parameter block for the "bazchem" flow type.

Here we create the function that will create the parameter block for the unit model:

.. testcode::

    # Function to create costing parameters for chemical addition unit model
    def build_chem_addition_cost_param_block(blk):

        # In this example, blk = m.fs.costing.chem_addition

        blk.chemical_capex_slope = Var(
            initialize=1.23e4,
            units=pyunits.USD_2020 / (pyunits.Mgallons / pyunits.day),
            doc="Base capital cost for chemical addition",
        )

        blk.chemical_capex_intercept = Var(
            initialize=5e3,
            units=pyunits.USD_2020,
            doc="Exponent for chemical addition capital cost scaling",
        )

        blk.factor_equip_replacement = Var(
            initialize=0.067,
            units=pyunits.year**-1,
            doc="Fraction of chemical addition equipment replaced per year",
        )

Then we create the function that will create the parameter block for the "bazchem" flow type and register the flow type with the costing package:

.. testcode::

    # Function to create costing parameters for bazchem flow type and register bazchem flow type with the costing package
    def build_bazchem_cost_param_block(blk):

        # In this example, blk = m.fs.costing.bazchem

        blk.unit_cost = Var(
            initialize=0.089,
            units=pyunits.USD_2023 / pyunits.kg,
            doc="Unit cost of bazchem",
        )

        costing_pkg = blk.parent_block()
        # Register the bazchem flow type with the costing package
        costing_pkg.register_flow_type("bazchem", blk.unit_cost)

Then create the costing model for the chemical addition unit that uses those parameters to calculate capital and operating costs.

.. testcode::
    
    # Create costing parameter blocks for chemical addition (unit model) and bazchem (chemical flow type)
    # and register them to be built on the flowsheet costing block via the @register_costing_parameter_block decorator
    @register_costing_parameter_block(
        build_rule=build_chem_addition_cost_param_block,
        parameter_block_name="chem_addition",
    )
    @register_costing_parameter_block(
        build_rule=build_bazchem_cost_param_block,
        parameter_block_name="bazchem",
    )
    def chem_addition_costing(blk):

        # In this example, blk = m.fs.chem_addition.costing 

        make_capital_cost_var(blk)
        blk.costing_package.add_cost_factor(blk, "TIC")
        make_fixed_operating_cost_var(blk)

        blk.capital_cost_constraint = Constraint(
            expr=blk.capital_cost
            == blk.cost_factor
            * pyunits.convert(
                blk.costing_package.chem_addition.chemical_capex_intercept,
                to_units=blk.costing_package.base_currency,
            )
            + pyunits.convert(
                blk.costing_package.chem_addition.chemical_capex_slope
                * blk.unit_model.properties[0].flow_vol_phase["Liq"],
                to_units=blk.costing_package.base_currency,
            )
        )
        blk.fixed_operating_cost_constraint = Constraint(
            expr=blk.fixed_operating_cost
            == blk.costing_package.chem_addition.factor_equip_replacement * blk.capital_cost
        )

        # Cost the flow of bazchem for chemical addition with the cost_flow method. 
        # Note that the flow passed to cost_flow must be convertable to units of cost/time.
        blk.costing_package.cost_flow(blk.unit_model.chem_mass_flow, "bazchem")

After creating the custom costing method, it is used in a flowsheet by passing the function name to the ``costing_method`` argument when creating the costing block on the flowsheet. 
For example, if the custom costing method function is named ``chem_addition_costing``, then we would create the costing block on the flowsheet with the line of code below.

.. code-block:: python

    m.fs.costing = WaterTAPCosting()
    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing, costing_method=chem_addition_costing
    )


Explanation
*************


Custom costing methods generally consist of two functions:

1. A function to build the costing parameter block(s) (``build_chem_addition_cost_param_block`` and ``build_bazchem_cost_param_block``). These functions define the parameters needed for the costing method and registers any flow types needed for variable cost calculations. In this example, there is one function to build costing parameters for the chemical addition unit and a separate function to build costing parameters for the "bazchem" flow type. Though not strictly necessary, convention is to have separate parameter blocks for unique flow types and unit processes. These two functions will:

    ``build_chem_addition_cost_param_block``:

    - Create variables for chemical addition capital cost calculation (``chemical_capex_base`` and ``chemical_capex_exponent``)
    - Create a variable for calculating fixed operating cost (``factor_equip_replacement``) as fraction of the capital cost per year

    ``build_bazchem_cost_param_block``:

    - Create a variable for the unit cost of the "bazchem" flow type (named ``unit_cost`` by convention)
    - Register the "bazchem" flow type with the costing package and assign cost as ``unit_cost``. This will be used to calculate operating costs based on the mass flow of bazchem.


2. A function to build the costing model (``chem_addition_costing``), which is decorated with the ``@register_costing_parameter_block`` decorator. This function creates the costing variables and constraints needed to calculate capital and operating costs, and also defines the variable cost calculations using the ``cost_flow`` method of the costing package :sup:`2`.

    - The first argument to the ``@register_costing_parameter_block`` decorator is the function that builds the costing parameter block, and the second argument is the desired name for the parameter block on the flowsheet costing block :sup:`3`. This is the name that will be used to access the parameters for this costing method from the flowsheet costing block. In this example for the chemical addition unit, the parameter block is named "chem_addition" and is accessed with ``m.fs.costing.chem_addition``.
    - Within the costing method function, we first create the necessary costing variables (capital cost and fixed operating cost :sup:`4`). Then we define the constraints that calculate capital and fixed operating cost using the parameters defined in the parameter block and any relevant unit model variables :sup:`5`. 
    - The ``cost_flow`` method is used to aggregate flows of the same type across multiple units (most commonly this is done with chemical and electricty flows).
    - Costing methods that calculate capital costs must provide a capital cost factor to be used to calculate direct and indirect capital costs.



.. important::

    :sup:`1` Users can create a custom costing method for *any* new or existing unit model in the same way. The flowsheet costing block will preferentially use the any method passed via the ``costing_method`` argument when creating the ``UnitModelCostingBlock`` before using the costing method defined by the unit model's ``default_costing_method`` attribute.

    :sup:`2` The flow cost set via the ``register_flow_type`` method multiplied by the flow passed to the ``cost_flow`` method *must* be convertable to cost / time units because it is considered a variable operating cost for that flow type.

    :sup:`3` Any Pyomo component can be added to the parameter blocks. But note that as part of the building process via the ``@register_costing_parameter_block`` decorator, any ``Var`` found will be fixed to their initialized value, thus it is not necessary to fix them at the flowsheet level.

    :sup:`4` For proper aggregation of capital and operating costs, the flowsheet costing block requires the following naming conventions:

        - The capital cost variable must be named ``capital_cost`` and constraint ``capital_cost_constraint``.
        - The fixed operating cost variable must be named ``fixed_operating_cost`` and constraint ``fixed_operating_cost_constraint``.

        For this reason, the imported utility functions ``make_capital_cost_var`` and ``make_fixed_operating_cost_var`` should be used.
    
    :sup:`5` In the costing model build function, any components located on the unit model that are used in the costing constraints can always be accessed via ``blk.unit_model`` and the costing package can be accessed via ``blk.costing_package`` regardless of their names on the flowsheet.

