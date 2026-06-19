.. _how_to_create_custom_costing_method:

How to create a custom costing method
=======================================

This guide provides an example of how to create a custom costing method for a new or existing unit model :sup:`1`. 
The custom unit model costing method presented here is adapted for a chemical addition unit in :ref:`how to use WaterTAP costing<how_to_use_watertap_costing>`. 

The code below shows an example of how to build a custom costing method for a new unit model that adds a chemical called "bazchem". 
We will create any variables, parameters, and constraints needed to calculate capital and operating costs for the new unit, and also register the "bazchem" flow type with the costing package to calculate variable operating costs based on the mass flow of bazchem.
This is the general structure of all the :ref:`default costing methods for existing WaterTAP unit models<detailed_unit_model_costing>` that are in the `watertap/costing/unit_models <https://github.com/watertap-org/watertap/tree/main/watertap/costing/unit_models>`_ directory.

Consider you have a flowsheet with a new unit model that does not have a defined costing method:

.. code-block:: python

    m.fs.unit = NewUnitModel(property_package=m.fs.properties)

Prior to adding the unit model costing block, we need to create a custom costing method for this new unit model.
To do this, we need to define the parameter block for the new unit model and the parameter block for the "bazchem" flow type.

First, define the functions that will create the parameter blocks for the unit model and the "bazchem" flow type.

.. code-block:: python

    # Function to create costing parameters for unit model
    def build_unit_model_cost_param_block(blk):

        # In this example, blk = m.fs.costing.unit_model

        blk.unit_capex_slope = Var(
            initialize=1.23e4,
            units=pyunits.USD_2020 / (pyunits.Mgallons / pyunits.day),
            doc="Slope for unit model capital cost equation",
        )

        blk.unit_capex_intercept = Var(
            initialize=5e3,
            units=pyunits.USD_2020,
            doc="Intercept for unit model capital cost equation",
        )

        blk.factor_equip_replacement = Var(
            initialize=0.067,
            units=pyunits.year**-1,
            doc="Fraction of unit model equipment replaced per year",
        )

    # Function to create costing parameters for bazchem flow type 
    # and register bazchem flow type with the costing package
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

Here, we register the "bazchem" flow type with the costing package in the unit model costing method.
This could also be done at the flowsheet level with steps presented in :ref:`how to cost a flow<how_to_cost_a_flow>` using the same costing package methods presented here.

Second, define the costing model for the new unit model that uses those parameters to calculate capital and operating costs.

.. code-block:: python
    
    # Create costing parameter blocks for the new unit model and bazchem (chemical flow type)
    # and register them to be built on the flowsheet costing block via the @register_costing_parameter_block decorator
    @register_costing_parameter_block(
        build_rule=build_unit_model_cost_param_block,
        parameter_block_name="unit_model",
    )
    @register_costing_parameter_block(
        build_rule=build_bazchem_cost_param_block,
        parameter_block_name="bazchem",
    )
    def unit_model_costing(blk):

        # In this example, blk = m.fs.unit_model.costing 

        make_capital_cost_var(blk)
        blk.costing_package.add_cost_factor(blk, "TIC")
        make_fixed_operating_cost_var(blk)

        blk.capital_cost_constraint = Constraint(
            expr=blk.capital_cost
            == blk.cost_factor
            * pyunits.convert(
                blk.costing_package.unit_model.unit_capex_intercept,
                to_units=blk.costing_package.base_currency,
            )
            + pyunits.convert(
                blk.costing_package.unit_model.unit_capex_slope
                * blk.unit_model.properties[0].flow_vol_phase["Liq"],
                to_units=blk.costing_package.base_currency,
            )
        )
        blk.fixed_operating_cost_constraint = Constraint(
            expr=blk.fixed_operating_cost
            == blk.costing_package.unit_model.factor_equip_replacement * blk.capital_cost
        )

        # Cost the flow of bazchem with the cost_flow method.
        blk.costing_package.cost_flow(blk.unit_model.chem_mass_flow, "bazchem")


See :ref:`how to add costing packages<how_to_add_watertap_costing_to_flowsheet>` for how to use this custom costing method when creating the unit model costing block.

To make this the default, set the unit model's ``default_costing_method`` attribute to return the custom costing method.
Setting this attribute would be done in the code that defines the unit model:

.. code-block:: python

    # Import the costing method
    from watertap.costing.unit_models import unit_model_costing

    class NewUnitModel(UnitModel):

        def build(self):
            # Add unit model build code here
            ...
            # Set the default costing method here
            self.default_costing_method = unit_model_costing

        # Or set the default costing method here
        @property
        def default_costing_method(self):
            return unit_model_costing

Additional details of the components of the costing method functions are presented in the section on :ref:`extensions over IDAES Costing Framework <extensions_over_idaes_costing_framework>`.

.. Costing method components
.. --------------------------

.. Custom costing methods generally consist of two functions:

.. 1. A function to build the costing parameter block(s) (``build_unit_model_cost_param_block`` and ``build_bazchem_cost_param_block``). These functions define the parameters needed for the costing method and registers any flow types needed for variable cost calculations. In this example, there is one function to build costing parameters for the unit model and a separate function to build costing parameters for the "bazchem" flow type. Though not strictly necessary, convention is to have separate parameter blocks for unique flow types and unit processes. These two functions are:

..     ``build_unit_model_cost_param_block``:

..     - Create variables for unit model capital cost calculation (``unit_capex_base`` and ``unit_capex_exponent``)
..     - Create a variable for calculating fixed operating cost (``factor_equip_replacement``) as a fraction of the capital cost per year

..     ``build_bazchem_cost_param_block``:

..     - Create a variable for the unit cost of the "bazchem" flow type (named ``unit_cost`` by convention)
..     - Register the "bazchem" flow type with the costing package and assign cost as ``unit_cost``. This will be used to calculate operating costs based on the mass flow of bazchem.


.. 2. A function to build the costing model (``unit_model_costing``), which is decorated with the ``@register_costing_parameter_block`` decorator. This function creates the costing variables and constraints needed to calculate capital and operating costs, and also defines the variable cost calculations using the ``cost_flow`` method of the costing package.

..     - The first argument to the ``@register_costing_parameter_block`` decorator is the function that builds the costing parameter block, and the second argument is the desired name for the parameter block on the flowsheet costing block :sup:`3`. This is the name that will be used to access the parameters for this costing method from the flowsheet costing block. In this example, the parameter block is named "unit_model" and is accessed with ``m.fs.costing.unit_model``.
..     - Within the costing method function, we first create the necessary costing variables (capital cost and fixed operating cost :sup:`2`). Then we define the constraints that calculate capital and fixed operating cost using the parameters defined in the parameter block and any relevant unit model variables :sup:`3`. 
..     - The ``cost_flow`` method is used to aggregate flows of the same type across multiple units (most commonly this is done with chemical and electricty flows).
..     - Costing methods that calculate capital costs *must* specify a capital cost factor via the ``add_cost_factor`` method (options are ``"TIC"``, ``"TPEC"``, or ``None``) to be used to calculate direct and indirect capital costs; see the :ref:`common costing parameters<common_global_costing_parameters>` for more information.


.. note::

    :sup:`1` Users can create a custom costing method for *any* unit model in the same way. The flowsheet costing block will preferentially use the any method passed via the ``costing_method`` argument when creating the ``UnitModelCostingBlock`` before using the costing method specified by the unit model's ``default_costing_method`` attribute.

    :sup:`2` For proper aggregation of capital and operating costs, the flowsheet costing block requires the following naming conventions:

        - The capital cost variable must be named ``capital_cost`` and constraint ``capital_cost_constraint``.
        - The fixed operating cost variable must be named ``fixed_operating_cost`` and constraint ``fixed_operating_cost_constraint``.

        For this reason, the imported utility functions ``make_capital_cost_var`` and ``make_fixed_operating_cost_var`` should be used.
    
    :sup:`3` In the costing model build function, any components located on the unit model that are used in the costing constraints can always be accessed via ``blk.unit_model`` and the costing package can be accessed via ``blk.costing_package`` regardless of their names on the flowsheet.

