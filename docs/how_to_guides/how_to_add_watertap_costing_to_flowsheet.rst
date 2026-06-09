.. _how_to_add_watertap_costing_to_flowsheet:

How to add WaterTAP costing to a flowsheet
===========================================

This guide provides a step-by-step example of how to add WaterTAP costing to a flowsheet for any unit model. 
The steps presented here is also used in :ref:`how to use WaterTAP costing<how_to_use_watertap_costing>`. 

How to
*******

Adding the WaterTAP costing package to a flowsheet can be done at any point in the flowsheet build prior to adding the unit model costing blocks and`` consists of two steps:
1. Create a costing block on the flowsheet.
2. Add costing blocks to unit models and specify the costing method to use for each unit model and specify the costing method to use (if not using the default WaterTAP costing method)

Consider the following flowsheet with a single unit model, ``unit``, that we want to add costing to.

.. code-block:: python

    from pyomo.environ import ConcreteModel
    from idaes.core import FlowsheetBlock, UnitModelCostingBlock
    from watertap.costing import WaterTAPCosting

    # Note: this is not a real unit model, just a placeholder for the purpose of this example
    from watertap.unit_models import UnitModel

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.unit = UnitModel()

To add costing to this flowsheet, we add a costing block to the flowsheet.
This is done by simply by creating an instance of the costing model and assigning it to a flowsheet attribute. 
Convention is to name this attribute ``m.fs.costing``. 
This is referred to as the "flowsheet costing block" (contrasted with a "unit model costing block"). 

.. code-block:: python

    m.fs.costing = WaterTAPCosting()


At this point, the flowsheet costing block only contains instructions to aggregate the costs from the individual unit model costing blocks into overall flowsheet-level costs.
To get costing results, we need to add the unit model costing block.
This is done by adding a ``UnitModelCostingBlock`` to the unit model. 
If we want to use the default WaterTAP costing method, we can simply add the costing block without specifying a costing method. 

.. code-block:: python

    m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

If we want to use a custom costing method, we can specify the function name of the custom costing method in the ``costing_method`` argument when creating the costing block.

.. code-block:: python

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing, costing_method=custom_costing_method
    )

In either case, if the costing method can accept keyword arguments, we can pass those as a dictionary when creating the costing block via the ``costing_method_arguments`` argument.

.. code-block:: python

    costing_method_arguments = {
        "arg1": value_1,
        "arg2": value_2
    }

    # If using a default costing method that accepts keyword arguments
    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments=costing_method_arguments,
    )

    # If using a custom costing method that accepts keyword arguments
    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=custom_costing_method,
        costing_method_arguments=costing_method_arguments,
    )

The same steps can be followed if using the :ref:`zero order costing model<zero_order_costing>`.