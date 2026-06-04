.. _how_to_create_custom_costing_method:

How to access costing results
===============================

This guide provides instructions on how to access costing results from a flowsheet with WaterTAP costing.

How to
*******

After building and solving a flowsheet, there are results associated with the flowsheet costing block and each individual unit model costing block.
In this example, we assume the flowsheet is structured in the following way:

.. code-block:: python

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.costing = WaterTAPCosting()
    m.fs.unit = UnitModel()
    m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

We call ``m.fs.costing`` the "flowsheet costing block" and ``m.fs.unit.costing`` the "unit model costing block".

Accessing System-Level Costing Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These results are found on the flowsheet costing block and for this example includes the following:

- Total capital cost: ``m.fs.costing.total_capital_cost``
- Total operating cost: ``m.fs.costing.total_operating_cost``
- Total electricity required: ``m.fs.costing.aggregate_flow_electricity``
- Total cost of electricity: ``m.fs.costing.aggregate_flow_costs["electricity"]``
- LCOW: ``m.fs.costing.LCOW``
- SEC: ``m.fs.costing.SEC``


.. testcode::

    total_capital_cost = value(m.fs.costing.total_capital_cost)
    total_operating_cost = value(m.fs.costing.total_operating_cost)
    LCOW = m.fs.costing.LCOW()
    SEC = m.fs.costing.SEC()