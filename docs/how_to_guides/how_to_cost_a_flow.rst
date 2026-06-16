.. _how_to_cost_a_flow:

How to cost a flow in WaterTAP costing
=======================================

In the WaterTAP costing package, variable operational costs for the system are calculated by collecting all the material and energy flows from each unit model on the flowsheet. 
These commonly include flows of power (electricity) and various chemicals.
If you are using a unit model with a default costing method, the necessary flows are already costed in that unit model costing method.
However, if you are developing a new unit model or using a unit model without a default costing method that has a material or energy flow, you will need to add that to the flowsheet.