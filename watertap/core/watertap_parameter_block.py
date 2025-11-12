import pandas as pd

from idaes.core.base.property_meta import PropertyClassMetadata
from idaes.core import PhysicalParameterBlock, declare_process_block_class


@declare_process_block_class("WaterTAPParameterBlock")
class WaterTAPParameterBlockData(PhysicalParameterBlock):
    """
    This is a WaterTAP-specific ParameterBlock class that extends the IDAES
    PhysicalParameterBlock with WaterTAP-specific functionality.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()

    def get_metadata(self):
        """Get property parameter metadata.

        If the metadata is not defined, this will instantiate a new
        metadata object and call `define_metadata()` to set it up.

        If the metadata is already defined, it will be simply returned.

        Returns:
            PropertyClassMetadata: The metadata
        """
        if self._metadata is None:
            pcm = PropertyClassMetadata()
            self.define_metadata(pcm)
            self._metadata = pcm

            # Check that the metadata was actually populated
            # Check requires looking at private attributes
            # pylint: disable-next=protected-access
            if pcm._properties is None or pcm._default_units is None:
                raise ValueError(
                    "Property package did not populate all expected metadata."
                )
        return self._metadata

    def print_property_metadata(self, return_df=False):
        """
        Print all supported properties from a WaterTAP/IDAES property package.

        Args:
            prop_pkg: The property model ParameterBlock (e.g., m.fs.properties)
            return_df: If True, returns a Pandas DataFrame instead of printing.
        """
        metadata = self.get_metadata()
        vars, units, docs = [], [], []

        for v in metadata.properties:
            vars.append(v._name)
            units.append(str(v._units))
            docs.append(v._doc)

        if return_df:
            return pd.DataFrame(
                {"Property Description": docs, "Model Attribute": vars, "Units": units}
            )

        # Pretty-print
        name_col = "Model Attribute"
        desc_col = "Property Description"
        units_col = "Units"

        name_w = max(len(name_col), max(len(n) for n in vars)) + 2
        desc_w = max(len(desc_col), max(len(d) for d in docs)) + 2
        units_w = max(len(units_col), max(len(u) for u in units)) + 2

        print(f"{desc_col:<{desc_w}}{name_col:<{name_w}}{units_col:<{units_w}}")
        print("-" * (name_w + desc_w + units_w))

        for n, d, u in zip(vars, docs, units):
            print(f"{d:<{desc_w}}{n:<{name_w}}{u:<{units_w}}")
