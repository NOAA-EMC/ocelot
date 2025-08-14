import os
import re
from typing import Dict, Union

import pyarrow as pa
import pyarrow.parquet as pq

import bufr
from bufr.obs_builder import add_encoder_type


# Encoder for Apache Parquet format
class Encoder(bufr.encoders.EncoderBase):
    def __init__(self, description: Union[str, bufr.encoders.Description]):
        if isinstance(description, str):
            self.description = bufr.encoders.Description(description)
        else:
            self.description = description
        super().__init__(self.description)

    def encode(
        self,
        container: bufr.DataContainer,
        output_template_path: str,
        append: bool = False,
    ) -> Dict:
        """Encode the DataContainer into parquet files.

        Parameters
        ----------
        container: bufr.DataContainer
            Container with all observation data.
        output_template_path: str
            Path template where the parquet file will be written.  Placeholders
            of the form ``{category/key}`` will be substituted using category
            information from the container.
        append: bool, optional
            If ``True`` the encoded data will be appended to an existing parquet
            file, otherwise a new file will be created or overwrite an existing
            one.
        """
        result: Dict = {}

        for category in container.all_sub_categories():
            substitutions = {}
            for idx, key in enumerate(container.get_category_map().keys()):
                substitutions[key] = category[idx]

            output_path = self._make_path(output_template_path, substitutions)
            dims = self.get_encoder_dimensions(container, category)

            table = self._build_table(container, category, dims)

            if append and os.path.exists(output_path):
                existing = pq.read_table(output_path)
                table = pa.concat_tables([existing, table])

            pq.write_table(table, output_path)
            result[tuple(category)] = table

        return result

    def _build_table(
        self,
        container: bufr.DataContainer,
        category: list,
        dims: bufr.encoders.EncoderDimensions,
    ) -> pa.Table:
        """Create a :class:`pyarrow.Table` for the given category."""
        data_dict = {}
        fields = []

        # File level metadata
        file_meta = {
            k.encode(): str(v).encode() for k, v in self.description.get_globals().items()
        }

        # Primary time dimension
        timestamps = container.get("variables/timestamp", category)
        data_dict["time"] = pa.array(timestamps)
        fields.append(pa.field("time", data_dict["time"].type))

        dim_label_map = {d.name().lower(): d.labels for d in dims.dims()}

        for var in self.description.get_variables():
            dim_names = [n.lower() for n in dims.dim_names_for_var(var["name"])]
            if not dim_names:
                dim_names = ["time"]
            else:
                dim_names[0] = "time"

            _, var_name = self._split_source_str(var["name"])

            if var_name.lower() in {"datetime", "time"}:
                continue

            source_key = var["source"].split("/")[-1]
            if source_key not in container.list():
                raise ValueError(f"Variable {var['source']} not found in the container")

            var_data = container.get(source_key, category)

            meta = self._field_metadata(var)

            if len(var_data.shape) == 1:
                array = pa.array(var_data)
                data_dict[var_name] = array
                fields.append(pa.field(var_name, array.type, metadata=meta))
            elif len(var_data.shape) == 2:
                labels = dim_label_map[dim_names[1]]
                for i in range(var_data.shape[1]):
                    col_name = f"{var_name}_{dim_names[1]}_{labels[i]}"
                    array = pa.array(var_data[:, i])
                    data_dict[col_name] = array
                    fields.append(pa.field(col_name, array.type, metadata=meta))
            else:
                raise ValueError(
                    f"Variable {var_name} has an invalid shape {var_data.shape}"
                )

        schema = pa.schema(fields, metadata=file_meta)
        return pa.Table.from_pydict(data_dict, schema=schema)

    def _field_metadata(self, var: dict) -> Dict[bytes, bytes]:
        meta: Dict[bytes, bytes] = {
            b"units": str(var["units"]).encode(),
            b"long_name": str(var["longName"]).encode(),
        }
        if "coordinates" in var:
            meta[b"coordinates"] = str(var["coordinates"]).encode()
        if "range" in var:
            meta[b"valid_range"] = str(var["range"]).encode()
        return meta

    def _make_path(self, prototype_path: str, sub_dict: dict) -> str:
        subs = re.findall(r"\{(?P<sub>\w+\/\w+)\}", prototype_path)
        for sub in subs:
            prototype_path = prototype_path.replace(f"{{{sub}}}", sub_dict[sub])
        return prototype_path

    def _split_source_str(self, source: str) -> (str, str):
        components = source.split("/")
        group_name = components[0] if len(components) > 1 else ""
        variable_name = components[-1]
        return group_name, variable_name


add_encoder_type("parquet-ocelot", Encoder)
