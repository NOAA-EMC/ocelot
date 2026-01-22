import re
from typing import Union
import zarr
import numpy as np

import bufr
from bufr.obs_builder import add_encoder_type

DEFAULT_CHUNK_SIZE = 5_000_000


# Encoder for Zarr format
class Encoder(bufr.encoders.EncoderBase):
    def __init__(self, description: Union[str, bufr.encoders.Description]):
        if isinstance(description, str):
            self.description = bufr.encoders.Description(description)
        else:
            self.description = description

        super(Encoder, self).__init__(self.description)

    def encode(self,
               container: bufr.DataContainer,
               output_template_path: str,
               append: bool = False) -> dict:

        if container is None or container.size() == 0:
            return {}

        result: dict = {}
        for category in container.all_sub_categories():
            cat_idx = 0
            substitutions = {}
            for key in container.get_category_map().keys():
                substitutions[key] = category[cat_idx]
                cat_idx += 1

            output_path = self._make_path(output_template_path, substitutions)

            store = zarr.DirectoryStore(output_path)
            root = zarr.group(store=store, overwrite=(not append))
            dims = self.get_encoder_dimensions(container, category)

            if 'time' not in root:
                self._add_attrs(root)
                self._init_dimensions(root, container, category, dims)
                self._init_variables(root, container, category, dims)
            else:
                self._append_data(root, container, category, dims)

            # Close the zarr file
            root.store.close()

            result[tuple(category)] = root

        return result

    def _add_attrs(self, root: zarr.Group):
        # Adds globals as attributes to the root group
        for key, data in self.description.get_globals().items():
            root.attrs[key] = data

    def _init_dimensions(self,
                         root: zarr.Group,
                         container: bufr.DataContainer,
                         category: list,
                         dims: bufr.encoders.EncoderDimensions):

        # Add the time dimension as the primary dimension
        timestamps = container.get('variables/timestamp', category)
        dim_store = root.create_dataset('time',
                                        shape=[len(timestamps)],
                                        chunks=[DEFAULT_CHUNK_SIZE],
                                        dtype=timestamps.dtype,
                                        compression='blosc',
                                        compression_opts=dict(cname='lz4',
                                                              clevel=3,
                                                              shuffle=1))
        dim_store[:] = timestamps
        dim_store.attrs['_ARRAY_DIMENSIONS'] = ['time']

        # Add the backing variables for the dimensions
        for dim in dims.dims():
            # Skip the location dimension as we will use the timestamp as the location
            if dim.name().lower() == 'location':
                continue

            dim_data = dim.labels
            dim_store = root.create_dataset(dim.name().lower(),
                                            shape=[len(dim_data)],
                                            chunks=[DEFAULT_CHUNK_SIZE],
                                            dtype=np.int32,
                                            compression='blosc',
                                            compression_opts=dict(cname='lz4',
                                                                  clevel=3,
                                                                  shuffle=1))
            dim_store[:] = dim_data
            dim_store.attrs['_ARRAY_DIMENSIONS'] = [dim.name().lower()]

    def _init_variables(self,
                        root: zarr.Group,
                        container: bufr.DataContainer,
                        category: list,
                        dims: bufr.encoders.EncoderDimensions):

        def add_variable(var, var_name, var_data):
            comp_level = var['compressionLevel'] if 'compressionLevel' in var else 3

            # if var_data type is a string type set the object_codec field
            if var_data.dtype.kind in ('U', 'S') or var_data.dtype == np.object_:
                # Create string zarr dataset
                store = root.create_dataset(var_name,
                                            shape=var_data.shape,
                                            # chunks=dims.chunks_for_var(var['name']),
                                            chunks=(DEFAULT_CHUNK_SIZE),
                                            dtype=object,
                                            object_codec=zarr.VLenUTF8(),
                                            compression='blosc',
                                            compression_opts=dict(cname='lz4',
                                                                  clevel=comp_level,
                                                                  shuffle=1))

            else:
                # Create the zarr dataset
                store = root.create_dataset(var_name,
                                            shape=var_data.shape,
                                            # chunks=dims.chunks_for_var(var['name']),
                                            chunks=(DEFAULT_CHUNK_SIZE),
                                            dtype=var_data.dtype,
                                            compression='blosc',
                                            compression_opts=dict(cname='lz4',
                                                                  clevel=comp_level,
                                                                  shuffle=1))

            store[:] = var_data

            # Add the attributes
            store.attrs['units'] = var['units']
            store.attrs['long_name'] = var['longName']

            if 'coordinates' in var:
                store.attrs['coordinates'] = var['coordinates']

            if 'range' in var:
                store.attrs['valid_range'] = var['range']

            store.attrs['_ARRAY_DIMENSIONS'] = ['time']

        for var in self.description.get_variables():
            # Associate the dimensions
            dim_names = dims.dim_names_for_var(var["name"])
            dim_names = [dim_name.lower() for dim_name in dim_names]
            dim_names[0] = 'time'

            _, var_name = self._split_source_str(var['name'])

            if var_name.lower() == 'datetime' or var_name.lower() == 'time':
                root.time.attrs['units'] = var['units']
                root.time.attrs['long_name'] = var['longName']
                continue  # Skip the time variable as it is a dimension

            if var["source"].split('/')[-1] not in container.list():
                raise ValueError(f'Variable {var["source"]} not found in the container')

            var_data = container.get(var['source'].split('/')[-1], category)

            if len(var_data.shape) == 1:
                add_variable(var, var_name, var_data)
            elif len(var_data.shape) == 2:
                for i in range(var_data.shape[1]):
                    dim_vals = root[dim_names[1]]
                    dim_val = dim_vals[i] if dim_vals[i] != 0 else i + 1
                    add_variable(var, f'{var_name}_{dim_names[1]}_{dim_val}', var_data[:, i])
            else:
                raise ValueError(f'Variable {var_name} has an invalid shape {var_data.shape}')

    def _append_data(self,
                     root: zarr.Group,
                     container: bufr.DataContainer,
                     category: list,
                     dims: bufr.encoders.EncoderDimensions) -> None:

        for var in self.description.get_variables():
            _, var_name = self._split_source_str(var['name'])

            if var_name == 'dateTime':
                var_name = 'time'

            dim_names = dims.dim_names_for_var(var["name"])
            dim_names = [dim_name.lower() for dim_name in dim_names]

            var_data = container.get(var["source"].split('/')[-1], category)

            if len(var_data.shape) == 1:
                root[var_name].append(var_data)
            elif len(var_data.shape) == 2:
                for i in range(var_data.shape[1]):
                    dim_vals = root[dim_names[1]]
                    dim_val = dim_vals[i] if dim_vals[i] != 0 else i + 1
                    root[f'{var_name}_{dim_names[1]}_{dim_val}'].append(var_data[:, i])
            else:
                raise ValueError(f'Variable {var_name} has an invalid shape {var_data.shape}')

    def _make_path(self, prototype_path: str, sub_dict: dict) -> str:
        subs = re.findall(r'\{(?P<sub>\w+\/\w+)\}', prototype_path)
        for sub in subs:
            prototype_path = prototype_path.replace(f'{{{sub}}}', sub_dict[sub])

        return prototype_path

    def _split_source_str(self, source: str) -> (str, str):
        components = source.split('/')
        group_name = components[0] if len(components) > 1 else ""
        variable_name = components[-1]

        return (group_name, variable_name)


add_encoder_type('zarr-ocelot', Encoder)
