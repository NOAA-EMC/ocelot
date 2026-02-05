#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import bufr
import yaml
import faulthandler

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

faulthandler.enable()

CRIS_PCA_YAML = map_path("cris_pca.yaml")

TAI93_EPOCH = np.datetime64("1993-01-01T00:00:00")
UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00")


class CrisPcaObsBuilder(ObsBuilder):
    """
    CrIS PCA netCDF reader.

    Reads a single cris_pca.yaml that defines:
      - observation: dim_path_map, variable_map (netCDF -> encoder names),
        global_pc_score_slice
      - encoder: BUFR schema (categories, variables)

    Flattens (atrack, xtrack, fov) into a 1D location dimension and fills
    a DataContainer with variables named per the encoder schema.
    """

    def __init__(self):
        super().__init__(CRIS_PCA_YAML, log_name=os.path.basename(__file__))

        with open(CRIS_PCA_YAML, "r") as f:
            full_yaml = yaml.safe_load(f)

        self.cris_pca_yaml = full_yaml

        obs = self.cris_pca_yaml.get("observation", {})
        self._dim_path_map = obs.get("dim_path_map", {})
        self._variable_map = obs.get("variable_map", [])
        pc_slice = obs.get("global_pc_score_slice", [1, 25])
        self._pc_score_start, self._pc_score_end = pc_slice[0], pc_slice[1]

        enc = self.cris_pca_yaml.get("encoder", {})
        self._encoder_variables = enc.get("variables", [])

    # -----------------------------------------------------
    def load_input(self, filename):
        self.log.info(f"*** load_input() CALLED: {filename}")
        ds = xr.open_dataset(filename, decode_times=False)
        return ds

    # -----------------------------------------------------
    def preprocess_dataset(self, ds):

        required = ["atrack", "xtrack", "fov"]
        for d in required:
            if d not in ds.sizes:
                raise RuntimeError(f"Missing dimension {d}")

        na = ds.sizes["atrack"]
        nx = ds.sizes["xtrack"]
        nf = ds.sizes["fov"]
        nlocs = na * nx * nf

        # Build indices
        a, x, f = xr.broadcast(
            xr.DataArray(np.arange(na), dims="atrack"),
            xr.DataArray(np.arange(nx), dims="xtrack"),
            xr.DataArray(np.arange(nf), dims="fov"),
        )

        xtrack = x.values.ravel()
        fov = f.values.ravel()

        scan_pos = nf * xtrack + fov

        out = xr.Dataset()
        out = out.assign_coords(location=np.arange(nlocs))

        out["scan_position"] = xr.DataArray(scan_pos, dims=("location",))

        # Flatten variables from netCDF into encoder names (from observation.variable_map)
        for entry in self._variable_map:
            v_in = entry["source"]
            v_out = entry["name"]
            if v_in in ds:
                out[v_out] = xr.DataArray(
                    ds[v_in].values.reshape(nlocs),
                    dims=("location",)
                )
        # Hardcode NOAA-20 sat id for now
        out["satelliteId"] = xr.DataArray(np.ones(nlocs)*225, dims=("location",))

        # Time
        time3d = xr.broadcast(ds["obs_time_tai93"], ds["lat"])[0]
        time_tai93 = time3d.values.reshape(nlocs)
        offset = (TAI93_EPOCH - UNIX_EPOCH) / np.timedelta64(1, "s")
        time_unix = time_tai93 + offset
        out["time"] = xr.DataArray(time_unix, dims=("location",))

        # Global PC scores
        npc = ds.sizes["npc_global"]
        out["global_pc_score"] = xr.DataArray(
                ds["global_pc_score"].values.reshape(nlocs, npc)[
                    :, self._pc_score_start : self._pc_score_end
                ],
            dims=("location", "npc_global")
        )

        return out

    # -----------------------------------------------------
    # 2) Build a DataContainer from the flattened Dataset
    # -----------------------------------------------------

    def _dims_for_var(self, varname, dims):
        """
        Map xarray dimension names (e.g. ('location', 'npc_global'))
        to BUFR query strings using observation.dim_path_map in cris_pca.yaml.
        """
        unknown = [d for d in dims if d not in self._dim_path_map]
        if unknown:
            raise RuntimeError(
                f"_dims_for_var: no mapping for dimension(s) {unknown} "
                f"(variable '{varname}'); known: {list(self._dim_path_map.keys())}"
            )
        return [self._dim_path_map[d] for d in dims]

    def make_obs(self, comm, input_path):
        ds = self.load_input(input_path)
        ds = self.preprocess_dataset(ds)

        container = bufr.DataContainer()

        for v in self._encoder_variables:
            name = v["name"]
            source = v["source"]

            if source not in ds:
                self.log.warning(f"WARNING: source '{source}' not in dataset, skipping")
                continue

            xr_dims = ds[source].dims
            dim_paths = self._dims_for_var(name, xr_dims)

            vals = ds[source].values
            self.log.debug(
                f"name={name} shape={getattr(vals, 'shape', None)} dim_paths={dim_paths}"
            )
            container.add(name, vals, dim_paths)

        return container


add_main_functions(CrisPcaObsBuilder)
