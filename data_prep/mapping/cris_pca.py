#!/usr/bin/env python3
import os
import numpy as np
import xarray as xr
import bufr
import yaml
import faulthandler

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

faulthandler.enable()
# Encoder YAML (BUFR schema) – separate from any mapping YAML
ENCODER_YAML = map_path("cris_pca.yaml")


class CrisPcaObsBuilder(ObsBuilder):
    """
    CrIS PCA netCDF reader:

      * DOES NOT use an ObsBuilder mapping YAML
      * DOES use a BUFR encoder YAML (cris_pca.yaml)
      * Flattens atrack/xtrack/fov -> location
      * Fills a DataContainer matching encoder variable names
    """

    def __init__(self):
        print("\n*** CrisPcaObsBuilder CONSTRUCTOR ***")
        print("    ENCODER_YAML =", ENCODER_YAML)

        # --- Load YAML FIRST (before calling super) ---
        with open(ENCODER_YAML, "r") as f:
            full_yaml = yaml.safe_load(f)

        self._encoder_yaml = full_yaml

        # Build dimension map
        # enc = full_yaml.get("encoder", {})
        dim_path_map = {}
        for dim in full_yaml.get("dimensions", []):
            n = dim["name"]
            p = dim["path"]
            dim_path_map[n] = p

        self._dim_path_map = dim_path_map

        print("    DIM PATH MAP:", self._dim_path_map)

        # NOW call parent (which calls _make_description)
        super().__init__(None, log_name=os.path.basename(__file__))

    # -----------------------------------------------------
    # 1) Return a Description using the encoder YAML file
    # -----------------------------------------------------
    def _make_description(self):
        print("*** _make_description(): using ENCODER_YAML ***")
        return bufr.encoders.Description(ENCODER_YAML)

    # -----------------------------------------------------
    def load_input(self, filename):
        print(f"*** load_input() CALLED: {filename}")
        ds = xr.open_dataset(filename, decode_times=False)
        print("    dims:", ds.sizes)
        return ds

    # -----------------------------------------------------
    def preprocess_dataset(self, ds):
        print("*** preprocess_dataset() CALLED ***")

        required = ["atrack", "xtrack", "fov"]
        for d in required:
            if d not in ds.sizes:
                raise RuntimeError(f"Missing dimension {d}")

        na = ds.sizes["atrack"]
        nx = ds.sizes["xtrack"]
        nf = ds.sizes["fov"]
        nlocs = na * nx * nf

        print(f"    atrack={na}, xtrack={nx}, fov={nf} -> nlocs={nlocs}")

        # Build indices
        a, x, f = xr.broadcast(
            xr.DataArray(np.arange(na), dims="atrack"),
            xr.DataArray(np.arange(nx), dims="xtrack"),
            xr.DataArray(np.arange(nf), dims="fov"),
        )

        xtrack = x.values.ravel()
        fov = f.values.ravel()

        scan_pos = 9 * xtrack + fov

        out = xr.Dataset()
        out = out.assign_coords(location=np.arange(nlocs))

        out["scan_position"] = xr.DataArray(scan_pos, dims=("location",))

        # Flatten lat/lon into encoder variable names
        for v_in, v_out in [("lat", "latitude"), ("lon", "longitude")]:
            if v_in in ds:
                print(f"    flattening {v_in} -> {v_out}")
                out[v_out] = xr.DataArray(
                    ds[v_in].values.reshape(nlocs),
                    dims=("location",)
                )

        # Time
        if "obs_time_tai93" in ds:
            print("    converting obs_time_tai93 -> UNIX seconds")

            time3d = xr.broadcast(ds["obs_time_tai93"], ds["lat"])[0]
            time_tai93 = time3d.values.reshape(nlocs)

            TAI93_EPOCH = np.datetime64("1993-01-01T00:00:00")
            UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00")

            offset = (TAI93_EPOCH - UNIX_EPOCH) / np.timedelta64(1, "s")
            time_unix = time_tai93 + offset

            out["time"] = xr.DataArray(time_unix, dims=("location",))

        # Global PC scores
        if "global_pc_score" in ds:
            npc = ds.sizes["npc_global"]
            print(f"    flattening global_pc_score to (location, {npc})")
            out["global_pc_score"] = xr.DataArray(
                ds["global_pc_score"].values.reshape(nlocs, npc),
                dims=("location", "npc_global")
            )

        print("*** preprocess complete, vars:", list(out.variables))
        return out

    # -----------------------------------------------------
    # 2) Build a DataContainer from the flattened Dataset
    # -----------------------------------------------------
    # -----------------------------------------------------
    # 2) Build a DataContainer from the flattened Dataset
    # -----------------------------------------------------

    def _dims_for_var(self, varname, dims):
        """
        Map xarray dimension names (e.g. ('location', 'npc_global'))
        to BUFR query strings using the 'dimensions' section in cris_pca.yaml.
        """
        dim_paths = []
        for d in dims:
            if d not in self._dim_path_map:
                raise RuntimeError(
                    f"_dims_for_var: no mapping for dimension '{d}' "
                    f"in encoder YAML; known: "
                    f"{list(self._dim_path_map.keys())}"
                )
            dim_paths.append(self._dim_path_map[d])

        print(f"    _dims_for_var({varname}, {dims}) -> {dim_paths}")
        return dim_paths

    def make_obs(self, comm, input_path):
        print("***** Entering make_obs *****")
        ds = self.load_input(input_path)
        ds = self.preprocess_dataset(ds)

        container = bufr.DataContainer()

        # Load YAML once more (or reuse self._encoder_yaml)
        enc = self._encoder_yaml["encoder"]
        variables = enc["variables"]

        print('VARIABLES=', variables)

        for v in variables:
            name = v["name"]
            source = v["source"]

            if source not in ds:
                print(f"WARNING: source '{source}' not in dataset, skipping")
                continue

            xr_dims = ds[source].dims
            dim_paths = self._dims_for_var(name, xr_dims)

            print(f"Adding {name} from {source} with dim_paths {dim_paths}")
            print("  shape =", ds[source].values.shape)

            container.add(
                name,
                ds[source].values,
                dim_paths
            )

        return container


add_main_functions(CrisPcaObsBuilder)
