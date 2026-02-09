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
        pc_slice = obs.get("global_pc_score_slice", [0, 24])
        self._pc_score_start, self._pc_score_end = pc_slice[0], pc_slice[1]

        # Subsampling config: optional in YAML
        subs = obs.get("subsample", {})
        # default behavior preserves previous behavior (random with p=1/17, seed=0)
        self._subsample_enabled = subs.get("enabled", True)
        self._subsample_method = subs.get("method", "random")  # "random" or "every_n"
        self._subsample_n = int(subs.get("n", 17))
        # probability used only for random method; default 1/n
        self._subsample_p = float(subs.get("p", 1.0 / self._subsample_n))
        self._subsample_seed = int(subs.get("seed", 0))

        enc = self.cris_pca_yaml.get("encoder", {})
        self._encoder_variables = enc.get("variables", [])

        # optional: write flattened Dataset to netCDF for debugging / inspection
        write_cfg = obs.get("write_netcdf", {})
        self._write_netcdf_enabled = bool(write_cfg.get("enabled", False))
        self._write_netcdf_path = write_cfg.get("path", "cris_pca_out.nc")
        self._write_netcdf_mode = write_cfg.get("mode", "w")

    # -----------------------------------------------------
    def load_input(self, filename):
        self.log.info(f"*** load_input() CALLED: {filename}")
        ds = xr.open_dataset(filename, decode_times=False)
        return ds

    # -----------------------------------------------------
    def preprocess_dataset(self, ds):
        self.log.debug(f"*** preprocess_dataset() CALLED with dataset: {ds}")
        required = ["atrack", "xtrack", "fov"]
        for d in required:
            if d not in ds.sizes:
                raise RuntimeError(f"Missing dimension {d}")

        na, nx, nf = ds.sizes["atrack"], ds.sizes["xtrack"], ds.sizes["fov"]
        nlocs = na * nx * nf

        out = xr.Dataset(coords={"location": np.arange(nlocs)})

        # scan_position (repeats for each atrack, by design)
        x = np.arange(nx, dtype=np.int32)
        f = np.arange(nf, dtype=np.int32)
        scan_pos_2d = (nf * x[:, None] + f[None, :]).ravel()           # length nx*nf
        scan_pos = np.tile(scan_pos_2d, na)                            # length nlocs
        out["scan_position"] = xr.DataArray(scan_pos, dims=("location",))

        # Flatten mapped variables safely
        for entry in self._variable_map:
            v_in = entry["source"]
            v_out = entry["name"]
            if v_in in ds:
                da = ds[v_in].stack(location=("atrack", "xtrack", "fov")).transpose("location")
                da = da.reset_index("location", drop=True)        # <-- removes MultiIndex
                da = da.assign_coords(location=out["location"])   # <-- use your integer location
                out[v_out] = da

        # Hardcode NOAA-20 sat id for now
        out["satelliteId"] = xr.DataArray(np.full(nlocs, 225, dtype=np.int32), dims=("location",))

        t3 = ds["obs_time_tai93"].expand_dims(fov=ds["fov"])
        t = t3.stack(location=("atrack", "xtrack", "fov")).transpose("location")
        t = t.reset_index("location", drop=True).assign_coords(location=out["location"])

        offset = (TAI93_EPOCH - UNIX_EPOCH) / np.timedelta64(1, "s")
        out["time"] = t + offset

        # Global PC scores (dims preserved correctly)
        pc = ds["global_pc_score"].stack(location=("atrack", "xtrack", "fov")).transpose("location", "npc_global")
        pc = pc.reset_index("location", drop=True).assign_coords(location=out["location"])
        pc = pc.isel(npc_global=slice(self._pc_score_start, self._pc_score_end))
        out["global_pc_score"] = pc

        # subsample
        out = self._apply_subsample(out)

        # optionally write to netCDF (path taken from YAML)
        if self._write_netcdf_enabled:
            out_path = os.path.expanduser(self._write_netcdf_path)
            dirpath = os.path.dirname(out_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            out.to_netcdf(out_path, mode=self._write_netcdf_mode)
            self.log.info(f"Wrote preprocessed dataset to {out_path}")
 
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
 
            da = ds[source]
            vals = np.asarray(da.values, dtype=np.float64, order="C")
            vals = np.ascontiguousarray(vals)

            container.add(name, vals, dim_paths)

        return container

    def _apply_subsample(self, out):
        """
        Apply subsampling to the flattened Dataset according to YAML config.
        Returns the (possibly) reduced Dataset with location coords re-assigned.
        """
        if not self._subsample_enabled:
            return out

        nloc = out.sizes["location"]
        if self._subsample_method == "every_n":
            # deterministic: keep every n-th record
            out = out.isel(location=slice(0, None, self._subsample_n))
        elif self._subsample_method == "random":
            # random subsample with probability p and deterministic seed
            rng = np.random.default_rng(self._subsample_seed)
            mask = rng.random(nloc) < self._subsample_p
            out = out.isel(location=mask)
        else:
            raise RuntimeError(f"unknown subsample method '{self._subsample_method}'")

        return out.assign_coords(location=np.arange(out.sizes["location"]))


add_main_functions(CrisPcaObsBuilder)
