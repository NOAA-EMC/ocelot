#!/usr/bin/env python3

import os

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
import settings

MAPPING_PATH = map_path('cris_pca.yaml')

class CrisPcaObsBuilder(ObsBuilder):
    """
    ObsBuilder subclass for CrIS PCA netCDF input.
    """

    def __init__(self):
        print('MAPPING PATH:',MAPPING_PATH)
        super().__init__(MAPPING_PATH,
                         log_name=os.path.basename(__file__))

    def load_input(self, filename):
        """Load the CrIS PCA netCDF file."""
        return xr.open_dataset(filename, decode_times=False)

    def preprocess_dataset(self, ds):
        """
        Convert the CrIS PCA dataset into a flattened 1-D location structure.
        """

        na = ds.dims["atrack"]
        nx = ds.dims["xtrack"]
        nf = ds.dims["fov"]
        nlocs = na * nx * nf

        # Build index arrays
        atrack_idx, xtrack_idx, fov_idx = xr.broadcast(
            xr.DataArray(np.arange(na), dims="atrack"),
            xr.DataArray(np.arange(nx), dims="xtrack"),
            xr.DataArray(np.arange(nf), dims="fov")
        )

        # Flatten indices
        atrack_1d = atrack_idx.values.flatten()
        xtrack_1d = xtrack_idx.values.flatten()
        fov_1d = fov_idx.values.flatten()

        # scan_position = 9 * xtrack + fov
        scan_position = 9 * xtrack_1d + fov_1d

        newds = xr.Dataset()
        newds = newds.assign_coords(location=np.arange(nlocs))
        newds["scan_position"] = xr.DataArray(scan_position, dims=("location",))

        # obs_time_tai93(atrack,xtrack) -> broadcast to fov -> 1D -> UNIX seconds
        if "obs_time_tai93" in ds:
            time3d = xr.broadcast(ds["obs_time_tai93"], ds["lat"])[0]
            time_tai93 = time3d.values.reshape(nlocs)

            TAI93_EPOCH = np.datetime64("1993-01-01T00:00:00")
            UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00")
            TAI93_TO_UNIX_SECONDS = (TAI93_EPOCH - UNIX_EPOCH) / np.timedelta64(1, "s")

            time_unix = time_tai93 + TAI93_TO_UNIX_SECONDS

            newds["obs_time_tai93"] = xr.DataArray(time_unix, dims=("location",))

        # 3D vars (atrack,xtrack,fov) -> 1D location
        vars_3d = ["lat", "lon", "pca_qc", "fov_obs_id"]
        for v in vars_3d:
            if v in ds:
                newds[v] = xr.DataArray(
                    ds[v].values.reshape(nlocs),
                    dims=("location",)
                )

        # 4D global_pc_score -> (location, npc_global)
        if "global_pc_score" in ds:
            n2 = ds.dims["npc_global"]
            newds["global_pc_score"] = xr.DataArray(
                ds["global_pc_score"].values.reshape(nlocs, n2),
                dims=("location", "npc_global")
            )

        return newds


add_main_functions(CrisPcaObsBuilder)

