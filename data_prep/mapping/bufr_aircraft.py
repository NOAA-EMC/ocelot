#!/usr/bin/env python3
"""Aircraft BUFR mapping for OCELOT data preparation.

"""

import os
import numpy as np
import numpy.ma as ma
import re
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrMapPath = map_path('bufr_aircraft_prepbufr.yaml')


class AircraftBuilder(ObsBuilder):

    def __init__(self):
        super().__init__(PrepbufrMapPath, log_name=os.path.basename(__file__))

    def _derive_obstype_wind(self, container_ot_wind):
        obsType_wind = ma.array(container_ot_wind)
        obsType_wind[(obsType_wind > 300) & (obsType_wind < 400)] -= 100
        obsType_wind[(obsType_wind > 400) & (obsType_wind < 500)] -= 200
        obsType_wind[(obsType_wind > 500) & (obsType_wind < 600)] -= 300
 
        return obsType_wind


    def _derive_obstype_other(self, container_ot_other):
        obsType_other = ma.array(container_ot_other)
        obsType_other[(obsType_other > 300) & (obsType_other < 400)] -= 200
        obsType_other[(obsType_other > 400) & (obsType_other < 500)] -= 300
        obsType_other[(obsType_other > 500) & (obsType_other < 600)] -= 400

        return obsType_other


    def _compute_ialr_if_masked(self, typ, ialr):
        """
        Compute instantaneousAltitudeRate (IALR) if it is masked.
        Parameters:
            typ: datatype
            ialr: instantaneousAltitudeRate
        Returns:
            Masked array of the updated instantaneousAltitudeRate
        """


        ialr_bc = ialr.copy()

        cond = ialr_bc.mask & (typ >= 330) & (typ < 340)

        ialr_bc.data[cond] = 0.0
        ialr_bc.mask[cond] = False

        return ialr_bc


    def make_obs(self, comm, input_path) -> bufr.DataContainer:
        if not input_path:
            return bufr.DataContainer()

        prep_container = bufr.Parser(input_path, PrepbufrMapPath).parse(comm)
        prep_container.apply_mask(~prep_container.get('driftCycleTime').mask)
        prep_container.apply_mask(~prep_container.get('driftLatitude').mask)

        reference_time = self._get_reference_time(input_path)

        self._add_timestamp('driftCycleTime',
                            'timestamp',
                            prep_container,
                            reference_time)
        
        container_ot_wind = prep_container.get('obsType').filled()
        container_ot_other = prep_container.get('obsType').filled()
        obsType_wind = self._derive_obstype_wind(container_ot_wind)
        obsType_other = self._derive_obstype_other(container_ot_other)

        missing_ot = bufr.get_missing_value(container_ot_wind.dtype)
        prep_container.add('obsType_wind', obsType_wind.filled(missing_ot),['*'])
        prep_container.add('obsType_other', obsType_other.filled(missing_ot),['*'])

        orig_ot = prep_container.get('obsType')
        ialr = prep_container.get('instantaneousAltitudeRate')
        ialr_paths = prep_container.get_paths('instantaneousAltitudeRate')
        ialr2 = ma.array(ialr)

        ialr_bc = self._compute_ialr_if_masked(orig_ot, ialr2)
        prep_container.replace('instantaneousAltitudeRate', ialr_bc)

        return prep_container

    def _make_description(self):
        description = bufr.encoders.Description(PrepbufrMapPath)

        # Add the quality flag variables
        description.add_variables([
            {
                'name': "time",
                'source': 'timestamp',
                'longName': "Datetime",
                'units': "seconds since 1970-01-01T00:00:00Z"
            },
            {
                'name': "obsType_wind",
                'source': 'obsType_wind',
                'longName': "ObsType for wind data"
            },
            {
                'name': "obsType_other",
                'source': 'obsType_other',
                'longName': "ObsType for non-wind data"
            }
            ])

        return description

    def _get_reference_time(self, input_path) -> np.datetime64:
        path_components = Path(input_path).parts
        m = re.match(r'\w+\.(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', path_components[-4])

        if not m.groups():
            raise Exception("Error: Path string did not match the expected pattern.")

        return np.datetime64(datetime(year=int(m.group('year')),
                                      month=int(m.group('month')),
                                      day=int(m.group('day')),
                                      hour=int(path_components[-3])))

    def _add_timestamp(self,
                       input_name: str,
                       output_name: str,
                       container: bufr.DataContainer,
                       reference_time: np.datetime64) -> None:
        cycle_times = np.array([3600 * t for t in container.get(input_name)]).astype('timedelta64[s]')
        time = (reference_time + cycle_times).astype('datetime64[s]').astype('int64')
        container.add(output_name, time, ['*'])


# Add main functions create_obs_file and create_obs_group
add_main_functions(AircraftBuilder)
