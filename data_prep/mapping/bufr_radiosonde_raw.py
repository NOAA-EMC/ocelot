#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrKey = 'prepbufr'
DumpKey = 'dump'

PrepbufrMapPath = map_path('bufr_radiosonde_prepbufr.yaml')
DumpMapPath = map_path('bufr_radiosonde_dump.yaml')

class RawRadiosondeBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({PrepbufrKey: PrepbufrMapPath,
                          DumpKey: DumpMapPath}, log_name=os.path.basename(__file__))

    # Override
    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        prep_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)
        container = bufr.Parser(input_dict[DumpKey], self.map_dict[DumpKey]).parse(comm)

        # Mask out missing time stamps
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        prep_container.apply_mask(~prep_container.get('launchCycleTime').mask)

        prep_container.all_gather(comm)

        # Add timestamps to the prep container
        reference_time = self._get_reference_time(input_dict[PrepbufrKey])
        self._add_timestamp('launchCycleTime',
                            'launchTime',
                            prep_container,
                            reference_time)

        self._add_timestamp('driftCycleTime',
                            'driftTime',
                            prep_container,
                            reference_time)

        # Add the prep quality flag fields to the combined container
        # Use the timestamp, latitude and longitude to match the observations

        prep_drift_lat =  prep_container.get('driftLatitude')

        prep_container.apply_mask(~prep_drift_lat.mask)
        prep_time = prep_container.get('launchTime')
        prep_lat = prep_container.get('launchLatitude')
        prep_lon = prep_container.get('launchLongitude')
        prep_id = prep_container.get('sequenceId')
        prep_pres = prep_container.get('airPressure')[:,0]
        prep_reason = prep_container.get('airPressureReasonCode').filled()
        prep_program = prep_container.get('airPressureProgramCode').filled()

        prep_pres.mask = (prep_reason != 100)

        prep_pres = prep_pres.filled()

        print (f'Num 100s: {float(np.sum(prep_reason == 100))/prep_reason.size:.3f}')

        print ("prep reason code samples:")
        print (prep_reason[:10, :])
        print ('')
        print ("prep program code samples:")
        print (prep_program[:10, :])
        print ('')
 
        prep_dict = {}
        for i, (t, lat, lon, oid) in enumerate(zip(prep_time, prep_lat, prep_lon, prep_id)):
            key = (t, np.round(lat, 2), np.round(lon, 2), int(oid))
            if key not in prep_dict:
                prep_dict[key] = []
            prep_dict[key].append(i)

        dump_pres = container.get('windDirection')

        container.apply_mask(~dump_pres.mask)
        dump_time = container.get('timestamp')
        dump_lat = container.get('latitude')
        dump_lon = container.get('longitude')
        dump_id = container.get('reportId')
        dump_pres = container.get('airPressure').filled()

        dump_dict = {}
        for i, (t, lat, lon, oid) in enumerate(zip(dump_time, dump_lat, dump_lon, dump_id)):
            key = (t, np.round(lat, 2), np.round(lon, 2), int(oid))
            if key in prep_dict:
                if key not in dump_dict:
                    dump_dict[key] = []
                dump_dict[key].append(i)

        # for key in dump_dict.keys():
        #     print(key, len(prep_dict[key]), len(dump_dict[key]))
        #     print ('  dump: ', end='')
        #     for i in dump_dict[key]:
        #         print (f' {dump_pres[i]:.2f}', end='')
        #     print('')
        #     print ('  prep: ', end='')
        #     for i in prep_dict[key]:
        #         print (f' {prep_pres[i]:.2f}', end='')
        #     print('')

        # Walk the pressure lelvels between the two containers to discover the common
        # pressure level runs.

        for key in dump_dict.keys():

            print(key, len(prep_dict[key]), len(dump_dict[key]))
            print ('  dump: ', end='')
            for i in dump_dict[key]:
                print (f' {dump_pres[i]:.2f}', end='')
            print('')
  
            # Split the dump data into runs (same logic)
            dump_runs = []
            run_start = 0
            for i in range(1, len(dump_dict[key])):
                if abs(dump_pres[dump_dict[key][i]] - dump_pres[dump_dict[key][i-1]]) > 250:
                    dump_runs.append(dump_dict[key][run_start:i])
                    run_start = i

            # print prep_pres in runs
            for r in dump_runs:
                print('  ** dump: ', end='')
                for i in r:
                    print (f' {dump_pres[i]:.2f}', end='')
                print('')


            # Split prepbufr data into runs (pressure values start high and go low). When the pressure jumps
            # it means a new run.

            print ('  prep: ', end='')
            for i in prep_dict[key]:
                print (f' {prep_pres[i]:.2f}', end='')
            print('')


            prep_runs = []
            run_start = 0
            for i in range(1, len(prep_dict[key])):
                if abs(prep_pres[prep_dict[key][i]] - prep_pres[prep_dict[key][i-1]]) > 250:
                    prep_runs.append(prep_dict[key][run_start:i])
                    run_start = i

            # print prep_pres in runs
            for r in prep_runs:
                print('  ** prep: ', end='')
                for i in r:
                    print (f' {prep_pres[i]:.2f}', end='')
                print('')
            

            # Loop through the dump runs and find the best matching prep run. The prep run should contain
            # all the pressure levels in the dump run, but may have extra values sprinkled in.
            for d_run in dump_runs:
                d_run = set([dump_pres[i] for i in d_run])
                for p_run in prep_runs:
                    p_run = set([prep_pres[i] for i in p_run])
                    run_count = len(d_run.intersection(p_run))
                    if run_count == len(d_run):
                        print (f' {key}: {d_run} || {p_run}')
                        break

            


 
        # Use hash table to find matching indices in combined container
        indices = [-1] * len(dump_time)
        obs_idx = 0
        last_key = None
        for i, (t, lat, lon, oid) in enumerate(zip(dump_time, dump_lat, dump_lon, dump_id)):
            key = (t, np.round(lat, 2), np.round(lon, 2), int(oid))

            if key != last_key:
                obs_idx = 0
                last_key = key

            if key in prep_dict:
                indices[i] = prep_dict[key][obs_idx]
                obs_idx += 1

        indices = np.array(indices)
        valid_mask = indices != -1
        indices = indices[valid_mask]
        container.apply_mask(~valid_mask)

        # Add the quality flags to the container
        for var in ['driftTime',
                    'driftLatitude',
                    'driftLongitude']:

            quality_flags = prep_container.get(var)[indices]
            container.add(var, quality_flags, ['*'])

        for var in ['height',
                    'airTemperatureQuality',
                    'specificHumidityQuality',
                    'dewPointTemperatureQuality',
                    'windQuality',
                    'airPressureQuality',
                    'heightQuality']:

            quality_flags = prep_container.get(var)[indices]
            container.add(var, quality_flags, ['*', '*/EVENT'])

        return container

    def _make_description(self):
        description = bufr.encoders.Description(self.map_dict[DumpKey])

        # Add the quality flag variables
        description.add_variables([
            {
                'name': "time",
                'source': 'driftTime',
                'longName': "Datetime",
                'units': "seconds since 1970-01-01T00:00:00Z"
            },
            {
                'name': "latitude",
                'source': 'driftLatitude',
                'longName': "Latitude",
                'units': "degree_north"
            },
            {
                'name': "longitude",
                'source': 'driftLongitude',
                'longName': "Longitude",
                'units': "degree_east"
            },
            {
                'name': "height",
                'source': 'height',
                'longName': "Height",
                'units': "meters"
            },
            {
                'name': "airTemperatureQuality",
                'source': 'airTemperatureQuality',
                'longName': "Air Temperature Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "specificHumidityQuality",
                'source': 'specificHumidityQuality',
                'longName': "Specific Humidity Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "dewPointTemperatureQuality",
                'source': 'dewPointTemperatureQuality',
                'longName': "Dew Point Temperature Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "windQuality",
                'source': 'windQuality',
                'longName': "Wind Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "airPressureQuality",
                'source': 'airPressureQuality',
                'longName': "Air Pressure Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "heightQuality",
                'source': 'heightQuality',
                'longName': "Height Quality Marker",
                'units': "quality_marker"
            }
        ])

        description.add_dimension('event', ['*', '*/EVENT'])

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

add_main_functions(RawRadiosondeBuilder)

