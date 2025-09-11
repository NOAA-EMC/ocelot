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
        prep_reason = prep_container.get('airPressureReasonCode')#[:,0]

        prep_container.apply_mask(prep_reason == 100)

        prep_time = prep_container.get('launchTime')
        prep_lat = prep_container.get('launchLatitude')
        prep_lon = prep_container.get('launchLongitude')
        prep_id = prep_container.get('sequenceId')
        prep_pres = prep_container.get('airPressure')#[:,0]
        prep_drift_time =  prep_container.get('driftTime')
        prep_drift_lat =  prep_container.get('driftLatitude')
        prep_drift_lon =  prep_container.get('driftLongitude')
        prep_reason = prep_container.get('airPressureReasonCode')#[:,0]

        prep_pres = prep_pres.filled()
 
        prep_dict = {}
        for i, (t, lat, lon) in enumerate(zip(prep_time, prep_lat, prep_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            if key not in prep_dict:
                prep_dict[key] = []
            prep_dict[key].append(i)

        dump_pres = container.get('airPressure')

        container.apply_mask(~dump_pres.mask)
        dump_time = container.get('timestamp')
        dump_lat = container.get('latitude')
        dump_lon = container.get('longitude')
        dump_id = container.get('reportId')
        dump_pres = container.get('airPressure').filled()

        dump_dict = {}
        for i, (t, lat, lon) in enumerate(zip(dump_time, dump_lat, dump_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            if key in prep_dict:
                if key not in dump_dict:
                    dump_dict[key] = []
                dump_dict[key].append(i)

        matching_idxs = np.array([-1]*len(dump_time))
        for key in dump_dict.keys():

            # Make prepbufr look-up table for this key
            prep_bufr_table = {}
            for i in prep_dict[key]:
                prep_bufr_table[prep_pres[i]] = i

            # Match dump pressures to prepbufr pressures for this key
            for i in dump_dict[key]:
                dump_pressure = dump_pres[i]
                if dump_pressure in prep_bufr_table:
                    matching_idxs[i] = prep_bufr_table[dump_pressure]

        

        # matching_idxs = np.array([-1]*len(dump_time))
        # for key in dump_dict.keys():

        #     # Split the dump data into runs (pressure values start high and go low). When the pressure jumps
        #     # it means a new run.

        #     dump_runs = []
        #     run_start = 0
        #     for i in range(1, len(dump_dict[key])):
        #         if dump_pres[dump_dict[key][i]] > dump_pres[dump_dict[key][i-1]]:
        #             dump_runs.append(dump_dict[key][run_start:i])
        #             run_start = i
        #         elif i == len(dump_dict[key]) - 1:
        #             dump_runs.append(dump_dict[key][run_start:])

        #     # Split prepbufr data into runs (pressure values start high and go low). When the pressure jumps
        #     # it means a new run.

        #     prep_runs = []
        #     run_start = 0
        #     for i in range(1, len(prep_dict[key])):
        #         if prep_pres[prep_dict[key][i]] > prep_pres[prep_dict[key][i-1]]:
        #             prep_runs.append(prep_dict[key][run_start:i])
        #             run_start = i
        #         elif i == len(prep_dict[key]) - 1:
        #             prep_runs.append(prep_dict[key][run_start:])

            

        #     # Print the runs for comparison

        #     def print_col(title, arr, width=10):
        #         if type(arr[0]) == str or type(arr[0]) == np.datetime64:
        #             col_str = f'{title:<{width}}' + ''.join([f'{arr[i]:{width}}' for i in range(len(arr))])
        #         else:
        #             col_str = f'{title:<{width}}' + ''.join([f'{arr[i]:{width}.2f}' for i in range(len(arr))])
        #         print(col_str)

        #     for idx, dump_run in enumerate(dump_runs):
        #         if len(prep_runs) <= idx:
        #             break
        #         print_col(f'dump_{idx}',[dump_pres[i] for i in dump_run])
        #         print_col(f'prep_{idx}', [prep_pres[i] for i in prep_runs[idx]])
        #         print_col(f'prep_time_{idx}', [datetime.fromtimestamp(prep_drift_time[i]).strftime("%H:%M:%S") for i in prep_runs[idx]])
        #     print ('----')

        #     # Line up the runs and make the outputs

        #     for idx, prep_run in enumerate(prep_runs):
        #         print ('Matching run', idx)
        #         dump_run = dump_runs[idx]
        #         prep_pos = 0

        #         for dump_pos in range(len(dump_run)):
        #             # print('*', len(prep_run), prep_pos, len(dump_run), dump_pos, ' | ', len(prep_pres), prep_run[prep_pos])
        #             prep_pressure = prep_pres[prep_run[prep_pos]]
        #             dump_pressure = dump_pres[dump_run[dump_pos]]
                    
        #             if prep_pressure < dump_pressure:
        #                 continue

        #             skip = False
        #             while not np.isclose(prep_pressure, dump_pressure):
        #                 if prep_pressure < dump_pressure:
        #                     skip = True  # No match for this dump pressure
        #                     break

        #                 prep_pos += 1
        #                 prep_pressure = prep_pres[prep_run[prep_pos]]

        #             if skip:
        #                 continue

        #             print (dump_pressure, end=' ')

        #             matching_idxs[dump_run[dump_pos]] = prep_run[prep_pos]

        #         print ('')

            # drift_dict = {}
            # for i in prep_dict[key]:
            #     if not prep_pres[i] in drift_dict:
            #         drift_dict[prep_pres[i]] = (prep_drift_time[i], prep_drift_lat[i], prep_drift_lon[i])
            #     else:
            #         print (drift_dict[prep_pres[i]], (prep_drift_time[i], prep_drift_lat[i], prep_drift_lon[i]))
            #         assert(drift_dict[prep_pres[i]] == (prep_drift_time[i], prep_drift_lat[i], prep_drift_lon[i]))


            # dump_drift_lat = np.zeros(len(dump_dict[key]))
            # prep_pos = 0
            # dump_pos = 0
            # for prep_idx, prep_run in enumerate(prep_runs):
            #     if prep_idx >= len(dump_runs):
            #         break

            #     dump_run = dump_runs[prep_idx]

 
        # # Use hash table to find matching indices in combined container
        # indices = [-1] * len(dump_time)
        # obs_idx = 0
        # last_key = None
        # for i, (t, lat, lon) in enumerate(zip(dump_time, dump_lat, dump_lon)):
        #     key = (t, np.round(lat, 2), np.round(lon, 2))

        #     if key != last_key:
        #         obs_idx = 0
        #         last_key = key

        #     if key in prep_dict:
        #         indices[i] = prep_dict[key][obs_idx]
        #         obs_idx += 1

        valid_mask = matching_idxs != -1
        matching_idxs = matching_idxs[valid_mask]
        container.apply_mask(valid_mask)

        # Add the quality flags to the container
        for var in ['driftTime',
                    'driftLatitude',
                    'driftLongitude']:

            quality_flags = prep_container.get(var)[matching_idxs]
            container.add(var, quality_flags, ['*'])

        for var in ['height',
                    'airTemperatureQuality',
                    'specificHumidityQuality',
                    'dewPointTemperatureQuality',
                    'windQuality',
                    'airPressureQuality',
                    'heightQuality']:

            quality_flags = prep_container.get(var)[matching_idxs]
            container.add(var, quality_flags, ['*'])

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

        # description.add_dimension('event', ['*', '*/EVENT'])

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

