#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrKey = 'prepbufr'
LowResDumpKey = 'low_res_dump'
HighResDumpKey = 'high_res_dump'

PrepbufrMapPath = map_path('bufr_radiosonde_prepbufr.yaml')
LowResDumpMapPath = map_path('bufr_radiosonde_adpupa.yaml')
HighResDumpMapPath = map_path('bufr_radiosonde_uprair.yaml')

class RawRadiosondeBuilder(ObsBuilder):
    last_flight_id: int = 0

    def __init__(self):
        super().__init__({PrepbufrKey: PrepbufrMapPath,
                          LowResDumpKey: LowResDumpMapPath,
                          HighResDumpKey: HighResDumpMapPath}, log_name=os.path.basename(__file__))

    # Override
    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        if 'high_res_dump' in input_dict:
            print ("Processing high resolution dump")
            container = self._process_high_res_dump(comm, input_dict)
        else:
            print ("Processing low resolution dump")
            container = self._process_low_res_dump(comm, input_dict)

        return container

    
    def _process_low_res_dump(self, comm, input_dict) -> bufr.DataContainer:
        prep_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)

        container = bufr.Parser(input_dict[LowResDumpKey], self.map_dict[LowResDumpKey]).parse(comm)
        container.apply_mask(~container.get('airPressure').mask)

        # Mask out missing time stamps
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        prep_container.apply_mask(~prep_container.get('launchCycleTime').mask)
        prep_container.apply_mask(~prep_container.get('driftLatitude').mask)
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

        prep_container.apply_mask(prep_container.get('airPressureReasonCode') == 100)

        prep_time = prep_container.get('launchTime')
        prep_lat = prep_container.get('launchLatitude')
        prep_lon = prep_container.get('launchLongitude')
        prep_pres = prep_container.get('airPressure')
        prep_drift_time = prep_container.get('driftTime')

        prep_pres = prep_pres.filled()
   
        prep_dict = {}
        for i, (t, lat, lon) in enumerate(zip(prep_time, prep_lat, prep_lon)):
            key = (t, self._floatToKey(lat), self._floatToKey(lon))
            if key not in prep_dict:
                prep_dict[key] = []
            prep_dict[key].append(i)

        dump_time = container.get('timestamp')
        dump_lat = container.get('latitude')
        dump_lon = container.get('longitude')
        dump_pres = container.get('airPressure')

        #round dump pressure to nearest .1 hPa
        dump_pres = np.round(dump_pres, 1)

        dump_dict = {}
        for i, (t, lat, lon) in enumerate(zip(dump_time, dump_lat, dump_lon)):
            key = (t, self._floatToKey(lat), self._floatToKey(lon))
            if key in prep_dict:
                if key not in dump_dict:
                    dump_dict[key] = []
                dump_dict[key].append(i)

        matching_idxs = []
        for flight_idx, key in enumerate(dump_dict.keys()):

            # Make prepbufr look-up table for this key
            prep_bufr_table = {}
            prep_bufr_times = set()
            for i in prep_dict[key]:
                prep_bufr_table[self._floatToKey(prep_pres[i])] = i
                prep_bufr_times.add((prep_drift_time[i], self._floatToKey(prep_pres[i])))

            # Make dump look-up table for this key
            dump_bufr_table = {}
            for i in dump_dict[key]:
                dump_bufr_table[self._floatToKey(dump_pres[i])] = i

            # Sort prep_bufr_times by first tuple element (drift time)
            prep_bufr_times = list(prep_bufr_times)
            prep_bufr_times.sort()

            # Match dump pressures to prepbufr pressuresk ordered by drift time
            for time, prep_pressure in prep_bufr_times:
                if prep_pressure in dump_bufr_table:
                    dump_idx = dump_bufr_table[prep_pressure]
                    prep_idx = prep_bufr_table[prep_pressure]
                    matching_idxs.append((dump_idx, prep_idx, flight_idx))

        # Make new container with only the matched indices
        new_container = bufr.DataContainer()
        for var in container.list():
            data = container.get(var)
            path = container.get_paths(var)
            matched_data = np.array([data[dump_idx] for dump_idx, prep_idx, flight_idx in matching_idxs])
            new_container.add(var, matched_data, path)

        # Add the prepbufr data to the new container
        for var in ['driftTime',
                    'driftLatitude',
                    'driftLongitude',
                    'height',
                    'airTemperatureQuality',
                    'specificHumidityQuality',
                    'dewPointTemperatureQuality',
                    'windQuality',
                    'airPressureQuality',
                    'heightQuality']:
            
            data = prep_container.get(var)
            path = prep_container.get_paths(var)
            matched_data = np.array([data[prep_idx] for dump_idx, prep_idx, flight_idx in matching_idxs])
            new_container.add(var, matched_data, path)

        # Add the flight ID
        flight_ids = np.array([RawRadiosondeBuilder.last_flight_id + flight_idx + 1 for dump_idx, prep_idx, flight_idx in matching_idxs])
        new_container.add('flightId', flight_ids, ['*'])

        RawRadiosondeBuilder.last_flight_id = flight_ids[-1]

        return new_container

    def _process_high_res_dump(self, comm, input_dict) -> bufr.DataContainer:
        prep_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)
        prep_container.apply_mask(~prep_container.get('launchCycleTime').mask)

        container = bufr.Parser(input_dict[HighResDumpKey], self.map_dict[HighResDumpKey]).parse(comm)

        container.apply_mask(~container.get('latitude').mask)
        container.apply_mask(~container.get('airPressure').mask)

        # launch_time = np.datetime64('1970-01-01T00:00:00') + container.get('timestamp').astype('timedelta64[s]')
        # drift_time = launch_time + container.get('driftDisplacementTime').astype('timedelta64[s]')
        # container.add('driftTime', drift_time.astype('datetime64[s]').astype('int64'), ['*'])

        reference_time = self._get_reference_time(input_dict[PrepbufrKey])
        self._add_timestamp('launchCycleTime',
                            'launchTime',
                            prep_container,
                            reference_time)

        self._add_timestamp('driftCycleTime',
                            'driftTime',
                            prep_container,
                            reference_time)


        # Mask out latitude and longitude missing in prepbufr from the container
        prep_time = prep_container.get('launchTime')
        prep_lat = prep_container.get('launchLatitude')
        prep_lon = prep_container.get('launchLongitude')
        prep_pres = prep_container.get('airPressure')
        prep_drift_time = prep_container.get('driftTime')

        dump_time = container.get('timestamp')
        dump_lat = container.get('latitude')
        dump_lon = container.get('longitude')

        # dump_time = np.int(np.ceil(dump_time / 1800) * 1800)

        dump_time_key = (np.round((dump_time + 1800) / 1800) * 1800).astype(int)
        dump_pres = np.round(container.get('airPressure'), 1)
        # prep_time_key = (np.ceil(prep_time / 3600) * 3600).astype(int)

        prep_dict = {}
        for i, (t, lat, lon) in enumerate(zip(prep_time, prep_lat, prep_lon)):
            key = (t, self._floatToKey(lat), self._floatToKey(lon))
            if key not in prep_dict:
                prep_dict[key] = []
            prep_dict[key].append(i)

        # print (f'prep_dict {len(list(prep_dict.keys()))}: {list(prep_dict.keys())[:100]}')

        dump_dict = {}
        for i, (t, lat, lon) in enumerate(zip(dump_time_key, dump_lat, dump_lon)):
            key = (t, self._floatToKey(lat), self._floatToKey(lon))
            if key in prep_dict:
                if key not in dump_dict:
                    dump_dict[key] = []
                dump_dict[key].append(i)

        # print (f'dump_dict {len(list(dump_dict.keys()))}: {list(dump_dict.keys())[:100]}')

        # num_matches = 0
        # for key in dump_dict.keys():
        #     if key in prep_dict:
        #         num_matches += 1

        # print (f'num_matches {num_matches} percent {100 * num_matches / len(list(dump_dict.keys())):.1f}%')

        matching_idxs = []
        for flight_idx, key in enumerate(dump_dict.keys()):

            # Make prepbufr look-up table for this key
            prep_bufr_table = {}
            prep_bufr_times = set()
            for i in prep_dict[key]:
                prep_bufr_table[self._floatToKey(prep_pres[i])] = i
                prep_bufr_times.add((prep_drift_time[i], self._floatToKey(prep_pres[i])))

            # Make dump look-up table for this key
            dump_bufr_table = {}
            for i in dump_dict[key]:
                dump_bufr_table[self._floatToKey(dump_pres[i])] = i

            # Sort prep_bufr_times by first tuple element (drift time)
            prep_bufr_times = list(prep_bufr_times)
            prep_bufr_times.sort()

            # Match dump pressures to prepbufr pressuresk ordered by drift time
            for time, prep_pressure in prep_bufr_times:
                if prep_pressure in dump_bufr_table:
                    dump_idx = dump_bufr_table[prep_pressure]
                    prep_idx = prep_bufr_table[prep_pressure]
                    matching_idxs.append((dump_idx, prep_idx, flight_idx))

            # print (f'match percent {100 * len(matching_idxs) / len(dump_bufr_table):.1f}% ')

        # Make new container with only the matched indices
        new_container = bufr.DataContainer()
        for var in container.list():
            data = container.get(var)
            path = container.get_paths(var)
            matched_data = np.array([data[dump_idx] for dump_idx, prep_idx, flight_idx in matching_idxs])
            new_container.add(var, matched_data, path)



        # driftLatitude = container.get('latitude') + container.get('driftLatitude')
        # driftLongitude = container.get('longitude') + container.get('driftLongitude')

        # container.replace('driftLatitude', driftLatitude)
        # container.replace('driftLongitude', driftLongitude)

        # print (drift_time)
        # print (container.get('driftLatitude'))
        # print (container.get('driftLongitude'))

        # Add the prepbufr data to the new container
        for var in ['driftTime',
                    'driftLatitude',
                    'driftLongitude',
                    'height',
                    'airTemperatureQuality',
                    'specificHumidityQuality',
                    'dewPointTemperatureQuality',
                    'windQuality',
                    'airPressureQuality',
                    'heightQuality']:
            
            data = prep_container.get(var)
            path = prep_container.get_paths(var)
            matched_data = np.array([data[prep_idx] for dump_idx, prep_idx, flight_idx in matching_idxs])
            if matched_data.dtype == np.dtype('float64'):
                matched_data = matched_data.astype('float32')
            new_container.add(var, matched_data, path)


        report_ids = new_container.get('reportId')
        flight_id_dict = {}
        flight_id = RawRadiosondeBuilder.last_flight_id
        for unique_rep in np.unique(report_ids):
            if unique_rep not in flight_id_dict:
                flight_id_dict[unique_rep] = flight_id + 1
                flight_id += 1

        # print ('ids ', report_ids.size)

        flight_ids = np.array([flight_id_dict[report_id] for report_id in report_ids])
        new_container.add('flightId', flight_ids, ['*'])

        # # Mask out flights with less than 10 observations
        # counts = np.bincount(flight_ids)
        # valid_flights = np.where(counts >= 10)[0]
        # container.apply_mask(~np.isin(flight_ids, valid_flights))

        return new_container

    def _make_description(self):
        description = bufr.encoders.Description(self.map_dict[LowResDumpKey])

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
            },
            {
                'name': "flightId",
                'source': 'flightId',
                'longName': "Flight Identifier",
                'units': ""
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

    def _floatToKey(self, f: float) -> str:
        return f"{np.round(f, 1):.2f}"

add_main_functions(RawRadiosondeBuilder)

