#!/usr/bin/env python3

import os
import numpy as np
import numpy.ma as ma
import re
import math
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrKey = 'prepbufr'
AdpsfcKey = 'adpsfc'
SfcshpKey = 'sfcshp'

PrepbufrMapPath = map_path('bufr_surface_obs_prepbufr.yaml')
AdpsfcMapPath = map_path('bufr_surface_obs_adpsfc.yaml')
SfcshpMapPath = map_path('bufr_surface_obs_sfcshp.yaml')


class RawAdpsfcBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({PrepbufrKey: PrepbufrMapPath,
                          AdpsfcKey: AdpsfcMapPath,
                          SfcshpKey: SfcshpMapPath}, log_name=os.path.basename(__file__))

    def _derive_specifichumidity_rh(self, container_airt, container_rh, container_pres):
        T = container_airt
        rh = container_rh
        pres = container_pres
        print(" NICKE 1 T max min ", T.max(), T.min(), len(T))
        print(" NICKE 1 rh max min ", rh.max(), rh.min(), len(rh))
        print(" NICKE 1 pres max min ", pres.max(), pres.min(), len(pres))
        missing_T = bufr.get_missing_value(T.dtype)
        missing_rh = bufr.get_missing_value(rh.dtype)
        missing_pres = bufr.get_missing_value(pres.dtype)
        mask = (
            (T == missing_T) | 
            (rh == missing_rh) | 
            (pres == missing_pres)
        )
        mask2 = (
             (T < -90) | (T > 60) |
             (rh < 0) | (rh > 100) |
             (pres < 300) | (pres > 1100)
        )
        T = ma.masked_array(T, mask=(mask | mask2))
        rh = ma.masked_array(rh, mask=(mask | mask2))
        pres = ma.masked_array(pres, mask=(mask | mask2))
        print(" NICKE 2 max min ", T.max(), T.min(), len(T))
        print(" NICKE 2 max min ", rh.max(), rh.min(), len(rh))
        print(" NICKE 2 max min ", pres.max(), pres.min(), len(pres))
        #es = 6.112 * ma.exp((17.67 * T ) / (T + 243.5))
        es = 6.1078 * ma.exp((17.269 * T ) / (T + 237.3))
        print("NICKE 3 max min for es", es.max(), es.min(), len(es))
        e = (rh / 100) * es
        print("NICKE 3 max min for e ", e.max(), e.min(), len(e))
        q_rh = ((0.622 * e ) / (pres - (.378 * e)))#.astype('float')
        print("NICKE 3 max min for q_rh", q_rh.max(), q_rh.min(), len(q_rh))
        print("NICKE 4 number of non-missing in q_rh:", q_rh.count())
        #q = ma.masked_array(q, mask=mask)
  
        return q_rh

    def _derive_specifichumidity_dewp(self, container_dewp, container_pres):
        dewp = container_dewp
        pres = container_pres
        print(" NICKE 1 dewp max min ", dewp.max(), dewp.min(), len(dewp))
        print(" NICKE 1 pres max min ", pres.max(), pres.min(), len(pres))
        missing_dewp= bufr.get_missing_value(dewp.dtype)
        missing_pres = bufr.get_missing_value(pres.dtype)
        mask = (
            (dewp == missing_dewp) | 
            (pres == missing_pres)
        )
        mask2 = (
             (dewp < -90) | (dewp > 50) |
             (pres < 300) | (pres > 1100)
        )
        dewp = ma.masked_array(dewp, mask=(mask | mask2))
        pres = ma.masked_array(pres, mask=(mask | mask2))
        print(" NICKE 2 max min ", dewp.max(), dewp.min(), len(dewp))
        print(" NICKE 2 max min ", pres.max(), pres.min(), len(pres))
        e = 6.1078 * ma.exp((17.269 * dewp ) / (dewp + 237.3))
        print("NICKE 3 max min for es", e.max(), e.min(), len(e))
        q_dewp = ((0.622 * e ) / (pres - (.378 * e)))#.astype('float')
        print("NICKE 3 max min for q_dewp", q_dewp.max(), q_dewp.min(), len(q_dewp))
        print("NICKE 4 number of non-missing in q_dewp:", q_dewp.count())

        return q_dewp


    # Override
    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        if PrepbufrKey not in input_dict or \
           AdpsfcKey not in input_dict or \
           SfcshpKey not in input_dict:
            return bufr.DataContainer()

        prepbufr_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)
        adpsfc_container = bufr.Parser(input_dict[AdpsfcKey], self.map_dict[AdpsfcKey]).parse(comm)
        sfcshp_container = bufr.Parser(input_dict[SfcshpKey], self.map_dict[SfcshpKey]).parse(comm)

        # Mask out missing time stamps
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        prepbufr_container.apply_mask(~prepbufr_container.get('obsTimeMinusCycleTime').mask)
        prepbufr_container.all_gather(comm)

        # Add timestamps to the prepbufr container
        reference_time = self._get_reference_time(input_dict[PrepbufrKey])
        self._add_timestamp(prepbufr_container, reference_time)

        # Create output container
        container = bufr.DataContainer()

        # Combine the ADPSFC and SFCSHP containers
        container.append(adpsfc_container)
        container.append(sfcshp_container)

        # Add the prepbufr quality flag fields to the combined container
        # Use the timestamp, latitude and longitude to match the observations
        prepbufr_time = prepbufr_container.get('timestamp').filled()
        prepbufr_lat = prepbufr_container.get('latitude').filled()
        prepbufr_lon = prepbufr_container.get('longitude').filled()

        container_time = container.get('timestamp').filled()
        container_lat = container.get('latitude').filled()
        container_lon = container.get('longitude').filled()

        # DO CALCULATION OF SPECIFIC HUMIDITY USING RELATIVE HUMIDITY
        print("NICKE START")
        container_airt = container.get('airTemperature').filled()
        container_rh = container.get('relativeHumidity').filled()
        container_pres = container.get('airPressure_bufr').filled()
        container_dewp = container.get('dewPointTemperature').filled()
        #ydr_paths = container.get_paths('airPressure_bufr')

        #print("double check 1 q", q.max(), min(q), len(q))
        q_rh = self._derive_specifichumidity_rh(container_airt, container_rh, container_pres)#.astype('float')
        q_dewp = self._derive_specifichumidity_dewp(container_dewp, container_pres)#.astype('float')

        rh_missing = ma.getmaskarray(q_rh).sum()
        dewp_missing = ma.getmaskarray(q_dewp).sum()

        print("which missing rh dewp", rh_missing, dewp_missing)
        # set q
        q = ma.where(~ma.getmaskarray(q_dewp), q_dewp, q_rh) # fill in q_dewp with q_rh where q_dewp is missing 
        #q = ma.where(~ma.getmaskarray(q_rh), q_rh, q_dewp) #
        #for i in range(len(q)):
        #    print(" NICKE q : ", q[i])
        print("double check 2 q", q.max(), q.min(), len(q))
        print("NICKE END 1")
        #qarr = np.array(q)
        #for i in range(len(qarr)):
        #    print("NICKE q ", qarr[i])
        missing_q = bufr.get_missing_value(q.dtype)
        #container.add('specificHumidity_new', q, ["*"]) 
        container.add('specificHumidity_new', q.filled(missing_q), ['*', '*/EVENT']) 

        # Make hash table for fast lookup
        prepbufr_dict = {}
        for i, (t, lat, lon) in enumerate(zip(prepbufr_time, prepbufr_lat, prepbufr_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            prepbufr_dict[key] = i

        # Use hash table to find matching indices in combined container
        indices = [-1] * len(container_time)
        for i, (t, lat, lon) in enumerate(zip(container_time, container_lat, container_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            if key in prepbufr_dict:
                indices[i] = prepbufr_dict[key]

        indices = np.array(indices)
        valid_mask = indices != -1
        indices = indices[valid_mask]
        container.apply_mask(valid_mask)

        # Add event data
        for var in ['airPressure_prepbufr',
                    #'specificHumidity_new',
                    'pressureMeanSeaLevel_prepbufr',
                    'height_prepbufr',
                    'stationElevation_prepbufr',
                    'specificHumidity',
                    'eastwardWind',
                    'northwardWind',
                    'airPressureQuality',
                    'pressureMeanSeaLevelQuality',
                    'heightQuality',
                    'airTemperatureQuality',
                    'dewPointTemperatureQuality',
                    'specificHumidityQuality',
                    'windQuality',
                    'obsType']:

            quality_flags = prepbufr_container.get(var)[indices]
            container.add(var, quality_flags, ['*', '*/EVENT'])
        #container.add('specificHumidity_new', q, ['*', '*/EVENT']) 

#        container_q_pb = container.get('specificHumidity').filled()
#        print("NICKE 10 num of q from pb", len(container_q_pb))
#        missing_q_pb = bufr.get_missing_value(container_q_pb.dtype)
#        mask = (container_q_pb == missing_q_pb)
#        q_pb = ma.masked_array(container_q_pb, mask=mask)
#        print("NICKE 11 num non-missing q: ", q_pb.count())
#        
#        m1 = ma.getmaskarray(q)     # mask for q
#        m2 = ma.getmaskarray(q_pb)  # mask for q_pb
#
#        both_present = (~m1) & (~m2)
#        only_q       = (~m1) & (m2)
#        only_q_pb    = (m1) & (~m2)
#        neither      = (m1) & (m2)
#        n_both_present = both_present.sum()
#        n_only_q       = only_q.sum()
#        n_only_q_pb    = only_q_pb.sum()
#        n_neither      = neither.sum()
#        
#        print("both present:", n_both_present)
#        print("only q:", n_only_q)
#        print("only q_pb:", n_only_q_pb)
#        print("neither:", n_neither)


        return container

    def _make_description(self):
        description = bufr.encoders.Description(self.map_dict[AdpsfcKey])

        # Add the quality flag variables
        description.add_variables([
            {
                'name': "specificHumidity_new",
                'source': 'specificHumidity_new',
                'longName': "specificHumidity_new",
                'units': "kg/kg",
            },
            {
                'name': 'height_prepbufr',
                'source': 'height_prepbufr',
                'longName': 'height_prepbufr',
                'units': 'm',
            },
            {
                'name': 'stationElevation_prepbufr',
                'source': 'stationElevation_prepbufr',
                'longName': 'stationElevation_prepbufr',
                'units': 'm',
            },
            {
                'name': 'airPressure_prepbufr',
                'source': 'airPressure_prepbufr',
                'longName': 'airPressure_prepbufr',
                'units': 'hPa',
            },
            {
                'name': 'pressureMeanSeaLevel_prepbufr',
                'source': 'pressureMeanSeaLevel_prepbufr',
                'longName': 'pressureMeanSeaLevel_prepbufr',
                'units': 'hPa',
            },
            {
                'name': "specificHumidity",
                'source': 'specificHumidity',
                'longName': "Specific Humidity",
                'units': "kg/kg"
            },
            {
                'name': "eastwardWind",
                'source': 'eastwardWind',
                'longName': "Eastward Wind",
                'units': "m/s"
            },
            {
                'name': "northwardWind",
                'source': 'northwardWind',
                'longName': "Northward Wind",
                'units': "m/s"
            },
            {
                'name': "heightQuality",
                'source': 'heightQuality',
                'longName': "Height Quality Marker",
            },
            {
                'name': "airPressureQuality",
                'source': 'airPressureQuality',
                'longName': "Air Pressure Quality Marker",
            },
            {
                'name': "pressureMeanSeaLevelQuality",
                'source': 'pressureMeanSeaLevelQuality',
                'longName': 'pressureMeanSeaLevel Quality Marker',
            },
            {
                'name': "airTemperatureQuality",
                'source': 'airTemperatureQuality',
                'longName': "Air Temperature Quality Marker",
            },
            {
                'name': "dewPointTemperatureQuality",
                'source': 'dewPointTemperatureQuality',
                'longName': "Dew Point Temperature Quality Marker",
            },
            {
                'name': "specificHumidityQuality",
                'source': 'specificHumidityQuality',
                'longName': "Specific Humidity Quality Marker",
            },
            {
                'name': "windQuality",
                'source': 'windQuality',
                'longName': "Wind Quality Marker",
            },
            {
                'name': "obsType",
                'source': 'obsType',
                'longName': "ObsType",
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

    def _add_timestamp(self, container: bufr.DataContainer, reference_time: np.datetime64) -> np.array:
        cycle_times = np.array([3600 * t for t in container.get('obsTimeMinusCycleTime')]).astype('timedelta64[s]')
        time = (reference_time + cycle_times).astype('datetime64[s]').astype('int64')
        container.add('timestamp', time, ['*'])



add_main_functions(RawAdpsfcBuilder)
