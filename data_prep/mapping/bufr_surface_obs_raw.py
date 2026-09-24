#!/usr/bin/env python3
"""Raw surface-observation BUFR mapping for OCELOT data preparation.

"""

import os
import numpy as np
import numpy.ma as ma
import re
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
from collections import Counter
from decimal import Decimal

import inspect
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

    def _elevation_check_sfcshp(self, container_late, container_lone, container_eleve, container_t29e):
        lat = container_late
        lon = container_lone
        eleve = container_eleve
        t29 = container_t29e

        eleve_missing = bufr.get_missing_value(eleve.dtype)
        sfland = np.isin(t29, [511, 512, 514, 540])
        region = (
            (lat >= 41.0) & (lat <= 50.0) &
            (lon >= -93.0) & (lon <= -75.0)
        )


        mask_to_unmask = eleve.mask & ~sfland & ~region
        eleve[mask_to_unmask] = 0.0
        eleve.mask[mask_to_unmask] = False

        ##mask where eleve=9999.0 meters
        eleve = ma.masked_equal(eleve, 9999.0)
        eleve.set_fill_value(eleve_missing)

        return eleve


    def _derive_p_fr_a(self, container_alt, container_pres5, container_elev5):
        missing_alt = bufr.get_missing_value(container_alt.dtype)
        missing_e5 = bufr.get_missing_value(container_elev5.dtype)

        alt = ma.masked_invalid(ma.masked_equal(container_alt, missing_alt))
        elev5 = ma.masked_invalid(ma.masked_equal(container_elev5, missing_e5))

        newpres = (alt**0.190284 - ((1013.25**0.190284) * 0.0065 / 288.15) * elev5) ** 5.2553026

        return newpres


    def _derive_pstn_frompmsl_sfclnd(self,container_lat6, container_lon6, container_elev6, container_t296, container_temp6, container_airp6, container_mslp6):
        lat = container_lat6
        lon = container_lon6
        elev = container_elev6
        t29 = container_t296
        temp = container_temp6
        airp = container_airp6.filled()
        pmsl = container_mslp6.filled()

        tmissing = bufr.get_missing_value(temp.dtype)
        pmissing = bufr.get_missing_value(pmsl.dtype)
        airpmissing = bufr.get_missing_value(airp.dtype)

        mask = (pmsl == pmissing) | (airp != airpmissing)
        sfland = np.isin(t29, [511, 512, 514, 540])

        # Geographic box
        region = (
            (lat >= 41.0) & (lat <= 50.0) &
            (lon >= -93.0) & (lon <= -75.0)
        )
        
        
        inside = region & (~sfland)
        mask |= inside & (elev <= 7.5)
        sfland |= inside & (elev > 7.5)

        pstn = np.full_like(pmsl, pmissing, dtype=float)
        
        # ELEV <= 7.5
        low = (elev <= 7.5) & (~mask)
        pstn[low] = pmsl[low]
        
        # IF(.NOT.SFLAND)
        high = (elev> 7.5) & (~mask)
        mask |= high & (~sfland)
        
        # Temperature used by PR()
        tt = np.full_like(temp, 288.15, dtype=float)

        good_temp = (temp != tmissing) #& (~np.isnan(temp))
        # Only change tt where temp is present
        tt[good_temp] = temp[good_temp]+273.15
       
        valid_high = high & (~mask)
        pstn[valid_high]  = pmsl[valid_high] * (((tt[valid_high] - (.0065 * elev[valid_high]))/tt[valid_high])**5.256)

        pstn[mask] = pmissing

        return pstn


    def _derive_specifichumidity_rh(self, container_airt, container_rh, container_pres):
        T = container_airt
        rh = container_rh
        pres = container_pres
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
        es = 6.1078 * ma.exp((17.269 * T ) / (T + 237.3))
        e = (rh / 100) * es
        q_rh = ((0.622 * e ) / (pres - (.378 * e)))#.astype('float')
  
        return q_rh


    def _derive_specifichumidity_dewp(self, container_airt, container_dewp, container_pres):
        T = container_airt
        dewp = container_dewp
        pres = container_pres
        missing_dewp= bufr.get_missing_value(dewp.dtype)
        missing_pres = bufr.get_missing_value(pres.dtype)
        mask = (
            (dewp == missing_dewp) | 
            (pres == missing_pres)
        )
        mask2 = (
             (dewp < -90) | (dewp > 50) |
             (pres < 300) | (pres > 1100) #|
             #(T - dewp > 60) 
        )
        dewp = ma.masked_array(dewp, mask=(mask | mask2))
        pres = ma.masked_array(pres, mask=(mask | mask2))
        e = 6.1078 * ma.exp((17.269 * dewp ) / (dewp + 237.3))
        q_dewp = ((0.622 * e ) / (pres - (.378 * e)))#.astype('float')

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
        prepbufr_elv = prepbufr_container.get('stationElevation_prepbufr').filled()
        prepbufr_sid = prepbufr_container.get('stationIdentification_prepbufr').filled()


        #for i in range(len(prepbufr_time)):
        #    print("NElata ", prepbufr_time[i])

        container_time = container.get('timestamp').filled()
        container_lat = container.get('latitude').filled()
        container_lon = container.get('longitude').filled()
        container_elv = container.get('height_bufr').filled()
        container_sid = container.get('stationIdentification_bufr').filled()

        #for i in range(len(container_time)):
        #    print("NElatb ", container_time[i])

        missing_pbelv = bufr.get_missing_value(prepbufr_elv.dtype)        
        missing_hbufr = bufr.get_missing_value(container_elv.dtype)        

        print("NE A")

        print("NE container lengths pb", len(prepbufr_time), "bufr", len(container_time))



        # ============================================================
        # Build PREPBUFR lookup
        # ============================================================
        
        prepbufr_dict = defaultdict(list)
        
        for i, (t, lat, lon, sid) in enumerate(
                zip(prepbufr_time, prepbufr_lat, prepbufr_lon, prepbufr_sid)):
        
            places = abs(Decimal(str(lat)).as_tuple().exponent)
        
            latr = np.round(lat, places)
            lonr = np.round(lon, places)
        
            key = (t, latr, lonr, sid)
        
            prepbufr_dict[key].append(i)
        
        
        # ============================================================
        # Build CONTAINER/BUFR lookup
        # ============================================================
        
        container_dict = defaultdict(list)
        
        for i, (t, lat, lon, sid) in enumerate(
                zip(container_time, container_lat, container_lon, container_sid)):
        
            places = abs(Decimal(str(lat)).as_tuple().exponent)
        
            latr = np.round(lat, places)
            lonr = np.round(lon, places)
        
            key = (t, latr, lonr, sid)
        
            container_dict[key].append(i)
        


        # ============================================================
        # Build the final BUFR/PREPBUFR pairing.
        #
        # A unique BUFR observation may have 1 or 2 PREPBUFR events.
        #
        # For duplicate BUFR observations:
        #
        #   2 BUFR + 2 PREPBUFR:
        #       keep BUFR #1 -> PREPBUFR #1
        #
        #   2 BUFR + 4 PREPBUFR:
        #       keep BUFR #1 -> PREPBUFR #1,#2
        #
        # We deliberately retain only the FIRST BUFR occurrence.
        # A second copy of that BUFR observation will be created
        # below when it has two PREPBUFR events.
        # ============================================================

        container_to_prepbufr = {}

        for key, container_indices in container_dict.items():

            prep_indices = prepbufr_dict.get(key, [])

            if not prep_indices:
                continue

            num_bufr = len(container_indices)
            num_prep = len(prep_indices)

            first_container_index = container_indices[0]

            if num_bufr == 1:

                # Unique BUFR observation:
                # associate all PREPBUFR events (1 or 2).
                if num_prep not in (1, 2):
                    print(
                        "WARNING: Unexpected unique BUFR/PREPBUFR count:",
                        "key =", key,
                        "BUFR =", num_bufr,
                        "PREPBUFR =", num_prep
                    )
                    continue

                matches = prep_indices

            elif num_bufr == 2:

                if num_prep == 2:

                    # B1 -> P1
                    # B2 -> P2
                    #
                    # Keep only B1/P1.
                    matches = prep_indices[:1]

                elif num_prep == 4:

                    # B1 -> P1,P2
                    # B2 -> P3,P4
                    #
                    # Keep only B1/P1,P2.
                    matches = prep_indices[:2]

                else:

                    print(
                        "WARNING: Unexpected BUFR/PREPBUFR count:",
                        "key =", key,
                        "BUFR =", num_bufr,
                        "PREPBUFR =", num_prep
                    )
                    continue

            else:

                print(
                    "WARNING: More than two duplicate BUFRs:",
                    "key =", key,
                    "BUFR =", num_bufr,
                    "PREPBUFR =", num_prep
                )
                continue

            container_to_prepbufr[first_container_index] = matches


        # ============================================================
        # Determine which retained BUFR observations have:
        #
        #   1 PREPBUFR event
        #   2 PREPBUFR events
        #
        # We use these masks to construct two DataContainers:
        #
        #   slot0 = every matched BUFR observation once
        #   slot1 = observations requiring a second event
        #
        # final = slot0 + slot1
        # ============================================================

        num_container_obs = len(container_time)

        match_count = np.zeros(
            num_container_obs,
            dtype=np.int32
        )

        for container_index, prep_indices in container_to_prepbufr.items():

            match_count[container_index] = len(prep_indices)


        slot0_mask = (
            match_count > 0
        ).astype(np.int32)

        slot1_mask = (
            match_count == 2
        ).astype(np.int32)


        # ============================================================
        # Make a deep copy of the original container.
        #
        # get_sub_container() itself shares DataObjects with the
        # original container, so we immediately append it to an empty
        # DataContainer.  DataContainer.append() copies the DataObjects
        # when the destination is empty.
        # ============================================================

        original_subcontainer = container.get_sub_container([])

        base_container = bufr.DataContainer()
        base_container.append(original_subcontainer)


        # ============================================================
        # The original BUFR container can contain event-level fields
        # that we are going to replace with PREPBUFR event data.
        #
        # Remove those fields before constructing the final container.
        #
        # Observation-level fields are retained.
        # ============================================================

        event_fields = []

        for field in base_container.list():

            paths = base_container.get_paths(field)

            if any(
                path == '*/EVENT'
                for path in paths
            ):
                event_fields.append(field)


        for field in event_fields:
            base_container.remove(field)


        # ============================================================
        # SLOT 0
        #
        # Every matched BUFR observation occurs once.
        # ============================================================

        slot0 = bufr.DataContainer()
        slot0.append(base_container)

        slot0.apply_mask(slot0_mask)


        # ============================================================
        # SLOT 1
        #
        # Only BUFR observations having TWO PREPBUFR events occur here.
        #
        # This is the physical copy of the BUFR observation.
        # ============================================================

        slot1 = bufr.DataContainer()
        slot1.append(base_container)

        slot1.apply_mask(slot1_mask)


        # ============================================================
        # Construct the final BUFR container.
        #
        # This physically duplicates the required BUFR observations.
        #
        # Example:
        #
        #   original:       A B C
        #   events:         2 1 2
        #
        #   slot0:          A B C
        #   slot1:          A   C
        #
        #   final:          A B C A C
        # ============================================================

        final_container = bufr.DataContainer()

        final_container.append(slot0)
        final_container.append(slot1)


        # Replace the working container.
        container = final_container

        print("==========================================")
        print("FINAL CONTAINER")
        print("container.size():", container.size())
        fields = container.list()

        for field in [
            'latitude',
            'longitude',
            'height_bufr',
            'airTemperature',
            'airPressure_bufr',
        ]:
            if field in fields:
                print(
                    field,
                    "shape =", container.get(field).shape,
                    "paths =", container.get_paths(field)
                )

        # ============================================================
        # Build PREPBUFR indices in EXACTLY the same order as the
        # physical BUFR container above.
        #
        # slot0:
        #   first PREPBUFR event for every retained BUFR observation
        #
        # slot1:
        #   second PREPBUFR event for every BUFR observation having 2
        #   events
        # ============================================================

        keep_container_indices = np.array(
            sorted(container_to_prepbufr.keys()),
            dtype=int
        )

        slot0_prepbufr_indices = np.array(
            [
                container_to_prepbufr[container_index][0]
                for container_index in keep_container_indices
                if len(container_to_prepbufr.get(container_index, [])) >= 1
            ],
            dtype=int
        )

        slot1_prepbufr_indices = np.array(
            [
                container_to_prepbufr[container_index][1]
                for container_index in keep_container_indices
                if len(container_to_prepbufr.get(container_index, [])) == 2
            ],
            dtype=int
        )

        prepbufr_indices = np.concatenate(
            [
                slot0_prepbufr_indices,
                slot1_prepbufr_indices
            ]
        )


        # ============================================================
        # Sanity check.
        # ============================================================

        print("==========================================")
        print("NE matched BUFR observations:",
              np.count_nonzero(match_count > 0))

        print("NE BUFR observations with 1 PREPBUFR event:",
              np.count_nonzero(match_count == 1))

        print("NE BUFR observations with 2 PREPBUFR events:",
              np.count_nonzero(match_count == 2))

        print("NE final BUFR observations:",
              container.size())

        print("NE final PREPBUFR events:",
              len(prepbufr_indices))

        if container.size() != len(prepbufr_indices):
            raise RuntimeError(
                "FATAL: Final BUFR/PREPBUFR dimension mismatch: "
                f"{container.size()} BUFR observations vs "
                f"{len(prepbufr_indices)} PREPBUFR events"
            )


        # ============================================================
        # Add PREPBUFR event data.
        #
        # These fields now correspond one-for-one with the physical
        # BUFR observations in the final container.
        # ============================================================

        for var in [
            'airPressure_prepbufr',
            'pressureMeanSeaLevel_prepbufr',
            'height_prepbufr',
            'stationElevation_prepbufr',
            'dewPointTemperature_prepbufr',
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
            'stationIdentification_prepbufr',
            'prepbufrDataLevelCategory',
            #'temperatureEventCode',
            'temperatureEventCode1',
            'temperatureEventCode2',
            'temperatureEventCode3',
            'specificHumidityEventCode',
            'observationSubTypeNum',
            'obsType'
        ]:

            quality_flags = (
                prepbufr_container.get(var)[prepbufr_indices]
            )

            # A PREPBUFR event is now associated with exactly one
            # final BUFR observation.
            #
            # If the selected PREPBUFR data is one-dimensional,
            # explicitly make it (Location, EVENT=1).
            if quality_flags.ndim == 1:
                quality_flags = quality_flags[:, np.newaxis]

            container.add(
                var,
                quality_flags,
                ['*', '*/EVENT']
            )

        print("==========================================")
        print("PREPBUFR FIELDS")

        for field in [
            'observationSubTypeNum',
            #'temperatureEventCode',
            'temperatureEventCode1',
            'temperatureEventCode2',
            'temperatureEventCode3',
            'specificHumidityEventCode',
            'airPressureQuality',
            'airTemperature',
        ]:
            print(
                field,
                "shape =", container.get(field).shape,
                "paths =", container.get_paths(field)
            )


        print("NE D")
        print("elevation check for sfcshp that is not sfclnd and not great lakes")

        container_late = container.get('latitude')
        container_lone = container.get('longitude')
        container_eleve = container.get('height_bufr')

        # T29 is now the PREPBUFR event associated with each
        # final BUFR observation.
        #
        # The PREPBUFR indices were constructed in exactly the
        # same order as the final DataContainer:
        #
        #   slot0 BUFR rows -> slot0 PREPBUFR events
        #   slot1 BUFR rows -> slot1 PREPBUFR events
        #
        container_t29e = (
            prepbufr_container
            .get('observationSubTypeNum')[prepbufr_indices]
        )

        if container_t29e.ndim > 1:
            container_t29e = np.squeeze(container_t29e)

        if len(container_late) != len(container_t29e):
            raise RuntimeError(
                "FATAL: Elevation-check dimensions do not match: "
                f"latitude={len(container_late)}, "
                f"longitude={len(container_lone)}, "
                f"height={len(container_eleve)}, "
                f"T29={len(container_t29e)}"
            )

        eleve = self._elevation_check_sfcshp(
            container_late,
            container_lone,
            container_eleve,
            container_t29e
        )

        container.replace(
            'height_bufr',
            eleve
        )


        #eleve = self._elevation_check_sfcshp(container_late, container_lone, container_eleve, container_t29e)
        #container.replace('height_bufr', eleve)
        print("NE E")
        # Altimeter and other things 
        container_alt = container.get('altimeter_bufr')#.filled()
        container_pres5 = container.get('airPressure_bufr')#.filled()
        container_elev5 = container.get('height_bufr')#.filled()

        missingvalue = bufr.get_missing_value(container_pres5.dtype)
        pres5_masked = ma.masked_invalid(ma.masked_equal(container_pres5, missingvalue))
        
        # Compute newpres (derive_p_fr_a should handle missing/masked internally)
        newpres = self._derive_p_fr_a(container_alt, container_pres5, container_elev5)
        newpres_masked = ma.masked_invalid(ma.masked_equal(newpres, missingvalue))
        
        # Safe fallback from reported pressure to altimeter-derived pressure
        pres5_is_valid = ~ma.getmaskarray(pres5_masked)
        newpres2_data = np.where(pres5_is_valid, pres5_masked, newpres_masked)
        newpres2_mask = ma.getmaskarray(pres5_masked) & ma.getmaskarray(newpres_masked)
        
        newpres2_masked = ma.array(newpres2_data, mask=newpres2_mask, fill_value=missingvalue)
        print("NE F")
        container.replace('airPressure_bufr', newpres2_masked)

 
        ############# PMSL for SFCLND 
        container_lat6 = container.get('latitude')#.filled()
        container_lon6 = container.get('longitude')#.filled()
        container_elev6 = container.get('height_bufr')#.filled()
        #container_t296 = container.get('observationSubTypeNum')#.filled()
        container_t296 = container_t29e
        container_temp6 = container.get('airTemperature')#.filled()
        container_airp6 = container.get('airPressure_bufr')#.filled()
        container_mslp6 = container.get('pressureMeanSeaLevel_bufr')#.filled()

        pstn_frompmsl_sfclnd = self._derive_pstn_frompmsl_sfclnd(
            container_lat6, container_lon6, container_elev6,
            container_t296, container_temp6, container_airp6, container_mslp6
        )
        pstn_masked = ma.masked_invalid(ma.masked_equal(pstn_frompmsl_sfclnd, missingvalue))
        pstn_masked = ma.masked_where((pstn_masked < 300.0) | (pstn_masked > 1100.0), pstn_masked)
        print("NE G")
        # 2. Check where newpres2 (Reported / Altimeter) is VALID
        pres2_is_valid = ~ma.getmaskarray(newpres2_masked)
        
        # 3. IF pres2 is valid -> use newpres2_masked
        #    ELSE -> fall back to pstn_masked (PMSL derived)
        newpres3_data = np.where(pres2_is_valid, newpres2_masked, pstn_masked)
        newpres3_mask = ma.getmaskarray(newpres2_masked) & ma.getmaskarray(pstn_masked)
        
        newpres3 = ma.array(newpres3_data, mask=newpres3_mask, dtype=container_alt.dtype)
        newpres3 = ma.masked_invalid(newpres3) # Catch any leftover NaNs from np.where selection
        final_array_for_zarr = newpres3.filled(missingvalue)

        # Update container with final pressure
        container.replace('airPressure_bufr', final_array_for_zarr)

         ########## TRY DOWN HERE
        container_airt2 = container.get('airTemperature').filled()
        container_rh2 = container.get('relativeHumidity').filled()
        container_pres2 = container.get('airPressure_bufr').filled()
        container_dewp2 = container.get('dewPointTemperature').filled()

        print("NE h")

        #print(f"DTYPE {container_dewp2.dtype}")
        missing_value=bufr.get_missing_value(container_dewp2.dtype)
        q_rh2 = self._derive_specifichumidity_rh(container_airt2, container_rh2, container_pres2).astype('float32')
        q_dewp2 = self._derive_specifichumidity_dewp(container_airt2, container_dewp2, container_pres2).astype('float32')

        rh_missing2 = ma.getmaskarray(q_rh2).sum()
        dewp_missing2 = ma.getmaskarray(q_dewp2).sum()

        #print("which missing rh2 dewp2", rh_missing2, dewp_missing2)
        # set q
        q2 = ma.where(~ma.getmaskarray(q_dewp2), q_dewp2, q_rh2) # fill in q_dewp with q_rh where q_dewp is missing 
        #print("double check 2 q2", q2.max(), q2.min(), len(q2))
        missing_q2 = bufr.get_missing_value(q2.dtype)
        #print(" NE len q2 ", len(q2))
        #container.add('specificHumidity_new2', q2.filled(missing_q2), ['*', '*/EVENT']) 
        container.add('specificHumidity_new', q2.filled(missing_value), ['*', '*/EVENT'])



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
                'name': 'dewPointTemperature_prepbufr',
                'source': 'dewPointTemperature_prepbufr',
                'longName': 'dewPointTemperature_prepbufr',
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
                'name': "prepbufrDataLevelCategory",
                'source': "variables/prepbufrDataLevelCategory",
                'longName': "prepbufrDataLevelCategory",
            },
            {
                'name': "obsType",
                'source': 'obsType',
                'longName': "ObsType",
            },
            {
                'name': "observationSubTypeNum",
                'source': "observationSubTypeNum",
                'longName': "Observation SubType Number",
            },
            {
                'name': "stationIdentification_prepbufr",
                'source': "stationIdentification_prepbufr",
                'longName': "stationIdentification_prepbufr",
            },
#            {
#                'name': "temperatureEventCode",
#                'source': 'temperatureEventCode',
#                'longName': 'temperatureEventCode',
#            },
            {
                'name': "temperatureEventCode1",
                'source': 'temperatureEventCode1',
                'longName': 'temperatureEventCode',
            },
            {
                'name': "temperatureEventCode2",
                'source': 'temperatureEventCode2',
                'longName': 'temperatureEventCode',
            },
            {
                'name': "temperatureEventCode3",
                'source': 'temperatureEventCode3',
                'longName': 'temperatureEventCode',
            },
            {
                'name': "specificHumidityEventCode",
                'source': 'specificHumidityEventCode',
                'longName': 'specificHumidityEventCode',
            }
        ])

        print("NE I")

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
        time = (np.round(time / 10) * 10).astype('int64')
        #for i in range(len(time)):
        #    print(f"NEtimeb {time[i]}")
        container.add('timestamp', time, ['*'])



add_main_functions(RawAdpsfcBuilder)
