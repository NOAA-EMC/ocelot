# FSOI Radiosonde Temperature Impact Study - Quick Reference

## Objective
Determine how **all observation types** (satellite + conventional) impact **radiosonde temperature forecast error**.

---

## Configuration Summary

**Input observations:** All types (ATMS, AMSUA, surface, radiosonde, aircraft, etc.)
**Target metric:** Radiosonde temperature forecast error only
**Forecast lead time:** +12 hours (step 0)
**Analysis:** Attribution to each input observation type/channel

---

## Quick Start

### 1. Test the Setup (5-10 minutes)
```bash
cd /scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/FSOI/ocelot/gnn_model

# Run validation tests
python FSOI/test_fsoi.py \
   --checkpoint checkpoints/
```

Expected output:
- ✓ Test 1: FSOI configuration loads correctly
- ✓ Test 2: Sequential dataset creates valid pairs
- ✓ Test 3: Gradient requirements set properly
- ✓ Test 4: Gradient computation successful
- ✓ Test 5: FSOI computation runs
- ✓ Test 6: Gradient verification passes

---

### 2. Run FSOI Computation

#### Option A: Interactive Testing (1-2 hours for 1 week)
```bash
cd /scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/FSOI/ocelot/gnn_model

python FSOI/fsoi_inference.py \
   --config FSOI/configs/fsoi_config_radiosonde_temp.yaml \
   --obs_config configs/observation_config.yaml \
   --checkpoint checkpoints/ \
   --start_date 2024-01-01 \
   --end_date 2024-01-07
```

#### Option B: Submit Batch Job (recommended for longer periods)
```bash
# Edit FSOI/scripts/run_fsoi_radiosonde_temp.sh to set:
# - Date range
# - Your email for notifications
# - Conda environment path

sbatch FSOI/scripts/run_fsoi_radiosonde_temp.sh
```

---

## Understanding the Results

### Output Files

The results are saved in `FSOI/fsoi_outputs/radiosonde_temp_impact/` (CSVs under `csv/`):

1. **csv/fsoi_by_instrument.csv** - Impact by observation type
   ```
   instrument,mean_impact,total_impact,count,percent_beneficial
   radiosonde,-0.0234,100,4280,98.2
   ATMS_ch7,-0.0156,80,5120,95.3
   surface_temp,-0.0089,40,4520,89.1
   AMSUA_ch5,-0.0071,35,4890,87.4
   ...
   ```

   **Interpretation:**
   - Negative impact = **beneficial** (reduces forecast error)
   - Positive impact = **detrimental** (increases forecast error)
   - Larger absolute value = stronger impact

2. **csv/fsoi_by_channel.csv** - Impact by specific channel/level
   ```
   instrument,channel,pressure_level,mean_impact,count
   radiosonde,0,850,−0.0345,520
   ATMS,7,NULL,-0.0156,640
   radiosonde,1,500,-0.0142,518
   ...
   ```

3. **csv/fsoi_by_time.csv** - Daily aggregates
   ```
   date,mean_impact,total_impact,count
   2024-01-01,-0.0198,120,6050
   2024-01-02,-0.0203,125,6180
   ...
   ```

---

## Key Questions Answered

### 1. Which observation types most reduce temperature forecast error?
Look at `fsoi_by_instrument.csv` sorted by `mean_impact`:
- Most negative = most beneficial
- Compare satellite vs. conventional

### 2. Which satellite channels are most important?
Look at `fsoi_by_channel.csv`:
- ATMS channels 5-8: temperature sounders (mid-upper troposphere)
- AMSUA channels 5-10: temperature sounders
- Compare to moisture channels

### 3. How does impact vary by pressure level?
For radiosondes in `fsoi_by_channel.csv`:
- Look at different pressure levels (1000, 850, 500, 250, 100 hPa)
- Upper levels vs. lower levels

### 4. Are conventional or satellite observations more important?
Aggregate by type:
```bash
# Sum conventional obs (surface, radiosonde, aircraft)
# vs. satellite obs (ATMS, AMSUA, AIRS, etc.)
```

---

## Expected Results

Based on typical FSOI studies for temperature forecasts:

1. **Strong Negative Impact (Very Beneficial):**
   - Other radiosondes (direct temperature measurement)
   - Temperature-sensitive satellite channels:
     - ATMS channels 5-8 (peak ~500-200 hPa)
     - AMSUA channels 5-10 (stratified levels)
   - Aircraft temperature reports

2. **Moderate Negative Impact (Beneficial):**
   - Surface temperature observations (for lower troposphere)
   - Some water vapor channels (indirect T constraint)

3. **Weak or Neutral Impact:**
   - Moisture-sensitive channels (less directly related to T)
   - Window channels (mostly surface emission)

4. **Potential Positive Impact (Detrimental):**
   - Observations with large errors
   - Observations from poorly represented regions
   - Conflicting observations

---

## Validation Checks

The first run includes **finite-difference gradient validation**:

```
[VALIDATION] Running finite-difference gradient check...
[FD Test 1/5] radiosonde obs 123, channel 2
  Autograd gradient: -0.003421
  Finite-diff gradient: -0.003419
  Relative error: 0.06% ✓
...
[VALIDATION] Finite-difference check PASSED ✓
```

If validation passes:
- Gradients are mathematically correct
- FSOI values are trustworthy
- Disable FD check for production runs (much faster)

If validation fails:
- Review implementation
- Check model/data alignment
- Contact developer

---

## Troubleshooting

### No observations in some windows
```
[WARNING] No analysis inputs found, skipping this pair
```
**Solution:** Check data availability for your date range

### Shape mismatches
```
[WARNING] Shape mismatch for radiosonde: pred=(50,10), ref=(50,8)
```
**Solution:** Check model configuration matches data

### Out of memory
```
CUDA out of memory
```
**Solutions:**
- Ensure `batch_size: 1`
- Use CPU for testing: `--device cpu`
- Reduce date range

---

## Next Steps

After this study:

1. **Extend analysis:**
   - Test other target variables (u_wind, v_wind, humidity)
   - Test different forecast lead times (+24h, +48h)
   - Regional analysis (tropics, mid-latitudes, polar)

2. **Data denial experiments:**
   - Remove least beneficial obs types
   - Retrain/test without those observations
   - Compare forecast skill

3. **Observation impact studies:**
   - Identify which channels to prioritize
   - Optimize observation network design
   - Guide satellite instrument development

---

## Configuration Customization

Edit `FSOI/configs/fsoi_config_radiosonde_temp.yaml`:

### Focus on specific pressure levels
```yaml
forecast:
  target_pressure_levels: [850, 500, 250, 100]  # Upper air only
```

### Include multiple variables
```yaml
forecast:
  target_variables: ["temperature", "u_wind", "v_wind"]
```

### Regional analysis
```yaml
aggregation:
  by_region: true
  regions:
    tropics:
      lat_range: [-30, 30]
```

### Longer time period
```yaml
data:
  start_date: "2024-01-01"
  end_date: "2024-12-31"  # Full year
```

---

## Reference

**FSOI Formula:**
```
FSOI(k) = δx(k) ⊙ (ga(k) + gb(k))

where:
  δx(k) = innovation (xa - xb)
  ga(k) = ∂e(xa)/∂xa  (analysis adjoint)
  gb(k) = ∂e(xb)/∂xb  (background adjoint)
  e(x)  = forecast error for radiosonde temperature
```

**Key Papers:**
- Langland & Baker (2004) - Original FSOI methodology
- Cardinali (2009) - ECMWF FSOI implementation
- Baker & Daley (2000) - Observation impact theory

---

## Contact

For questions or issues:
- Review `FSOI_FIXES.md` for implementation details
- Check `FSOI_QUICKSTART.md` for general guidance
- See `FSOI/ocelot/gnn_model/README.md` for full documentation
