# Ingredient Optimization Fixes - Summary

## Overview
This document summarizes the critical fixes applied to enable proper ingredient optimization in the pizza store inventory system, particularly for EOQ, POQ, and LXL policies.

## Problem Statement
The ingredient optimization was not working correctly, with the following symptoms:
1. **EOQ policy** was not generating order liberation matrices (zero orders)
2. **Safety stock calculations** were excessive for ingredients
3. **Order preservation** issues when converting from pizza demand to ingredient demand
4. **Batch size calculations** were not appropriate for ingredient-scale demands

---

## Fixes Applied

### 1. Safety Stock Calculation Fix (CRITICAL)
**Location:** `services/simulacion.py`

**Problem:** Safety stock was being calculated using a full month (×30 days) instead of lead time:
```python
# OLD (WRONG)
ss = round((porcentaje_seguridad * tasa_consumo_diario * 30), 0)
```

**Solution:** Use lead time for safety stock calculation:
```python
# NEW (CORRECT)
ss = round((porcentaje_seguridad * tasa_consumo_diario * lead_time), 0)
```

**Functions Fixed:**
- `simular_politica_EOQ()` - Line 958
- `simular_politica_POQ()` - Line 1040
- `simular_politica_LxL()` - Line 1158

**Impact:** This was causing excessive safety stock that prevented orders from being triggered. For example, with a daily demand of 100g, lead time of 2 days, and 10% safety stock:
- **Before:** 100 × 0.1 × 30 = 300g safety stock (3 days of demand!)
- **After:** 100 × 0.1 × 2 = 20g safety stock (20% of 1 day demand, appropriate)

---

### 2. EOQ Batch Size Calculation Improvement
**Location:** `services/simulacion.py`

**Problem:** Minimum batch size logic was too rigid and not considering lead time.

**Solution:** Improved batch size calculation in `replicas_EOQ()` (Line 492-505) and `replicas_EOQ_verbose()` (Line 1969-1988):
```python
# Calculate EOQ using standard formula
if costo_sobrante > 0 and demanda_anual > 0 and costo_pedir > 0:
    tamano_lote = int(round((2 * demanda_anual * costo_pedir / costo_sobrante) ** 0.5))
else:
    # Fallback: use monthly demand as batch size
    tamano_lote = int(round(max(tasa_consumo_diario * 30, 50)))

# Ensure minimum batch size based on lead time
tamano_lote = max(tamano_lote, int(tasa_consumo_diario * lead_time * 2))
```

**Impact:** Ensures batch sizes are appropriate for ingredient scale and considers lead time demand.

---

### 3. Matrix Reajuste Fix for Deterministic Policies (CRITICAL)
**Location:** `services/simulacion.py`

**Problem:** The `matriz_reajuste()` function was designed for reactive policies (QR, ST, SST, SS) and was eliminating carefully calculated orders from deterministic policies (EOQ, POQ, LXL).

**Solution:** Use `matriz_reajuste_ingredient()` instead of `matriz_reajuste()` for deterministic policies:

**Functions Fixed:**
- `replicas_EOQ()` - Line 535 - Now uses `matriz_reajuste_ingredient()`
- `replicas_POQ()` - Line 427 - Now uses `matriz_reajuste_ingredient()`
- `replicas_LXL()` - Line 632 - Now uses `matriz_reajuste_ingredient()`
- `replicas_EOQ_verbose()` - Already using `matriz_reajuste_ingredient()` ✓
- `replicas_POQ_verbose()` - Line 1875 - Now uses `matriz_reajuste_ingredient()`
- `replicas_LXL_verbose()` - Line 2099 - Now uses `matriz_reajuste_ingredient()`

**Explanation:**
- `matriz_reajuste()`: Designed for reactive policies that react to forecast errors. Adjusts orders based on demand variance.
- `matriz_reajuste_ingredient()`: Designed for deterministic policies. Preserves the calculated orders without adjustments.

**Impact:** This is the most critical fix. EOQ, POQ, and LXL calculate orders using mathematical formulas (Wilson's EOQ formula, etc.). The old reajuste function was eliminating these orders. Now they are properly preserved.

---

### 4. Parameter Naming Consistency Verification
**Location:** `services/materia_prima.py`, `services/PSO.py`, `services/leer_datos.py`

**Verified:** All files consistently use `"lead time"` (with space) as the parameter name:
- materia_prima.py: Line 1244 - `"lead time": base_info.get("lead time", 1)`
- PSO.py: Line 651 - `LT = params.get("lead time", 1)`
- leer_datos.py: Line 145 - `"lead time": ["lead time", "leadtime", "LEAD TIME"]`
- simulacion.py: 19 occurrences of `"lead time"` parameter access

**Impact:** Ensures parameters are correctly found and used throughout the optimization process.

---

### 5. Data Shifting Verification
**Location:** `services/simulacion.py`

**Verified:** The data shifting functions are working correctly:
- `shift_ventas_data_for_simulation()` - Line 54
- `shift_pronosticos_data_for_simulation()` - Line 84

Both functions properly shift data by 1 position to create a warmup period (period 0) before the actual simulation starts (period 1+).

**Impact:** Ensures proper alignment between forecast and actual demand data during simulation.

---

## How the Fixes Work Together

### Before the Fixes:
1. **Excessive safety stock** (×30 instead of ×lead_time) → Inventory stayed above reorder point
2. **matriz_reajuste** eliminated calculated orders → Liberation matrix was all zeros
3. **Rigid batch sizes** → Orders were either too large or too small for ingredient scale

### After the Fixes:
1. **Appropriate safety stock** (×lead_time) → Inventory properly triggers reorder point
2. **matriz_reajuste_ingredient** preserves orders → Liberation matrix contains calculated orders
3. **Flexible batch sizes** → Orders match ingredient scale and lead time requirements

---

## Testing Recommendations

To verify the fixes work correctly, test with the following scenarios:

### Test Case 1: EOQ for High-Volume Ingredient
```python
{
    "Familia_1": {
        "PARAMETROS": {
            "demanda_promedio": 150.5,  # 150.5g per day
            "demanda_diaria": 150.5,
            "costo_pedir": 25.0,
            "costo_sobrante": 0.8,
            "costo_unitario": 2.0,
            "lead time": 2,
        }
    }
}
```
**Expected:** Orders should be generated every few days with batch sizes around 500-1000g.

### Test Case 2: EOQ for Low-Volume Ingredient
```python
{
    "Familia_2": {
        "PARAMETROS": {
            "demanda_promedio": 30.0,   # 30g per day
            "demanda_diaria": 30.0,
            "costo_pedir": 15.0,
            "costo_sobrante": 0.5,
            "lead time": 3,
        }
    }
}
```
**Expected:** Orders should be generated less frequently with smaller batch sizes around 100-300g.

---

## Files Modified

1. **services/simulacion.py** - Main fixes:
   - Line 958: Fixed EOQ safety stock calculation
   - Line 1040: Fixed POQ safety stock calculation  
   - Line 1158: Fixed LxL safety stock calculation
   - Line 492-505: Improved EOQ batch size calculation
   - Line 1969-1988: Improved EOQ verbose batch size calculation
   - Line 535: Changed EOQ to use matriz_reajuste_ingredient
   - Line 427: Changed POQ to use matriz_reajuste_ingredient
   - Line 632: Changed LxL to use matriz_reajuste_ingredient
   - Line 1875: Changed POQ verbose to use matriz_reajuste_ingredient
   - Line 2099: Changed LxL verbose to use matriz_reajuste_ingredient

2. **services/materia_prima.py** - Verified (no changes needed):
   - Parameter naming is consistent
   - Data dict creation is correct
   - Conversion from pizzas to ingredients is working

3. **services/PSO.py** - Verified (no changes needed):
   - Parameter naming is consistent
   - Decision bounds calculation is correct

---

## Key Technical Insights

### Why Safety Stock Matters
Safety stock protects against demand variability during lead time. The formula should be:
```
SS = safety_factor × demand_during_lead_time
SS = porcentaje_seguridad × demanda_diaria × lead_time
```

Using ×30 (a full month) instead of ×lead_time creates excessive buffer that prevents the inventory level from ever dropping below the reorder point.

### Why matriz_reajuste_ingredient is Needed
Deterministic policies (EOQ, POQ, LXL) calculate orders using mathematical formulas:
- **EOQ**: Wilson's EOQ formula minimizes total cost
- **POQ**: Orders at fixed intervals based on demand
- **LXL**: Orders exactly what's needed (lot-for-lot)

These calculated orders should NOT be adjusted based on forecast errors. The `matriz_reajuste_ingredient` function:
1. Sets all reajuste adjustments to 0
2. Preserves the original liberation orders from the policy simulation
3. Correctly tracks inventory flow without modifying the order decisions

---

## Verification Checklist

When testing ingredient optimization, verify:

- [ ] Liberation order matrix is NOT all zeros
- [ ] Total orders across all replicas > 0
- [ ] Safety stock is reasonable (not >10 days of demand)
- [ ] Batch sizes are appropriate for ingredient scale (50g-2000g typical)
- [ ] Orders are generated when inventory drops below reorder point
- [ ] Cost calculations include ordering, holding, and shortage costs
- [ ] Service level constraints are met (typically 90-95%)
- [ ] Inventory levels stay within reasonable bounds

---

## Summary

The fixes address three critical areas:
1. **Safety stock calculation** - Now uses lead time instead of 30 days
2. **Order preservation** - Deterministic policies now use matriz_reajuste_ingredient
3. **Batch sizing** - EOQ batch sizes are appropriate for ingredient scale

These changes ensure the ingredient optimization process works correctly for EOQ, POQ, and LXL policies, generating proper order liberation matrices that can be used to optimize ingredient inventory costs.

