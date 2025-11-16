# Storage Key System - Optimization Results

## Problem Solved
Previously, optimization results were stored using only `Familia_{cluster_id}` as the key, which caused **data overwriting** when optimizing different ingredient sets or different punto de venta.

## New Solution: Composite Keys

### Storage Key Format

#### For Segundo Eslabón (Points of Sale / Ingredients):
```
{punto_venta}_{ingredient_code}_{policy}
```

**Examples:**
- `Terraplaza_MM001_EOQ` - Pollo from Terraplaza optimized with EOQ
- `Torres_MM002_QR` - Queso from Torres optimized with QR
- `Terraplaza_MM001_POQ` - Same ingredient (Pollo), same PV, different policy

#### For Primer Eslabón (Factory / Raw Materials):
```
Fabrica_{raw_material_code}_{policy}
```

**Examples:**
- `Fabrica_SAL_EOQ` - Salt optimized with EOQ
- `Fabrica_PEPPER_QR` - Pepper optimized with QR

## Benefits

### ✅ No Overwriting
- Run optimization for Terraplaza ingredients → saved
- Run optimization for Torres ingredients → saved separately
- Both sets of results are **preserved**

### ✅ Multiple Policies per Ingredient
- Test EOQ for Pollo → saved as `Terraplaza_POLLO_EOQ`
- Test QR for Pollo → saved as `Terraplaza_POLLO_QR`
- Compare results for the **same ingredient** with different policies

### ✅ Multiple Clustering Sessions
- Cluster `{POLLO, QUESO, CARNE}` and optimize → saved with ingredient codes
- Later cluster `{HARINA, TOMATE, SAL}` and optimize → saved with different codes
- **No conflicts**

### ✅ Ready for Primer Eslabón
- Optimize all ingredients from Terraplaza → saved with `Terraplaza_*` keys
- Optimize all ingredients from Torres → saved with `Torres_*` keys
- Primer eslabón can aggregate **both** sets of results automatically

## Data Structure Stored

```python
st.app_state[st.STATE_OPTIMIZATION_RESULTS][storage_key] = {
    'policy': 'EOQ',
    'optimization_result': {...},  # Full PSO results
    'cluster_info': {...},  # Clustering information
    'timestamp': pd.Timestamp.now(),
    'punto_venta_usado': 'Terraplaza',  # For filtering
    'ingredient_code': 'MM001',  # For identification
    'eslabon': 'segundo'  # 'primero' or 'segundo'
}
```

## Retrieval Logic - ENHANCED ⚡

### For Primer Eslabón (Factory Optimization)
The `get_second_eslabon_liberation_orders()` function in `primer_eslabon.py` uses a **two-method approach** for fast and reliable retrieval:

#### Method 1: Direct Key Prefix Lookup (Primary - Fast ⚡)
1. **Creates PV prefix**: `f"{punto_venta}_"` (e.g., `"Terraplaza_"`)
2. **Filters keys by prefix**: `if key.startswith(pv_prefix)`
3. **Directly extracts liberation matrices** using stored metadata
4. **Returns**: `{ingredient_code: liberation_matrix}` dictionary

**Advantages:**
- ✅ **Fast**: O(n) single pass, no nested iterations
- ✅ **Reliable**: Uses direct string matching on keys
- ✅ **Clear**: Keys like `Terraplaza_MM001_EOQ` are self-documenting

#### Method 2: Metadata Fallback (Backward Compatibility)
If Method 1 finds no results (e.g., old data format):
1. **Iterates all keys** in `STATE_OPTIMIZATION_RESULTS`
2. **Checks metadata fields**: `punto_venta_usado`, `ingredient_code`
3. **Falls back to nested result structure** if needed

This allows factory optimization to easily aggregate ingredients from:
- **Terraplaza**: All keys starting with `Terraplaza_*` → extracted instantly
- **Torres**: All keys starting with `Torres_*` → extracted instantly

### Validation Logic - ENHANCED ⚡

The `validate_second_eslabon_optimization_complete()` function now:
1. **Shows all stored keys** for debugging
2. **Counts ingredients per PV** (e.g., "Terraplaza: 3 ingrediente(s)")
3. **Uses composite key prefix** for faster validation
4. **Clear terminal output** showing exactly what's available

### Helper Function: List Available Ingredients

New function `list_available_ingredient_optimizations()` provides:
- **Organized view** of all ingredients by PV
- **Policy information** for each ingredient
- **Useful before** running factory optimization to verify data availability

Example output:
```
📊 INGREDIENTES OPTIMIZADOS DISPONIBLES:
============================================================

  🏪 Terraplaza: 3 ingredientes
     • MM001_EOQ
     • MM002_EOQ
     • MM003_QR

  🏪 Torres: 2 ingredientes
     • MM001_EOQ
     • MM004_POQ

============================================================
```

## Example Workflow

### Step 1: Optimize Terraplaza Ingredients
```
User actions:
1. Select "Eslabón 2 - Puntos de Venta"
2. Select "Terraplaza"
3. Cluster ingredients: {POLLO, QUESO, CARNE}
4. Optimize Familia_1 (rep: POLLO) with EOQ

Stored as: "Terraplaza_MM001_EOQ"
```

### Step 2: Optimize Torres Ingredients
```
User actions:
1. Select "Eslabón 2 - Puntos de Venta"
2. Select "Torres"
3. Cluster ingredients: {POLLO, QUESO, CARNE}
4. Optimize Familia_1 (rep: POLLO) with EOQ

Stored as: "Torres_MM001_EOQ"
```

### Step 3: Optimize Factory (Primer Eslabón)
```
User actions:
1. Select "Eslabón 1 - Fábrica"
2. Cluster raw materials: {SAL, PEPPER, SPICES}
3. Optimize Familia_1 with EOQ

System automatically:
- Retrieves "Terraplaza_MM001_EOQ" liberation orders
- Retrieves "Torres_MM001_EOQ" liberation orders
- Converts both to raw material demands
- Aggregates demands: Terraplaza + Torres
- Optimizes factory inventory
```

## Terminal Output

When storing, you'll see:
```
✅ Results stored with key: 'Terraplaza_MM001_EOQ'
```

When retrieving for factory optimization, you'll see:
```
📥 OBTENCIÓN: Órdenes de liberación segundo eslabón
   Punto de venta: Terraplaza
   🔍 Revisando 5 resultados almacenados
   🔑 Claves disponibles: ['Terraplaza_MM001_EOQ', 'Terraplaza_MM002_EOQ', 'Torres_MM001_EOQ', ...]
   ✅ Ingrediente 'POLLO' (MM001): (30, 100)
   ✅ Ingrediente 'QUESO' (MM002): (30, 100)
   📦 Total ingredientes encontrados: 2
```

## Summary

This new storage system ensures that:
- ✅ Each optimization result has a **unique identifier**
- ✅ Multiple PVs can be optimized **independently**
- ✅ Results are **preserved across sessions**
- ✅ Factory optimization can **aggregate all ingredient demands**
- ✅ Users can **compare different policies** for the same ingredient

No more data loss! 🎉

