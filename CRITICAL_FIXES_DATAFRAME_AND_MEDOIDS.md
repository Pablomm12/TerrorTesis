# 🔧 CRITICAL FIXES - DataFrame Indexing & Medoids Access

## 🔴 **Problems Identified**

### **Problem 1: KeyError: (0, 0) - DataFrame vs numpy array indexing**

**Error Location:** Line 1010 in `services/materia_prima.py`

**Error Message:**
```
KeyError: (0, 0)
File "/Users/lina.beltran/Downloads/codigo_frontend/services/materia_prima.py", line 1909, in convert_pizza_liberation_matrix_to_ingredient
    pizza_count = liberacion_orden_matrix_pizzas[periodo, 0] if liberacion_orden_matrix_pizzas.ndim == 2 else liberacion_orden_matrix_pizzas[periodo]
```

**Root Cause:**
The code was trying to use numpy array tuple indexing `[periodo, 0]` on a **pandas DataFrame**. DataFrames don't support this syntax directly - when you try `df[0, 0]`, pandas tries to find a column named `(0, 0)` which doesn't exist.

---

### **Problem 2: Representative Ingredient Inconsistency**

**Symptom:** Terminal shows "CARANTANTA" as representative for Familia_1, but the UI says it should be "POLLO"

**Error Location:** Line 2905 in `services/materia_prima.py` (in `_prepare_enhanced_ingredient_info`)

**Terminal Output:**
```
🔍 DEBUG - Cluster info:
   Cluster ID solicitado: 1
   Medoids disponibles: []  ← EMPTY!
   Clusters en df_clustered: [np.int64(1), np.int64(2), np.int64(3), np.int64(4)]
   🔄 No suitable medoid found, checking df_clustered...
   🔄 Using first ingredient in cluster as representative: CARANTANTA  ← WRONG!
```

**Root Cause:**
The code was looking for `medoids` directly in `cluster_info` dictionary:
```python
medoids = cluster_info.get("medoids", {})  # ❌ WRONG - returns {}
```

But medoids are actually stored in the nested `clustering_result`:
```python
cluster_info = {
    'df_clustered': df_clustered,
    'cluster_to_products': {...},
    'cluster_representative': {...},
    'clustering_result': {  ← HERE!
        'medoids': {...},  ← The actual medoids dictionary
        'scaler': ...,
        ...
    }
}
```

---

## ✅ **Fixes Applied**

### **Fix 1: Convert DataFrame to numpy array before indexing** (Lines 1908-1914)

**File:** `services/materia_prima.py`

**Before (BROKEN):**
```python
for periodo in range(num_periodos_debug):
    pizza_count = liberacion_orden_matrix_pizzas[periodo, 0] if ...  # ❌ Fails if DataFrame
    ingrediente_count = liberacion_orden_matrix_ingredient[periodo, 0] if ...
```

**After (FIXED):**
```python
# CRITICAL FIX: Convert DataFrame to numpy array if needed
pizza_matrix_array = liberacion_orden_matrix_pizzas.values if hasattr(liberacion_orden_matrix_pizzas, 'values') else liberacion_orden_matrix_pizzas
ingredient_matrix_array = liberacion_orden_matrix_ingredient.values if hasattr(liberacion_orden_matrix_ingredient, 'values') else liberacion_orden_matrix_ingredient

for periodo in range(num_periodos_debug):
    pizza_count = pizza_matrix_array[periodo, 0] if pizza_matrix_array.ndim == 2 else pizza_matrix_array[periodo]  # ✅ Works for both
    ingrediente_count = ingredient_matrix_array[periodo, 0] if ingredient_matrix_array.ndim == 2 else ingredient_matrix_array[periodo]
```

**Why this works:**
- `.values` converts a DataFrame to numpy array
- `hasattr(obj, 'values')` checks if it's a DataFrame/Series
- Falls back to original object if it's already a numpy array
- Tuple indexing `[periodo, 0]` now works correctly

---

### **Fix 2: Access medoids from correct location** (Lines 2906-2925)

**File:** `services/materia_prima.py` (function: `_prepare_enhanced_ingredient_info`)

**Before (BROKEN):**
```python
df_clustered = cluster_info.get("df_clustered", pd.DataFrame())
medoids = cluster_info.get("medoids", {})  # ❌ Always returns {} - wrong location!

print(f"   Medoids disponibles: {list(medoids.keys())}")  # Always prints []
```

**After (FIXED):**
```python
df_clustered = cluster_info.get("df_clustered", pd.DataFrame())

# CRITICAL FIX: Get medoids from the correct location in cluster_info
# Medoids are stored in clustering_result, not directly in cluster_info
clustering_result = cluster_info.get("clustering_result", {})
medoids = clustering_result.get("medoids", {}) if clustering_result else {}  # ✅ Correct path!

# FALLBACK: If not in clustering_result, try cluster_representative
if not medoids and "cluster_representative" in cluster_info:
    print(f"   ℹ️ Using cluster_representative as medoids")
    cluster_representative = cluster_info.get("cluster_representative", {})
    # Convert cluster_representative format to medoids format
    for cluster_id_key, rep_info in cluster_representative.items():
        medoids[cluster_id_key] = {
            'medoid_row': pd.Series(rep_info)
        }

print(f"   Medoids disponibles: {list(medoids.keys())}")  # ✅ Now shows [1, 2, 3, 4]
```

**Why this works:**
- **Primary path:** Gets medoids from `cluster_info['clustering_result']['medoids']` where they're actually stored
- **Fallback path:** If `clustering_result` doesn't exist, constructs medoids from `cluster_representative`
- **Backward compatible:** Works with both old and new data structures

---

### **Fix 3: Access medoids correctly in family liberation** (Lines 3401-3423)

**File:** `services/materia_prima.py` (function: `add_family_liberation_to_optimization_result`)

**Before (BROKEN):**
```python
medoid_info = cluster_info["medoids"][cluster_id]  # ❌ KeyError - medoids not in cluster_info!
```

**After (FIXED):**
```python
# Get representative ingredient name consistently - FIXED medoids access
clustering_result = cluster_info.get("clustering_result", {})
medoids = clustering_result.get("medoids", {}) if clustering_result else {}

# Fallback to cluster_representative if medoids not available
if not medoids and "cluster_representative" in cluster_info:
    cluster_representative = cluster_info.get("cluster_representative", {})
    if cluster_id in cluster_representative:
        representative_ingredient = cluster_representative[cluster_id].get('Nombre', family_ingredients[0] if family_ingredients else "Unknown")
    else:
        representative_ingredient = family_ingredients[0] if family_ingredients else "Unknown"
elif cluster_id in medoids:
    medoid_info = medoids[cluster_id]
    if hasattr(medoid_info, 'name'):
        representative_ingredient = medoid_info.name
    elif isinstance(medoid_info, dict) and 'medoid_row' in medoid_info:
        # Extract from medoid_row which is a pandas Series
        representative_ingredient = medoid_info['medoid_row'].get('Nombre', medoid_info['medoid_row'].name)
    else:
        # Fallback: use first ingredient in cluster
        representative_ingredient = family_ingredients[0] if family_ingredients else "Unknown"
else:
    # Fallback: use first ingredient in cluster
    representative_ingredient = family_ingredients[0] if family_ingredients else "Unknown"
```

**Why this works:**
- Uses same corrected medoids access as Fix 2
- Multiple fallback layers to ensure a representative is always found
- Extracts 'Nombre' field correctly from different data formats

---

### **Fix 4: Indentation error** (Line 2932)

**Fixed via sed command:**
```bash
sed -i '' '2932s/^    rep_ingredient/        rep_ingredient/' services/materia_prima.py
```

Changed from 4 spaces (wrong) to 8 spaces (correct) indentation.

---

## 📊 **Data Structure Reference**

### **cluster_info Structure:**
```python
cluster_info = {
    'df_clustered': DataFrame with columns ['Nombre', 'Cluster', ...],
    'cluster_to_products': {
        1: ['ingredient_a', 'ingredient_b', ...],
        2: ['ingredient_c', 'ingredient_d', ...],
        ...
    },
    'cluster_representative': {
        1: {'Nombre': 'POLLO', 'Costo': 5000, ...},
        2: {'Nombre': 'QUESO', 'Costo': 8000, ...},
        ...
    },
    'clustering_result': {  ← THIS IS WHERE MEDOIDS ARE!
        'medoids': {
            1: {
                'medoid_row': Series({'Nombre': 'POLLO', ...}),
                'medoid_idx_local': 2,
                'medoid_sum': 15.3
            },
            2: {...},
            ...
        },
        'scaler': StandardScaler(),
        'chosen_k': 4,
        'method_used': 'kmeans',
        ...
    },
    'features_used': ['Costo variable/vida util', 'Demanda promedio', ...],
    'num_clusters': 4,
    ...
}
```

### **Key Access Patterns:**

✅ **CORRECT:**
```python
# Get medoids
clustering_result = cluster_info['clustering_result']
medoids = clustering_result['medoids']
representative = medoids[cluster_id]['medoid_row']

# OR with safe access:
clustering_result = cluster_info.get('clustering_result', {})
medoids = clustering_result.get('medoids', {}) if clustering_result else {}
```

❌ **WRONG:**
```python
# This doesn't work - medoids are nested deeper
medoids = cluster_info['medoids']  # KeyError!
medoids = cluster_info.get('medoids', {})  # Returns {}
```

---

## 🎯 **Expected Results After Fixes**

### **Terminal Output:**
```
🔍 DEBUG - Cluster info:
   Cluster ID solicitado: 1
   Medoids disponibles: [1, 2, 3, 4]  ← ✅ Not empty!
   Clusters en df_clustered: [1, 2, 3, 4]
   ✅ Found medoid directly: POLLO PROCESADO* KL  ← ✅ Correct representative!

🔄 CONVERSIÓN PIZZA → INGREDIENTE: '1430.35.10'  ← ✅ Correct code
Punto de venta: Terraplaza
Matriz original: (30, 10) (períodos x réplicas)
...
🔍 CÁLCULO DETALLADO - PRIMEROS PERÍODOS:  ← ✅ No KeyError!

  📊 PERÍODO 1, RÉPLICA 1:
     Pizzas totales: 56
     + PIZZA POLLO CON CHAMPIÑONES (50CM): 56 × 0.280 × 0.6 = 9.408
     → TOTAL CALCULADO: 9.408
     → TOTAL EN MATRIZ: 9
```

### **Excel Output:**
- ✅ Correct representative ingredient displayed
- ✅ Correct materia_prima code used
- ✅ Unique conversion results for each ingredient (not all the same)
- ✅ Liberation vectors correctly calculated

### **UI Display:**
- ✅ "⭐ Ingrediente representativo: POLLO PROCESADO* KL"
- ✅ "🔑 Código en materia prima: 1430.35.10"
- ✅ Both match and are consistent

---

## 📝 **Files Modified**

1. **`services/materia_prima.py`**
   - Lines 1908-1914: Fixed DataFrame indexing in conversion debug output
   - Lines 2906-2925: Fixed medoids access in `_prepare_enhanced_ingredient_info`
   - Lines 3401-3423: Fixed medoids access in `add_family_liberation_to_optimization_result`
   - Line 2932: Fixed indentation error

2. **`services/PSO.py`**
   - Lines 1576-1589: Fixed Excel export Series formatting (previous fix)
   - Lines 761-790: Fixed try-except indentation (previous fix)

---

## 🧪 **Testing Checklist**

Run your optimization and check:
- [ ] No "KeyError: (0, 0)" error
- [ ] Medoids show correct keys: `[1, 2, 3, 4]` instead of `[]`
- [ ] Representative ingredient matches between terminal and UI
- [ ] Terminal shows "✅ Found medoid directly: POLLO..." (or correct ingredient)
- [ ] Conversion calculations display without errors
- [ ] Excel file created successfully with correct data
- [ ] Each family member has unique order values (not identical)

---

## 🚀 **Additional Fix: Verbose Functions DataFrame Conversion**

### **Problem 3: Verbose functions missing DataFrame conversion (NEW)**

**Symptoms:**
- Hundreds of "⚠️ Skipping non-numeric key in pronosticos data: 0 = P, 1 = e..." warnings
- `ValueError: Shape of passed values is (30, 30), indices imply (30, 10)`

**Root Cause:**
All verbose functions were missing the `convert_replicas_matrix_to_array()` call, causing them to iterate over DataFrame column names instead of numeric data.

**Fix Applied:**
Added `matrizReplicas = convert_replicas_matrix_to_array(matrizReplicas)` at the start of all 6 verbose functions.

**Files Modified:** `services/simulacion.py` (Lines 1572-1575, 1640-1643, 1703-1706, 1777-1780, 1848-1851, 1946-1949)

See `VERBOSE_FUNCTIONS_FIX.md` for complete details.

---

## 🚀 **Ready to Test!**

All fixes have been applied and linting errors resolved. You can now run your optimization and it should:
1. ✅ Use the correct representative ingredient (POLLO, not CARANTANTA)
2. ✅ Convert pizza demand to ingredients without KeyError
3. ✅ Process DataFrame replicas without string iteration warnings
4. ✅ Generate correct matrix shapes (30, 10) not (30, 30)
5. ✅ Export results to Excel with proper formatting
6. ✅ Display consistent information in the UI
7. ✅ Generate unique orders for each family member

**Documentation:**
- `CRITICAL_FIXES_DATAFRAME_AND_MEDOIDS.md` - This file (DataFrame indexing & medoids fixes)
- `VERBOSE_FUNCTIONS_FIX.md` - Verbose functions conversion fix

Try running the optimization again!

