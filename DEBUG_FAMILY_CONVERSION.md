# Debugging Family Ingredient Conversion

## Date: November 14, 2024

## 🎯 Your Concerns

1. **All family ingredients show SAME orders** → Should be DIFFERENT
2. **INPUT demand sheet shows wrong ingredient** → Should show representative

## 🔍 What to Look For in Terminal Output

### 1. Representative Ingredient Identification

```bash
# LOOK FOR THIS SECTION:
🎯 Optimizando Cluster 1
📋 Política: EOQ
⭐ Ingrediente representativo: POLLO  ← THIS SHOULD MATCH
📦 Ingredientes en el cluster: POLLO, TOCINO, JAMÓN
```

**CHECK**:
- ✅ Representative ingredient name is clear and recognizable
- ✅ All cluster ingredients are listed
- ✅ These are actual ingredient NAMES, not numbers like "1, 2, 3"

### 2. Family Liberation Generation

```bash
# LOOK FOR THIS SECTION:
🏭 FAMILY LIBERATION GENERATION
📦 Cluster ID: 1
⭐ Representative: POLLO
🏢 Pizza Punto Venta: Terraplaza
⚙️ Policy: EOQ
📈 Optimized params: {'porcentaje': 0.186}
```

**CHECK**:
- ✅ Representative matches the one from optimization
- ✅ Punto Venta is correct (not "Familia_1")

### 3. Individual Ingredient Processing (CRITICAL!)

For EACH ingredient in the family, you should see:

```bash
======================================================================
🧪 Processing ingredient: 'POLLO'
======================================================================
   🔍 Ingredient code type: <class 'str'>
   🔍 Ingredient code value: 'POLLO'
   🔍 Searching in materia_prima keys: ['POLLO', 'TOCINO', 'JAMÓN', ...]
   ✅ 'POLLO' FOUND in materia_prima  ← MUST SAY "FOUND"
   
   📊 Ingredient data_dict created:
      demanda_diaria: 156.42g  ← UNIQUE for POLLO
      demanda_promedio: 4692.60g
   
   🔄 Converting pizza replicas to POLLO replicas
   📊 Pizza replicas shape: (10, 30)
   📊 Pizza demand range: 3-8 pizzas
   
   ✅ Ingredient replicas shape: (10, 30)
   📊 Ingredient demand range: 120.50-210.30g  ← UNIQUE for POLLO
   📊 Average conversion factor: 15.6420g per pizza  ← UNIQUE for POLLO
   
   📈 Ingredient replicas statistics:
      Average: 156.42g  ← UNIQUE for POLLO
      Range: 120.50g - 210.30g
      Unique values: 300  ← Should be HIGH (different values)
      First 5 values of first replica: [145.2, 167.8, 134.5, 189.2, 172.3]
```

**Then for NEXT ingredient:**

```bash
======================================================================
🧪 Processing ingredient: 'TOCINO'
======================================================================
   🔍 Ingredient code type: <class 'str'>
   🔍 Ingredient code value: 'TOCINO'
   ✅ 'TOCINO' FOUND in materia_prima
   
   📊 Ingredient data_dict created:
      demanda_diaria: 89.23g  ← DIFFERENT from POLLO! ✅
      demanda_promedio: 2676.90g
   
   ✅ Ingredient replicas shape: (10, 30)
   📊 Ingredient demand range: 68.30-118.90g  ← DIFFERENT from POLLO! ✅
   📊 Average conversion factor: 8.9230g per pizza  ← DIFFERENT! ✅
   
   📈 Ingredient replicas statistics:
      Average: 89.23g  ← DIFFERENT from POLLO! ✅
      Range: 68.30g - 118.90g  ← DIFFERENT!
      Unique values: 300
      First 5 values of first replica: [82.8, 95.7, 76.7, 108.1, 98.2]  ← DIFFERENT! ✅
```

### 4. Liberation Final Vectors (CRITICAL!)

```bash
   🎯 LIBERATION FINAL for 'POLLO':
      Total orders: 4692g  ← Should be ~demanda_diaria × 30
      Periods with orders: 3
      Unique order values: 2  ← 0 and the order size
      First 10 periods: [0, 0, 0, 0, 1564, 0, 0, 0, 0, 0]
      Non-zero periods: [4, 13, 22]

   🎯 LIBERATION FINAL for 'TOCINO':
      Total orders: 2677g  ← DIFFERENT from POLLO! ✅
      Periods with orders: 3
      Unique order values: 2
      First 10 periods: [0, 0, 0, 0, 892, 0, 0, 0, 0, 0]  ← DIFFERENT! ✅
      Non-zero periods: [4, 13, 22]  ← Same timing (EOQ), but DIFFERENT quantities
```

## ❌ Problem Indicators

### Problem 1: Ingredient Not Found

```bash
❌ 'POLLO' NOT FOUND in materia_prima
🔍 Potential matches: ['Pollo Desmenuzado', 'Pollo en Cubos']
```

**CAUSE**: Ingredient name in clustering doesn't match materia_prima keys
**FIX**: The mapping code in PSO.py will try to find matches, but you may need to check ingredient names

### Problem 2: Same Conversion Factors

```bash
🧪 Processing ingredient: 'POLLO'
   📊 Average conversion factor: 15.6420g per pizza

🧪 Processing ingredient: 'TOCINO'
   📊 Average conversion factor: 15.6420g per pizza  ← SAME! ❌
```

**CAUSE**: Both ingredients mapping to same recipe or using same conversion
**FIX**: Check that recipes contain both ingredients with different quantities

### Problem 3: Identical Liberation Vectors

```bash
🎯 LIBERATION FINAL for 'POLLO':
   First 10 periods: [0, 0, 0, 0, 1564, 0, 0, 0, 0, 0]

🎯 LIBERATION FINAL for 'TOCINO':
   First 10 periods: [0, 0, 0, 0, 1564, 0, 0, 0, 0, 0]  ← IDENTICAL! ❌
```

**CAUSE**: Using same replicas_matrix for both (not converting individually)
**FIX**: Bug in conversion - need to investigate `create_replicas_matrix_for_ingredient`

### Problem 4: Numeric Codes Instead of Names

```bash
🧪 Processing ingredient: '1'  ← Should be 'POLLO'!
   🔍 Ingredient code type: <class 'int'>
   🔍 Ingredient code value: '1'
   ❌ '1' NOT FOUND in materia_prima
```

**CAUSE**: Clustering returning indices instead of names
**FIX**: Check that df_clustered has 'Nombre' column and it's being used

## ✅ Success Indicators

### All These Should Be TRUE:

1. **Ingredient Identification**:
   - ✅ Each ingredient shows as FOUND in materia_prima
   - ✅ Ingredient codes are names (strings), not numbers
   - ✅ All family ingredients are processed

2. **Unique Conversions**:
   - ✅ Each ingredient has DIFFERENT demanda_diaria
   - ✅ Each ingredient has DIFFERENT conversion factor
   - ✅ Each ingredient's replicas have DIFFERENT averages
   - ✅ Each ingredient's replicas have DIFFERENT ranges

3. **Unique Liberation Vectors**:
   - ✅ Each ingredient has DIFFERENT total orders
   - ✅ Each ingredient has DIFFERENT order quantities
   - ✅ Order TIMING may be same (EOQ uses same parameters)
   - ✅ But order SIZES must be proportional to demand

4. **Excel Export**:
   - ✅ INPUT sheet shows correct representative ingredient
   - ✅ FAMILIA_Resumen shows different "Vector_Final_Órdenes" for each
   - ✅ FAM_xxx sheets show different order values

## 🔬 Example: Correct Behavior

### Family with 3 ingredients:
- **POLLO**: 15g per pizza
- **TOCINO**: 8g per pizza  
- **JAMÓN**: 12g per pizza

### If pizza demand = 10 per period:

**Ingredient replicas (converted):**
- POLLO: 150g per period (10 pizzas × 15g)
- TOCINO: 80g per period (10 pizzas × 8g)
- JAMÓN: 120g per period (10 pizzas × 12g)

**EOQ orders (example with porcentaje=0.2):**
- POLLO: Orders of ~4500g every 9 periods
- TOCINO: Orders of ~2400g every 9 periods
- JAMÓN: Orders of ~3600g every 9 periods

**Key insight**: Same TIMING (every 9 periods), DIFFERENT QUANTITIES (proportional to usage)

## 📋 Checklist for Debugging

When you run the optimization, check:

### Step 1: Clustering Phase
- [ ] Ingredient names are printed clearly (not numbers)
- [ ] All ingredients in family are actual ingredient names

### Step 2: Optimization Phase  
- [ ] Representative ingredient is named clearly
- [ ] INPUT demand sheet name matches representative

### Step 3: Family Liberation Phase
- [ ] Each ingredient says "FOUND in materia_prima"
- [ ] Each ingredient has different demanda_diaria
- [ ] Each ingredient has different conversion factor
- [ ] Each ingredient has different replica averages

### Step 4: Liberation Vectors
- [ ] Each ingredient has different total_orders
- [ ] Order quantities are different (check "First 10 periods")
- [ ] Order timing may be same (normal for EOQ with same params)

### Step 5: Excel Validation
- [ ] Open FAMILIA_Resumen sheet
- [ ] Check "Vector_Final_Órdenes" column
- [ ] All ingredients should have DIFFERENT values
- [ ] Check FAM_xxx sheets for each ingredient
- [ ] Order quantities should match their converted demand

## 🚨 If You See Problems

### Copy These Terminal Sections:

1. The "🏭 FAMILY LIBERATION GENERATION" section
2. ALL "🧪 Processing ingredient" sections (for each ingredient)
3. ALL "🎯 LIBERATION FINAL" sections (for each ingredient)
4. The "📋 DEBUG - INPUT Demand sheet" section

This will show:
- Which ingredients are being processed
- Whether they're found in recipes
- What conversion factors are used
- What liberation vectors are generated

## 🔧 Quick Fixes

### If all ingredients show same orders:

1. **Check conversion factors in terminal** - are they different?
   - If YES: Bug in simulation (report)
   - If NO: Bug in conversion (recipes issue)

2. **Check ingredient names** - are they actual names or numbers?
   - If numbers: Clustering not returning names properly
   - If names: Check they exist in materia_prima

3. **Check INPUT sheet name** - does it match representative?
   - If NO: ingredient_info has wrong ingredient_code
   - Check "📋 DEBUG - INPUT Demand sheet" output

## 📊 Expected Terminal Output Structure

```
1. Clustering results
2. Optimization of representative
   └─ INPUT demand sheet debug
3. Verbose results generation
4. Family liberation generation
   ├─ Ingredient 1 processing
   │  ├─ Found in materia_prima?
   │  ├─ Conversion factors
   │  ├─ Replica statistics
   │  └─ Liberation final
   ├─ Ingredient 2 processing
   │  ├─ Found in materia_prima?
   │  ├─ Conversion factors
   │  ├─ Replica statistics
   │  └─ Liberation final
   └─ Ingredient 3 processing
      └─ ...
5. Excel export
6. Success message
```

---

**Status**: Debug logging added to track conversion process  
**Next step**: Run optimization and check terminal output against this guide

