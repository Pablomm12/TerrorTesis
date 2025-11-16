"""
Primer Eslabón (First Link / Factory) Optimization Module

This module handles the optimization of raw materials (first eslabon) used in the factory
to create processed ingredients (second eslabon).

Workflow:
1. Get liberation orders from second eslabon ingredients (both PVs)
2. Convert second eslabon orders → first eslabon raw material demands
3. Aggregate demands from multiple products and PVs
4. Create replicas matrix for PSO optimization
5. Run PSO optimization for raw materials

Author: AI Assistant
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from presentation import state as st


def validate_second_eslabon_optimization_complete(
    punto_venta_list: List[str],
    selected_raw_materials: List[str] = None,
    recetas_primero: dict = None,
    verbose: bool = True
) -> Tuple[bool, List[str], Dict[str, List[str]]]:
    """
    Validates that second eslabon optimization has been completed for the SPECIFIC
    ingredients needed to produce the selected raw materials.
    
    🎯 SMART VALIDATION: Only checks ingredients that produce the selected raw materials,
       not ALL ingredients from the PV.
    
    Parameters:
    -----------
    punto_venta_list : list
        List of punto de venta names to check (e.g., ['Terraplaza', 'Torres'])
    selected_raw_materials : list, optional
        List of raw material codes selected for clustering (e.g., ['SAL', 'AZUCAR'])
        If provided, only validates ingredients that produce these raw materials
    recetas_primero : dict, optional
        Recipe dictionary mapping second eslabon → first eslabon
        Required if selected_raw_materials is provided
    verbose : bool
        Print validation status
        
    Returns:
    --------
    tuple
        (is_complete: bool, missing_pvs: list, required_ingredients: dict)
        - is_complete: True if all required ingredients are optimized
        - missing_pvs: List of PV names that are missing optimization results
        - required_ingredients: {pv: [ingredient_codes]} - which ingredients are needed per PV
    """
    
    if verbose:
        print(f"\n🔍 VALIDACIÓN: Optimización Segundo Eslabón")
        print(f"   Puntos de venta a verificar: {punto_venta_list}")
    
    # Step 1: Determine which second eslabon ingredients are needed
    required_ingredients_per_pv = {}  # {pv: [ingredient_codes_needed]}
    
    if selected_raw_materials and recetas_primero:
        # 🎯 SMART MODE: Only validate specific ingredients that produce selected raw materials
        if verbose:
            print(f"   🎯 Modo inteligente: Validando solo ingredientes necesarios")
            print(f"   📦 Materias primas seleccionadas: {selected_raw_materials}")
            print(f"   🔍 DEBUG - Revisando recetas_primero ({len(recetas_primero)} ingredientes)")
        
        # Find which second eslabon ingredients produce these raw materials
        required_second_eslabon_codes = set()
        all_found_raw_materials = set()  # Track what we find
        
        for ingredient_code, recipe_data in recetas_primero.items():
            if not isinstance(recipe_data, dict):
                continue
            
            # Check if this ingredient produces any of the selected raw materials
            recipes = recipe_data.get('recetas', {})
            if not recipes and verbose:
                # DEBUG: Show ingredients without recipes
                print(f"   ⚠️  Ingrediente '{ingredient_code}' sin recetas")
                
            for flavor_name, flavor_recipe in recipes.items():
                if not isinstance(flavor_recipe, dict):
                    continue
                
                raw_materials_produced = flavor_recipe.get('materias_primas', {})
                
                # DEBUG: Track all raw materials found
                for rm_code in raw_materials_produced.keys():
                    all_found_raw_materials.add(rm_code)
                
                # Check if any selected raw material is in this recipe
                for rm_code in selected_raw_materials:
                    if rm_code in raw_materials_produced:
                        required_second_eslabon_codes.add(ingredient_code)
                        if verbose:
                            print(f"   ✅ Ingrediente '{ingredient_code}' produce '{rm_code}'")
                        break
        
        if verbose:
            print(f"   📋 Ingredientes de segundo eslabón necesarios: {list(required_second_eslabon_codes)}")
            
            # DEBUG: Show if raw materials weren't found
            not_found = set(selected_raw_materials) - all_found_raw_materials
            if not_found:
                print(f"   ⚠️  MATERIAS PRIMAS NO ENCONTRADAS EN RECETAS: {list(not_found)}")
                print(f"   💡 Materias primas disponibles (primeras 10): {list(all_found_raw_materials)[:10]}")
        
        # Same ingredients needed for all PVs
        for pv in punto_venta_list:
            required_ingredients_per_pv[pv] = list(required_second_eslabon_codes)
    else:
        # ⚠️ FALLBACK MODE: Validate that PV has ANY optimization (old behavior)
        if verbose:
            print(f"   ⚠️  Modo básico: Validando que PV tenga alguna optimización")
        # Will just check if PV has any results (required_ingredients_per_pv stays empty)
    
    # Step 2: Get optimization results from state
    try:
        opt_results_mp = st.app_state.get(st.STATE_OPTIMIZATION_RESULTS, {})
        
        if not opt_results_mp:
            if verbose:
                print(f"   ❌ No hay resultados de optimización de ingredientes disponibles")
            return False, punto_venta_list, required_ingredients_per_pv
        
        if verbose:
            print(f"   📊 Total de resultados almacenados: {len(opt_results_mp)}")
            # DEBUG: Show what's actually stored
            print(f"   🔑 DEBUG - Claves almacenadas:")
            for key in opt_results_mp.keys():
                result_data = opt_results_mp[key]
                if isinstance(result_data, dict):
                    pv = result_data.get('punto_venta_usado', 'N/A')
                    eslabon = result_data.get('eslabon', 'segundo')
                    ing_code = result_data.get('ingredient_code', 'N/A')
                    opt_result = result_data.get('optimization_result', {})
                    has_matrix = 'liberacion_orden_matrix' in opt_result
                    matrix_shape = 'N/A'
                    if has_matrix:
                        matrix = opt_result['liberacion_orden_matrix']
                        matrix_shape = matrix.shape if hasattr(matrix, 'shape') else str(type(matrix))
                    print(f"      • {key} → PV:{pv}, Eslabón:{eslabon}, Código:{ing_code}")
                    print(f"        ⚙️  Has liberation_matrix: {has_matrix}, Shape: {matrix_shape}")
            
    except Exception as e:
        if verbose:
            print(f"   ❌ Error accediendo a resultados: {e}")
        return False, punto_venta_list, required_ingredients_per_pv
    
    # Step 3: Check which required ingredients are optimized
    missing_pvs = []
    found_ingredients_per_pv = {pv: [] for pv in punto_venta_list}
    
    # Storage key format: {punto_venta}_{ingredient_code}
    for key, result_data in opt_results_mp.items():
        if not isinstance(result_data, dict):
            continue
        
        # ✅ CRITICAL: Skip first eslabon results (we only care about second eslabon ingredients)
        eslabon = result_data.get('eslabon', 'segundo')
        if eslabon == 'primero' or key.startswith('Fabrica_'):
            continue
        
        pv_usado = result_data.get('punto_venta_usado')
        ingredient_code = result_data.get('ingredient_code')
        opt_result = result_data.get('optimization_result', {})
        
        if pv_usado in punto_venta_list and 'liberacion_orden_matrix' in opt_result:
            found_ingredients_per_pv[pv_usado].append(ingredient_code)
    
    # Step 4: Validate
    if required_ingredients_per_pv:
        # SMART MODE: Check if specific required ingredients are present
        if verbose:
            print(f"\n   📋 VALIDACIÓN DETALLADA:")
        
        for pv in punto_venta_list:
            required = set(required_ingredients_per_pv.get(pv, []))
            found = set(found_ingredients_per_pv.get(pv, []))
            missing = required - found
            
            if verbose:
                print(f"   {'✅' if not missing else '❌'} {pv}:")
                print(f"      Necesarios: {list(required)}")
                print(f"      Optimizados: {list(found)}")
                if missing:
                    print(f"      ⚠️  FALTAN: {list(missing)}")
            
            if missing:
                missing_pvs.append(pv)
    else:
        # FALLBACK MODE: Just check if PV has any optimization
        for pv in punto_venta_list:
            if not found_ingredients_per_pv.get(pv):
                missing_pvs.append(pv)
                if verbose:
                    print(f"   ❌ {pv}: Sin optimización de ingredientes")
            else:
                if verbose:
                    count = len(found_ingredients_per_pv[pv])
                    print(f"   ✅ {pv}: {count} ingrediente(s) optimizado(s)")
    
    is_complete = len(missing_pvs) == 0
    
    if verbose:
        if is_complete:
            print(f"   ✅ VALIDACIÓN COMPLETA: Todos los ingredientes necesarios optimizados")
        else:
            print(f"   ⚠️  VALIDACIÓN INCOMPLETA: Faltan ingredientes en {len(missing_pvs)} PVs")
    
    return is_complete, missing_pvs, required_ingredients_per_pv


def convert_second_eslabon_to_first_eslabon(
    liberacion_orden_matrix_second: np.ndarray,
    second_eslabon_code: str,
    recetas_primero: dict,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Converts liberation orders from a second eslabon ingredient (processed ingredient)
    into raw material demands for first eslabon (factory raw materials).
    
    Similar to pizza→ingredient conversion, but now: processed ingredient→raw materials.
    
    Parameters:
    -----------
    liberacion_orden_matrix_second : np.ndarray
        Liberation order matrix for the second eslabon ingredient
        Shape: (periods x replicas) or (replicas x periods)
    second_eslabon_code : str
        Code of the second eslabon ingredient (e.g., "1430.05.02" for processed chicken)
    recetas_primero : dict
        Recipes that define how second eslabon products are made from first eslabon materials
        Structure: {second_eslabon_code: {'ingredientes': {raw_material_code: {'cantidad': X, ...}}}}
    verbose : bool
        Print conversion details
        
    Returns:
    --------
    dict
        Dictionary mapping raw material codes to their demand matrices
        {raw_material_code: np.ndarray (periods x replicas)}
        
    Example:
    --------
    Input: 100kg processed chicken needed in period 1
    Recipe: Chicken = 2g Salt/kg + 1g Pepper/kg + 0.5g Spices/kg
    Output: {'SALT_CODE': [200g, ...], 'PEPPER_CODE': [100g, ...], 'SPICES_CODE': [50g, ...]}
    """
    
    if verbose:
        print(f"\n🔄 CONVERSIÓN: Segundo Eslabón → Primer Eslabón")
        print(f"   Ingrediente: {second_eslabon_code}")
        print(f"   Matriz original: {liberacion_orden_matrix_second.shape}")
    
    # Ensure matrix is (periods x replicas) format
    matrix = liberacion_orden_matrix_second
    if hasattr(matrix, 'values'):
        matrix = matrix.values
    elif not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Detect and transpose if needed
    if matrix.shape[0] > matrix.shape[1]:
        # Likely (replicas x periods), transpose to (periods x replicas)
        matrix = matrix.T
        if verbose:
            print(f"   🔄 Transpuesta a (períodos x réplicas): {matrix.shape}")
    
    # Get recipe for this second eslabon ingredient
    if second_eslabon_code not in recetas_primero:
        if verbose:
            print(f"   ❌ ERROR: Receta no encontrada para '{second_eslabon_code}'")
            print(f"   Recetas disponibles: {list(recetas_primero.keys())[:5]}...")
        return {}
    
    receta = recetas_primero[second_eslabon_code]
    ingredientes = receta.get('ingredientes', {})
    
    if not ingredientes:
        if verbose:
            print(f"   ⚠️  Receta sin ingredientes para '{second_eslabon_code}'")
        return {}
    
    if verbose:
        print(f"   📋 Receta '{receta.get('nombre', second_eslabon_code)}':")
        for mp_code, mp_info in ingredientes.items():
            cantidad = mp_info.get('cantidad', 0)
            unidad = mp_info.get('unidad', '')
            nombre = mp_info.get('nombre', mp_code)
            print(f"      - {nombre} ({mp_code}): {cantidad} {unidad}")
    
    # Create demand matrices for each raw material
    raw_material_demands = {}
    
    for mp_code, mp_info in ingredientes.items():
        cantidad_por_unidad = mp_info.get('cantidad', 0)
        
        if cantidad_por_unidad <= 0:
            if verbose:
                print(f"   ⚠️  Cantidad <= 0 para {mp_code}, omitiendo")
            continue
        
        # Convert: second_eslabon_orders * cantidad_per_unit = raw_material_demand
        raw_material_matrix = matrix * cantidad_por_unidad
        
        # Round to integers (grams are discrete)
        raw_material_matrix = np.round(raw_material_matrix).astype(int)
        
        # Ensure minimum of 1g where there was demand
        raw_material_matrix = np.where(raw_material_matrix > 0, 
                                      np.maximum(raw_material_matrix, 1), 
                                      0)
        
        raw_material_demands[mp_code] = raw_material_matrix
        
        if verbose:
            total_demand = raw_material_matrix.sum()
            avg_period = raw_material_matrix.mean(axis=1).mean()
            print(f"   ✅ {mp_info.get('nombre', mp_code)}: {total_demand:.0f}g total, {avg_period:.1f}g/período promedio")
    
    if verbose:
        print(f"   📦 Total materias primas generadas: {len(raw_material_demands)}")
    
    return raw_material_demands


def aggregate_demands_from_multiple_sources(
    demand_dicts_list: List[Dict[str, np.ndarray]],
    source_names: List[str] = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Aggregates raw material demands from multiple sources (products and/or PVs).
    
    This function sums demands period-by-period across all sources.
    For example: Salt demand from Chicken + Salt demand from Meat = Total Salt demand
    
    Parameters:
    -----------
    demand_dicts_list : list
        List of demand dictionaries, each from a different source
        Each dict: {raw_material_code: np.ndarray (periods x replicas)}
    source_names : list, optional
        Names of sources for debugging (e.g., ['Chicken', 'Meat'])
    verbose : bool
        Print aggregation details
        
    Returns:
    --------
    dict
        Aggregated demands: {raw_material_code: np.ndarray (periods x replicas)}
        
    Example:
    --------
    Source 1 (Chicken): {'SALT': [[10, 15], [20, 25]], 'PEPPER': [[5, 8], [10, 12]]}
    Source 2 (Meat):    {'SALT': [[5, 8],  [10, 12]], 'SPICES': [[2, 3], [4, 5]]}
    Result:             {'SALT': [[15, 23], [30, 37]], 'PEPPER': [[5, 8], [10, 12]], 'SPICES': [[2, 3], [4, 5]]}
    """
    
    if verbose:
        print(f"\n➕ AGREGACIÓN: Consolidando demandas")
        print(f"   Fuentes: {len(demand_dicts_list)}")
        if source_names:
            for i, name in enumerate(source_names, 1):
                print(f"      {i}. {name}")
    
    if not demand_dicts_list:
        return {}
    
    aggregated_demands = {}
    
    for i, demand_dict in enumerate(demand_dicts_list):
        source_name = source_names[i] if source_names and i < len(source_names) else f"Fuente {i+1}"
        
        for mp_code, demand_matrix in demand_dict.items():
            if mp_code not in aggregated_demands:
                # First time seeing this raw material
                aggregated_demands[mp_code] = demand_matrix.copy()
                if verbose:
                    print(f"   ✅ {mp_code}: Inicializado desde {source_name}")
            else:
                # Add to existing demand
                aggregated_demands[mp_code] += demand_matrix
                if verbose:
                    print(f"   ➕ {mp_code}: Agregado desde {source_name}")
    
    if verbose:
        print(f"\n   📊 Resumen agregación:")
        for mp_code, demand_matrix in aggregated_demands.items():
            total_demand = demand_matrix.sum()
            avg_period = demand_matrix.mean(axis=1).mean()
            print(f"      {mp_code}: {total_demand:.0f}g total, {avg_period:.1f}g/período promedio")
        print(f"   📊 Total materias primas agregadas: {len(aggregated_demands)}")
    
    return aggregated_demands


def list_available_ingredient_optimizations(verbose: bool = True) -> dict:
    """
    Lists all available ingredient optimizations organized by punto de venta.
    Useful for debugging and validation before running first eslabon optimization.
    
    Storage key format: {punto_venta}_{ingredient_code} (NO policy in key)
    ✅ Each ingredient has ONE optimization (overwrites if re-optimized with different policy)
    
    Returns:
    --------
    dict
        {
            'Terraplaza': ['MM001 (EOQ)', 'MM002 (QR)', ...],
            'Torres': ['MM001 (EOQ)', 'MM003 (POQ)', ...],
            ...
        }
    """
    from presentation import state as st
    
    opt_results = st.app_state.get(st.STATE_OPTIMIZATION_RESULTS, {})
    
    if not opt_results:
        if verbose:
            print("⚠️ No hay resultados de optimización almacenados")
        return {}
    
    # Organize by PV
    by_pv = {}
    
    for key, result_data in opt_results.items():
        if not isinstance(result_data, dict):
            continue
        
        # Get PV from metadata
        pv_usado = result_data.get('punto_venta_usado')
        ingredient_code = result_data.get('ingredient_code')
        policy = result_data.get('policy')
        eslabon = result_data.get('eslabon', 'segundo')
        
        # Skip factory optimizations
        if eslabon == 'primero':
            continue
        
        if pv_usado and ingredient_code:
            if pv_usado not in by_pv:
                by_pv[pv_usado] = []
            
            # Format: ingredient_code (policy) - shows which policy was used
            entry = f"{ingredient_code} ({policy})" if policy else ingredient_code
            by_pv[pv_usado].append(entry)
    
    if verbose:
        print(f"\n📊 INGREDIENTES OPTIMIZADOS DISPONIBLES:")
        print(f"{'='*60}")
        if by_pv:
            for pv, ingredients in sorted(by_pv.items()):
                print(f"\n  🏪 {pv}: {len(ingredients)} ingredientes")
                for ing in sorted(ingredients):
                    print(f"     • {ing}")
        else:
            print("  ⚠️ No hay ingredientes optimizados")
        print(f"{'='*60}\n")
    
    return by_pv


def get_second_eslabon_liberation_orders(
    punto_venta: str,
    policy: str = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Retrieves liberation order matrices for all optimized second eslabon ingredients
    from a specific punto de venta.
    
    Parameters:
    -----------
    punto_venta : str
        Point of sale name (e.g., 'Terraplaza')
    policy : str, optional
        Specific policy to retrieve. If None, gets the most recent policy.
    verbose : bool
        Print retrieval details
        
    Returns:
    --------
    dict
        {ingredient_code: liberation_matrix (periods x replicas)}
    """
    
    if verbose:
        print(f"\n📥 OBTENCIÓN: Órdenes de liberación segundo eslabón")
        print(f"   Punto de venta: {punto_venta}")
        print(f"   Política: {policy or 'Última disponible'}")
    
    try:
        # Get from STATE_OPTIMIZATION_RESULTS (ingredient optimization)
        opt_results_mp = st.app_state.get(st.STATE_OPTIMIZATION_RESULTS, {})
        
        if verbose:
            print(f"   🔍 Revisando {len(opt_results_mp)} resultados almacenados")
            if opt_results_mp:
                print(f"   🔑 Claves disponibles: {list(opt_results_mp.keys())}")
        
        liberation_orders = {}
        ingredient_policies = {}  # Track which policy was used for each ingredient
        
        # SIMPLIFIED RETRIEVAL: Use composite key format for direct lookup
        # Key format: {punto_venta}_{ingredient_code} (NO policy - only one optimization per ingredient)
        # ✅ If ingredient re-optimized with different policy, storage overwrites previous result
        
        # Direct key-based lookup (fast and reliable)
        pv_prefix = f"{punto_venta}_"
        
        for key, result_data in opt_results_mp.items():
            if not isinstance(result_data, dict):
                continue
            
            # ✅ CRITICAL: Skip first eslabon results
            if key.startswith('Fabrica_'):
                continue
            
            eslabon = result_data.get('eslabon', 'segundo')
            if eslabon == 'primero':
                continue
            
            # Check if key starts with the PV prefix (direct composite key match)
            if key.startswith(pv_prefix):
                # Extract data using stored metadata
                ingredient_code = result_data.get('ingredient_code')
                opt_result = result_data.get('optimization_result', {})
                
                # Verify this is the correct PV (double-check)
                pv_usado = result_data.get('punto_venta_usado', opt_result.get('punto_venta_usado', ''))
                
                if pv_usado != punto_venta:
                    if verbose:
                        print(f"   ⚠️ Clave coincide pero PV no: esperado '{punto_venta}', encontrado '{pv_usado}'")
                    continue
                
                # Check if it has liberation matrix
                if 'liberacion_orden_matrix' in opt_result and ingredient_code:
                    liberation_matrix = opt_result['liberacion_orden_matrix']
                    policy_used = result_data.get('policy', 'Unknown')
                    
                    liberation_orders[ingredient_code] = liberation_matrix
                    ingredient_policies[ingredient_code] = policy_used
                    
                    if verbose:
                        shape = liberation_matrix.shape if hasattr(liberation_matrix, 'shape') else 'N/A'
                        ingredient_name = opt_result.get('ingredient_display_name', ingredient_code)
                        print(f"   ✅ Ingrediente '{ingredient_name}' ({ingredient_code})")
                        print(f"      Política: {policy_used}, Forma: {shape}")
        
        # Method 2: Fallback - Check all keys by metadata (for backward compatibility)
        if not liberation_orders:
            if verbose:
                print(f"   🔄 Método directo no encontró resultados, usando búsqueda por metadata...")
            
            for key, result_data in opt_results_mp.items():
                if not isinstance(result_data, dict):
                    continue
                
                # Skip first eslabon results
                if key.startswith('Fabrica_'):
                    continue
                
                eslabon = result_data.get('eslabon', 'segundo')
                if eslabon == 'primero':
                    continue
                
                # Skip keys we already checked
                if key.startswith(pv_prefix):
                    continue
                
                # Check if this result is from the specified PV using metadata
                opt_result = result_data.get('optimization_result', {})
                pv_usado = result_data.get('punto_venta_usado') or opt_result.get('punto_venta_usado', '')
                
                if pv_usado != punto_venta:
                    continue
                
                # Check if it has the liberation matrix
                if 'liberacion_orden_matrix' in opt_result:
                    ingredient_code = result_data.get('ingredient_code') or opt_result.get('ingredient_mp_code')
                    
                    if ingredient_code:
                        liberation_matrix = opt_result['liberacion_orden_matrix']
                        liberation_orders[ingredient_code] = liberation_matrix
                        
                        if verbose:
                            shape = liberation_matrix.shape if hasattr(liberation_matrix, 'shape') else 'N/A'
                            ingredient_name = opt_result.get('ingredient_display_name', ingredient_code)
                            print(f"   ✅ Ingrediente '{ingredient_name}' ({ingredient_code}): {shape}")
        
        if liberation_orders:
            if verbose:
                print(f"   📦 Total ingredientes encontrados: {len(liberation_orders)}")
                print(f"   📋 Ingredientes: {list(liberation_orders.keys())}")
            return liberation_orders
        
        # No ingredient optimization found
        if verbose:
            print(f"   ❌ No se encontraron ingredientes optimizados para {punto_venta}")
            print(f"   💡 Debe optimizar segundo eslabón (ingredientes) para '{punto_venta}' antes de primer eslabón")
            print(f"   💡 Busque claves que comiencen con: '{pv_prefix}'")
        
        return {}
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Error obteniendo órdenes: {e}")
            import traceback
            traceback.print_exc()
        return {}


def create_first_eslabon_replicas_matrix(
    punto_venta_list: List[str],
    recetas_primero: dict,
    policy: str = None,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Main function to create replicas matrices for first eslabon raw materials.
    
    This orchestrates the entire conversion process:
    1. Get liberation orders from all PVs (second eslabon)
    2. Convert each second eslabon ingredient → raw materials
    3. Aggregate demands across all products and PVs
    
    Parameters:
    -----------
    punto_venta_list : list
        List of PV names to aggregate from (e.g., ['Terraplaza', 'Torres'])
    recetas_primero : dict
        First eslabon recipes
    policy : str, optional
        Policy to use for retrieval
    verbose : bool
        Print process details
        
    Returns:
    --------
    dict
        {raw_material_code: replicas_matrix (replicas x periods)}
        Ready for PSO optimization
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🏭 CREACIÓN MATRIZ RÉPLICAS: PRIMER ESLABÓN (FÁBRICA)")
        print(f"{'='*70}")
        print(f"   Puntos de venta: {', '.join(punto_venta_list)}")
    
    # Step 1: Validate prerequisites (fallback mode - checks any optimization)
    is_complete, missing_pvs, _ = validate_second_eslabon_optimization_complete(
        punto_venta_list, verbose=verbose
    )
    
    if not is_complete:
        raise ValueError(
            f"Segundo eslabón no optimizado para: {', '.join(missing_pvs)}. "
            f"Por favor, complete la optimización de ingredientes antes de proceder."
        )
    
    # Step 2: Collect all demand dictionaries
    all_demand_dicts = []
    source_names = []
    
    for pv in punto_venta_list:
        # Get liberation orders for this PV
        pv_liberation_orders = get_second_eslabon_liberation_orders(pv, policy, verbose=verbose)
        
        if not pv_liberation_orders:
            if verbose:
                print(f"   ⚠️  Sin órdenes de liberación para {pv}, omitiendo")
            continue
        
        # Convert each second eslabon ingredient to raw materials
        for ingredient_code, liberation_matrix in pv_liberation_orders.items():
            raw_material_demands = convert_second_eslabon_to_first_eslabon(
                liberation_matrix,
                ingredient_code,
                recetas_primero,
                verbose=verbose
            )
            
            if raw_material_demands:
                all_demand_dicts.append(raw_material_demands)
                source_names.append(f"{pv}_{ingredient_code}")
    
    if not all_demand_dicts:
        raise ValueError("No se pudieron generar demandas de primer eslabón")
    
    # Step 3: Aggregate all demands
    aggregated_demands = aggregate_demands_from_multiple_sources(
        all_demand_dicts,
        source_names,
        verbose=verbose
    )
    
    # Step 4: Convert to (replicas x periods) format for PSO
    final_replicas_matrices = {}
    
    if verbose:
        print(f"\n🔄 CONVERSIÓN FINAL: (períodos x réplicas) → (réplicas x períodos)")
    
    for mp_code, demand_matrix in aggregated_demands.items():
        # demand_matrix is currently (periods x replicas)
        # PSO needs (replicas x periods)
        replicas_matrix = demand_matrix.T
        
        final_replicas_matrices[mp_code] = replicas_matrix
        
        if verbose:
            print(f"   ✅ {mp_code}: {replicas_matrix.shape} (réplicas x períodos)")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ PROCESO COMPLETO")
        print(f"   Materias primas generadas: {len(final_replicas_matrices)}")
        print(f"   Listas para optimización PSO")
        print(f"{'='*70}\n")
    
    return final_replicas_matrices


# Export main functions
__all__ = [
    'validate_second_eslabon_optimization_complete',
    'convert_second_eslabon_to_first_eslabon',
    'aggregate_demands_from_multiple_sources',
    'get_second_eslabon_liberation_orders',
    'create_first_eslabon_replicas_matrix'
]
