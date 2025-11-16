"""
Family Liberation Generator

This module provides functionality to generate liberation order vectors for entire ingredient families
based on optimization results from a representative ingredient. It converts pizza demand data to 
ingredient demand data and applies optimized parameters to generate liberation schedules.

Functions:
- generate_family_liberation_vectors: Main function to generate liberation for all family ingredients
- convert_pizza_to_ingredient_data: Convert pizza demand data to ingredient-specific data
- create_ingredient_data_dict_from_pizza: Create data_dict structure for ingredient optimization
- export_family_liberation_to_excel: Export liberation vectors to Excel files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime

def sanitize_excel_sheet_name(name: str, max_length: int = 31) -> str:
    """
    Sanitize a string for use as Excel sheet name.
    Excel sheet names cannot contain: \\ / ? * [ ] :
    And must be <= 31 characters.
    """
    if not name:
        return "Sheet1"
    
    # Replace invalid characters
    invalid_chars = ['\\\\', '/', '?', '*', '[', ']', ':']
    sanitized = name
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Ensure not empty after sanitization
    if not sanitized.strip():
        sanitized = "Sheet1"
    
    return sanitized

def convert_pizza_to_ingredient_data(
    ingredient_mp_code: str,
    pizza_data_dict: Dict,
    punto_venta: str,
    recetas_primero: Dict,
    recetas_segundo: Dict,
    materia_prima: Dict
) -> Dict:
    """
    Convert pizza demand data to ingredient-specific data for simulation.
    
    Parameters:
    - ingredient_mp_code: Code or NAME of the ingredient to convert
    - pizza_data_dict: Original pizza data dictionary
    - punto_venta: Point of sale name
    - recetas_primero: First level recipes
    - recetas_segundo: Second level recipes (pizza recipes)
    - materia_prima: Raw materials information
    
    Returns:
    - ingredient_data_dict: Converted data dictionary for ingredient simulation
    """
    if punto_venta not in pizza_data_dict:
        raise ValueError(f"Punto de venta '{punto_venta}' not found in pizza_data_dict")
    
    # CRITICAL: Resolve ingredient name to code
    from services.materia_prima import convert_pizza_demand_to_ingredient_demand, find_ingredient_code_in_materia_prima
    
    actual_mp_code, mp_info = find_ingredient_code_in_materia_prima(ingredient_mp_code, materia_prima)
    if actual_mp_code is None:
        print(f"⚠️ WARNING: Ingredient '{ingredient_mp_code}' not found in materia_prima - using as-is")
        actual_mp_code = ingredient_mp_code
        mp_info = {}
    else:
        print(f"✅ Resolved ingredient '{ingredient_mp_code}' → code '{actual_mp_code}'")
    
    pizza_sheets = pizza_data_dict[punto_venta]
    pizza_parametros = pizza_sheets["PARAMETROS"]
    pizza_resultados = pizza_sheets.get("RESULTADOS", {})
    
    # Convert pizza demand to ingredient demand using the resolved code
    pizza_ventas = pizza_resultados.get("ventas", {})
    ingredient_demand, conversion_factor = convert_pizza_demand_to_ingredient_demand(
        pizza_ventas, actual_mp_code, recetas_primero, recetas_segundo
    )
    
    # Calculate actual average daily demand from ingredient_demand
    if ingredient_demand:
        ingredient_daily_demands = [demand for demand in ingredient_demand.values() if isinstance(demand, (int, float))]
        actual_daily_demand = sum(ingredient_daily_demands) / len(ingredient_daily_demands) if ingredient_daily_demands else conversion_factor
    else:
        actual_daily_demand = conversion_factor
    
    # Use the already-resolved ingredient_info
    ingredient_info = mp_info if mp_info else {}
    
    # Create ingredient-specific parameters
    ingredient_parametros = {
        "inventario_inicial": ingredient_info.get("inventario_inicial", 0),
        "lead time": ingredient_info.get("lead_time", pizza_parametros.get("lead time", 1)),
        "MOQ": ingredient_info.get("MOQ", 0),
        "backorders": pizza_parametros.get("backorders", 1),
        
        # Cost parameters for ingredient
        "costo_pedir": ingredient_info.get("costo_pedir", 25),
        "costo_unitario": ingredient_info.get("costo_unitario", 2),
        "costo_faltante": ingredient_info.get("costo_faltante", 10),
        "costo_sobrante": ingredient_info.get("costo_sobrante", 1),
        
        # FIXED: use actual average daily demand in ingredient units
        "demanda_diaria": actual_daily_demand,  # Actual average daily demand
        "demanda_promedio": actual_daily_demand * 30,  # Monthly demand
    }
    
    # Create ingredient-specific results
    ingredient_resultados = {
        "ventas": ingredient_demand,
        "T": pizza_resultados.get("T", 2),
    }
    
    # Build ingredient data dictionary
    ingredient_data_dict = {
        punto_venta: {
            "PARAMETROS": ingredient_parametros,
            "RESULTADOS": ingredient_resultados
        }
    }
    
    return ingredient_data_dict


def create_replicas_matrix_for_ingredient(
    ingredient_mp_code: str,
    pizza_replicas_matrix: np.ndarray,
    recetas_primero: Dict,
    recetas_segundo: Dict,
    materia_prima: Dict = None,
    verbose: bool = False
) -> np.ndarray:
    """
    Convert pizza replicas matrix to ingredient replicas matrix.
    Each ingredient gets its own unique converted replicas based on its specific recipes.
    
    Parameters:
    - ingredient_mp_code: Code or NAME of the ingredient
    - pizza_replicas_matrix: Matrix of pizza demand replicas
    - recetas_primero: First level recipes
    - recetas_segundo: Second level recipes
    - materia_prima: Raw materials dict (optional, for name resolution)
    - verbose: Print conversion details
    
    Returns:
    - ingredient_replicas_matrix: Converted replicas matrix for ingredient
    """
    from services.materia_prima import convert_pizza_demand_to_ingredient_demand, find_ingredient_code_in_materia_prima
    
    # Resolve ingredient name to code if materia_prima is provided
    actual_mp_code = ingredient_mp_code
    if materia_prima:
        resolved_code, _ = find_ingredient_code_in_materia_prima(ingredient_mp_code, materia_prima)
        if resolved_code:
            actual_mp_code = resolved_code
            if verbose and resolved_code != ingredient_mp_code:
                print(f"   🔄 Resolved '{ingredient_mp_code}' → '{actual_mp_code}'")
    
    if verbose:
        print(f"   🔄 Converting pizza replicas to {actual_mp_code} replicas")
        print(f"   📊 Pizza replicas shape: {pizza_replicas_matrix.shape}")
        print(f"   📊 Pizza demand range: {pizza_replicas_matrix.min():.0f}-{pizza_replicas_matrix.max():.0f} pizzas")
    
    ingredient_replicas = []
    conversion_factors = []
    
    for replica_row in pizza_replicas_matrix:
        # Convert this replica to a demand dictionary
        pizza_demand_dict = {i: replica_row[i] for i in range(len(replica_row))}
        
        # Convert to ingredient demand using ingredient-specific recipes (with RESOLVED code)
        ingredient_demand_dict, conversion_factor = convert_pizza_demand_to_ingredient_demand(
            pizza_demand_dict, actual_mp_code, recetas_primero, recetas_segundo
        )
        
        # Store conversion factor for diagnostics
        conversion_factors.append(conversion_factor)
        
        # Convert back to array format
        ingredient_replica = np.array([ingredient_demand_dict.get(i, 0) for i in range(len(replica_row))])
        ingredient_replicas.append(ingredient_replica)
    
    ingredient_replicas_matrix = np.array(ingredient_replicas)
    
    if verbose:
        print(f"   ✅ Ingredient replicas shape: {ingredient_replicas_matrix.shape}")
        print(f"   📊 Ingredient demand range: {ingredient_replicas_matrix.min():.2f}-{ingredient_replicas_matrix.max():.2f}g")
        print(f"   📊 Average conversion factor: {np.mean(conversion_factors):.4f}g per pizza")
        print(f"   📊 Unique values per period: {[len(np.unique(ingredient_replicas_matrix[:, i])) for i in range(min(5, ingredient_replicas_matrix.shape[1]))][:5]}")
    
    return ingredient_replicas_matrix


def generate_family_liberation_vectors(
    family_ingredients: List[str],
    representative_ingredient: str,
    optimized_params: Dict,
    policy: str,
    pizza_data_dict: Dict,
    pizza_replicas_matrix: np.ndarray,
    punto_venta: str,
    recetas_primero: Dict,
    recetas_segundo: Dict,
    materia_prima: Dict,
    verbose: bool = True,
    representative_liberation_final: np.ndarray = None
) -> Dict[str, Dict]:
    """
    Generate liberation order vectors for all ingredients in a family based on 
    optimization results from representative ingredient.
    
    Parameters:
    - family_ingredients: List of ingredient codes in the family
    - representative_ingredient: Code of the representative ingredient that was optimized
    - optimized_params: Optimization results (e.g., {"Q": 4, "R": 1} for QR policy)
    - policy: Policy type ("QR", "ST", "SST", "SS", "EOQ", "POQ", "LXL")
    - pizza_data_dict: Original pizza data dictionary
    - pizza_replicas_matrix: Matrix of pizza demand replicas
    - punto_venta: Point of sale name
    - recetas_primero: First level recipes
    - recetas_segundo: Second level recipes
    - materia_prima: Raw materials information
    - verbose: Whether to print debug information
    
    Returns:
    - family_liberation_results: Dict with liberation vectors for each ingredient
        {
            "ingredient_code": {
                "liberation_df": DataFrame with liberation vectors,
                "summary_metrics": DataFrame with performance indicators,
                "optimized_params": Dict with applied parameters
            }
        }
    """
    from services.simulacion import (
        replicas_QR_verbose, replicas_ST_verbose, replicas_SST_verbose,
        replicas_SS_verbose, replicas_EOQ_verbose, replicas_POQ_verbose,
        replicas_LXL_verbose
    )
    
    family_liberation_results = {}
    
    if verbose:
        print(f"\n🏭 GENERATING FAMILY LIBERATION VECTORS")
        print(f"📊 Family: {len(family_ingredients)} ingredients")
        print(f"🎯 Representative: {representative_ingredient}")
        print(f"⚙️ Policy: {policy}")
        print(f"📈 Optimized params: {optimized_params}")
        print("="*60)
    
    # Get the verbose simulation function based on policy
    simulation_functions = {
        "QR": replicas_QR_verbose,
        "ST": replicas_ST_verbose,
        "SST": replicas_SST_verbose,
        "SS": replicas_SS_verbose,
        "EOQ": replicas_EOQ_verbose,
        "POQ": replicas_POQ_verbose,
        "LXL": replicas_LXL_verbose
    }
    
    if policy not in simulation_functions:
        raise ValueError(f"Unsupported policy: {policy}")
    
    simulation_function = simulation_functions[policy]
    
    for ingredient_code in family_ingredients:
        if verbose:
            print(f"\n{'='*70}")
            print(f"🧪 Processing ingredient: '{ingredient_code}'")
            print(f"{'='*70}")
        
        try:
            # CRITICAL DEBUG: Check ingredient code format
            if verbose:
                print(f"   🔍 Ingredient code type: {type(ingredient_code)}")
                print(f"   🔍 Ingredient code value: '{ingredient_code}'")
                print(f"   🔍 Searching in materia_prima keys: {list(materia_prima.keys())[:10] if materia_prima else 'None'}...")
                if ingredient_code in materia_prima:
                    print(f"   ✅ '{ingredient_code}' FOUND in materia_prima")
                else:
                    print(f"   ❌ '{ingredient_code}' NOT FOUND in materia_prima")
                    # Try fuzzy matching
                    potential_matches = [k for k in materia_prima.keys() if str(ingredient_code).upper() in str(k).upper()]
                    if potential_matches:
                        print(f"   🔍 Potential matches: {potential_matches[:5]}")
            
            # Convert pizza data to ingredient-specific data
            ingredient_data_dict = convert_pizza_to_ingredient_data(
                ingredient_code, pizza_data_dict, punto_venta,
                recetas_primero, recetas_segundo, materia_prima
            )
            
            if verbose:
                ingredient_params = ingredient_data_dict[punto_venta]["PARAMETROS"]
                print(f"   📊 Ingredient data_dict created:")
                # CRITICAL FIX: Ensure we convert to float to avoid Series formatting issues
                demanda_diaria = ingredient_params.get('demanda_diaria', 0)
                demanda_promedio = ingredient_params.get('demanda_promedio', 0)
                
                # Convert Series to float
                if hasattr(demanda_diaria, 'iloc'):  # It's a Series
                    demanda_diaria = float(demanda_diaria.iloc[0]) if len(demanda_diaria) > 0 else 0.0
                elif hasattr(demanda_diaria, '__float__'):
                    demanda_diaria = float(demanda_diaria)
                else:
                    demanda_diaria = float(demanda_diaria) if demanda_diaria else 0.0
                    
                if hasattr(demanda_promedio, 'iloc'):  # It's a Series
                    demanda_promedio = float(demanda_promedio.iloc[0]) if len(demanda_promedio) > 0 else 0.0
                elif hasattr(demanda_promedio, '__float__'):
                    demanda_promedio = float(demanda_promedio)
                else:
                    demanda_promedio = float(demanda_promedio) if demanda_promedio else 0.0
                
                # Now safe to format
                print(f"      demanda_diaria: {demanda_diaria:.2f}g")
                print(f"      demanda_promedio: {demanda_promedio:.2f}g")
            
            # Convert replicas matrix to ingredient units using ingredient-specific recipes
            ingredient_replicas = create_replicas_matrix_for_ingredient(
                ingredient_code, pizza_replicas_matrix, 
                recetas_primero, recetas_segundo, materia_prima=materia_prima, verbose=verbose
            )
            
            if verbose:
                print(f"   📊 Converted replicas shape: {ingredient_replicas.shape}")
                avg_demand = np.mean(ingredient_replicas)
                min_demand = np.min(ingredient_replicas)
                max_demand = np.max(ingredient_replicas)
                unique_values = len(np.unique(ingredient_replicas))
                print(f"   📈 Ingredient replicas statistics:")
                print(f"      Average: {avg_demand:.2f}g")
                print(f"      Range: {min_demand:.2f}g - {max_demand:.2f}g")
                print(f"      Unique values: {unique_values}")
                print(f"      First 5 values of first replica: {ingredient_replicas[0, :5]}")
                
                # CRITICAL DEBUG: Show that this ingredient has DIFFERENT values than pizza
                if pizza_replicas_matrix is not None:
                    pizza_avg = np.mean(pizza_replicas_matrix)
                    ratio = avg_demand / pizza_avg if pizza_avg > 0 else 0
                    print(f"   🔄 CONVERSION CHECK:")
                    print(f"      Pizza matrix average: {pizza_avg:.2f}")
                    print(f"      Ingredient matrix average: {avg_demand:.2f}g")
                    print(f"      Conversion ratio: {ratio:.4f} (should be different for each ingredient)")
                    
                    # Show first period comparison
                    pizza_first_period = pizza_replicas_matrix[0, 0] if pizza_replicas_matrix.ndim == 2 else pizza_replicas_matrix[0]
                    ingredient_first_period = ingredient_replicas[0, 0]
                    print(f"      First period: Pizza={pizza_first_period:.0f} → Ingredient={ingredient_first_period:.2f}g")
            
            # CRITICAL: Check if this is the representative ingredient with pre-calculated liberation
            is_representative = (str(ingredient_code).upper() == str(representative_ingredient).upper())
            use_precalculated = is_representative and representative_liberation_final is not None
            
            if use_precalculated:
                if verbose:
                    print(f"   ⭐ This is the REPRESENTATIVE ingredient - using pre-calculated liberation vector")
                    print(f"   ✅ Liberation vector already calculated by verbose function")
                    precalc_sum = np.sum(representative_liberation_final)
                    precalc_periods = np.sum(representative_liberation_final > 0)
                    print(f"   📊 Pre-calculated vector: total={precalc_sum:.0f}g, active periods={precalc_periods}")
                
                # Still need to run simulation to get summary_metrics and liberation_df for Excel
                # But we'll use the pre-calculated liberation_final instead of the one from simulation
                if policy == "QR":
                    Q, R = optimized_params["Q"], optimized_params["R"]
                    summary_metrics, liberation_df, resultados_replicas, _ = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, Q, R
                    )
                    applied_params = {"Q": Q, "R": R}
                elif policy == "ST":
                    S, T = optimized_params["S"], optimized_params["T"]
                    summary_metrics, liberation_df, resultados_replicas, _ = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, S, T
                    )
                    applied_params = {"S": S, "T": T}
                elif policy == "SST":
                    s, S, T = optimized_params["s"], optimized_params["S"], optimized_params["T"]
                    summary_metrics, liberation_df, resultados_replicas, _ = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, s, S, T
                    )
                    applied_params = {"s": s, "S": S, "T": T}
                elif policy == "SS":
                    s, S = optimized_params["s"], optimized_params["S"]
                    summary_metrics, liberation_df, resultados_replicas, _ = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, S, s
                    )
                    applied_params = {"s": s, "S": S}
                elif policy in ["EOQ", "POQ", "LXL"]:
                    porcentaje_seguridad = optimized_params.get("porcentaje", optimized_params.get("porcentaje_seguridad", 0.35))
                    summary_metrics, liberation_df, resultados_replicas, _ = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, porcentaje_seguridad
                    )
                    applied_params = {"porcentaje": porcentaje_seguridad}
                
                # Use the pre-calculated liberation_final
                liberation_final = representative_liberation_final
                
                # CRITICAL FIX: Ensure liberation_final is numpy array, not pandas Series
                if liberation_final is not None:
                    if hasattr(liberation_final, 'values'):
                        liberation_final = liberation_final.values
                    elif not isinstance(liberation_final, np.ndarray):
                        liberation_final = np.array(liberation_final)
                
            else:
                # Not the representative - calculate normally
                if verbose:
                    print(f"   🔧 Applying {policy} policy with params {optimized_params}")
                    print(f"   📊 Using ingredient-specific replicas: avg={np.mean(ingredient_replicas):.2f}g")
                
                if policy == "QR":
                    Q, R = optimized_params["Q"], optimized_params["R"]
                    if verbose:
                        print(f"   ⚙️ Running QR simulation with Q={Q}, R={R} for {ingredient_code}")
                    summary_metrics, liberation_df, resultados_replicas, liberation_final = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, Q, R
                    )
                    applied_params = {"Q": Q, "R": R}
                    
                elif policy == "ST":
                    S, T = optimized_params["S"], optimized_params["T"]
                    if verbose:
                        print(f"   ⚙️ Running ST simulation with S={S}, T={T} for {ingredient_code}")
                    summary_metrics, liberation_df, resultados_replicas, liberation_final = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, S, T
                    )
                    applied_params = {"S": S, "T": T}
                    
                elif policy == "SST":
                    s, S, T = optimized_params["s"], optimized_params["S"], optimized_params["T"]
                    summary_metrics, liberation_df, resultados_replicas, liberation_final = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, s, S, T
                    )
                    applied_params = {"s": s, "S": S, "T": T}
                    
                elif policy == "SS":
                    s, S = optimized_params["s"], optimized_params["S"]
                    summary_metrics, liberation_df, resultados_replicas, liberation_final = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, S, s
                    )
                    applied_params = {"s": s, "S": S}
                    
                elif policy in ["EOQ", "POQ", "LXL"]:
                    # FIXED: Handle both "porcentaje" and "porcentaje_seguridad" parameter names
                    porcentaje_seguridad = optimized_params.get("porcentaje", optimized_params.get("porcentaje_seguridad", 0.35))
                    summary_metrics, liberation_df, resultados_replicas, liberation_final = simulation_function(
                        ingredient_replicas, ingredient_data_dict, punto_venta, porcentaje_seguridad
                    )
                    applied_params = {"porcentaje": porcentaje_seguridad}
            
            # CRITICAL FIX: Ensure liberation_final is numpy array, not pandas Series
            if liberation_final is not None:
                if hasattr(liberation_final, 'values'):
                    liberation_final = liberation_final.values
                elif not isinstance(liberation_final, np.ndarray):
                    liberation_final = np.array(liberation_final)
            
            # CRITICAL DEBUG: Check liberation_final before storing
            if verbose:
                # CRITICAL FIX: Ensure all values are proper scalars, not Series
                liberation_final_sum = float(np.sum(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0.0
                liberation_final_periods = int(np.sum(liberation_final > 0)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0
                liberation_final_unique = int(len(np.unique(liberation_final))) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0
                print(f"   🎯 LIBERATION FINAL for '{ingredient_code}':")
                print(f"      Total orders: {liberation_final_sum:.0f}g")
                print(f"      Periods with orders: {liberation_final_periods}")
                print(f"      Unique order values: {liberation_final_unique}")
                if liberation_final is not None and hasattr(liberation_final, '__iter__'):
                    print(f"      First 10 periods: {liberation_final[:10]}")
                    print(f"      Non-zero periods: {[i for i, v in enumerate(liberation_final) if v > 0]}")
            
            # Store results
            family_liberation_results[ingredient_code] = {
                "liberation_df": liberation_df,  # Matrix of orders by period and replica  
                "liberation_final": liberation_final,  # FINAL VECTOR - this goes to "Órdenes_Finales" sheet
                "summary_metrics": summary_metrics,
                "optimized_params": applied_params,
                "resultados_replicas": resultados_replicas,  # Individual replica results
                "ingredient_data_dict": ingredient_data_dict,
                "ingredient_replicas": ingredient_replicas
            }
            
            if verbose:
                total_orders = float(liberation_df.sum().sum())
                periods_with_orders = int((liberation_df > 0).any(axis=1).sum())
                
                # CRITICAL FIX: Extract avg_cost safely - it might be a Series
                avg_cost_raw = summary_metrics.loc["Costo total", "Promedio Indicadores"]
                if hasattr(avg_cost_raw, 'iloc'):
                    avg_cost = float(avg_cost_raw.iloc[0]) if len(avg_cost_raw) > 0 else 0.0
                elif hasattr(avg_cost_raw, '__float__'):
                    avg_cost = float(avg_cost_raw)
                else:
                    avg_cost = float(avg_cost_raw) if avg_cost_raw else 0.0
                
                # Debug liberation_final vector
                liberation_vector_sum = float(sum(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0.0
                liberation_vector_length = int(len(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__len__') else 0
                
                print(f"   ✅ Success! Matrix total: {total_orders:.0f}g, Vector final: {liberation_vector_sum:.0f}g ({liberation_vector_length} períodos)")
                print(f"   📊 Active periods: {periods_with_orders}, Total cost: {avg_cost:.2f}")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Error processing {ingredient_code}: {str(e)}")
            family_liberation_results[ingredient_code] = {
                "error": str(e),
                "liberation_df": None,
                "summary_metrics": None,
                "optimized_params": None
            }
    
    if verbose:
        successful = sum(1 for r in family_liberation_results.values() if "error" not in r)
        print(f"\n📊 SUMMARY: {successful}/{len(family_ingredients)} ingredients processed successfully")
        
        # CRITICAL: Show conversion differences between ingredients
        print(f"\n🔍 INGREDIENT-SPECIFIC CONVERSION VERIFICATION:")
        print(f"="*70)
        print(f"{'Ingredient':<30} {'Avg Demand':<15} {'Total Orders':<15} {'Unique'}")
        print(f"-"*70)
        
        for ingredient_code, results in family_liberation_results.items():
            if "error" not in results:
                ingredient_replicas = results.get("ingredient_replicas")
                liberation_final = results.get("liberation_final", [])
                
                if ingredient_replicas is not None and hasattr(ingredient_replicas, 'mean'):
                    # CRITICAL FIX: Ensure all values are proper scalars
                    avg_demand = float(np.mean(ingredient_replicas))
                    total_orders = float(sum(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0.0
                    unique_vals = int(len(np.unique(ingredient_replicas)))
                    
                    print(f"{str(ingredient_code):<30} {avg_demand:<15.2f} {total_orders:<15.0f} {unique_vals}")
        
        print(f"="*70)
        print(f"✅ Each ingredient should have DIFFERENT values above")
        print(f"⚠️  If all values are identical, the conversion is NOT working correctly")
    
    return family_liberation_results


def export_family_liberation_to_excel(
    family_liberation_results: Dict[str, Dict],
    family_id: int,
    policy: str,
    punto_venta: str,
    output_dir: str = "optimization_results"
) -> str:
    """
    Export family liberation vectors to Excel file.
    
    Parameters:
    - family_liberation_results: Results from generate_family_liberation_vectors
    - family_id: ID of the ingredient family
    - policy: Policy used for optimization
    - punto_venta: Point of sale name
    - output_dir: Directory to save the Excel file
    
    Returns:
    - output_path: Path to the created Excel file
    """
    import os
    from datetime import datetime
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"family_{family_id}_liberation_{policy}_{punto_venta}_{timestamp}.xlsx"
    output_path = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # Summary sheet
        summary_data = []
        for ingredient_code, results in family_liberation_results.items():
            if "error" not in results:
                metrics = results["summary_metrics"]
                params = results["optimized_params"]
                
                row = {
                    "Ingredient": ingredient_code,
                    "Policy": policy,
                    "Parameters": str(params),
                    "Total_Cost": metrics.loc["Costo total", "Promedio Indicadores"],
                    "Avg_Inventory": metrics.loc["Inventario promedio", "Promedio Indicadores"],
                    "Service_Level": metrics.loc["Proporción demanda satisfecha", "Promedio Indicadores"],
                    "Total_Orders": results["liberation_df"].sum().sum()
                }
                summary_data.append(row)
            else:
                summary_data.append({
                    "Ingredient": ingredient_code,
                    "Policy": policy,
                    "Parameters": "ERROR",
                    "Error": results["error"]
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Individual liberation sheets
        for ingredient_code, results in family_liberation_results.items():
            if "error" not in results:
                # Sanitize ingredient name for sheet names
                clean_ingredient = sanitize_excel_sheet_name(str(ingredient_code), 20)
                
                # Liberation vectors sheet
                results["liberation_df"].to_excel(
                    writer, sheet_name=f"{clean_ingredient}_Liberation"
                )
                
                # Metrics sheet
                results["summary_metrics"].to_excel(
                    writer, sheet_name=f"{clean_ingredient}_Metrics"
                )
    
    print(f"📊 Family liberation results exported to: {output_path}")
    return output_path


def apply_representative_optimization_to_family(
    cluster_info: Dict,
    family_id: int,
    optimization_result: Dict,
    pizza_data_dict: Dict,
    pizza_replicas_matrix: np.ndarray,
    punto_venta: str,
    recetas_primero: Dict,
    recetas_segundo: Dict,
    materia_prima: Dict,
    output_dir: str = "optimization_results"
) -> Dict:
    """
    Apply optimization results from representative ingredient to entire family.
    
    Parameters:
    - cluster_info: Clustering information with family assignments
    - family_id: ID of the family to process
    - optimization_result: Results from PSO optimization of representative ingredient
    - pizza_data_dict: Original pizza data dictionary
    - pizza_replicas_matrix: Matrix of pizza demand replicas
    - punto_venta: Point of sale name
    - recetas_primero: First level recipes
    - recetas_segundo: Second level recipes
    - materia_prima: Raw materials information
    - output_dir: Directory for output files
    
    Returns:
    - complete_family_results: Dict with all family processing results
    """
    
    # Get family ingredients
    df_clustered = cluster_info["df_clustered"]
    family_ingredients = df_clustered[df_clustered["Cluster"] == family_id].index.tolist()
    
    # Get representative ingredient
    representative_ingredient = cluster_info["medoids"][family_id].name
    
    # Extract optimization parameters
    policy = optimization_result.get("policy", "EOQ")
    best_params = optimization_result.get("best_params", {})
    
    print(f"\n🎯 APPLYING REPRESENTATIVE OPTIMIZATION TO FAMILY {family_id}")
    print(f"👥 Family ingredients: {family_ingredients}")
    print(f"🏆 Representative: {representative_ingredient}")
    print(f"⚙️ Policy: {policy}")
    print(f"📊 Optimized parameters: {best_params}")
    
    # Generate liberation vectors for entire family
    family_liberation_results = generate_family_liberation_vectors(
        family_ingredients=family_ingredients,
        representative_ingredient=representative_ingredient,
        optimized_params=best_params,
        policy=policy,
        pizza_data_dict=pizza_data_dict,
        pizza_replicas_matrix=pizza_replicas_matrix,
        punto_venta=punto_venta,
        recetas_primero=recetas_primero,
        recetas_segundo=recetas_segundo,
        materia_prima=materia_prima,
        verbose=True
    )
    
    # Export to Excel
    excel_path = export_family_liberation_to_excel(
        family_liberation_results=family_liberation_results,
        family_id=family_id,
        policy=policy,
        punto_venta=punto_venta,
        output_dir=output_dir
    )
    
    # Compile complete results
    complete_family_results = {
        "family_id": family_id,
        "family_ingredients": family_ingredients,
        "representative_ingredient": representative_ingredient,
        "policy": policy,
        "optimized_params": best_params,
        "liberation_results": family_liberation_results,
        "excel_export_path": excel_path,
        "optimization_result": optimization_result
    }
    
    return complete_family_results


# Example usage function
def example_usage():
    """
    Example of how to use the family liberation generation functions.
    """
    
    # This would typically be called after clustering and optimization:
    """
    # 1. After clustering ingredients into families
    df_clustered, cluster_info = perform_ingredient_clustering(
        selected_ingredients, materia_prima, recetas_primero, recetas_segundo
    )
    
    # 2. After optimizing representative ingredient for family 1
    optimization_result = optimize_cluster_policy(
        policy="QR", 
        cluster_id=1, 
        cluster_info=cluster_info, 
        data_dict_MP=data_dict_MP,
        punto_venta="Terraplaza"
    )
    
    # 3. Apply optimization to entire family
    family_results = apply_representative_optimization_to_family(
        cluster_info=cluster_info,
        family_id=1,
        optimization_result=optimization_result,
        pizza_data_dict=original_pizza_data,
        pizza_replicas_matrix=pizza_replicas,
        punto_venta="Terraplaza",
        recetas_primero=recetas_primero,
        recetas_segundo=recetas_segundo,
        materia_prima=materia_prima
    )
    
    # 4. Access liberation vectors
    for ingredient, results in family_results["liberation_results"].items():
        liberation_df = results["liberation_df"]
        print(f"Liberation orders for {ingredient}:")
        print(liberation_df.head())
    """
    
    pass


if __name__ == "__main__":
    print("Family Liberation Generator - Use as imported module")
    example_usage()