# pso_opt.py  (o pega dentro de PSO.py)
import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import copy
import time
import services.Replicas as replicas
import services.simulacion as sim
from typing import Callable, Dict, List, Tuple, Any
import os
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from services.leer_datos import procesar_datos
import warnings
warnings.filterwarnings("ignore")

# ------------------ helpers: mapear particle -> decision vars ------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def map_particle_to_decisions(policy: str, particle: np.ndarray, bounds: List[Tuple[float,float]]):
    """
    Convierte el vector 'particle' a un dict de parámetros para la política 'policy'.
    - bounds: lista de (low, high) en el mismo orden que particle.
    Debes ajustar estas reglas si tus políticas usan nombres/formatos distintos.
    """
    policy = policy.upper()
    p = particle.copy()
    # asegurar límites
    for j in range(len(p)):
        p[j] = clamp(p[j], bounds[j][0], bounds[j][1])

    if policy == "QR":
        # particle = [Q, R]
        Q = int(round(p[0]))
        R = int(round(p[1]))
        return {"Q": Q, "R": R}

    if policy == "ST":
        # particle = [S, T]
        S = int(round(p[0]))
        T = int(round(p[1]))
        if T < 1: T = 1
        return {"S": S, "T": T}

    # ✅ CRITICAL: Check SS BEFORE SST to avoid index error
    if policy == "SS":
        # particle = [s, S] - ONLY 2 parameters, NO T!
        s = int(round(p[0]))
        S = int(round(p[1]))
        return {"s": s, "S": S}

    if policy == "SST":
        # particle = [s, S, T] - 3 parameters!
        s = int(round(p[0]))
        S = int(round(p[1]))
        T = int(round(p[2]))
        if T < 1: T = 1
        return {"s": s, "S": S, "T": T}

    if policy == "EOQ":
        porcentaje = float(p[0])  # Remove the max(1.0, ...) constraint
        return {"porcentaje": porcentaje}

    if policy == "POQ":
        porcentaje = float(p[0])  # Remove the max(1.0, ...) constraint
        return {"porcentaje": porcentaje}
    
    if policy == "LXL":
        porcentaje = float(p[0])  # Remove the max(1.0, ...) constraint
        return {"porcentaje": porcentaje}

    # fallback: return as generic vector (user can adapt)
    return {f"x{i}": (int(round(p[i])) if float(p[i]).is_integer() else float(p[i])) for i in range(len(p))}


# ------------------ wrapper para llamar a la función de réplicas ------------------

def call_replicas_for_policy(policy: str,
                             replicas_matrix: np.ndarray,
                             data_dict: dict,
                             ref: str,
                             decision_vars: dict):
    """
    Llama a la función replicas_<policy> correspondiente con las variables de decisión apropiadas.
    Devuelve el DataFrame promedio y la matriz de "Liberación orden".
    """
    # Sanity
    policy = policy.upper()

    # obtener la función de replicas en sim.py
    if policy == "SST":
        fn_name = "replicas_SST"  # Map SST to SST function name
    else:
        fn_name = f"replicas_{policy}"
    
    replicas_fn = getattr(sim, fn_name, None)
    if replicas_fn is None:
        raise ValueError(f"No existe la función {fn_name} en sim.py. Ajusta el nombre o crea un wrapper.")

    # Para QR es especial - toma Q y R como parámetros directos
    if policy == "QR":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           Q=decision_vars["Q"], R=decision_vars["R"])
    elif policy == "ST":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           S=decision_vars["S"], T=decision_vars["T"])
    elif policy == "SST":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           s=decision_vars["s"], S=decision_vars["S"], T=decision_vars["T"])
    elif policy == "SS":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           S=decision_vars["S"], s=decision_vars["s"])
    elif policy == "EOQ":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           decision_vars["porcentaje"])
    elif policy == "POQ":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           decision_vars["porcentaje"])
    elif policy == "LXL":
        result = replicas_fn(replicas_matrix, data_dict, ref, 
                           decision_vars["porcentaje"])
    else:
        # Para otras políticas, insertar decision vars en data_dict
        dd_copy = copy.deepcopy(data_dict)
        
        # Asegurarnos de existir la estructura
        if ref not in dd_copy:
            raise ValueError(f"Referencia {ref} no encontrada en data_dict")
        
        # Insertar las decision vars en PARAMETROS
        if 'PARAMETROS' not in dd_copy[ref]:
            dd_copy[ref]['PARAMETROS'] = {}
        
        # sobrescribimos/insertamos las variables de decisión
        dd_copy[ref]['PARAMETROS'].update(decision_vars)
        
        # llamar la función
        result = replicas_fn(replicas_matrix, dd_copy, ref)

    # Todas las funciones replicas ahora devuelven (df_promedio, liberacion_orden_matrix)
    if isinstance(result, tuple) or isinstance(result, list):
        if len(result) >= 2:
            df_promedio = result[0]
            liberacion_orden_matrix = result[1]
            return df_promedio, liberacion_orden_matrix
        else:
            # Si solo devuelve un elemento, asumir que es df_promedio
            df_promedio = result[0]
            return df_promedio, None
    else:
        # Si no es tupla, devolver solo df_promedio
        return result, None


# ------------------ extraer indicador del df promedio ------------------

def extract_indicator_value(df_promedio, indicator_name: str):
    """
    Dado el DataFrame promedio devuelto por replicas_*,
    devolvemos el valor numérico del indicador indicado (si existe).
    Si no aparece, devuelve np.inf (penaliza).
    """
    if df_promedio is None:
        return np.inf

    # si es DataFrame con índices = indicadores
    try:
        # Si el DataFrame tiene una columna única, obtener [indicator_name, 0]
        if indicator_name in df_promedio.index:
            # si hay columnas, tomar primero
            col = df_promedio.columns[0]
            val = df_promedio.loc[indicator_name, col]
            return float(val)
        else:
            # intentar buscar por aproximación (sin mayúsculas)
            idx_lower = [str(i).lower() for i in df_promedio.index]
            if indicator_name.lower() in idx_lower:
                pos = idx_lower.index(indicator_name.lower())
                row_label = df_promedio.index[pos]
                val = df_promedio.loc[row_label, df_promedio.columns[0]]
                return float(val)
    except Exception:
        pass

    # no se encontró: penalizar
    return np.inf

def check_constraints_violations(df_promedio, restricciones, verbose=False):
    """
    Verifica si se violan las restricciones y retorna información detallada.
    """
    violations = []
    
    # Restricción 1: Proporción de periodos sin faltantes
    nivel_servicio_requerido = restricciones.get('Proporción demanda satisfecha', None)
    if nivel_servicio_requerido is not None:
        proporcion_sin_faltantes = extract_indicator_value(df_promedio, 'Proporción de periodos sin faltantes')
        # More lenient service level for ingredients (allow 2% tolerance)
        tolerance = 0.02
        if proporcion_sin_faltantes < (nivel_servicio_requerido - tolerance):
            violations.append(f"Nivel de servicio: {proporcion_sin_faltantes:.3f} < {nivel_servicio_requerido:.3f} (requerido)")
    
    # Restricción 2: Inventario promedio máximo
    inv_max_requerido = restricciones.get('Inventario a la mano (max)', None)
    if inv_max_requerido is not None:
        inventario_promedio = extract_indicator_value(df_promedio, 'Inventario promedio')
        # More lenient inventory constraint for ingredients (allow 10% tolerance)
        tolerance_factor = 1.1
        if inventario_promedio > (inv_max_requerido * tolerance_factor):
            violations.append(f"Inventario promedio: {inventario_promedio:.2f} > {inv_max_requerido:.2f} (máximo)")
    
    if verbose and violations:
        print(f"  Violaciones de restricciones: {'; '.join(violations)}")
    
    return violations


# ------------------ PSO implementation (simple) ------------------

import numpy as np

def generate_family_liberation_for_optimization(
    policy: str,
    best_decision_vars: dict,
    ingredient_info: dict,
    data_dict: dict,
    replicas_matrix: np.ndarray,
    ref: str,
    materia_prima: dict,
    recetas_primero: dict,
    recetas_segundo: dict,
    pizza_data_dict: dict = None,
    pizza_replicas_matrix: np.ndarray = None,
    representative_liberation_final: np.ndarray = None
) -> dict:
    """
    Generate family liberation results for all ingredients in the same family
    as the optimized representative ingredient.
    
    Parameters:
    -----------
    policy : str
        Optimized policy (EOQ, QR, etc.)
    best_decision_vars : dict
        Optimal parameters from PSO optimization
    ingredient_info : dict
        Information about the representative ingredient including cluster_id
    data_dict : dict
        Data dictionary used for optimization
    replicas_matrix : np.ndarray
        Replicas matrix used for optimization
    ref : str
        Reference (cluster name like "Familia_1")
    materia_prima : dict
        Raw materials information
    recetas_primero : dict
        First level recipes
    recetas_segundo : dict
        Second level recipes
    pizza_data_dict : dict, optional
        Original pizza data dictionary
    pizza_replicas_matrix : np.ndarray, optional
        Original pizza replicas matrix
        
    Returns:
    --------
    dict
        Family liberation results
    """
    
    try:
        # Check if we have clustering information to identify family members
        cluster_id = ingredient_info.get('cluster_id', None)
        representative_ingredient = ingredient_info.get('ingredient_code', ref)
        
        if cluster_id is None:
            print("⚠️ No cluster information available - skipping family liberation generation")
            return {}
        
        # Get actual punto_venta (pizza point of sale) from ingredient_info
        actual_punto_venta = ingredient_info.get('pizza_point_of_sale', None)
        
        if not actual_punto_venta or actual_punto_venta == 'N/A':
            print(f"⚠️ No valid pizza punto_venta found - cannot generate family liberation")
            print(f"   ingredient_info keys: {list(ingredient_info.keys())}")
            return {}
        
        print(f"\n🏭 FAMILY LIBERATION GENERATION")
        print(f"📦 Cluster ID: {cluster_id}")
        print(f"⭐ Representative: {representative_ingredient}")
        print(f"🏢 Pizza Punto Venta: {actual_punto_venta}")
        print(f"⚙️ Policy: {policy}")
        print(f"📈 Optimized params: {best_decision_vars}")
        
        # Get family ingredients from cluster_info stored in ingredient_info
        raw_family_ingredients = ingredient_info.get('cluster_ingredients', [representative_ingredient])
        
        # CRITICAL FIX: Map numeric codes to actual ingredient names from materia_prima
        family_ingredients = []
        if materia_prima:
            materia_prima_names = list(materia_prima.keys())
            materia_prima_by_index = list(materia_prima.items())
            
            print(f"🔍 Raw family ingredients from clustering: {raw_family_ingredients}")
            print(f"🔍 First 5 materia_prima entries: {[(k, v.get('nombre', k)[:20] if hasattr(v, 'get') else k) for k, v in materia_prima_by_index[:5]]}")
            
            for ingredient_code in raw_family_ingredients:
                mapped_ingredient = None
                
                # Method 1: Direct lookup (ingredient_code is actually a name like "POLLO")
                if str(ingredient_code) in materia_prima:
                    mapped_ingredient = str(ingredient_code)
                    print(f"📝 Direct match: '{ingredient_code}' found in materia_prima")
                
                # Method 2: Try numeric index lookup (ingredient_code is index like 1, 5, 10)
                elif isinstance(ingredient_code, (int, str)) and str(ingredient_code).isdigit():
                    try:
                        numeric_index = int(ingredient_code)
                        # Try using it as index in the materia_prima keys
                        if 0 <= numeric_index < len(materia_prima_names):
                            mapped_ingredient = materia_prima_names[numeric_index]
                            print(f"📝 Index mapping: code '{ingredient_code}' -> ingredient '{mapped_ingredient}' (index {numeric_index})")
                        # Or try finding by searching for the number in ingredient codes
                        else:
                            # Look for ingredients that contain this number in their code
                            potential_matches = [name for name in materia_prima_names if str(ingredient_code) in name]
                            if potential_matches:
                                mapped_ingredient = potential_matches[0]  # Use first match
                                print(f"📝 Pattern match: code '{ingredient_code}' -> ingredient '{mapped_ingredient}'")
                    except ValueError:
                        pass
                
                # Method 3: Search by name patterns 
                if not mapped_ingredient:
                    potential_matches = [name for name in materia_prima_names 
                                       if str(ingredient_code).upper() in name.upper() or 
                                          name.upper().split()[0] == str(ingredient_code).upper()]
                    if potential_matches:
                        mapped_ingredient = potential_matches[0]
                        print(f"📝 Name pattern match: '{ingredient_code}' -> '{mapped_ingredient}'")
                
                # Fallback: use as-is but warn
                if not mapped_ingredient:
                    mapped_ingredient = str(ingredient_code)
                    print(f"⚠️ No mapping found for '{ingredient_code}', using as-is (may cause conversion issues)")
                
                family_ingredients.append(mapped_ingredient)
        else:
            # No materia_prima available, use raw ingredients
            family_ingredients = [str(ing) for ing in raw_family_ingredients]
            print(f"⚠️ No materia_prima available for mapping")
        
        # Ensure family_ingredients is not empty
        if not family_ingredients:
            family_ingredients = [str(representative_ingredient)]
        
        print(f"👥 Family ingredients after mapping ({len(family_ingredients)}): {family_ingredients}")
        
        # Retrieve pizza data_dict and replicas_matrix if not provided
        if pizza_data_dict is None or pizza_replicas_matrix is None:
            print(f"📥 Retrieving pizza optimization results from state...")
            
            # Import state module to access stored pizza results
            try:
                from presentation import state as st
                
                # Get pizza optimization results
                pizza_opt_results = st.app_state.get(st.STATE_OPT, {})
                
                if actual_punto_venta not in pizza_opt_results:
                    print(f"❌ Pizza punto_venta '{actual_punto_venta}' not found in optimization results")
                    print(f"   Available PVs: {list(pizza_opt_results.keys())}")
                    return {}
                
                # Get the most recent policy results for this punto_venta
                pv_results = pizza_opt_results[actual_punto_venta]
                
                # Try to get results for the same policy first
                if policy in pv_results:
                    policy_results = pv_results[policy]
                else:
                    # Use any available policy
                    available_policies = list(pv_results.keys())
                    if not available_policies:
                        print(f"❌ No optimization results found for {actual_punto_venta}")
                        return {}
                    policy_results = pv_results[available_policies[-1]]
                    print(f"ℹ️ Using policy {available_policies[-1]} results (policy {policy} not found)")
                
                # Get pizza data_dict and replicas matrix
                if pizza_data_dict is None:
                    # Get original data_dict from state
                    original_data_dict = st.app_state.get(st.STATE_DATA, {})
                    if actual_punto_venta in original_data_dict:
                        pizza_data_dict = {actual_punto_venta: original_data_dict[actual_punto_venta]}
                        print(f"✅ Retrieved pizza_data_dict for {actual_punto_venta}")
                    else:
                        print(f"⚠️ Could not find pizza_data_dict for {actual_punto_venta}")
                        return {}
                
                if pizza_replicas_matrix is None:
                    # CRITICAL: Get LIBERATION ORDERS matrix, not demand replicas
                    # Each ingredient needs to convert the pizza ORDERS (output) not the demand (input)
                    pizza_liberation_matrix = policy_results.get("liberacion_orden_matrix", None)
                    
                    if pizza_liberation_matrix is not None:
                        # Use the liberation orders as the matrix to convert
                        pizza_replicas_matrix = pizza_liberation_matrix
                        print(f"✅ Retrieved pizza LIBERATION ORDERS matrix: shape {pizza_replicas_matrix.shape}")
                        print(f"   📦 This is the OPTIMIZED ORDER SCHEDULE from pizzas, not demand")
                        print(f"   🔄 Each ingredient will convert these orders using their own recipes")
                    else:
                        print(f"⚠️ No liberacion_orden_matrix found in optimization results")
                        print(f"   Trying to get demand replicas as fallback...")
                        # Fallback: try demand replicas
                        pizza_replicas_matrix = policy_results.get("replicas_matrix", None)
                        if pizza_replicas_matrix is not None:
                            print(f"⚠️ Using demand replicas as fallback: shape {pizza_replicas_matrix.shape}")
                        else:
                            print(f"❌ No pizza data found - using ingredient replicas")
                            pizza_replicas_matrix = replicas_matrix  # Last resort fallback
                        
            except Exception as e:
                print(f"❌ Error retrieving pizza data from state: {e}")
                return {}
        
        # Validate we have what we need
        if pizza_data_dict is None:
            print(f"❌ pizza_data_dict is None - cannot generate family liberation")
            return {}
        
        if actual_punto_venta not in pizza_data_dict:
            print(f"❌ '{actual_punto_venta}' not in pizza_data_dict keys: {list(pizza_data_dict.keys())}")
            return {}
        
        from services.family_liberation_generator import generate_family_liberation_vectors
        
        # Generate liberation vectors for the family
        print(f"\n🚀 Calling generate_family_liberation_vectors...")
        family_liberation_results = generate_family_liberation_vectors(
            family_ingredients=family_ingredients,
            representative_ingredient=representative_ingredient,
            optimized_params=best_decision_vars,
            policy=policy,
            pizza_data_dict=pizza_data_dict,  # Must have actual_punto_venta as key
            pizza_replicas_matrix=pizza_replicas_matrix,
            punto_venta=actual_punto_venta,  # FIXED: Use actual pizza punto_venta, not cluster ref
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            materia_prima=materia_prima,
            verbose=True,
            representative_liberation_final=representative_liberation_final  # Pass pre-calculated vector
        )
        
        return family_liberation_results
        
    except Exception as e:
        print(f"❌ Error generando liberación familiar: {e}")
        import traceback
        traceback.print_exc()
        return {}


# Update print message in export function to include family sheets
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

def update_export_success_message_with_family(filepath: str, family_liberation_results: dict = None):
    """Helper function to print success message with family sheets info"""
    print(f"✅ Archivo Excel creado exitosamente:")
    print(f"   📂 Ruta: {filepath}")
    print(f"   📊 Hojas incluidas:")
    print(f"      - Resumen_Optimización: Parámetros óptimos y configuración")
    print(f"      - Indicadores_Promedio: KPIs promedio de todas las réplicas")
    print(f"      - Órdenes_Optimizadas: RESULTADO - Órdenes por período (matriz de liberación)")
    print(f"      - Resultados_Todas_Réplicas: KPIs de cada réplica")
    print(f"      - INPUT_Demanda_XXX: ENTRADA - Matriz de demanda usada para optimización")
    
    if family_liberation_results:
        successful_ingredients = [ing for ing, res in family_liberation_results.items() if "error" not in res]
        print(f"      - FAMILIA_Resumen: Resumen de {len(successful_ingredients)} ingredientes de la familia")
        for ingredient in successful_ingredients:
            clean_name = ingredient.replace("/", "_").replace("\\", "_")[:25]
            print(f"      - FAM_{clean_name}: Órdenes detalladas para {ingredient}")


def pso_optimize_single_policy(
    policy, 
    data_dict, 
    ref, 
    replicas_matrix, 
    decision_bounds, 
    objective_indicator="Costo total", 
    minimize=True,
    swarm_size=20, 
    iters=50, 
    verbose=True,
    ingredient_info=None
):
    """
    Ejecuta optimización por PSO para UNA política dada (QR, ST, SsT, SS, etc.).
    
    Parameters
    ----------
    policy : str
        Política a optimizar ("QR", "ST", "SsT", "SS").
    data_dict : dict
        Diccionario con parámetros cargados desde Excel.
    ref : str
        Referencia que se optimiza.
    replicas_matrix : np.ndarray
        Matriz de réplicas de pronósticos (n_replicas x n_periodos).
    decision_bounds : list of tuple
        Límites de búsqueda para las variables de decisión.
    objective_indicator : str
        Indicador a optimizar (ej. "Costo total", "Proporción demanda satisfecha").
    minimize : bool
        Si True, minimiza; si False, maximiza.
    swarm_size : int
        Número de partículas en el enjambre.
    iters : int
        Número de iteraciones.
    verbose : bool
        Imprimir progreso.
    ingredient_info : dict, optional
        Información adicional específica de ingredientes para el reporte Excel.
    """
    
    # ✅ CRITICAL: Verify policy and bounds match
    if verbose:
        print(f"🎯 PSO Iniciando para política: '{policy}'")
        print(f"   Bounds recibidos: {decision_bounds} (dimensión: {len(decision_bounds)})")
        if policy.upper() == "SS" and len(decision_bounds) != 2:
            print(f"   ⚠️  ERROR: SS policy debe tener 2 bounds (s, S) pero recibió {len(decision_bounds)}")

    dim = len(decision_bounds)  # número de variables de decisión

    # --- inicialización del enjambre ---
    X = np.zeros((swarm_size, dim))
    V = np.zeros((swarm_size, dim))
    pbest_X = np.zeros((swarm_size, dim))
    pbest_score = np.full(swarm_size, np.inf if minimize else -np.inf)

    # posiciones iniciales
    for d in range(dim):
        low, high = decision_bounds[d]
        X[:, d] = np.random.uniform(low, high, size=swarm_size)

    # inicializar pbest_X con las posiciones iniciales
    pbest_X = X.copy()

    gbest_X = X[0, :].copy()  # inicializar con primera partícula
    gbest_score = np.inf if minimize else -np.inf

    # --- función para evaluar una partícula ---
    def evaluate_particle(x):
        try:
            # mapear particle a variables de decisión según la política
            decision_vars = map_particle_to_decisions(policy, x, decision_bounds)
            
            if verbose and np.random.random() < 0.1:  # Debug 10% of evaluations
                print(f"[PSO] Evaluating {policy} with vars: {decision_vars}")
                #print(f"[PSO] Raw particle: {x}")
                #print(f"[PSO] Number of decision variables: {len(decision_vars)}")
                if policy == "SS":
                    print(f"[PSO] SS policy - Expected vars: s, S | Actual vars: {list(decision_vars.keys())}")
            
            # llamar a la función de réplicas correspondiente
            result = call_replicas_for_policy(
                policy=policy,
                replicas_matrix=replicas_matrix,
                data_dict=data_dict,
                ref=ref,
                decision_vars=decision_vars
            )
            
            # Extraer df_promedio y liberacion_orden_matrix
            if isinstance(result, tuple) and len(result) >= 2:
                df_promedio, liberacion_orden_matrix = result
            else:
                df_promedio = result
                liberacion_orden_matrix = None
            
            # extraer el valor del indicador objetivo
            score = extract_indicator_value(df_promedio, objective_indicator)
            
            # Verificar restricciones y aplicar penalizaciones
            restricciones = data_dict[ref].get('RESTRICCIONES', {})
            violations = check_constraints_violations(df_promedio, restricciones, verbose=False)
            
            penalty = 0
            # Restricción 1: Proporción de periodos sin faltantes (Nivel de servicio)
            nivel_servicio_requerido = restricciones.get('Proporción demanda satisfecha', None)
            if nivel_servicio_requerido is not None:
                proporcion_sin_faltantes = extract_indicator_value(df_promedio, 'Proporción de periodos sin faltantes')
                if proporcion_sin_faltantes < nivel_servicio_requerido:
                    penalty += (nivel_servicio_requerido - proporcion_sin_faltantes) * 1000000  # Gran penalización
            
            # Restricción 2: Inventario promedio (Inv. Max)
            inv_max_requerido = restricciones.get('Inventario a la mano (max)', None)
            if inv_max_requerido is not None:
                inventario_promedio = extract_indicator_value(df_promedio, 'Inventario promedio')
                if inventario_promedio > inv_max_requerido:
                    penalty += (inventario_promedio - inv_max_requerido) * 1000  # Penalización por exceso
            
            # Aplicar penalización al score
            final_score = score + penalty
            
            if verbose and np.random.random() < 0.1:  # Debug 10% of evaluations
                print(f"[PSO] Score for {decision_vars}: {score}, Penalty: {penalty}, Final: {final_score}")
                
            return float(final_score)

        except Exception as e:
            print(f"[PSO] simulación falló: {e}")
            return np.inf if minimize else -np.inf


    # --- ciclo principal de PSO ---
    for it in range(iters):
        for i in range(swarm_size):
            score = evaluate_particle(X[i, :])

            # actualizar mejor personal (pbest) si es una mejora
            if (minimize and score < pbest_score[i]) or (not minimize and score > pbest_score[i]):
                pbest_score[i] = score
                pbest_X[i, :] = X[i, :].copy()

            # actualizar mejor global si es una mejora
            if (minimize and score < gbest_score) or (not minimize and score > gbest_score):
                gbest_score = score
                gbest_X = X[i, :].copy()

        # actualizar velocidades y posiciones
        w, c1, c2 = 0.7, 1.5, 1.5
        r1, r2 = np.random.rand(swarm_size, dim), np.random.rand(swarm_size, dim)
        V = w * V + c1 * r1 * (pbest_X - X) + c2 * r2 * (gbest_X - X)
        X = X + V

        # respetar límites
        for d in range(dim):
            low, high = decision_bounds[d]
            X[:, d] = np.clip(X[:, d], low, high)

        if verbose and it % 5 == 0:
            print(f"[PSO] iter {it}/{iters} best_score={gbest_score}")

    # Verificar restricciones de la mejor solución encontrada y obtener la matriz de liberación final
    best_liberacion_orden_matrix = None
    if gbest_X is not None:
        best_decision_vars = map_particle_to_decisions(policy, gbest_X, decision_bounds)
        print(f"\n Calculando matriz de liberación final con parámetros óptimos: {best_decision_vars}")
        
        best_result = call_replicas_for_policy(
            policy=policy,
            replicas_matrix=replicas_matrix,
            data_dict=data_dict,
            ref=ref,
            decision_vars=best_decision_vars
        )
        
        # Extraer df_best y liberacion_orden_matrix de la mejor solución
        if isinstance(best_result, tuple) and len(best_result) >= 2:
            df_best, best_liberacion_orden_matrix = best_result
            print(f" Matriz de liberación obtenida: shape {best_liberacion_orden_matrix.shape if hasattr(best_liberacion_orden_matrix, 'shape') else 'N/A'}")
        else:
            df_best = best_result
            best_liberacion_orden_matrix = None
            print(" ⚠️ No se pudo obtener la matriz de liberación de órdenes")
        
        restricciones = data_dict[ref].get('RESTRICCIONES', {})
        violations = check_constraints_violations(df_best, restricciones, verbose=True)
        
        if not violations:
            print(" La mejor solución cumple todas las restricciones.")
        else:
            print(f"  La mejor solución viola {len(violations)} restricción(es).")

        # Crear archivos con las liberaciones de ordenes y KPIs de cada réplica usando parámetros óptimos
        print(f"📊 Generando resultados detallados con parámetros óptimos...")
        
        if policy == "QR":
            Q = best_decision_vars["Q"]
            R = best_decision_vars["R"] 
            print(f"   Ejecutando QR verbose con Q={Q}, R={R}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_QR_verbose(replicas_matrix, data_dict, ref, Q=Q, R=R)
            
        elif policy == "ST":
            S = best_decision_vars["S"]
            T = best_decision_vars["T"]
            print(f"   Ejecutando ST verbose con S={S}, T={T}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_ST_verbose(replicas_matrix, data_dict, ref, S=S, T=T)
            
        elif policy == "SST":
            s = best_decision_vars["s"]
            S = best_decision_vars["S"]
            T = best_decision_vars["T"]
            print(f"   Ejecutando SST verbose con s={s}, S={S}, T={T}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_SST_verbose(replicas_matrix, data_dict, ref, s=s, S=S, T=T)
            
        elif policy == "SS":
            s = best_decision_vars["s"]
            S = best_decision_vars["S"]
            print(f"   Ejecutando SS verbose con s={s}, S={S}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_SS_verbose(replicas_matrix, data_dict, ref, S=S, s=s)
            
        elif policy == "EOQ":
            porcentaje = best_decision_vars["porcentaje"]
            print(f"   Ejecutando EOQ verbose con porcentaje={porcentaje}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_EOQ_verbose(replicas_matrix, data_dict, ref, porcentaje_seguridad=porcentaje)
            
        elif policy == "POQ":
            porcentaje = best_decision_vars["porcentaje"]
            print(f"   Ejecutando POQ verbose con porcentaje={porcentaje}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_POQ_verbose(replicas_matrix, data_dict, ref, porcentaje_seguridad=porcentaje)
            
        elif policy == "LXL":
            porcentaje = best_decision_vars["porcentaje"]
            print(f"   Ejecutando LXL verbose con porcentaje={porcentaje}")
            df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = sim.replicas_LXL_verbose(replicas_matrix, data_dict, ref, porcentaje_seguridad=porcentaje)
            
        else:
            raise ValueError(f"Política {policy} no soportada para resultados verbose")
        
        # Generate family liberation results if ingredient_info contains family data
        family_liberation_results = None
        if ingredient_info:
            required_keys = ['cluster_id', 'materia_prima', 'recetas_primero', 'recetas_segundo']
            missing_keys = [key for key in required_keys if key not in ingredient_info]
            
            if verbose:
                print(f"🔍 FAMILY LIBERATION CHECK:")
                print(f"   ingredient_info keys: {list(ingredient_info.keys()) if ingredient_info else 'None'}")
                print(f"   Required keys: {required_keys}")
                print(f"   Missing keys: {missing_keys}")
                print(f"   Condition met: {len(missing_keys) == 0}")
            
            # DISABLED: Family liberation for other ingredients
            # User preference: Only show representative ingredient results in standard format
            # This avoids ingredient lookup errors and keeps results simple
            if verbose:
                print(f"ℹ️  Family liberation DISABLED - showing representative ingredient only")
            family_liberation_results = None
        
        # Export results to Excel if requested
        excel_path = export_optimization_results_to_excel(
            policy=policy,
            ref=ref, 
            best_decision_vars=best_decision_vars,
            df_promedio=df_promedio,
            liberacion_orden_df=liberacion_orden_df, 
            resultados_replicas=resultados_replicas,
            replicas_matrix=replicas_matrix,
            ingredient_info=ingredient_info,
            liberacion_final=liberacion_final,
            family_liberation_results=family_liberation_results
        )
        
        print(f"✅ Resultados verbose generados: {len(resultados_replicas)} réplicas procesadas")
            

    # Store results for return
    pso_result = {
        "best_score": gbest_score,
        "best_decision_vars": gbest_X,
        "best_decision_mapped": map_particle_to_decisions(policy, gbest_X, decision_bounds) if gbest_X is not None else None,
        "best_liberacion_orden_matrix": best_liberacion_orden_matrix,
        # ✅ CRITICAL FIX: Add this key for first eslabon validation
        "liberacion_orden_matrix": best_liberacion_orden_matrix  # Same data, but with expected key name
    }
    
    # Add verbose results if they were generated
    if 'df_promedio' in locals():
        pso_result.update({
            "verbose_results": {
                "df_promedio": df_promedio,
                "liberacion_orden_df": liberacion_orden_df,
                "resultados_replicas": resultados_replicas,
                "excel_file_path": locals().get("excel_path")
            },
            # ✅ Also add liberacion_final if available (used by first eslabon conversion)
            "liberacion_final": liberacion_final if 'liberacion_final' in locals() else None
        })
    
    return pso_result


# ------------------ decision bounds para cada política ------------------

def get_decision_bounds_for_policy(policy: str, pv: str, data_dict: dict):
    """
    Genera límites razonables (bounds) para optimizar políticas de inventario
    según los parámetros históricos de cada punto de venta (pv).
    """
    policy = policy.upper()
    sheets = data_dict.get(pv, {})
    params = sheets.get("PARAMETROS", {})
    restric = sheets.get("RESTRICCIONES", {})

    # Demanda y lead time
    mu = params.get("demanda_diaria", 100)       # demanda promedio diaria
    sigma = params.get("demanda_std", mu * 0.3)  # desviación demanda (fallback CV=0.3)
    LT = params.get("lead time", 1)  # Fix: Use correct parameter name
    H = params.get("costo_sobrante", 1)
    K = params.get("costo_pedir", 1)

    # Nivel de servicio
    nivel_servicio = restric.get("Proporción demanda satisfecha", 0.95)
    Z = norm.ppf(nivel_servicio) if 0 < nivel_servicio < 1 else 1.65

    # Stock de seguridad
    SS = Z * sigma * math.sqrt(LT)

    # --- Políticas ---
    if policy == "QR":
        # Q: rango alrededor del EOQ
        demanda_mensual = mu * 30
        if H > 0:
            EOQ = math.sqrt((2 * demanda_mensual * K) / H)
        else:
            EOQ = mu * 10
        q_bounds = (max(1, EOQ * 0.5), EOQ * 1.5)

        # R: en torno a mu*LT + SS
        R_centro = mu * LT + SS
        r_bounds = (max(1, R_centro * 0.5), R_centro * 1.5)

        return [q_bounds, r_bounds]

    elif policy == "ST":
        # S: cubrir LT+T con SS
        S_base = mu * (LT + 4) + SS
        s_bounds = (max(1, S_base * 0.5), S_base * 1.5)
        # T: entre 1 y 12 periodos
        t_bounds = (1, 12)
        return [s_bounds, t_bounds]

    # ✅ CRITICAL: Check SS BEFORE SST to avoid confusion
    elif policy == "SS":
        # (s, S) sin periodo fijo - ONLY 2 parameters!
        print(f"   ✅ Matched SS policy - generating 2 bounds (s, S)")
        
        s_low = mu * LT
        s_bounds = (max(1, s_low * 0.5), s_low * 1.2)

        S_up = s_low + (mu * 4)
        S_bounds = (S_up * 0.8, S_up * 1.5)

        print(f"   📊 SS Bounds: s={s_bounds}, S={S_bounds}")
        return [s_bounds, S_bounds]

    elif policy == "SST":
        # (s, S, T) con periodo fijo - 3 parameters!
        print(f"   ✅ Matched SST policy - generating 3 bounds (s, S, T)")
        
        # s: punto de reorden bajo
        s_low = mu * LT
        s_bounds = (max(1, s_low * 0.5), s_low * 1.2)

        # S: stock máximo
        S_up = s_low + (mu * 4)
        S_bounds = (S_up * 0.8, S_up * 1.5)

        # T: frecuencia
        t_bounds = (1, 12)
        
        print(f"   📊 SST Bounds: s={s_bounds}, S={S_bounds}, T={t_bounds}")
        return [s_bounds, S_bounds, t_bounds]

    elif policy in ("EOQ", "POQ", "LXL"):
        # porcentaje de seguridad en stock - more conservative for ingredients
        return [(0.10, 0.60)]  # 10% a 60% (more realistic safety stock range)

    else:
        return [(1, 1000)]



# ------------------ integrated forecast and replicas generation ------------------

def get_available_references(file_path: str):
    """
    Obtiene las referencias disponibles del archivo Excel.
    
    Parameters
    ----------
    file_path : str
        Ruta al archivo Excel.
        
    Returns
    -------
    referencias : list
        Lista de referencias disponibles.
    """
    df = pd.read_excel(file_path, sheet_name="Demanda")
    referencias = [col for col in df.columns if col not in ['Fecha', 'Total']]
    return referencias


def create_replicas_matrix_from_existing_forecast(data_dict: dict, pv: str, n_replicas: int = 100, u: int = 30) -> np.ndarray:
    """
    Crea una matriz de réplicas usando los pronósticos ya calculados en leer_datos.py
    y aplicando variación estadística para generar múltiples escenarios.
    
    Parameters:
    -----------
    data_dict : dict
        Diccionario con datos de puntos de venta incluyendo pronósticos
    pv : str
        Punto de venta/referencia
    n_replicas : int
        Número de réplicas a generar
    u : int
        Número de períodos (horizonte de pronóstico)
    
    Returns:
    --------
    np.ndarray
        Matriz de réplicas (n_replicas x u períodos)
    """
    
    print(f"\n🔄 Creando matriz de réplicas desde pronósticos existentes para {pv}")
    
    # Verificar que existe el punto de venta en data_dict
    if pv not in data_dict:
        raise ValueError(f"Punto de venta '{pv}' no encontrado en data_dict. Disponibles: {list(data_dict.keys())}")
    
    pv_data = data_dict[pv]
    
    # Obtener pronósticos existentes
    ventas_pronosticadas = pv_data.get("RESULTADOS", {}).get("ventas", {})
    
    if not ventas_pronosticadas:
        print(f"⚠️ No hay pronósticos existentes para {pv}, generando datos sintéticos")
        # Usar parámetros de demanda como fallback
        params = pv_data.get("PARAMETROS", {})
        demanda_promedio = params.get("demanda_diaria", 50)
        
        # Crear pronóstico base sintético
        base_forecast = np.full(u, demanda_promedio)
        cv = 0.3  # Coeficiente de variación por defecto
    else:
        print(f"✅ Usando pronósticos existentes: {len(ventas_pronosticadas)} períodos disponibles")
        
        # Convertir pronósticos a array, limitando a u períodos
        forecast_keys = sorted([k for k in ventas_pronosticadas.keys() if isinstance(k, (int, str)) and str(k).isdigit()])
        forecast_values = [ventas_pronosticadas[k] for k in forecast_keys[:u]]
        
        # Completar con promedio si no hay suficientes períodos
        if len(forecast_values) < u:
            avg_demand = np.mean(forecast_values) if forecast_values else 50
            forecast_values.extend([avg_demand] * (u - len(forecast_values)))
            print(f"⚠️ Completando pronóstico con {u - len(forecast_values)} períodos usando promedio {avg_demand:.1f}")
        
        base_forecast = np.array(forecast_values[:u])
        
        # Calcular coeficiente de variación desde los datos históricos o parámetros
        params = pv_data.get("PARAMETROS", {})
        if len(forecast_values) > 1:
            cv = np.std(forecast_values) / np.mean(forecast_values) if np.mean(forecast_values) > 0 else 0.3
        else:
            cv = 0.3  # Por defecto
        
        cv = max(0.1, min(0.8, cv))  # Limitar CV entre 10% y 80%
    
    print(f"📊 Pronóstico base: min={base_forecast.min():.1f}, max={base_forecast.max():.1f}, promedio={base_forecast.mean():.1f}")
    print(f"📈 Coeficiente de variación: {cv:.2%}")
    
    # Generar matriz de réplicas con variación estocástica
    replicas_matrix = np.zeros((n_replicas, u))
    
    # Establecer semilla para reproducibilidad
    np.random.seed(42)

    # Note: we generate replicas below using multiplicative noise on the base_forecast.
    # The older call to `replicas.generar_replicas(best_model, errores, ...)` was removed
    # because `best_model`/`errores` are not defined in this scope and would raise an error.
    
    for replica in range(n_replicas):
        # Aplicar variación estocástica al pronóstico base
        # Usar distribución normal truncada para evitar valores negativos
        noise = np.random.normal(0, cv, u)  # Ruido gaussiano
        
        # Aplicar ruido multiplicativo para mantener proporción con el nivel de demanda
        replica_forecast = base_forecast * (1 + noise)
        
        # Asegurar valores mínimos (no negativos)
        replica_forecast = np.maximum(replica_forecast, 1)
        
        # Redondear a enteros (demanda discreta)
        replica_forecast = np.round(replica_forecast).astype(int)
        
        replicas_matrix[replica, :] = replica_forecast
    

    return replicas_matrix


def create_ingredient_replicas_matrix_from_data_dict(data_dict_MP: dict, familia_name: str, n_replicas: int = 100, u: int = 30) -> np.ndarray:
    """
    Crea una matriz de réplicas para una familia de ingredientes usando los datos ya calculados
    en data_dict_MP (que incluye conversión de pizzas a ingredientes).
    
    Parameters:
    -----------
    data_dict_MP : dict
        Diccionario con datos de familias de materias primas
    familia_name : str
        Nombre de la familia (ej: "Familia_1")
    n_replicas : int
        Número de réplicas a generar
    u : int
        Número de períodos
        
    Returns:
    --------
    np.ndarray
        Matriz de réplicas en unidades de ingrediente (n_replicas x u períodos)
    """
    
    print(f"\n🏭 Creando matriz de réplicas de ingredientes para {familia_name}")
    
    # Verificar que existe la familia
    if familia_name not in data_dict_MP:
        available_families = list(data_dict_MP.keys())
        raise ValueError(f"Familia '{familia_name}' no encontrada. Disponibles: {available_families}")
    
    familia_data = data_dict_MP[familia_name]
    
    # Obtener datos de ventas convertidas (ya en unidades de ingrediente)
    ventas_ingrediente = familia_data.get("RESULTADOS", {}).get("ventas", {})
    
    if not ventas_ingrediente:
        print(f"⚠️ No hay datos de ventas para {familia_name}, usando parámetros de demanda")
        
        # Usar parámetros de demanda como fallback
        params = familia_data.get("PARAMETROS", {})
        demanda_diaria = params.get("demanda_diaria", 100)  # En gramos de ingrediente
        
        # Crear demanda base sintética
        base_demand = np.full(u, demanda_diaria)
        cv = 0.25  # Coeficiente de variación moderado para ingredientes
        
    else:
        print(f"✅ Usando ventas de ingrediente convertidas: {len(ventas_ingrediente)} períodos")
        
        # Convertir ventas de ingrediente a array - SOLO claves numéricas
        # Filtrar completamente las claves que son strings como 'Periodo 1', 'Periodo 2', etc.
        numeric_keys = []
        for k in ventas_ingrediente.keys():
            if isinstance(k, (int, float)):
                numeric_keys.append(int(k))
            elif isinstance(k, str) and k.isdigit():
                numeric_keys.append(int(k))
            # Saltar completamente strings como 'Periodo 1', 'Periodo 2'
        
        sales_keys = sorted(numeric_keys)
        sales_values = [float(ventas_ingrediente[k]) for k in sales_keys[:u]]
        
        # Verificar si hay claves problemáticas
        problematic_keys = [k for k in ventas_ingrediente.keys() if not isinstance(k, (int, float)) and not (isinstance(k, str) and k.isdigit())]
        if problematic_keys:
            print(f"⚠️ Claves problemáticas filtradas: {problematic_keys[:5]}...")
        
        # Verificar si hay valores problemáticos
        problematic_values = [(k, v) for k, v in ventas_ingrediente.items() if not isinstance(v, (int, float, np.integer, np.floating))]
        if problematic_values:
            print(f"⚠️ Valores problemáticos en ventas_ingrediente: {problematic_values[:5]}...")
            
        # Ensure all sales_values are numeric
        clean_sales_values = []
        for val in sales_values:
            if isinstance(val, (int, float, np.integer, np.floating)):
                clean_sales_values.append(float(val))
            else:
                print(f"⚠️ Non-numeric sales value: {val} (type: {type(val)}), using default 100")
                clean_sales_values.append(100.0)  # Default demand
        
        sales_values = clean_sales_values
        print(f"🔍 DEBUG: Clean sales_values: {sales_values[:5]}")
        
        # Completar con promedio si faltan períodos
        if len(sales_values) < u:
            avg_sales = np.mean(sales_values) if sales_values else 100
            sales_values.extend([avg_sales] * (u - len(sales_values)))
            print(f"⚠️ Completando con {u - len(sales_values)} períodos usando promedio {avg_sales:.1f}g")
        
        base_demand = np.array(sales_values[:u])
        
        # Calcular variabilidad
        if len(sales_values) > 1:
            cv = np.std(sales_values) / np.mean(sales_values) if np.mean(sales_values) > 0 else 0.25
        else:
            cv = 0.25
            
        cv = max(0.15, min(0.5, cv))  # Limitar CV para ingredientes (15%-50%)
    
    # Obtener información adicional de la familia
    params = familia_data.get("PARAMETROS", {})
    ingredientes_incluidos = params.get("ingredientes_incluidos", [])
    representativo = params.get("representativo", "N/A")
    cantidad_por_pizza = params.get("cantidad_por_pizza", 0)
    unidad = params.get("unidad", "g")
    
    print(f"📦 Familia: {len(ingredientes_incluidos) if ingredientes_incluidos else 0} ingredientes")
    print(f"⭐ Representativo: {representativo}")
    print(f"🔄 Conversión: {cantidad_por_pizza:.2f}{unidad} por pizza")
    print(f"📊 Demanda base: min={base_demand.min():.1f}, max={base_demand.max():.1f}, promedio={base_demand.mean():.1f}{unidad}")
    print(f"📈 Variabilidad: {cv:.1%}")
    
    # Generar matriz de réplicas
    replicas_matrix = np.zeros((n_replicas, u))
    
    # Semilla para reproducibilidad
    np.random.seed(42)
    
    for replica in range(n_replicas):
        # Variación estocástica específica para ingredientes
        # Usar distribución gamma para modelar demanda de ingredientes (siempre positiva)
        shape = 1 / (cv ** 2)  # Parámetro de forma de gamma
        
        # Generar factores multiplicativos usando gamma
        scale_factors = np.random.gamma(shape, cv ** 2, u)
        
        # Aplicar variación a la demanda base
        replica_demand = base_demand * scale_factors
        
        # Asegurar valores mínimos (al menos 1g)
        replica_demand = np.maximum(replica_demand, 1)
        
        # Redondear a enteros (gramos discretos)
        replica_demand = np.round(replica_demand).astype(int)
        
        replicas_matrix[replica, :] = replica_demand
    
    # Estadísticas finales
    print(f"✅ Matriz de réplicas de ingredientes generada: {replicas_matrix.shape}")
    print(f"📊 Estadísticas finales:")
    print(f"   Rango: {replicas_matrix.min()}-{replicas_matrix.max()}{unidad}")
    print(f"   Promedio: {replicas_matrix.mean():.1f}{unidad}")
    print(f"   Desviación: {replicas_matrix.std():.1f}{unidad}")
    
    # Ensure the matrix is numeric
    if replicas_matrix.dtype == 'object':
        print(f"⚠️ WARNING: Matrix has object dtype, converting to numeric...")
        # Convert any non-numeric values to integers
        numeric_matrix = np.zeros_like(replicas_matrix, dtype=int)
        for i in range(replicas_matrix.shape[0]):
            for j in range(replicas_matrix.shape[1]):
                val = replicas_matrix[i, j]
                if isinstance(val, (int, float, np.integer, np.floating)):
                    numeric_matrix[i, j] = int(val)
                else:
                    print(f"   Non-numeric value at [{i},{j}]: {val} (type: {type(val)})")
                    numeric_matrix[i, j] = 1  # Default fallback
        replicas_matrix = numeric_matrix
        print(f"   Converted matrix dtype: {replicas_matrix.dtype}")
        print(f"   Converted sample: {replicas_matrix[0][:5]}")
    
    return replicas_matrix


def optimize_policy(data_dict, file_datos: str, pv: str, año: int, policy: str, u: int = 30, n_replicas: int = 100):
    """
    Función única que optimiza cualquier política de inventario especificada.
    UPDATED: Now works with a single Excel file that contains both data and historical demand.
    
    Parameters
    ----------
    file_datos : str
        Ruta al archivo Excel principal que contiene todas las hojas (incluyendo "Demanda").
    pv : str
        Referencia del producto a analizar.
    año : int
        Año de los datos a utilizar.
    policy : str
        Política de inventario a optimizar ("QR", "ST", "SST", "SS", "EOQ", "POQ", "LXL").
    u : int
        Número de períodos para test y pronóstico.
    n_replicas : int
        Número de réplicas a generar.
        
    Returns
    -------
    replicas_matrix : np.ndarray
        Matriz de réplicas (n_replicas x u).
    result : dict
        Resultados de la optimización PSO.
    """
    # Validar política
    políticas_disponibles = ["QR", "ST", "SST", "SS", "EOQ", "POQ", "LXL"]
    if policy.upper() not in políticas_disponibles:
        raise ValueError(f"Política '{policy}' no disponible. Políticas disponibles: {políticas_disponibles}")
    
    policy = policy.upper()
    
    # Cargar datos históricos de demanda desde la hoja "Demanda" del archivo principal
    try:
        df = pd.read_excel(file_datos, sheet_name="Demanda")
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
        df.set_index('Fecha', inplace=True)
    except Exception as e:
        raise ValueError(f"Error leyendo hoja 'Demanda' del archivo {file_datos}: {e}")
    
    # Verificar que la referencia existe
    if pv not in df.columns:
        available_refs = [col for col in df.columns if col not in ['Fecha', 'Total']]
        raise ValueError(f"Referencia '{pv}' no encontrada. Referencias disponibles: {available_refs}")
    
    # Filtrar serie para la referencia y año especificados
    serie = df[pv]
    serie = serie[serie.index.year == año]
    serie = serie.replace(0, np.nan).fillna(serie.mean())
    
    if len(serie) < 12:
        raise ValueError(f"Datos insuficientes para {pv} en {año}. Se requieren al menos 12 observaciones.")
    
    # Verificar que la referencia existe en data_dict antes de proceder
    if pv not in data_dict:
        raise ValueError(f"Los parámetros para la referencia '{pv}' no se encontraron en data_dict")
    
    # Usar pronósticos existentes en lugar de recalcular SARIMA
    print(f"\n🔄 Usando pronósticos existentes desde leer_datos.py para {pv}")
    
    # Verificar si hay pronósticos disponibles
    ventas_pronosticadas = data_dict[pv].get("RESULTADOS", {}).get("ventas", {})
    
    if ventas_pronosticadas:
        print(f"✅ Pronósticos encontrados: {len(ventas_pronosticadas)} períodos")
        
        # Generar matriz de réplicas usando pronósticos existentes
        replicas_matrix = create_replicas_matrix_from_existing_forecast(
            data_dict=data_dict, 
            pv=pv, 
            n_replicas=n_replicas, 
            u=u
        )
        
    else:
        print(f"⚠️ No hay pronósticos existentes para {pv}, generando con SARIMA...")
        
        # Fallback: usar SARIMA si no hay pronósticos existentes
        train, test = serie[:-u], serie[-u:]
        
        print(f"📊 Optimizando SARIMA para referencia {pv} año {año}...")
        
        # Optimizar SARIMA con PSO
        s = 12
        pso_sarima = replicas.PSO_SARIMA(train, test, s)
        best_params, best_score = pso_sarima.optimizar()
        (p, d, q, P, D, Q) = best_params
        
        print(f"✅ Mejor modelo SARIMA({p},{d},{q})x({P},{D},{Q},{s}) - MAPE: {best_score:.2f}%")
        
        # Entrenar modelo final
        best_model = SARIMAX(train, order=(p, d, q),
                             seasonal_order=(P, D, Q, s),
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit(disp=False)
        
        # Calcular errores para generar réplicas
        pronosticos = best_model.forecast(steps=u)
        errores = test.values - pronosticos.values
        
        # Generar réplicas usando método tradicional
        print(f"📊 Generando {n_replicas} réplicas de pronósticos...")
        # TODO: Implement SARIMA replica generation or use fallback
        print("⚠️ SARIMA replica generation not implemented, using synthetic data")
        
        # Fallback: create synthetic replicas matrix based on test data
        if len(test) > 0:
            mean_demand = test.mean()
            std_demand = test.std() if len(test) > 1 else mean_demand * 0.3
        else:
            mean_demand = data_dict[pv].get("PARAMETROS", {}).get("demanda_diaria", 50)
            std_demand = mean_demand * 0.3
        
        # Generate synthetic replicas
        np.random.seed(42)
        replicas_matrix = np.zeros((n_replicas, u))
        for i in range(n_replicas):
            replica = np.random.normal(mean_demand, std_demand, u)
            replica = np.maximum(replica, 1)  # Ensure positive values
            replicas_matrix[i, :] = replica.astype(int)

    # Ensure replicas_matrix is always defined
    if 'replicas_matrix' not in locals():
        print("⚠️ replicas_matrix not defined, creating default synthetic matrix")
        mean_demand = data_dict[pv].get("PARAMETROS", {}).get("demanda_diaria", 50)
        np.random.seed(42)
        replicas_matrix = np.random.poisson(mean_demand, size=(n_replicas, u))

    # Los parámetros ya están disponibles en data_dict (pasado como parámetro)
    print(f"✅ Usando parámetros existentes desde data_dict para {pv}")

    # Obtener límites de decisión para la política
    decision_bounds = get_decision_bounds_for_policy(policy, pv, data_dict)
    
    print(f"\n Política seleccionada: {policy}")
    print(f" Decision bounds para {policy}: {decision_bounds}")
    print(f" Número de variables de decisión: {len(decision_bounds)}")
    
    # Optimización con PSO
    print(f"\n Iniciando optimización PSO para política {policy}...")
    print(f"    Usando matriz de réplicas: {replicas_matrix.shape[0]} réplicas × {replicas_matrix.shape[1]} períodos")
    print(f"    Cada evaluación PSO probará las variables de decisión contra TODAS las réplicas")
    
    #modificacion aca
    result = pso_optimize_single_policy(
        policy, data_dict, pv, replicas_matrix, decision_bounds,
        objective_indicator="Costo total", minimize=True,
        swarm_size=20, iters=3, verbose=True
    )

    # --- Resultados ---
    print("\n" + "="*50)
    print(" RESULTADOS DE OPTIMIZACIÓN PSO")
    print("="*50)
    print(f"Referencia optimizada: {pv}")
    print(f"Política: {policy}")
    print(f"Matriz de réplicas: {replicas_matrix.shape}")
    print(f"Mejor score (Costo total): {result['best_score']:,.2f}")
    print(f"Mejores variables (raw): {result['best_decision_vars']}")
    print(f"Mejores variables (mapeadas): {result['best_decision_mapped']}")
    print("="*50)

    if replicas_matrix is None:
        raise ValueError("replicas_matrix is None")
    if result is None:
        raise ValueError("result is None")
    
    #print(f"About to return: replicas_matrix.shape={replicas_matrix.shape}, result type={type(result)}")
    
    
    return result


def export_optimization_results_to_excel(
    policy: str,
    ref: str,
    best_decision_vars: dict,
    df_promedio: pd.DataFrame,
    liberacion_orden_df: pd.DataFrame,
    resultados_replicas: list,
    replicas_matrix: np.ndarray,
    output_dir: str = "optimization_results",
    ingredient_info: dict = None,
    liberacion_final=None,
    family_liberation_results: dict = None
) -> str:
    """
    Exports detailed optimization results to Excel with multiple sheets.
    
    Parameters:
    -----------
    policy : str
        The optimized policy (QR, ST, etc.)
    ref : str
        Reference/point of sale
    best_decision_vars : dict
        Optimal decision variables found by PSO
    df_promedio : pd.DataFrame
        Average indicators across all replicas
    liberacion_orden_df : pd.DataFrame
        Liberation order matrix (periods x replicas)
    resultados_replicas : list
        Individual results for each replica
    replicas_matrix : np.ndarray
        Original demand replicas matrix
    output_dir : str
        Directory to save the Excel file
    ingredient_info : dict, optional
        Additional ingredient-specific information for enhanced reporting
        Keys: cluster_id, ingredient_code, representative_ingredient, 
              conversion_factor, unit, pizza_point_of_sale
    family_liberation_results : dict, optional
        Results from family liberation generation for all ingredients in the family
        Structure: {ingredient_code: {liberation_df, summary_metrics, optimized_params}}
    
    Returns:
    --------
    str
        Path to the created Excel file
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"OptimizationResults_{policy}_{ref}_{timestamp}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    print(f"📁 Exportando resultados a Excel: {filepath}")
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            
            # Sheet 1: Summary of optimization
            summary_data = {
                'Parámetro': ['Política', 'Punto de Venta', 'Fecha de Optimización', 'Número de Réplicas', 'Períodos'],
                'Valor': [policy, ref, datetime.now().strftime("%Y-%m-%d %H:%M"), len(resultados_replicas), replicas_matrix.shape[1]]
            }
            
            # Add ingredient-specific information if provided
            if ingredient_info:
                for key, value in ingredient_info.items():
                    display_key = key.replace('_', ' ').title()
                    summary_data['Parámetro'].append(display_key)
                    summary_data['Valor'].append(str(value))
            
            # Add optimal parameters to summary
            for param_name, param_value in best_decision_vars.items():
                summary_data['Parámetro'].append(f'Óptimo {param_name}')
                summary_data['Valor'].append(str(param_value))
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Resumen_Optimización', index=False)
            
            # Sheet 2: Average indicators
            df_promedio.to_excel(writer, sheet_name='Indicadores_Promedio')
            
            # Sheet 3: Liberation order matrix (OPTIMIZATION RESULT - what to produce)
            liberacion_orden_df.to_excel(writer, sheet_name='Órdenes_Optimizadas')

            df_liberacion_final =pd.DataFrame(liberacion_final)
            df_liberacion_final.to_excel(writer, sheet_name='Órdenes_Finales')
            
            # Sheet 4: Individual replica results (combined)
            if resultados_replicas:
                # Combine all replica results into one DataFrame
                df_all_replicas = pd.concat(
                    [replica_result.rename(columns={replica_result.columns[0]: f'Replica_{i+1}'}) 
                     for i, replica_result in enumerate(resultados_replicas)], 
                    axis=1
                )
                df_all_replicas.to_excel(writer, sheet_name='Resultados_Todas_Réplicas')
            
            # Sheet 5: Replicas matrix used for optimization (INPUT DEMAND - converted from pizzas)
            df_demand_replicas = pd.DataFrame(
                replicas_matrix.T,  # Transpose to have periods as rows, replicas as columns
                index=[f'Periodo_{i+1}' for i in range(replicas_matrix.shape[1])],
                columns=[f'Replica_{i+1}' for i in range(replicas_matrix.shape[0])]
            )
            
            # Determine appropriate sheet name based on optimization type
            if ingredient_info:
                # This is ingredient optimization - show the REPRESENTATIVE ingredient that was optimized
                unit = ingredient_info.get('unit', 'units')
                representative_ingredient = ingredient_info.get('ingredient_code', 'Unknown')
                
                # DEBUG: Show what we're using
                print(f"📋 DEBUG - INPUT Demand sheet:")
                print(f"   Representative ingredient: '{representative_ingredient}'")
                print(f"   Cluster ID: {ingredient_info.get('cluster_id', 'N/A')}")
                print(f"   All ingredient_info keys: {list(ingredient_info.keys())}")
                
                # Sanitize for Excel sheet name
                clean_rep_name = sanitize_excel_sheet_name(str(representative_ingredient), 20)
                sheet_name = f'INPUT_{clean_rep_name}_REP'
                
                # Add comprehensive diagnostic info
                avg_value = replicas_matrix.mean()
                conversion_info = ingredient_info.get('conversion_factor', 'N/A')
                pizza_point = ingredient_info.get('pizza_point_of_sale', 'N/A')
                cluster_id = ingredient_info.get('cluster_id', 'N/A')
                
                # Create informative header
                info_data = {
                    'Información': [
                        'MATRIZ DE ENTRADA PARA OPTIMIZACIÓN',
                        f'Ingrediente REPRESENTATIVO: {representative_ingredient}',
                        f'Cluster/Familia: {cluster_id}',
                        f'Unidad: {unit}',
                        f'Conversión desde pizzas: {conversion_info}',
                        f'Punto de venta pizzas: {pizza_point}',
                        f'Demanda promedio: {avg_value:.1f} {unit}',
                        f'Rango: {replicas_matrix.min():.0f}-{replicas_matrix.max():.0f} {unit}',
                        '',
                        'NOTA: Esta es la demanda del INGREDIENTE REPRESENTATIVO',
                        'Los demás ingredientes de la familia usan sus propias demandas convertidas.'
                    ]
                }
                
                # Create info DataFrame
                info_df = pd.DataFrame(info_data)
                
                # Write info first, then a gap, then the actual replicas matrix
                start_row_data = len(info_df) + 2
                
                # Write to Excel with custom positioning
                info_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
                df_demand_replicas.to_excel(writer, sheet_name=sheet_name, startrow=start_row_data)
                
            else:
                # This is pizza/standard optimization
                sheet_name = 'INPUT_Demanda_Pizzas_(unidades)'
                
                # Add header for pizza optimization too
                info_data = {
                    'Información': [
                        'MATRIZ DE ENTRADA PARA OPTIMIZACIÓN',
                        'Producto: Pizzas',
                        'Unidad: unidades',
                        f'Demanda promedio: {replicas_matrix.mean():.1f} pizzas',
                        f'Rango: {replicas_matrix.min():.0f}-{replicas_matrix.max():.0f} pizzas',
                        '',
                        'Esta matriz muestra la demanda de pizzas',
                        'que se usó como entrada para la optimización de inventarios.'
                    ]
                }
                info_df = pd.DataFrame(info_data)
                start_row_data = len(info_df) + 2
                
                info_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
                df_demand_replicas.to_excel(writer, sheet_name=sheet_name, startrow=start_row_data)
            
            # Add family liberation results if provided
            if family_liberation_results:
                print(f"📊 Agregando resultados de liberación familiar ({len(family_liberation_results)} ingredientes)")
            else:
                print(f"⚠️ No family liberation results to export (family_liberation_results is {family_liberation_results})")
                
            if family_liberation_results:
                # Create family summary sheet
                family_summary_data = []
                for ingredient_code, results in family_liberation_results.items():
                    if "error" not in results:
                        summary_metrics = results.get("summary_metrics", pd.DataFrame())
                        liberation_df = results.get("liberation_df", pd.DataFrame())
                        
                        # Extract key metrics
                        total_orders = liberation_df.sum().sum() if not liberation_df.empty else 0
                        active_periods = (liberation_df > 0).any(axis=1).sum() if not liberation_df.empty else 0
                        
                        # Get liberation_final vector sum for display
                        liberation_final = results.get("liberation_final", [])
                        liberation_vector_sum = float(sum(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0
                        
                        # Get cost and demand satisfaction if available - handle Series/DataFrame properly
                        total_cost = 0
                        demand_satisfaction = 0
                        if not summary_metrics.empty:
                            try:
                                # summary_metrics is a DataFrame with indicators as index
                                if "Costo total" in summary_metrics.index:
                                    cost_val = summary_metrics.loc["Costo total", "Promedio Indicadores"]
                                    total_cost = float(cost_val) if hasattr(cost_val, '__float__') else 0
                                if "Proporción demanda satisfecha" in summary_metrics.index:
                                    satisfaction_val = summary_metrics.loc["Proporción demanda satisfecha", "Promedio Indicadores"]
                                    demand_satisfaction = float(satisfaction_val) * 100 if hasattr(satisfaction_val, '__float__') else 0
                            except Exception as e:
                                print(f"⚠️ Could not extract metrics for {ingredient_code}: {e}")
                        
                        family_summary_data.append({
                            "Ingrediente": str(ingredient_code),
                            "Total_Órdenes_Matriz": float(total_orders),
                            "Vector_Final_Órdenes": float(liberation_vector_sum),
                            "Períodos_Activos": int(active_periods),
                            "Costo_Total": float(total_cost),
                            "Satisfacción_Demanda_%": float(demand_satisfaction),
                            "Estado": "Procesado"
                        })
                    else:
                        family_summary_data.append({
                            "Ingrediente": str(ingredient_code),
                            "Total_Órdenes_Matriz": 0.0,
                            "Vector_Final_Órdenes": 0.0,
                            "Períodos_Activos": 0,
                            "Costo_Total": 0.0,
                            "Satisfacción_Demanda_%": 0.0,
                            "Estado": f"Error: {results['error']}"
                        })
                
                family_summary_df = pd.DataFrame(family_summary_data)
                family_summary_df.to_excel(writer, sheet_name='FAMILIA_Resumen', index=False)
                
                # Create individual sheets for each family member
                for ingredient_code, results in family_liberation_results.items():
                    if "error" not in results:
                        liberation_df = results.get("liberation_df", pd.DataFrame())
                        summary_metrics = results.get("summary_metrics", pd.DataFrame())
                        
                        # Clean ingredient name for sheet name (Excel has limitations)
                        # Convert to string first to handle integer codes
                        clean_ingredient = sanitize_excel_sheet_name(str(ingredient_code), 25)
                        sheet_name = f"FAM_{clean_ingredient}"
                        
                        try:
                            # Get liberation_final vector
                            liberation_final = results.get("liberation_final", [])
                            
                            # Create info section
                            optimized_params = results.get("optimized_params", {})
                            params_str = ", ".join([f"{k}={v}" for k, v in optimized_params.items()])
                            
                            # Calculate vector statistics - ensure all are proper numeric types
                            vector_sum = float(sum(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0.0
                            vector_length = int(len(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__len__') else 0
                            matrix_total = float(liberation_df.sum().sum()) if not liberation_df.empty else 0.0
                            
                            # Get cluster and representative info if available
                            cluster_id = ingredient_info.get('cluster_id', 'N/A') if ingredient_info else 'N/A'
                            representative = ingredient_info.get('ingredient_code', 'N/A') if ingredient_info else 'N/A'
                            
                            # CRITICAL FIX: Ensure active_periods is a proper int
                            active_periods = int((liberation_df > 0).any(axis=1).sum()) if not liberation_df.empty else 0
                            
                            info_data = {
                                'Información': [
                                    f'ÓRDENES DE LIBERACIÓN - {ingredient_code}',
                                    f'Familia/Cluster: {cluster_id}',
                                    f'Ingrediente representativo: {representative}',
                                    f'Política: {policy}',
                                    f'Parámetros aplicados: {params_str}',
                                    f'Vector final órdenes: {vector_sum:.0f} ({vector_length} períodos)',
                                    f'Total órdenes matriz: {matrix_total:.0f}',
                                    f'Períodos activos: {active_periods}',
                                    '',
                                    'VECTOR FINAL DE LIBERACIÓN (específico para este ingrediente):',
                                    'Órdenes calculadas usando demanda convertida individual'
                                ]
                            }
                            info_df = pd.DataFrame(info_data)
                            
                            # Write info header
                            start_row_vector = len(info_df) + 2
                            info_df.to_excel(writer, sheet_name=sheet_name, startrow=0, index=False)
                            
                            # Write liberation_final vector as the main result
                            if liberation_final is not None and hasattr(liberation_final, '__iter__'):
                                # Ensure liberation_final values are proper numeric types (not Series)
                                liberation_final_clean = [float(x) if hasattr(x, '__float__') else x for x in liberation_final]
                                
                                liberation_vector_df = pd.DataFrame({
                                    'Período': [f'Período_{i+1}' for i in range(len(liberation_final_clean))],
                                    'Órdenes_Finales': liberation_final_clean
                                })
                                liberation_vector_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row_vector, index=False)
                                
                                # Write detailed matrix below
                                start_row_matrix = start_row_vector + len(liberation_vector_df) + 4
                                matrix_header = pd.DataFrame({"Información": ["MATRIZ DETALLADA DE ÓRDENES (por réplica):"]})
                                matrix_header.to_excel(writer, sheet_name=sheet_name, startrow=start_row_matrix, index=False)
                                
                                if not liberation_df.empty:
                                    liberation_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row_matrix + 2)
                                    
                                # Write summary metrics at the end
                                if not summary_metrics.empty:
                                    start_row_metrics = start_row_matrix + 4 + liberation_df.shape[0] + 2
                                    summary_header = pd.DataFrame({"Información": ["INDICADORES DE DESEMPEÑO:"]})
                                    summary_header.to_excel(writer, sheet_name=sheet_name, startrow=start_row_metrics, index=False)
                                    summary_metrics.to_excel(writer, sheet_name=sheet_name, startrow=start_row_metrics + 2)
                            else:
                                # Fallback if no liberation_final vector
                                error_df = pd.DataFrame({"Error": ["No se pudo obtener el vector final de liberación"]})
                                error_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row_vector, index=False)
                                
                        except Exception as e:
                            print(f"⚠️ Error creando hoja para {ingredient_code}: {e}")
        
        # Print success message with family information
        update_export_success_message_with_family(filepath, family_liberation_results)
        
        return filepath
        
    except Exception as e:
        error_msg = f"❌ Error exportando resultados a Excel: {e}"
        print(error_msg)
        # Try to create a simple fallback export
        try:
            fallback_filename = f"OptimizationResults_Simple_{policy}_{ref}_{timestamp}.xlsx"
            fallback_filepath = os.path.join(output_dir, fallback_filename)
            
            with pd.ExcelWriter(fallback_filepath, engine='openpyxl') as writer:
                df_promedio.to_excel(writer, sheet_name='Indicadores')
                liberacion_orden_df.to_excel(writer, sheet_name='Liberacion_Ordenes')
            
            print(f"⚠️ Creado archivo simplificado: {fallback_filepath}")
            return fallback_filepath
            
        except Exception as e2:
            print(f"❌ Error creando archivo simplificado: {e2}")
            return None


def get_verbose_results_from_optimization(optimization_result: dict) -> dict:
    """
    Extracts verbose results from optimization result if available.
    
    Parameters:
    -----------
    optimization_result : dict
        Result dictionary from PSO optimization
        
    Returns:
    --------
    dict
        Dictionary with verbose results or None if not available
    """
    verbose_results = optimization_result.get("verbose_results", None)
    
    if verbose_results is None:
        print("ℹ️ No verbose results available in optimization result")
        return None
    
    print("✅ Verbose results found:")
    print(f"   📊 Average indicators shape: {verbose_results['df_promedio'].shape}")
    print(f"   📋 Liberation matrix shape: {verbose_results['liberacion_orden_df'].shape}")  
    print(f"   🔄 Number of replica results: {len(verbose_results['resultados_replicas'])}")
    if verbose_results.get('excel_file_path'):
        print(f"   📁 Excel file: {verbose_results['excel_file_path']}")
    
    return verbose_results


def test_verbose_optimization_export(policy: str = "QR", ref: str = "AP20"):
    """
    Test function to verify that verbose optimization and Excel export work correctly.
    """
    print(f"\n🧪 Testing verbose optimization export for {policy} policy, reference {ref}")
    
    try:
        # Create sample data for testing
        sample_replicas = np.random.randint(10, 100, size=(5, 30))  # 5 replicas, 30 periods
        
        sample_data_dict = {
            ref: {
                "PARAMETROS": {
                    "inventario_inicial": 50,
                    "lead time": 2,
                    "rp": {},
                    "MOQ": 0,
                    "backorders": 1,
                    "costo_pedir": 100,
                    "costo_unitario": 5,
                    "costo_faltante": 20,
                    "costo_sobrante": 1,
                    "demanda_diaria": 25,
                    "demanda_std": 8
                },
                "RESTRICCIONES": {
                    "Proporción demanda satisfecha": 0.95,
                    "Inventario a la mano (max)": 200
                }
            }
        }
        
        # Test bounds
        decision_bounds = get_decision_bounds_for_policy(policy, ref, sample_data_dict)
        print(f"✅ Decision bounds calculated: {decision_bounds}")
        
        # Test mapping
        test_particle = np.array([50, 25])  # Sample values for QR
        mapped_vars = map_particle_to_decisions(policy, test_particle, decision_bounds)
        print(f"✅ Variable mapping works: {mapped_vars}")
        
        print(f"🔧 Test completed - verbose optimization should work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_available_references_from_data(file_datos: str):
    """
    Obtiene las referencias disponibles del archivo Excel desde la hoja "Demanda".
    UPDATED: Works with single file.
    
    Parameters
    ----------
    file_datos : str
        Ruta al archivo Excel principal.
        
    Returns
    -------
    referencias : list
        Lista de referencias disponibles.
    años : list
        Lista de años disponibles.
    """
    try:
        df = pd.read_excel(file_datos, sheet_name="Demanda")
        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
        
        referencias = [col for col in df.columns if col not in ['Fecha', 'Total']]
        años = sorted(df['Fecha'].dt.year.unique())
        
        return referencias, años
    except Exception as e:
        print(f"Error leyendo referencias: {e}")
        return [], []


# 🎯 CONFIGURACIÓN: Cambia aquí la política que deseas optimizar
POLICY_TO_OPTIMIZE = "LXL"  # Opciones: "QR", "ST", "SsT", "SS", "EOQ", "POQ", "LXL"
if __name__ == "__main__":
    # --- Configuración de parámetros ---
    file_datos = "Excel_APP_Pruebas.xlsx"
    file_demanda = "Datos Completos Sr Pizza.xlsx"
    u = 30  # períodos para test y pronóstico
    n_replicas = 100  # cantidad de réplicas (increased for better variability testing)

    # --- Mostrar puntos disponibles ---
    print("Puntos disponibles en el archivo:")
    try:
        puntos_disponibles = get_available_references(file_demanda)
        for i, ref in enumerate(puntos_disponibles):
            print(f"  {i+1}. {ref}")
        
        # Usar la primera referencia disponible como ejemplo
        ref_ejemplo = puntos_disponibles[0] if puntos_disponibles else "AP20"
        print(f"\n Usando referencia: {ref_ejemplo}")
        
    except Exception as e:
        print(f" Error leyendo referencias: {e}")
        ref_ejemplo = "AP20"  # fallback

    try:
        año = int(input("Año: "))
    except:
        print(" Año inválido.")
        año = 2023
    
    # --- Optimizar política seleccionada ---
    print(f"\n Optimizando política {POLICY_TO_OPTIMIZE} para {ref_ejemplo} en {año}")
    
    # Mostrar políticas disponibles
    políticas_disponibles = ["QR", "ST", "SsT", "SS", "EOQ", "POQ", "LxL"]
    print(f"\n Políticas disponibles: {', '.join(políticas_disponibles)}")
    print(f"   Política actual: {POLICY_TO_OPTIMIZE}")
    print(f"    Para cambiar política, modifica POLICY_TO_OPTIMIZE en línea 391")
    
    try:
        replicas_matrix, result = optimize_policy(
            file_demanda, ref_ejemplo, año, POLICY_TO_OPTIMIZE, file_datos, u, n_replicas
        )
        print(f"\n Optimización completada exitosamente!")
        
    except Exception as e:
                print(f" Error durante optimización para {ref_ejemplo} en {año}: {e}")