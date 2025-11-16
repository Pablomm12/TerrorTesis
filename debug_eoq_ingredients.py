"""
Debug script to identify why EOQ optimization for ingredients gives 0 orders
"""

import numpy as np
import pandas as pd
import os
import sys

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def debug_eoq_parameters():
    """
    Debug the EOQ parameters for ingredient families
    """
    from services.leer_datos import procesar_datos
    from services.materia_prima import perform_ingredient_clustering, create_ingredient_data_dict
    
    print("🔍 DEBUGGING EOQ INGREDIENTS - CHECKING PARAMETERS")
    print("="*60)
    
    try:
        # Load data
        file_datos = "Código Juana.xlsx"
        if not os.path.exists(file_datos):
            print(f"❌ File not found: {file_datos}")
            return
        
        print("📁 Loading data...")
        data_dict, materia_prima, recetas_primero, recetas_segundo = procesar_datos(file_datos)
        
        # Test with a small set of ingredients
        selected_ingredients = list(materia_prima.keys())[:6]  # Take first 6 ingredients
        print(f"🧪 Testing with {len(selected_ingredients)} ingredients: {selected_ingredients}")
        
        # Perform clustering (use 2 families for testing)
        print("\n📊 Performing clustering...")
        cluster_result = perform_ingredient_clustering(
            selected_ingredients=selected_ingredients,
            materia_prima=materia_prima,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            k_clusters=2  # Force 2 families for testing
        )
        
        df_clustered, cluster_info = cluster_result
        
        # Create ingredient data dict
        print("\n🏭 Creating ingredient data dict...")
        data_dict_MP = create_ingredient_data_dict(
            selected_ingredients=selected_ingredients,
            cluster_info=cluster_info,
            materia_prima=materia_prima,
            recetas_primero=recetas_primero,
            recetas_segundo=recetas_segundo,
            data_dict_pizzas=data_dict
        )
        
        # Analyze each family's parameters
        print("\n" + "="*60)
        print("📋 ANALYZING FAMILY PARAMETERS")
        print("="*60)
        
        for familia_name, familia_data in data_dict_MP.items():
            print(f"\n🎯 FAMILY: {familia_name}")
            print("-" * 40)
            
            params = familia_data.get("PARAMETROS", {})
            resultados = familia_data.get("RESULTADOS", {})
            
            # Key EOQ parameters
            demanda_promedio = params.get("demanda_promedio", 0)
            demanda_diaria = params.get("demanda_diaria", 0)
            costo_pedir = params.get("costo_pedir", 0)
            costo_sobrante = params.get("costo_sobrante", 0)
            costo_unitario = params.get("costo_unitario", 0)
            inventario_inicial = params.get("inventario_inicial", 0)
            lead_time = params.get("lead time", 0)
            
            print(f"📊 Critical Parameters:")
            print(f"   demanda_promedio: {demanda_promedio}")
            print(f"   demanda_diaria: {demanda_diaria}")
            print(f"   costo_pedir: {costo_pedir}")
            print(f"   costo_sobrante: {costo_sobrante}")
            print(f"   costo_unitario: {costo_unitario}")
            print(f"   inventario_inicial: {inventario_inicial}")
            print(f"   lead_time: {lead_time}")
            
            # Calculate EOQ manually to verify
            if demanda_promedio > 0 and costo_pedir > 0 and costo_sobrante > 0:
                demanda_anual = demanda_promedio * 365
                eoq_batch_size = int(round((2 * demanda_anual * costo_pedir / costo_sobrante) ** 0.5))
                print(f"\n🧮 Manual EOQ Calculation:")
                print(f"   Demanda anual: {demanda_anual:.1f}")
                print(f"   EOQ batch size: {eoq_batch_size}")
                
                if eoq_batch_size == 0:
                    print(f"   ❌ PROBLEM: EOQ batch size is 0!")
                    print(f"   → Check if costs are too small relative to demand")
                else:
                    print(f"   ✅ EOQ batch size looks reasonable")
            else:
                print(f"\n❌ PROBLEM: Missing or zero critical parameters")
                print(f"   → demanda_promedio: {demanda_promedio > 0}")
                print(f"   → costo_pedir: {costo_pedir > 0}")
                print(f"   → costo_sobrante: {costo_sobrante > 0}")
            
            # Check sales data
            ventas = resultados.get("ventas", {})
            if ventas:
                numeric_sales = [v for k, v in ventas.items() if isinstance(k, (int, float)) and isinstance(v, (int, float))]
                if numeric_sales:
                    print(f"\n📈 Sales Data:")
                    print(f"   Number of periods: {len(numeric_sales)}")
                    print(f"   Average sales: {np.mean(numeric_sales):.2f}")
                    print(f"   Sales range: {min(numeric_sales):.2f} - {max(numeric_sales):.2f}")
                    
                    if max(numeric_sales) == 0:
                        print(f"   ❌ PROBLEM: All sales are 0!")
                else:
                    print(f"\n❌ PROBLEM: No numeric sales data found")
                    print(f"   ventas keys: {list(ventas.keys())[:10]}")
                    print(f"   ventas sample: {dict(list(ventas.items())[:5])}")
            else:
                print(f"\n❌ PROBLEM: No sales data (RESULTADOS.ventas)")
        
        return data_dict_MP, cluster_info
        
    except Exception as e:
        print(f"❌ Error during debug: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def debug_eoq_simulation():
    """
    Debug the actual EOQ simulation to see why it returns 0 orders
    """
    print("\n" + "="*60)
    print("🔧 DEBUGGING EOQ SIMULATION LOGIC")
    print("="*60)
    
    # Test with simple synthetic data
    from services.simulacion import replicas_EOQ_verbose
    
    # Create a simple test case
    test_data_dict = {
        "Familia_Test": {
            "PARAMETROS": {
                "demanda_promedio": 100,  # 100g per day
                "demanda_diaria": 100,
                "costo_pedir": 50,  # $50 ordering cost
                "costo_sobrante": 2,  # $2 holding cost per unit
                "costo_unitario": 5,  # $5 unit cost
                "costo_faltante": 10,  # $10 stockout cost
                "inventario_inicial": 50,
                "lead time": 3,
                "backorders": 1
            },
            "RESULTADOS": {
                "ventas": {i: 90 + np.random.normal(0, 10) for i in range(30)}  # Random sales around 100
            }
        }
    }
    
    # Create test replicas matrix (10 replicas, 30 periods)
    test_replicas = np.random.normal(100, 15, (10, 30))  # Average 100g demand with variation
    test_replicas = np.maximum(test_replicas, 0)  # Ensure non-negative
    
    print(f"🧪 Test replicas matrix shape: {test_replicas.shape}")
    print(f"📊 Test replicas statistics:")
    print(f"   Average: {test_replicas.mean():.2f}")
    print(f"   Min: {test_replicas.min():.2f}")
    print(f"   Max: {test_replicas.max():.2f}")
    
    # Test EOQ with different safety percentages
    safety_percentages = [0.1, 0.2, 0.3]  # 10%, 20%, 30%
    
    for safety_pct in safety_percentages:
        print(f"\n🔍 Testing with safety percentage: {safety_pct*100:.0f}%")
        
        try:
            result = replicas_EOQ_verbose(
                matrizReplicas=test_replicas,
                data_dict=test_data_dict,
                punto_venta="Familia_Test",
                porcentaje_seguridad=safety_pct
            )
            
            if result and len(result) >= 2:
                df_promedio, liberacion_orden_df = result[:2]
                
                # Check liberation orders
                if liberacion_orden_df is not None:
                    total_orders = liberacion_orden_df.values.sum()
                    max_order = liberacion_orden_df.values.max()
                    
                    print(f"   📋 Liberation orders:")
                    print(f"      Total orders: {total_orders}")
                    print(f"      Max single order: {max_order}")
                    print(f"      Liberation matrix shape: {liberacion_orden_df.shape}")
                    
                    if total_orders == 0:
                        print(f"   ❌ PROBLEM: No orders generated!")
                        print(f"   📋 Liberation matrix sample:")
                        print(liberacion_orden_df.head())
                    else:
                        print(f"   ✅ Orders generated successfully")
                        
                        # Show sample periods with orders
                        periods_with_orders = (liberacion_orden_df.values > 0).any(axis=1)
                        num_periods_with_orders = periods_with_orders.sum()
                        print(f"   📅 Periods with orders: {num_periods_with_orders}/{len(periods_with_orders)}")
                else:
                    print(f"   ❌ PROBLEM: No liberation order DataFrame returned")
                
                # Check indicators
                if df_promedio is not None:
                    print(f"   📊 Indicators:")
                    for idx in df_promedio.index:
                        print(f"      {idx}: {df_promedio.loc[idx].iloc[0]:.2f}")
                else:
                    print(f"   ❌ PROBLEM: No indicators DataFrame returned")
            else:
                print(f"   ❌ PROBLEM: EOQ function returned invalid result")
                
        except Exception as e:
            print(f"   ❌ ERROR in EOQ simulation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Starting EOQ Ingredients Debug")
    
    # Debug 1: Check parameters
    data_dict_MP, cluster_info = debug_eoq_parameters()
    
    # Debug 2: Check simulation logic
    debug_eoq_simulation()
    
    print(f"\n" + "="*60)
    print("🎯 DEBUG SUMMARY")
    print("="*60)
    print("1. Check the output above for:")
    print("   - Zero or missing demanda_promedio/demanda_diaria")
    print("   - Zero or missing costo_pedir/costo_sobrante")
    print("   - Zero sales data in RESULTADOS.ventas")
    print("   - Issues in EOQ batch size calculation")
    print("2. Most likely issues:")
    print("   - Conversion from pizzas to ingredients giving 0 values")
    print("   - Missing cost parameters in ingredient data")
    print("   - Incorrect demand calculation in family aggregation")
    print("="*60)