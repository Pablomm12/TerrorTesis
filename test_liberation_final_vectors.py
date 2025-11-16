#!/usr/bin/env python3
"""
Test script to verify that family liberation correctly extracts and uses
the liberation_final vector from verbose simulation functions.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

def test_liberation_final_extraction():
    """
    Test that we correctly extract liberation_final from verbose functions
    """
    
    print("🧪 TESTING LIBERATION_FINAL VECTOR EXTRACTION")
    print("="*60)
    
    try:
        # Import needed modules
        from services.family_liberation_generator import convert_pizza_to_ingredient_data
        from services.simulacion import replicas_EOQ_verbose
        import numpy as np
        
        # Create test data
        test_ingredient_data = {
            "TestPV": {
                "PARAMETROS": {
                    "inventario_inicial": 1000,
                    "lead time": 2,
                    "MOQ": 0,
                    "backorders": 1,
                    "costo_pedir": 50,
                    "costo_unitario": 0.05,
                    "costo_faltante": 0.25,
                    "costo_sobrante": 0.01,
                    "demanda_diaria": 2000,  # 2kg per day
                    "demanda_promedio": 60000  # 60kg per month
                },
                "RESULTADOS": {
                    "ventas": {i: 1800 + (i % 200) for i in range(30)},  # Variable demand
                    "T": 3
                }
            }
        }
        
        # Create test replicas matrix
        np.random.seed(42)
        test_replicas = np.random.poisson(2000, size=(5, 30))  # 5 replicas, 30 days
        
        print(f"📊 Test data created:")
        print(f"   Daily demand: {test_ingredient_data['TestPV']['PARAMETROS']['demanda_diaria']}g")
        print(f"   Replicas matrix: {test_replicas.shape}")
        print(f"   Average replica demand: {test_replicas.mean():.1f}g/day")
        
        # Test EOQ verbose function directly
        print(f"\n🔄 Testing EOQ verbose function directly:")
        
        df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = replicas_EOQ_verbose(
            test_replicas, test_ingredient_data, "TestPV", porcentaje_seguridad=0.95
        )
        
        print(f"✅ EOQ verbose function returned 4 values:")
        print(f"   df_promedio: {type(df_promedio)} - {df_promedio.shape}")
        print(f"   liberacion_orden_df: {type(liberacion_orden_df)} - {liberacion_orden_df.shape}")  
        print(f"   resultados_replicas: {type(resultados_replicas)} - {len(resultados_replicas)} replicas")
        print(f"   liberacion_final: {type(liberacion_final)} - Length: {len(liberacion_final) if hasattr(liberacion_final, '__len__') else 'N/A'}")
        
        # Analyze liberation_final
        if liberacion_final is not None:
            if hasattr(liberacion_final, '__iter__'):
                vector_sum = float(np.sum(liberacion_final))
                vector_max = float(np.max(liberacion_final)) if len(liberacion_final) > 0 else 0
                vector_min = float(np.min(liberacion_final)) if len(liberacion_final) > 0 else 0
                non_zero_periods = int(np.sum(liberacion_final > 0))
                
                print(f"\n📈 LIBERATION_FINAL ANALYSIS:")
                print(f"   Vector sum: {vector_sum:.0f}g")
                print(f"   Vector range: {vector_min:.0f} - {vector_max:.0f}g")
                print(f"   Non-zero periods: {non_zero_periods}/{len(liberacion_final)}")
                print(f"   First 5 periods: {liberacion_final[:5]}")
                print(f"   Last 5 periods: {liberacion_final[-5:]}")
                
                # Compare with matrix total
                matrix_total = liberacion_orden_df.sum().sum()
                print(f"\n🔍 COMPARISON:")
                print(f"   Matrix total (all replicas): {matrix_total:.0f}g")
                print(f"   Vector total (final): {vector_sum:.0f}g")
                print(f"   Ratio (vector/matrix): {vector_sum/matrix_total*100:.1f}%")
                
                # Verify the vector makes sense
                if vector_sum > 0 and non_zero_periods > 0:
                    print(f"✅ Liberation vector looks valid!")
                    return True
                else:
                    print(f"❌ Liberation vector appears invalid (sum={vector_sum}, active={non_zero_periods})")
                    return False
            else:
                print(f"❌ liberation_final is not iterable: {liberacion_final}")
                return False
        else:
            print(f"❌ liberation_final is None")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_family_liberation_integration():
    """
    Test the family liberation generator with the liberation_final vector
    """
    
    print(f"\n🏭 TESTING FAMILY LIBERATION INTEGRATION")
    print("="*50)
    
    try:
        from services.family_liberation_generator import generate_family_liberation_vectors
        
        # Mock minimal data for testing
        family_ingredients = ["I_TEST_INGREDIENT"]
        representative_ingredient = "I_TEST_INGREDIENT"
        optimized_params = {"porcentaje_seguridad": 0.95}
        policy = "EOQ"
        
        pizza_data_dict = {
            "TestPV": {
                "PARAMETROS": {
                    "lead time": 2,
                    "backorders": 1,
                    "demanda_diaria": 20,
                    "demanda_promedio": 600
                },
                "RESULTADOS": {
                    "ventas": {i: 15 + (i % 10) for i in range(30)},
                    "T": 3
                }
            }
        }
        
        np.random.seed(42)
        pizza_replicas_matrix = np.random.poisson(20, size=(5, 30))
        
        # Mock recipes
        recetas_segundo = {
            "Pizza_Test": {
                "nombre": "Test Pizza",
                "Proporción ventas": 1.0,
                "ingredientes": {
                    "I_TEST_INGREDIENT": {"cantidad": 100}
                }
            }
        }
        
        materia_prima = {
            "I_TEST_INGREDIENT": {
                "nombre": "Test Ingredient",
                "inventario_inicial": 1000,
                "lead_time": 2,
                "MOQ": 0,
                "costo_pedir": 50,
                "costo_unitario": 0.05,
                "costo_faltante": 0.25,
                "costo_sobrante": 0.01
            }
        }
        
        print(f"🔄 Calling family liberation generator...")
        
        family_results = generate_family_liberation_vectors(
            family_ingredients=family_ingredients,
            representative_ingredient=representative_ingredient,
            optimized_params=optimized_params,
            policy=policy,
            pizza_data_dict=pizza_data_dict,
            pizza_replicas_matrix=pizza_replicas_matrix,
            punto_venta="TestPV",
            recetas_primero={},
            recetas_segundo=recetas_segundo,
            materia_prima=materia_prima,
            verbose=True
        )
        
        print(f"\n📊 Family liberation results:")
        for ingredient, results in family_results.items():
            if "error" not in results:
                liberation_final = results.get("liberation_final", [])
                liberation_df = results.get("liberation_df", pd.DataFrame())
                
                vector_sum = float(np.sum(liberation_final)) if liberation_final is not None and hasattr(liberation_final, '__iter__') else 0
                matrix_total = liberation_df.sum().sum() if not liberation_df.empty else 0
                
                print(f"   🧪 {ingredient}:")
                print(f"      Liberation vector sum: {vector_sum:.0f}")
                print(f"      Liberation matrix total: {matrix_total:.0f}")
                print(f"      Vector length: {len(liberation_final) if liberation_final is not None and hasattr(liberation_final, '__len__') else 0}")
                
                if vector_sum > 0:
                    print(f"      ✅ Vector captured successfully!")
                else:
                    print(f"      ❌ Vector appears empty or invalid")
                    return False
            else:
                print(f"   ❌ {ingredient}: {results['error']}")
                return False
        
        print(f"✅ Family liberation integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Family liberation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 TESTING LIBERATION_FINAL VECTOR HANDLING")
    print("="*70)
    
    # Test 1: Direct verbose function test
    test1_success = test_liberation_final_extraction()
    
    # Test 2: Family liberation integration
    test2_success = test_family_liberation_integration()
    
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Test 1 (Direct EOQ verbose): {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"   Test 2 (Family integration): {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"💡 Liberation_final vectors are being properly extracted and stored!")
        print(f"📊 Excel exports will show the final liberation vector in each family sheet!")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print(f"🔧 Check the verbose function return values and family liberation logic")
    
    sys.exit(0 if (test1_success and test2_success) else 1)