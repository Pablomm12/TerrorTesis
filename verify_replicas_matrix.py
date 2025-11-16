"""
Verification script to check the actual replicas matrix values in ingredient optimization
"""

import numpy as np
import pandas as pd
import os
import sys

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def verify_ingredient_replicas_matrix():
    """
    Verifies that ingredient optimization is using properly converted replicas matrices
    """
    
    print("🔍 VERIFYING INGREDIENT REPLICAS MATRIX CONVERSION")
    print("="*60)
    
    try:
        from services.PSO import create_ingredient_replicas_matrix_from_data_dict
        from services.materia_prima import optimize_cluster_policy
        
        # Test 1: Direct function test with known conversion
        print("\n1. 🧪 Testing direct ingredient replicas generation...")
        
        # Create test data with known pizza-to-ingredient conversion
        test_data_dict_MP = {
            'Familia_Test': {
                'PARAMETROS': {
                    'demanda_diaria': 127.5,  # Should be in grams (5 pizzas × 25.5g/pizza)
                    'demanda_std': 38.25,     # 30% CV
                    'cantidad_por_pizza': 25.5,  # grams per pizza
                    'unidad': 'g',
                    'representativo': 'Test_Ingredient'
                },
                'RESULTADOS': {
                    'ventas': {
                        # These should be ingredient amounts (grams), not pizza counts
                        0: 127.5, 1: 153.0, 2: 102.0, 3: 178.5, 4: 127.5,  # 5,6,4,7,5 pizzas converted
                        5: 119.85, 6: 161.15, 7: 135.15, 8: 140.25, 9: 114.75,  # More conversions
                        10: 165.75, 11: 122.4, 12: 153.0, 13: 127.5, 14: 135.15
                    }
                }
            }
        }
        
        # Generate replicas matrix
        replicas_matrix = create_ingredient_replicas_matrix_from_data_dict(
            data_dict_MP=test_data_dict_MP,
            familia_name='Familia_Test',
            n_replicas=6,
            u=15
        )
        
        print(f"✅ Generated replicas matrix shape: {replicas_matrix.shape}")
        print(f"📊 Value statistics:")
        print(f"   Min: {replicas_matrix.min():.1f}")
        print(f"   Max: {replicas_matrix.max():.1f}")
        print(f"   Average: {replicas_matrix.mean():.1f}")
        print(f"   Std Dev: {replicas_matrix.std():.1f}")
        
        # Check if values are consistent with ingredient units
        expected_pizza_equivalent = replicas_matrix.mean() / 25.5
        print(f"\n🔍 Analysis:")
        print(f"   Average value: {replicas_matrix.mean():.1f}g")
        print(f"   Equivalent pizzas: {expected_pizza_equivalent:.1f} pizzas")
        print(f"   Expected range: 100-180g (4-7 pizzas × 25.5g/pizza)")
        
        if 100 <= replicas_matrix.mean() <= 180:
            print(f"   ✅ VALUES CORRECT: Matrix contains ingredient amounts (grams)")
            matrix_status = "CORRECT"
        elif 3 <= replicas_matrix.mean() <= 8:
            print(f"   ❌ VALUES WRONG: Matrix contains pizza amounts (units)")
            matrix_status = "WRONG - PIZZA UNITS"
        else:
            print(f"   ⚠️ VALUES UNCLEAR: Unexpected range")
            matrix_status = "UNCLEAR"
            
        # Show sample values for manual inspection
        print(f"\n📋 Sample matrix values (first 3 replicas, first 5 periods):")
        print(replicas_matrix[:3, :5])
        
        # Test 2: Check the Excel export enhancement
        print(f"\n2. 📊 Testing enhanced Excel export...")
        
        from services.PSO import export_optimization_results_to_excel
        
        # Create mock optimization results
        mock_ingredient_info = {
            'cluster_id': 1,
            'ingredient_code': 'TEST_ING_001',
            'representative_ingredient': 'Test Ingredient',
            'conversion_factor': '25.5g per pizza',
            'unit': 'g',
            'pizza_point_of_sale': 'TestPV',
            'cluster_size': 3,
            'optimization_type': 'Ingredient Cluster Optimization'
        }
        
        # Mock other required data
        mock_df_promedio = pd.DataFrame({'Valor': [1500, 0.95, 140]}, index=['Costo total', 'Nivel servicio', 'Inventario'])
        mock_liberation_df = pd.DataFrame(np.random.randint(80, 120, (15, 6)), 
                                        index=[f'Periodo_{i+1}' for i in range(15)],
                                        columns=[f'Replica_{i+1}' for i in range(6)])
        mock_replica_results = [pd.DataFrame({'Valor': [1480 + i*10, 0.94 + i*0.005]}, 
                                           index=['Costo total', 'Nivel servicio']) for i in range(6)]
        
        # Test Excel export
        try:
            excel_path = export_optimization_results_to_excel(
                policy='LXL',
                ref='Familia_Test',
                best_decision_vars={'porcentaje': 0.15},
                df_promedio=mock_df_promedio,
                liberacion_orden_df=mock_liberation_df,
                resultados_replicas=mock_replica_results,
                replicas_matrix=replicas_matrix,  # Use the ingredient matrix we generated
                output_dir='test_results',
                ingredient_info=mock_ingredient_info
            )
            
            if excel_path and os.path.exists(excel_path):
                print(f"✅ Enhanced Excel file created: {os.path.basename(excel_path)}")
                
                # Check sheet names
                xl_file = pd.ExcelFile(excel_path)
                sheet_names = xl_file.sheet_names
                
                ingredient_sheets = [s for s in sheet_names if 'Ingredientes' in s or '(g)' in s]
                if ingredient_sheets:
                    print(f"   ✅ Found ingredient-specific sheet: {ingredient_sheets}")
                    
                    # Read the ingredient sheet to verify values
                    sheet_name = ingredient_sheets[0]
                    df_check = pd.read_excel(excel_path, sheet_name=sheet_name)
                    print(f"   📊 Sheet '{sheet_name}' contains {df_check.shape[0]} rows × {df_check.shape[1]} columns")
                    
                    # Check if values are in grams range
                    numeric_data = df_check.select_dtypes(include=[np.number])
                    if not numeric_data.empty:
                        avg_in_sheet = numeric_data.mean().mean()
                        print(f"   📈 Average value in sheet: {avg_in_sheet:.1f}")
                        
                        if 100 <= avg_in_sheet <= 180:
                            print(f"   ✅ Excel contains INGREDIENT values (grams)")
                        elif 3 <= avg_in_sheet <= 8:
                            print(f"   ❌ Excel contains PIZZA values (units) - THIS IS THE BUG!")
                        else:
                            print(f"   ⚠️ Excel values unclear")
                else:
                    print(f"   ⚠️ No ingredient-specific sheet found")
                    print(f"   Available sheets: {sheet_names}")
            
        except Exception as e:
            print(f"   ⚠️ Excel export test failed: {e}")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("🎯 VERIFICATION SUMMARY")
    print("="*60)
    
    print("🔍 WHAT TO CHECK IN YOUR ACTUAL EXCEL FILE:")
    print("   1. Open the ingredient optimization Excel file")
    print("   2. Look at 'Demanda_Réplicas' or 'Demanda_Ingredientes' sheet")
    print("   3. Check the values in the matrix:")
    print("      - Should be ~100-200 (grams of ingredient)")
    print("      - Should NOT be ~3-8 (pizza units)")
    print("   4. Compare with pizza optimization Excel - values should be different")
    
    print(f"\n💡 IF VALUES ARE WRONG (showing pizza units instead of grams):")
    print("   - The issue is in the replicas matrix generation for ingredients")
    print("   - Need to ensure proper conversion in optimize_cluster_policy()")
    print("   - Check that create_ingredient_replicas_matrix_from_data_dict() is working")
    
    print(f"\n✅ IF VALUES ARE CORRECT (showing grams):")
    print("   - The issue is only in Excel sheet labeling/naming")
    print("   - My recent enhancement should fix this")
    print("   - Sheet should now be named 'Demanda_Ingredientes_(g)'")

if __name__ == "__main__":
    verify_ingredient_replicas_matrix()