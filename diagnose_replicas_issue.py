"""
Diagnostic script to check replicas matrix content in ingredient optimization
"""

import numpy as np
import pandas as pd
import os
import sys

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def diagnose_replicas_matrix_issue():
    """
    Diagnoses if the replicas matrix in ingredient optimization Excel export
    contains pizza units or ingredient units
    """
    
    print("🔍 DIAGNOSING REPLICAS MATRIX ISSUE IN INGREDIENT OPTIMIZATION")
    print("="*70)
    
    # Test the create_ingredient_replicas_matrix_from_data_dict function
    print("\n1. 🧪 Testing ingredient replicas matrix generation...")
    
    try:
        from services.PSO import create_ingredient_replicas_matrix_from_data_dict
        
        # Create mock data_dict_MP with ingredient-converted data
        mock_data_dict_MP = {
            'Familia_1': {
                'PARAMETROS': {
                    'demanda_diaria': 150.5,  # grams of ingredient per day
                    'demanda_std': 45.2,
                    'representativo': 'Tomate_Cherry',
                    'cantidad_por_pizza': 25.5,  # grams per pizza
                    'unidad': 'g'
                },
                'RESULTADOS': {
                    'ventas': {
                        0: 148.2, 1: 152.7, 2: 145.8, 3: 155.1, 4: 150.0,  # grams
                        5: 147.5, 6: 153.2, 7: 149.8, 8: 151.3, 9: 146.9,
                        10: 154.0, 11: 148.5, 12: 152.1, 13: 150.2, 14: 149.7
                    }
                }
            }
        }
        
        # Generate ingredient replicas matrix
        replicas_matrix = create_ingredient_replicas_matrix_from_data_dict(
            data_dict_MP=mock_data_dict_MP,
            familia_name='Familia_1',
            n_replicas=5,
            u=15
        )
        
        print(f"✅ Ingredient replicas matrix generated: {replicas_matrix.shape}")
        print(f"📊 Value range: {replicas_matrix.min():.1f} - {replicas_matrix.max():.1f}")
        print(f"📈 Average value: {replicas_matrix.mean():.1f}")
        
        # Check if values are in ingredient range (should be ~150g, not ~6 pizzas)
        expected_ingredient_range = (120, 180)  # grams
        pizza_range = (3, 8)  # pizzas
        
        avg_value = replicas_matrix.mean()
        
        if expected_ingredient_range[0] <= avg_value <= expected_ingredient_range[1]:
            print(f"✅ VALUES LOOK CORRECT: Average {avg_value:.1f}g is in expected ingredient range")
            matrix_type = "INGREDIENT (grams)"
        elif pizza_range[0] <= avg_value <= pizza_range[1]:
            print(f"❌ VALUES LOOK WRONG: Average {avg_value:.1f} looks like pizza units, not grams")
            matrix_type = "PIZZA (units)"
        else:
            print(f"⚠️ VALUES UNCLEAR: Average {avg_value:.1f} doesn't match expected ranges")
            matrix_type = "UNKNOWN"
            
        print(f"🔍 Matrix type identified: {matrix_type}")
        
    except Exception as e:
        print(f"❌ Error testing ingredient replicas generation: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Excel export labeling
    print(f"\n2. 📋 Checking Excel export labeling...")
    
    try:
        from services.PSO import export_optimization_results_to_excel
        
        # Check if we can improve the labeling in Excel
        print("💡 RECOMMENDATIONS FOR EXCEL EXPORT:")
        print("   - Sheet 'Demanda_Réplicas' should be renamed based on optimization type")
        print("   - For ingredients: 'Demanda_Ingredientes_Réplicas' (units: grams)")
        print("   - For pizzas: 'Demanda_Pizzas_Réplicas' (units: pizzas)")
        print("   - Add unit information in sheet name or header")
        
    except Exception as e:
        print(f"⚠️ Could not check Excel export function: {e}")
    
    # Test the actual conversion process
    print(f"\n3. 🔄 Testing pizza-to-ingredient conversion...")
    
    try:
        # Simulate pizza liberation matrix (6 pizzas per day average)
        pizza_matrix = np.array([
            [5, 6, 7, 6, 5],
            [6, 5, 8, 7, 6], 
            [7, 6, 5, 6, 7],
            [6, 7, 6, 5, 6]
        ])  # 4 periods x 5 replicas, values in pizzas
        
        print(f"🍕 Mock pizza liberation matrix:")
        print(f"   Shape: {pizza_matrix.shape}")
        print(f"   Range: {pizza_matrix.min()} - {pizza_matrix.max()} pizzas")
        print(f"   Average: {pizza_matrix.mean():.1f} pizzas per period")
        
        # Apply conversion factor (25.5g per pizza)
        conversion_factor = 25.5  # grams per pizza
        ingredient_matrix = pizza_matrix * conversion_factor
        
        print(f"\n🧪 Converted to ingredient matrix:")
        print(f"   Shape: {ingredient_matrix.shape}")
        print(f"   Range: {ingredient_matrix.min():.1f} - {ingredient_matrix.max():.1f} grams")
        print(f"   Average: {ingredient_matrix.mean():.1f} grams per period")
        print(f"   Conversion: {pizza_matrix.mean():.1f} pizzas × {conversion_factor}g = {ingredient_matrix.mean():.1f}g")
        
        # This should be what appears in the Excel for ingredient optimization
        print(f"\n📋 EXPECTED IN EXCEL 'Demanda_Réplicas' for ingredients:")
        print(f"   Values around {ingredient_matrix.mean():.0f}g per period (NOT ~6 pizzas)")
        
    except Exception as e:
        print(f"❌ Error in conversion test: {e}")
    
    print(f"\n" + "="*70)
    print("🎯 DIAGNOSIS SUMMARY:")
    print("="*70)
    
    print("✅ WHAT SHOULD HAPPEN:")
    print("   1. Pizza optimization → Excel shows pizza units (3-8 pizzas per period)")
    print("   2. Ingredient optimization → Excel shows grams (120-180g per period)")
    print("   3. Different sheet names or headers to distinguish the two")
    
    print("\n❌ WHAT MIGHT BE WRONG:")
    print("   1. Same replicas_matrix being passed to Excel for both cases")
    print("   2. Pizza matrix not being converted to ingredient units")
    print("   3. Excel sheet not distinguishing between pizza/ingredient units")
    
    print("\n🔧 HOW TO VERIFY:")
    print("   1. Run ingredient optimization and check Excel 'Demanda_Réplicas' sheet")
    print("   2. Values should be ~150g per period, not ~6 pizzas per period")
    print("   3. Compare with pizza optimization Excel - should be different values")
    
    print("\n💡 SOLUTION IF CONFIRMED:")
    print("   1. Ensure replicas_matrix passed to Excel is properly converted")
    print("   2. Add unit labels to Excel sheet names")
    print("   3. Add conversion info to summary sheet")

if __name__ == "__main__":
    diagnose_replicas_matrix_issue()