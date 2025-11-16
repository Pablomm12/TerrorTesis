#!/usr/bin/env python3
"""
Test the family liberation fix to ensure Excel exports are now coherent
"""

import sys
import os
import json
import numpy as np
import pandas as pd

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

def test_family_liberation_fix():
    """Test that the family liberation fix produces coherent results"""
    
    print("🧪 TESTING FAMILY LIBERATION FIX")
    print("=" * 50)
    
    try:
        from services.family_liberation_generator import (
            convert_pizza_to_ingredient_data,
            generate_family_liberation_order_vectors
        )
        from services.simulacion import replicas_EOQ_verbose
        from services.materia_prima import convert_pizza_demand_to_ingredient_demand
        
        print("✅ All modules imported successfully")
        
        # Mock realistic data
        pizza_data_dict = {
            "TestPV": {
                "PARAMETROS": {
                    "lead time": 2,
                    "backorders": 1,
                    "costo_pedir": 100,
                    "costo_unitario": 8,
                    "costo_faltante": 50,
                    "costo_sobrante": 2
                },
                "RESULTADOS": {
                    "ventas": {i: 15 + i % 10 for i in range(30)},  # 30 days of realistic pizza sales
                    "T": 3
                }
            }
        }
        
        # Mock recipes with realistic proportions
        recetas_segundo = {
            "Pizza_Margherita": {
                "nombre": "Margherita",
                "Proporción ventas": 0.35,  # 35% of sales
                "ingredientes": {
                    "I_TOMATE": {"cantidad": 120},  # 120g tomato per pizza
                    "I_QUESO": {"cantidad": 180}    # 180g cheese per pizza
                }
            },
            "Pizza_Pepperoni": {
                "nombre": "Pepperoni", 
                "Proporción ventas": 0.40,  # 40% of sales
                "ingredientes": {
                    "I_TOMATE": {"cantidad": 100},  # 100g tomato per pizza
                    "I_QUESO": {"cantidad": 150}    # 150g cheese per pizza
                }
            },
            "Pizza_Vegetariana": {
                "nombre": "Vegetariana",
                "Proporción ventas": 0.25,  # 25% of sales
                "ingredientes": {
                    "I_TOMATE": {"cantidad": 90},   # 90g tomato per pizza
                    "I_QUESO": {"cantidad": 140}    # 140g cheese per pizza
                }
            }
        }
        
        recetas_primero = {}
        
        materia_prima = {
            "I_TOMATE": {
                "nombre": "Pasta de Tomate",
                "inventario_inicial": 2000,
                "lead_time": 3,
                "MOQ": 5000,
                "costo_pedir": 75,
                "costo_unitario": 0.03,  # $0.03 per gram
                "costo_faltante": 0.15,
                "costo_sobrante": 0.005
            },
            "I_QUESO": {
                "nombre": "Queso Mozzarella",
                "inventario_inicial": 1500,
                "lead_time": 2,
                "MOQ": 3000,
                "costo_pedir": 60,
                "costo_unitario": 0.08,  # $0.08 per gram
                "costo_faltante": 0.40,
                "costo_sobrante": 0.02
            }
        }
        
        print("\n📊 TESTING INDIVIDUAL INGREDIENT CONVERSION:")
        
        # Test the fixed conversion for tomato
        ingredient_data_dict = convert_pizza_to_ingredient_data(
            "I_TOMATE", pizza_data_dict, "TestPV",
            recetas_primero, recetas_segundo, materia_prima
        )
        
        params = ingredient_data_dict["TestPV"]["PARAMETROS"]
        print(f"I_TOMATE - demanda_diaria: {params['demanda_diaria']:.2f}g/day")
        print(f"I_TOMATE - costo_unitario: ${params['costo_unitario']:.4f}/g")
        
        # Check if daily demand is reasonable (should be much larger than conversion factor)
        # Expected conversion factor = 0.35*120 + 0.40*100 + 0.25*90 = 42+40+22.5 = 104.5g per pizza
        # Average pizza sales = ~18 pizzas/day, so expected daily demand ~1,881g/day
        
        expected_factor = 0.35*120 + 0.40*100 + 0.25*90
        avg_pizza_sales = sum(pizza_data_dict["TestPV"]["RESULTADOS"]["ventas"].values()) / 30
        expected_daily = avg_pizza_sales * expected_factor
        
        print(f"Expected conversion factor: {expected_factor:.2f}g per pizza")
        print(f"Average pizza sales: {avg_pizza_sales:.1f} pizzas/day") 
        print(f"Expected daily demand: {expected_daily:.2f}g/day")
        
        # Verify the fix worked
        demand_ratio = params['demanda_diaria'] / expected_factor
        if demand_ratio > 10:  # Should be much higher than conversion factor
            print("✅ FIX SUCCESS: Daily demand properly calculated")
            print(f"   Daily demand is {demand_ratio:.1f}x the conversion factor")
        else:
            print("❌ FIX FAILED: Still using conversion factor instead of daily demand")
            return False
        
        print("\n🔄 TESTING EOQ SIMULATION WITH FIXED DATA:")
        
        # Create a simple replicas matrix for testing
        test_replicas = np.random.poisson(params['demanda_diaria'], size=(5, 30))  # 5 replicas, 30 days
        
        # Test EOQ simulation with the fixed ingredient data
        df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = replicas_EOQ_verbose(
            test_replicas, ingredient_data_dict, "TestPV", porcentaje_seguridad=0.95
        )
        
        print(f"EOQ Results:")
        print(f"  Total cost: ${df_promedio['Total_Cost'].iloc[0]:,.2f}")
        print(f"  Demand satisfaction: {df_promedio['Demand_Satisfaction'].iloc[0]:.1f}%")
        print(f"  Liberation final: {liberacion_final}")
        
        # Check if results are reasonable
        total_cost = df_promedio['Total_Cost'].iloc[0]
        demand_satisfaction = df_promedio['Demand_Satisfaction'].iloc[0]
        
        if total_cost > 1000 and demand_satisfaction > 90:  # Reasonable thresholds
            print("✅ EOQ SIMULATION SUCCESS: Coherent results")
        else:
            print("❌ EOQ SIMULATION FAILED: Incoherent results")
            return False
        
        print("\n🎯 OVERALL TEST RESULT: ✅ SUCCESS")
        print("The family liberation fix is working correctly!")
        print("Excel exports should now show coherent results.")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_family_liberation_fix()
    sys.exit(0 if success else 1)