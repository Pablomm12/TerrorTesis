"""
Debug test for ingredient EOQ zero cost issue
"""

import sys
import os
sys.path.append('.')

from services.family_liberation_generator import create_ingredient_data_dict_from_pizza
from services.simulacion import replicas_EOQ_verbose
from services.leer_datos import procesar_datos
import pandas as pd
import numpy as np

def test_ingredient_eoq_zero_cost():
    """Debug test to identify why ingredient EOQ shows zero costs"""
    
    print("=" * 80)
    print("DEBUG TEST: INGREDIENT EOQ ZERO COST ISSUE")
    print("=" * 80)
    
    try:
        # Load base data using the actual loader
        print("Loading data with procesar_datos...")
        data_dict, materia_prima, recetas_primero, recetas_segundo = procesar_datos('Configuracion.xlsx')
        
        print(f"Loaded data for {len(data_dict)} points of sale")
        print(f"Available points: {list(data_dict.keys())}")
        
        # Pick a common ingredient to test
        test_ingredient = 'Aceite de canola'
        print(f"\nTesting ingredient: {test_ingredient}")
        
        # Convert pizza data to ingredient data
        ingredient_data_dict = create_ingredient_data_dict_from_pizza(
            test_ingredient, data_dict, materia_prima, recetas_primero, recetas_segundo
        )
        
        print(f"Created ingredient data dict successfully")
        print(f"Ingredient data keys: {list(ingredient_data_dict.keys())}")
        
        if test_ingredient in ingredient_data_dict:
            ingredient_params = ingredient_data_dict[test_ingredient]["PARAMETROS"]
            ingredient_results = ingredient_data_dict[test_ingredient]["RESULTADOS"]
            
            print(f"\nIngredient parameters:")
            for key, value in ingredient_params.items():
                print(f"  {key}: {value}")
                
            print(f"\nIngredient sales data:")
            ventas = ingredient_results.get("ventas", {})
            if ventas:
                sample_ventas = {k: v for i, (k, v) in enumerate(ventas.items()) if i < 10}
                print(f"  Sample (first 10): {sample_ventas}")
                total_demand = sum(ventas.values())
                print(f"  Total demand: {total_demand}")
            else:
                print("  NO VENTAS DATA FOUND!")
                
        # Create simple replicas matrix for testing
        num_periods = 30
        num_replicas = 5
        
        # Generate realistic demand values
        base_demand = 100.0  # 100g per period base demand
        replicas_matrix = np.random.normal(base_demand, base_demand * 0.2, (num_replicas, num_periods))
        replicas_matrix = np.maximum(replicas_matrix, 1)  # Ensure positive demand
        
        print(f"\nCreated replicas matrix: {replicas_matrix.shape}")
        print(f"Sample demand values: {replicas_matrix[0][:10]}")
        
        # Run EOQ simulation with debug output
        print(f"\n" + "="*50)
        print("RUNNING EOQ SIMULATION WITH DEBUG")
        print("="*50)
        
        resultado_df, liberacion_df = replicas_EOQ_verbose(
            matrizReplicas=replicas_matrix,
            data_dict=ingredient_data_dict,
            punto_venta=test_ingredient,
            porcentaje_seguridad=0.05
        )
        
        print(f"\n" + "="*50)
        print("EOQ SIMULATION RESULTS")
        print("="*50)
        
        print(f"Resultado indicators:")
        print(resultado_df)
        
        print(f"\nLiberacion orders shape: {liberacion_df.shape}")
        print(f"Total orders: {liberacion_df.sum().sum()}")
        
        # Check specific cost values
        total_cost = resultado_df.loc['Costo total', 'Promedio Indicadores']
        print(f"\nFinal total cost: {total_cost}")
        
        if total_cost == 0:
            print("PROBLEM IDENTIFIED: Total cost is zero")
            print("This suggests the simulation matrix has no activity")
        else:
            print("SUCCESS: Non-zero cost detected")
            
    except Exception as e:
        print(f"ERROR in test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingredient_eoq_zero_cost()