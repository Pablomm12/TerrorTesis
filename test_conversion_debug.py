#!/usr/bin/env python3
"""
Test script to debug pizza to ingredient conversion
"""
import numpy as np
import sys
import os

# Add the current directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.materia_prima import convert_pizza_liberation_matrix_to_ingredient

def test_manual_conversion():
    """Test conversion with manual data that matches the user's case"""
    
    print("🎯 PRUEBA MANUAL DE CONVERSIÓN")
    print("="*60)
    
    # Create test data that matches user's description
    # User said: "flavor B being 9% of total demand, 0.6 units per pizza"
    recetas_segundo = {
        "PIZZA_A": {
            "nombre": "Pizza A", 
            "Proporción ventas": 50,  # 50% of sales
            "ingredientes": {}  # No target ingredient
        },
        "PIZZA_B": {
            "nombre": "Pizza B",
            "Proporción ventas": 9,   # 9% of sales - USER'S CASE
            "ingredientes": {
                "TEST_INGREDIENT": {"cantidad": 0.6}  # 0.6 units per pizza - USER'S CASE
            }
        },
        "PIZZA_C": {
            "nombre": "Pizza C",
            "Proporción ventas": 41,  # 41% of sales (total = 100%)
            "ingredientes": {}  # No target ingredient
        }
    }
    
    recetas_primero = {}  # No first level recipes for this test
    
    # User's pizza matrix: [0, 649, 9]
    test_pizza_matrix = np.array([[0, 649, 9]]).T  # Shape: (3, 1) - periods x replicas
    
    print("DATOS DE PRUEBA:")
    print("  Pizza matrix input: [0, 649, 9] (3 períodos, 1 réplica)")
    print("  Ingrediente: TEST_INGREDIENT")
    print("  Pizza B: 9% de ventas, 0.6 unidades por pizza")
    print("  Cálculo esperado:")
    print("    Período 1: 0 × 0.09 × 0.6 = 0")
    print("    Período 2: 649 × 0.09 × 0.6 = 35.046 ≈ 35")
    print("    Período 3: 9 × 0.09 × 0.6 = 0.486 ≈ 1")
    print()
    
    # Test the conversion
    try:
        converted_matrix = convert_pizza_liberation_matrix_to_ingredient(
            test_pizza_matrix, 
            "TEST_INGREDIENT", 
            "TEST_PV", 
            recetas_primero, 
            recetas_segundo
        )
        
        print(f"\nRESULTADO ACTUAL:")
        print(f"  Período 1: {converted_matrix[0, 0]}")
        print(f"  Período 2: {converted_matrix[1, 0]}")
        print(f"  Período 3: {converted_matrix[2, 0]}")
        
        print(f"\nCOMPARACIÓN:")
        expected = [0, 35, 1]
        actual = [converted_matrix[0, 0], converted_matrix[1, 0], converted_matrix[2, 0]]
        
        for i, (exp, act) in enumerate(zip(expected, actual)):
            status = "✅" if abs(exp - act) <= 1 else "❌"
            print(f"  Período {i+1}: Esperado={exp}, Actual={act} {status}")
        
    except Exception as e:
        print(f"ERROR en conversión: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_flavors():
    """Test with multiple flavors containing the same ingredient"""
    
    print("\n🧪 PRUEBA CON MÚLTIPLES SABORES")
    print("="*60)
    
    # Test case: ingredient appears in multiple flavors
    recetas_segundo = {
        "PIZZA_A": {
            "nombre": "Pizza A", 
            "Proporción ventas": 30,  
            "ingredientes": {
                "SHARED_INGREDIENT": {"cantidad": 0.5}  # 0.5 units in flavor A
            }
        },
        "PIZZA_B": {
            "nombre": "Pizza B",
            "Proporción ventas": 20,   
            "ingredientes": {
                "SHARED_INGREDIENT": {"cantidad": 0.3}  # 0.3 units in flavor B
            }
        },
        "PIZZA_C": {
            "nombre": "Pizza C",
            "Proporción ventas": 50,  
            "ingredientes": {}  # No shared ingredient
        }
    }
    
    recetas_primero = {}
    
    # Simple test: 100 pizzas total
    test_pizza_matrix = np.array([[100]]).T  # Shape: (1, 1) 
    
    print("DATOS DE PRUEBA:")
    print("  100 pizzas totales")
    print("  Pizza A: 30% ventas, 0.5 unidades del ingrediente")
    print("  Pizza B: 20% ventas, 0.3 unidades del ingrediente") 
    print("  Pizza C: 50% ventas, 0 unidades del ingrediente")
    print("  Cálculo esperado:")
    print("    A: 100 × 0.30 × 0.5 = 15")
    print("    B: 100 × 0.20 × 0.3 = 6")
    print("    C: 100 × 0.50 × 0 = 0")
    print("    TOTAL: 15 + 6 + 0 = 21")
    
    try:
        converted_matrix = convert_pizza_liberation_matrix_to_ingredient(
            test_pizza_matrix, 
            "SHARED_INGREDIENT", 
            "TEST_PV", 
            recetas_primero, 
            recetas_segundo
        )
        
        print(f"\nRESULTADO ACTUAL: {converted_matrix[0, 0]}")
        print(f"RESULTADO ESPERADO: 21")
        status = "✅" if abs(21 - converted_matrix[0, 0]) <= 1 else "❌"
        print(f"STATUS: {status}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_manual_conversion()
    test_multiple_flavors()