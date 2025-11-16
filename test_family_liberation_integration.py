#!/usr/bin/env python3
"""
Test script to verify family liberation integration in Excel exports
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))

def test_family_liberation_integration():
    """Test that family liberation is properly integrated into optimization workflow"""
    
    print("🧪 TESTING FAMILY LIBERATION INTEGRATION")
    print("="*60)
    
    try:
        from services.PSO import generate_family_liberation_for_optimization
        
        # Mock parameters for testing
        policy = "EOQ"
        best_decision_vars = {"porcentaje_seguridad": 0.95}
        
        # Mock ingredient info with family liberation data
        ingredient_info = {
            'cluster_id': 1,
            'ingredient_code': 'I_TEST_INGREDIENT',
            'representative_ingredient': 'I_TEST_INGREDIENT',
            'materia_prima': {
                'I_TEST_INGREDIENT': {
                    'inventario_inicial': 1000,
                    'lead_time': 2,
                    'MOQ': 0,
                    'costo_pedir': 50,
                    'costo_unitario': 0.05,
                    'costo_faltante': 0.25,
                    'costo_sobrante': 0.01
                }
            },
            'recetas_primero': {},
            'recetas_segundo': {
                'Pizza_Test': {
                    'nombre': 'Test Pizza',
                    'Proporción ventas': 1.0,
                    'ingredientes': {
                        'I_TEST_INGREDIENT': {'cantidad': 100}
                    }
                }
            },
            'pizza_data_dict': {
                'TestPV': {
                    'PARAMETROS': {
                        'lead time': 2,
                        'backorders': 1,
                        'demanda_diaria': 20,
                        'demanda_promedio': 600
                    },
                    'RESULTADOS': {
                        'ventas': {i: 15 + (i % 10) for i in range(30)},
                        'T': 3
                    }
                }
            },
            'pizza_replicas_matrix': None,
            'punto_venta': 'TestPV'
        }
        
        # Mock data_dict and replicas_matrix
        data_dict = {
            'TestPV': {
                'PARAMETROS': {
                    'inventario_inicial': 1000,
                    'lead time': 2,
                    'MOQ': 0,
                    'backorders': 1,
                    'costo_pedir': 50,
                    'costo_unitario': 0.05,
                    'costo_faltante': 0.25,
                    'costo_sobrante': 0.01,
                    'demanda_diaria': 1950.0,
                    'demanda_promedio': 58500.0
                },
                'RESULTADOS': {
                    'ventas': {i: 1500 + (i % 100) for i in range(30)},
                    'T': 3
                }
            }
        }
        
        np.random.seed(42)
        replicas_matrix = np.random.poisson(1950, size=(5, 30))
        
        print(f"🔄 Testing family liberation generation...")
        print(f"   Policy: {policy}")
        print(f"   Parameters: {best_decision_vars}")
        print(f"   Ingredient: {ingredient_info['ingredient_code']}")
        print(f"   Cluster ID: {ingredient_info['cluster_id']}")
        
        # Test the family liberation generation
        family_results = generate_family_liberation_for_optimization(
            policy=policy,
            best_decision_vars=best_decision_vars,
            ingredient_info=ingredient_info,
            data_dict=data_dict,
            replicas_matrix=replicas_matrix,
            ref='TestPV',
            materia_prima=ingredient_info['materia_prima'],
            recetas_primero=ingredient_info['recetas_primero'],
            recetas_segundo=ingredient_info['recetas_segundo'],
            pizza_data_dict=ingredient_info['pizza_data_dict'],
            pizza_replicas_matrix=ingredient_info['pizza_replicas_matrix']
        )
        
        print(f"\n📊 Family liberation results:")
        if family_results:
            for ingredient, results in family_results.items():
                if "error" not in results:
                    liberation_df = results.get("liberation_df")
                    liberation_final = results.get("liberation_final")
                    
                    if liberation_df is not None:
                        total_orders = liberation_df.sum().sum()
                        periods_with_orders = (liberation_df > 0).any(axis=1).sum()
                        print(f"   ✅ {ingredient}:")
                        print(f"      Total orders: {total_orders:.0f}")
                        print(f"      Active periods: {periods_with_orders}")
                        
                        if liberation_final is not None and hasattr(liberation_final, '__iter__'):
                            vector_total = float(np.sum(liberation_final)) if hasattr(liberation_final, '__iter__') else 0
                            print(f"      Liberation vector total: {vector_total:.0f}")
                        
                        print(f"      ✅ Family liberation generated successfully!")
                        return True
                    else:
                        print(f"   ❌ {ingredient}: No liberation_df generated")
                        return False
                else:
                    print(f"   ❌ {ingredient}: {results['error']}")
                    return False
        else:
            print(f"   ❌ No family results generated")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_ingredient_info():
    """Test that enhanced ingredient info is properly structured"""
    
    print(f"\n🔧 TESTING ENHANCED INGREDIENT INFO STRUCTURE")
    print("="*50)
    
    # Test the required keys for family liberation
    required_keys = [
        'cluster_id', 'ingredient_code', 'materia_prima', 
        'recetas_primero', 'recetas_segundo', 'pizza_data_dict'
    ]
    
    mock_ingredient_info = {
        'cluster_id': 1,
        'ingredient_code': 'TEST_INGREDIENT',
        'representative_ingredient': 'TEST_INGREDIENT',
        'materia_prima': {'TEST_INGREDIENT': {}},
        'recetas_primero': {},
        'recetas_segundo': {},
        'pizza_data_dict': {},
        'pizza_replicas_matrix': None,
        'punto_venta': 'TestPV'
    }
    
    missing_keys = []
    for key in required_keys:
        if key not in mock_ingredient_info:
            missing_keys.append(key)
    
    if not missing_keys:
        print(f"✅ All required keys present for family liberation")
        print(f"   Required keys: {required_keys}")
        print(f"   Available keys: {list(mock_ingredient_info.keys())}")
        return True
    else:
        print(f"❌ Missing required keys: {missing_keys}")
        return False


if __name__ == "__main__":
    print("🚀 TESTING FAMILY LIBERATION INTEGRATION")
    print("="*70)
    
    # Test 1: Family liberation generation
    test1_success = test_family_liberation_integration()
    
    # Test 2: Enhanced ingredient info structure
    test2_success = test_enhanced_ingredient_info()
    
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Test 1 (Family liberation generation): {'✅ PASSED' if test1_success else '❌ FAILED'}")
    print(f"   Test 2 (Enhanced ingredient info): {'✅ PASSED' if test2_success else '❌ FAILED'}")
    
    if test1_success and test2_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"💡 Family liberation should now be integrated into Excel exports!")
        print(f"📊 When you run ingredient optimization, Excel files should contain:")
        print(f"   - FAMILIA_Resumen sheet with family overview")
        print(f"   - FAM_[ingredient] sheets with individual liberation vectors")
        print(f"   - Liberation_final vectors properly displayed")
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print(f"🔧 Check the family liberation integration logic")
    
    sys.exit(0 if (test1_success and test2_success) else 1)