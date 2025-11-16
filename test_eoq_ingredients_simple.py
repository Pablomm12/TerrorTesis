"""
Simplified test to check EOQ ingredients optimization with working data
"""

import numpy as np
import pandas as pd
import os
import sys

# Add services to path  
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def create_test_ingredient_family_data():
    """
    Create a test ingredient family with all proper parameters
    """
    
    # Create test family data that mimics the real structure
    test_data_MP = {
        "Familia_1": {
            "PARAMETROS": {
                # Key demand parameters
                "demanda_promedio": 150.5,  # 150.5g per day average
                "demanda_diaria": 150.5,
                
                # Cost parameters  
                "costo_pedir": 25.0,      # $25 ordering cost
                "costo_sobrante": 0.8,    # $0.8 holding cost per gram per day
                "costo_unitario": 2.0,    # $2 per gram
                "costo_faltante": 15.0,   # $15 shortage cost per gram
                
                # Inventory parameters
                "inventario_inicial": 200,  # 200g initial inventory
                "lead time": 2,             # 2 day lead time
                "backorders": 1,
                
                # Additional parameters that might be needed
                "cantidad_por_pizza": 12.5,  # 12.5g per pizza
                "representativo": "Queso Mozzarella",
                "unidad": "g",
                
                # MOQ and other constraints
                "MOQ": 500,  # Minimum order quantity 500g
            },
            "RESULTADOS": {
                # Sales data for 30 periods (grams of ingredient needed)
                "ventas": {i: 140 + np.random.normal(0, 20) for i in range(30)}  # Random around 150g
            },
            "RESTRICCIONES": {
                "Proporción demanda satisfecha": 0.95,  # 95% service level
                "Inventario a la mano (max)": 1000      # Max 1000g inventory
            }
        },
        
        "Familia_2": {
            "PARAMETROS": {
                "demanda_promedio": 85.2,    # 85.2g per day average
                "demanda_diaria": 85.2,
                "costo_pedir": 30.0,
                "costo_sobrante": 0.5,
                "costo_unitario": 3.5,
                "costo_faltante": 20.0,
                "inventario_inicial": 100,
                "lead time": 3,
                "backorders": 1,
                "cantidad_por_pizza": 8.0,   # 8g per pizza
                "representativo": "Salsa de Tomate",
                "unidad": "g",
                "MOQ": 250,
            },
            "RESULTADOS": {
                "ventas": {i: 80 + np.random.normal(0, 15) for i in range(30)}  # Random around 85g
            },
            "RESTRICCIONES": {
                "Proporción demanda satisfecha": 0.90,
                "Inventario a la mano (max)": 500
            }
        }
    }
    
    return test_data_MP


def test_eoq_optimization_with_correct_data():
    """
    Test EOQ optimization with properly structured ingredient data
    """
    from services.PSO import create_ingredient_replicas_matrix_from_data_dict, pso_optimize_single_policy, get_decision_bounds_for_policy
    
    print("🧪 TESTING EOQ WITH PROPERLY STRUCTURED INGREDIENT DATA")
    print("="*65)
    
    # Create test data
    test_data_MP = create_test_ingredient_family_data()
    
    # Test each family
    for familia_name, familia_data in test_data_MP.items():
        print(f"\n🎯 Testing Family: {familia_name}")
        print("-" * 40)
        
        params = familia_data["PARAMETROS"]
        
        # Show key parameters
        print(f"📊 Key Parameters:")
        print(f"   Demand avg: {params['demanda_promedio']:.1f}g/day")
        print(f"   Order cost: ${params['costo_pedir']}")
        print(f"   Hold cost: ${params['costo_sobrante']}/g/day")
        print(f"   Unit cost: ${params['costo_unitario']}/g")
        print(f"   Lead time: {params['lead time']} days")
        
        # Calculate theoretical EOQ
        D = params['demanda_promedio'] * 365  # Annual demand  
        K = params['costo_pedir']             # Order cost
        H = params['costo_sobrante']          # Holding cost
        
        theoretical_eoq = int(round((2 * D * K / H) ** 0.5))
        print(f"📈 Theoretical EOQ: {theoretical_eoq}g")
        
        # Generate replicas matrix
        print(f"\n🔄 Generating replicas matrix...")
        try:
            replicas_matrix = create_ingredient_replicas_matrix_from_data_dict(
                data_dict_MP=test_data_MP,
                familia_name=familia_name,
                n_replicas=8,
                u=30
            )
            
            print(f"✅ Replicas matrix created: {replicas_matrix.shape}")
            print(f"📊 Matrix stats: avg={replicas_matrix.mean():.1f}, min={replicas_matrix.min():.1f}, max={replicas_matrix.max():.1f}")
            
            # Check for zero values
            zero_count = (replicas_matrix == 0).sum()
            if zero_count > 0:
                print(f"⚠️ Found {zero_count} zero values in matrix")
            
        except Exception as e:
            print(f"❌ Error creating replicas matrix: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Test decision bounds
        print(f"\n🎯 Testing decision bounds...")
        try:
            bounds = get_decision_bounds_for_policy("EOQ", familia_name, test_data_MP)
            print(f"✅ EOQ bounds: {bounds}")
            
            # Check if bounds are reasonable
            if len(bounds) == 1 and bounds[0][0] > 0 and bounds[0][1] > bounds[0][0]:
                print(f"✅ Bounds look reasonable")
            else:
                print(f"⚠️ Bounds may be problematic: {bounds}")
                
        except Exception as e:
            print(f"❌ Error getting decision bounds: {e}")
            continue
        
        # Test PSO optimization
        print(f"\n🚀 Running PSO optimization...")
        try:
            result = pso_optimize_single_policy(
                policy="EOQ",
                data_dict=test_data_MP,
                ref=familia_name,
                replicas_matrix=replicas_matrix,
                decision_bounds=bounds,
                objective_indicator="Costo total",
                minimize=True,
                swarm_size=10,  # Small swarm for testing
                iters=5,        # Few iterations for testing
                verbose=True
            )
            
            print(f"\n✅ Optimization completed!")
            print(f"📊 Results:")
            print(f"   Best score (cost): {result.get('best_score', 'N/A')}")
            print(f"   Best decision vars: {result.get('best_decision_mapped', 'N/A')}")
            
            # Check liberation matrix
            liberation_matrix = result.get('best_liberacion_orden_matrix')
            if liberation_matrix is not None:
                total_orders = np.sum(liberation_matrix)
                max_order = np.max(liberation_matrix)
                
                print(f"📦 Liberation Matrix:")
                print(f"   Total orders: {total_orders:.1f}g")
                print(f"   Max single order: {max_order:.1f}g")
                print(f"   Matrix shape: {liberation_matrix.shape}")
                
                if total_orders == 0:
                    print(f"❌ PROBLEM: Zero total orders!")
                    print(f"📋 Sample matrix values:")
                    print(liberation_matrix[:5, :3])  # Show first 5 periods, 3 replicas
                else:
                    print(f"✅ Orders generated successfully")
                    
                    # Count periods with orders
                    periods_with_orders = (liberation_matrix > 0).any(axis=1).sum()
                    print(f"📅 Periods with orders: {periods_with_orders}/{liberation_matrix.shape[0]}")
            else:
                print(f"❌ PROBLEM: No liberation matrix returned")
                
        except Exception as e:
            print(f"❌ Error in PSO optimization: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    print("🚀 TESTING EOQ OPTIMIZATION FOR INGREDIENTS")
    print("🎯 Using synthetic test data with proper structure")
    print("="*65)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run test
    test_eoq_optimization_with_correct_data()
    
    print(f"\n" + "="*65)
    print("🎯 TEST SUMMARY")
    print("="*65)
    print("Check the output above for:")
    print("1. ✅ Replicas matrix generation (should have reasonable values)")
    print("2. ✅ Decision bounds (should be reasonable percentage range)")  
    print("3. ✅ PSO optimization (should not crash)")
    print("4. ✅ Liberation matrix (should have non-zero total orders)")
    print("")
    print("If any step fails, the issue is in that specific component.")
    print("If all steps pass, the problem is in your actual data preparation.")


if __name__ == "__main__":
    main()