"""
Test script to verify EOQ ingredient optimization fixes
"""

import numpy as np
import pandas as pd
import os
import sys

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

def test_fixed_eoq_ingredients():
    """
    Test the fixed EOQ optimization with proper ingredient data structure
    """
    print("🧪 TESTING FIXED EOQ OPTIMIZATION FOR INGREDIENTS")
    print("="*60)
    
    # Create test family with CORRECT parameter structure for ingredients
    test_data_MP = {
        "Familia_1": {
            "PARAMETROS": {
                # KEY FIX: Use demanda_diaria (not demanda_promedio) for ingredients
                "demanda_diaria": 120.0,      # 120g per day (this is the key fix!)
                "demanda_promedio": 3600.0,   # 120 * 30 = 3600g per month
                
                # Realistic ingredient costs
                "costo_pedir": 30.0,      # $30 ordering cost
                "costo_sobrante": 0.5,    # $0.5 holding cost per gram per day  
                "costo_unitario": 2.5,    # $2.5 per gram
                "costo_faltante": 12.0,   # $12 shortage cost per gram
                
                # Inventory parameters
                "inventario_inicial": 150,  # 150g initial inventory
                "lead time": 2,             # 2 day lead time
                "backorders": 1,
                
                # Metadata  
                "cantidad_por_pizza": 15.0,  # 15g per pizza
                "representativo": "Queso Mozzarella Test",
                "unidad": "g",
                "MOQ": 500,
            },
            "RESULTADOS": {
                # Sales data for 30 periods (grams of ingredient needed)
                "ventas": {i: max(80, 100 + np.random.normal(0, 20)) for i in range(30)}
            },
            "RESTRICCIONES": {
                "Proporción demanda satisfecha": 0.90,  # 90% service level (less strict)
                "Inventario a la mano (max)": 2000      # Max 2000g inventory
            }
        }
    }
    
    # Display test parameters
    params = test_data_MP["Familia_1"]["PARAMETROS"]
    print(f"📊 Test Parameters:")
    print(f"   demanda_diaria: {params['demanda_diaria']:.1f}g/day (KEY FIX)")
    print(f"   demanda_promedio: {params['demanda_promedio']:.1f}g/month")
    print(f"   costo_pedir: ${params['costo_pedir']}")
    print(f"   costo_sobrante: ${params['costo_sobrante']}/g/day")
    print(f"   inventario_inicial: {params['inventario_inicial']}g")
    
    # Calculate theoretical EOQ
    D = params['demanda_diaria'] * 365  # Annual demand  
    K = params['costo_pedir']           # Order cost
    H = params['costo_sobrante']        # Holding cost
    
    theoretical_eoq = int(round((2 * D * K / H) ** 0.5))
    print(f"📈 Theoretical EOQ: {theoretical_eoq}g")
    print(f"   Annual demand (D): {D:.0f}g")
    print(f"   Order cost (K): ${K}")
    print(f"   Holding cost (H): ${H}/g/day")
    
    # Test the simulation directly
    print(f"\n🔬 Testing EOQ simulation directly...")
    
    try:
        from services.simulacion import replicas_EOQ_verbose
        
        # Create test replicas matrix (8 replicas, 30 periods)
        # Use ingredient demand values (not pizza values)
        np.random.seed(42)
        base_demand = params['demanda_diaria']
        test_replicas = np.random.normal(base_demand, base_demand * 0.2, (8, 30))
        test_replicas = np.maximum(test_replicas, 10)  # Ensure minimum 10g
        
        print(f"✅ Test replicas matrix: {test_replicas.shape}")
        print(f"📊 Replicas stats: avg={test_replicas.mean():.1f}g, min={test_replicas.min():.1f}g, max={test_replicas.max():.1f}g")
        
        # Run EOQ simulation with different safety percentages
        for safety_pct in [0.15, 0.25, 0.35]:
            print(f"\n🎯 Testing with safety percentage: {safety_pct*100:.0f}%")
            
            result = replicas_EOQ_verbose(
                matrizReplicas=test_replicas,
                data_dict=test_data_MP,
                punto_venta="Familia_1", 
                porcentaje_seguridad=safety_pct
            )
            
            if result and len(result) >= 2:
                df_promedio, liberacion_orden_df = result[:2]
                
                # Analyze results
                if liberacion_orden_df is not None:
                    total_orders = liberacion_orden_df.values.sum()
                    max_order = liberacion_orden_df.values.max()
                    num_periods_with_orders = (liberacion_orden_df.values > 0).any(axis=1).sum()
                    
                    print(f"   📦 Liberation orders:")
                    print(f"      Total orders: {total_orders:.0f}g")
                    print(f"      Max single order: {max_order:.0f}g")
                    print(f"      Periods with orders: {num_periods_with_orders}/30")
                    
                    if total_orders == 0:
                        print(f"   ❌ STILL GETTING ZERO ORDERS - Problem persists")
                    else:
                        print(f"   ✅ Orders generated successfully!")
                        
                        # Check if orders are reasonable
                        expected_total_demand = test_replicas.sum()
                        order_ratio = total_orders / expected_total_demand
                        print(f"      Order/Demand ratio: {order_ratio:.2f}")
                        
                        if order_ratio < 0.5 or order_ratio > 3.0:
                            print(f"   ⚠️ Order ratio seems unusual (expected ~1.0)")
                        
                        # Show first few periods for debugging
                        print(f"   📋 Sample liberation matrix (first 5 periods):")
                        print(liberacion_orden_df.head().to_string())
                        
                # Check indicators
                if df_promedio is not None:
                    print(f"   📊 Key Indicators:")
                    key_indicators = ['Inventario promedio', 'Costo total', 'Proporción demanda satisfecha']
                    for indicator in key_indicators:
                        if indicator in df_promedio.index:
                            value = df_promedio.loc[indicator].iloc[0]
                            print(f"      {indicator}: {value:.2f}")
                        
            else:
                print(f"   ❌ EOQ function returned invalid result")
                
    except Exception as e:
        print(f"❌ Error in EOQ simulation test: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*60)
    print("🎯 TEST ANALYSIS")
    print("="*60) 
    print("If you're still seeing 0 orders, check:")
    print("1. ✅ Parameter lookup fixed (demanda_diaria vs demanda_promedio)")
    print("2. ⚠️ EOQ batch size calculation")
    print("3. ⚠️ Matrix reajuste function zeroing out orders")
    print("4. ⚠️ Inventory initialization problems")
    print("5. ⚠️ Safety stock calculation")
    print("\nRun this test and check the debug output for more details.")
    print("="*60)


if __name__ == "__main__":
    test_fixed_eoq_ingredients()