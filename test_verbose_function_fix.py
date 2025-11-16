#!/usr/bin/env python3
"""
Quick test to verify that all verbose functions return 4 values correctly
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))

def test_verbose_functions_4_returns():
    """Test that all verbose functions return exactly 4 values"""
    
    print("🧪 TESTING ALL VERBOSE FUNCTIONS - 4 RETURN VALUES")
    print("="*60)
    
    try:
        from services import simulacion as sim
        
        # Create minimal test data
        test_data = {
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
                    "demanda_diaria": 100,
                    "demanda_promedio": 3000
                },
                "RESULTADOS": {
                    "ventas": {i: 80 + (i % 20) for i in range(30)},
                    "T": 3
                }
            }
        }
        
        # Create minimal replicas matrix
        np.random.seed(42)
        test_replicas = np.random.poisson(100, size=(3, 30))
        
        print(f"📊 Test setup: replicas={test_replicas.shape}, demand_daily=100")
        
        # Test each verbose function
        verbose_functions = [
            ("QR", lambda: sim.replicas_QR_verbose(test_replicas, test_data, "TestPV", Q=500, R=200)),
            ("ST", lambda: sim.replicas_ST_verbose(test_replicas, test_data, "TestPV", S=800, T=7)),
            ("SST", lambda: sim.replicas_SST_verbose(test_replicas, test_data, "TestPV", s=200, S=800, T=7)),
            ("SS", lambda: sim.replicas_SS_verbose(test_replicas, test_data, "TestPV", S=800, s=200)),
            ("EOQ", lambda: sim.replicas_EOQ_verbose(test_replicas, test_data, "TestPV", porcentaje_seguridad=0.95)),
            ("POQ", lambda: sim.replicas_POQ_verbose(test_replicas, test_data, "TestPV", porcentaje_seguridad=0.95)),
            ("LXL", lambda: sim.replicas_LXL_verbose(test_replicas, test_data, "TestPV", porcentaje_seguridad=0.95))
        ]
        
        results = {}
        all_passed = True
        
        for policy_name, func in verbose_functions:
            try:
                print(f"\n🔄 Testing {policy_name} verbose function...")
                result = func()
                
                if isinstance(result, tuple) and len(result) == 4:
                    df_promedio, liberacion_orden_df, resultados_replicas, liberacion_final = result
                    
                    print(f"   ✅ {policy_name}: Returns 4 values correctly")
                    print(f"      df_promedio: {type(df_promedio)} shape={getattr(df_promedio, 'shape', 'N/A')}")
                    print(f"      liberacion_orden_df: {type(liberacion_orden_df)} shape={getattr(liberacion_orden_df, 'shape', 'N/A')}")
                    print(f"      resultados_replicas: {type(resultados_replicas)} len={len(resultados_replicas) if hasattr(resultados_replicas, '__len__') else 'N/A'}")
                    print(f"      liberacion_final: {type(liberacion_final)} len={len(liberacion_final) if hasattr(liberacion_final, '__len__') else 'N/A'}")
                    
                    # Quick validation
                    if liberacion_final is not None and hasattr(liberacion_final, '__iter__'):
                        total_liberation = float(np.sum(liberacion_final)) if hasattr(liberacion_final, '__iter__') else 0
                        print(f"      Total liberation: {total_liberation:.0f}")
                        
                    results[policy_name] = "✅ PASS"
                else:
                    print(f"   ❌ {policy_name}: Expected 4 values, got {len(result) if hasattr(result, '__len__') else type(result)}")
                    results[policy_name] = "❌ FAIL"
                    all_passed = False
                    
            except Exception as e:
                print(f"   ❌ {policy_name}: Error - {e}")
                results[policy_name] = f"❌ ERROR: {e}"
                all_passed = False
        
        print(f"\n📋 SUMMARY:")
        for policy, status in results.items():
            print(f"   {policy:>3}: {status}")
            
        if all_passed:
            print(f"\n🎉 ALL VERBOSE FUNCTIONS WORKING CORRECTLY!")
            print(f"💡 PSO optimization should now work without unpacking errors!")
        else:
            print(f"\n⚠️ SOME FUNCTIONS FAILED - CHECK THE RESULTS ABOVE")
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 VERBOSE FUNCTION FIX VERIFICATION")
    print("="*50)
    
    success = test_verbose_functions_4_returns()
    
    if success:
        print(f"\n✅ FIX SUCCESSFUL!")
        print(f"📈 The 'too many values to unpack' error should be resolved!")
        print(f"🎯 You can now run ingredient optimization without unpacking errors!")
    else:
        print(f"\n❌ FIX NOT COMPLETE")
        print(f"🔧 Some verbose functions still need adjustment")
    
    sys.exit(0 if success else 1)