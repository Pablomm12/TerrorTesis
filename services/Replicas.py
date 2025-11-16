import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# =========================
# FUNCIÓN DE MÉTRICAS
# =========================
def calcular_metricas(y_true, y_pred):
    errores = y_true - y_pred
    mad = np.mean(np.abs(errores))
    mse = np.mean(errores**2)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return mad, mse, mape, errores

def graficar_resultados(test, pronosticos, titulo="Forecast"):
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test, label="Real")
    plt.plot(test.index, pronosticos, '--', label="Pronóstico")
    plt.title(titulo)
    plt.legend()
    plt.grid()
    plt.show()

# =========================
# FUNCIÓN PARA RÉPLICAS
# =========================
def generar_replicas(modelo, errores, pasos, n_replicas=30):
    """
    Genera réplicas de pronósticos usando los errores históricos.
    """
    todos_pronosticos = np.zeros((n_replicas, pasos))

    for r in range(n_replicas):
        pronostico_base = modelo.forecast(steps=pasos)
        ruido = np.random.choice(errores, size=pasos, replace=True)
        pronostico_con_ruido = pronostico_base + ruido
        pronostico_con_ruido = np.round(pronostico_con_ruido).astype(int)
        pronostico_con_ruido[pronostico_con_ruido <= 0] = 1
        todos_pronosticos[r] = pronostico_con_ruido

    promedios = np.mean(todos_pronosticos, axis=0)
    desviaciones = np.std(todos_pronosticos, axis=0)

    print("\n═ Pronósticos Promedio con Réplicas ═")
    print(f"Réplicas generadas: {n_replicas}")
    print("-" * 60)
    print("{:<8} {:<20} {:<20}".format("Periodo", "Promedio", "Desviación"))

    for i in range(pasos):
        print("{:<8} {:<20.2f} {:<20.2f}".format(
            i + 1, promedios[i], desviaciones[i]
        ))
    print("-" * 60)

    return todos_pronosticos, promedios, desviaciones

# =========================
# FUNCIÓN BONDAD DE AJUSTE
# =========================
def evaluar_distribuciones(errores):
    distribuciones = {
        "Normal": stats.norm,
        "Uniforme": stats.uniform,
        "Poisson": stats.poisson
    }
    resultados = {}

    for nombre, dist in distribuciones.items():
        try:
            params = dist.fit(errores) if nombre != "Poisson" else (np.mean(errores),)
            loglik = np.sum(dist.logpmf(errores, *params)) if nombre == "Poisson" else np.sum(dist.logpdf(errores, *params))
            k = len(params)
            aic = 2 * k - 2 * loglik
            resultados[nombre] = {"AIC": aic, "params": params}
        except Exception as e:
            resultados[nombre] = {"AIC": np.inf, "params": None}
    
    mejor = min(resultados, key=lambda k: resultados[k]["AIC"])
    
    print("\n📊 Evaluación de distribuciones para los errores:")
    for d, r in resultados.items():
        print(f"{d}: AIC = {r['AIC']:.2f} | Params = {r['params']}")
    print(f"\n✅ Mejor ajuste: {mejor}\n")
    
    return mejor, resultados[mejor]

# =========================
# OPTIMIZACIÓN PSO
# =========================
class PSO_SARIMA:
    def __init__(self, train, test, s, n_particles=12, n_iter=10, random_state=42):
        self.train = train
        self.test = test
        self.s = s
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.rng = np.random.default_rng(random_state)
        self.bounds = {
            "p": (0, 3), "d": (0, 2), "q": (0, 3),
            "P": (0, 2), "D": (0, 1), "Q": (0, 2)
        }

    def evaluar(self, params):
        p, d, q, P, D, Q = map(int, params)
        try:
            model = SARIMAX(self.train,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, self.s),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
            forecast = model.forecast(steps=len(self.test))
            return mean_absolute_percentage_error(self.test, forecast) * 100
        except:
            return np.inf

    def optimizar(self):
        keys = list(self.bounds.keys())
        lows = np.array([self.bounds[k][0] for k in keys], dtype=float)
        highs = np.array([self.bounds[k][1] for k in keys], dtype=float)

        pos = np.vstack([
            self.rng.integers(lows, highs + 1).astype(float)
            for _ in range(self.n_particles)
        ])
        vel = self.rng.uniform(-1, 1, pos.shape)

        p_best = pos.copy()
        p_best_scores = np.array([self.evaluar(x) for x in pos])
        g_idx = np.argmin(p_best_scores)
        g_best = p_best[g_idx].copy()
        g_best_score = p_best_scores[g_idx]

        for _ in range(self.n_iter):
            for i in range(self.n_particles):
                inertia = 0.5
                c1, c2 = 1.5, 1.5
                r1, r2 = self.rng.random(2)
                vel[i] = (inertia * vel[i] +
                          c1 * r1 * (p_best[i] - pos[i]) +
                          c2 * r2 * (g_best - pos[i]))
                pos[i] += vel[i]
                pos[i] = np.clip(pos[i], lows, highs)
                pos[i] = np.round(pos[i])
                score = self.evaluar(pos[i])
                if score < p_best_scores[i]:
                    p_best[i] = pos[i].copy()
                    p_best_scores[i] = score
                    if score < g_best_score:
                        g_best = pos[i].copy()
                        g_best_score = score

        return tuple(map(int, g_best)), g_best_score

# =========================
# PIPELINE PRINCIPAL
# =========================
file_path = "Datos Completos Sr Pizza.xlsx"
df = pd.read_excel(file_path, sheet_name="Totales")
df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True)
df.set_index('Fecha', inplace=True)

