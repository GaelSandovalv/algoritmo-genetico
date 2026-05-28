"""Pruebas del algoritmo genetico para alineamiento de secuencias."""
import random

from algoritmo_genetico import (
    generar_secuencias, validar_integridad, calcular_fitness,
    igualar_longitud, crear_individuo, crear_poblacion,
    seleccion_ruleta, cruza_un_punto, mutacion_mover_gap,
    ag_base,
    seleccion_torneo,
    insertar_bloque_gaps, mover_bloque_gaps, eliminar_columnas_gaps, mutacion_bloques,
    ag_mejorado,
    graficar, comparar,
    diversidad_poblacion,
    ejecutar_benchmark,
    graficar_convergencia_promedio,
    graficar_tiempo,
    graficar_diversidad,
    ejecutar_sensibilidad,
    graficar_sensibilidad,
    parsear_args,
)


def test_generar_secuencias_es_determinista():
    assert generar_secuencias(42) == generar_secuencias(42)


def test_generar_secuencias_cantidad_y_alfabeto():
    secuencias = generar_secuencias(42)
    assert len(secuencias) == 4
    for s in secuencias:
        assert len(s) > 0
        assert all(base in "ACGT" for base in s)


def test_validar_integridad_correcta():
    originales = ["ACGT", "ACG"]
    individuo = ["AC-GT", "AC--G"]
    assert validar_integridad(individuo, originales) is True


def test_validar_integridad_detecta_letra_alterada():
    originales = ["ACGT", "ACG"]
    individuo = ["AC-GA", "AC--G"]   # la ultima letra cambio de T a A
    assert validar_integridad(individuo, originales) is False


def test_fitness_todas_coinciden():
    # 2 secuencias iguales, 2 columnas, 1 par -> +1 +1 = 2
    assert calcular_fitness(["AC", "AC"]) == 2


def test_fitness_con_desajuste_y_gap():
    # par unico: col0 'A'/'A' = +1 ; col1 '-'/'C' = -2  -> total -1
    assert calcular_fitness(["A-", "AC"]) == -1


def test_igualar_longitud_rellena_con_gaps_finales():
    resultado = igualar_longitud(["AC", "ACGT", "A"])
    assert resultado == ["AC--", "ACGT", "A---"]


def test_crear_individuo_respeta_integridad_y_longitud():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    individuo = crear_individuo(originales, longitud, rng)
    assert all(len(fila) == longitud for fila in individuo)
    assert validar_integridad(individuo, originales) is True


def test_crear_poblacion_tamano():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    poblacion = crear_poblacion(originales, 10, longitud, rng)
    assert len(poblacion) == 10


def test_seleccion_ruleta_devuelve_individuo_de_la_poblacion():
    rng = random.Random(0)
    poblacion = [["AC"], ["GT"], ["TT"]]
    fitnesses = [1, 5, 2]
    elegido = seleccion_ruleta(poblacion, fitnesses, rng)
    assert elegido in poblacion


def test_cruza_un_punto_preserva_integridad():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    p1 = crear_individuo(originales, longitud, rng)
    p2 = crear_individuo(originales, longitud, rng)
    hijo = cruza_un_punto(p1, p2, rng)
    assert validar_integridad(hijo, originales) is True
    assert len(set(len(f) for f in hijo)) == 1   # todas las filas igual largo


def test_mutacion_mover_gap_preserva_integridad():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    individuo = crear_individuo(originales, longitud, rng)
    mutado = mutacion_mover_gap(individuo, rng)
    assert validar_integridad(mutado, originales) is True


def test_ag_base_mantiene_integridad_y_devuelve_historial():
    originales = generar_secuencias(42)
    mejor, historial = ag_base(originales, tam_poblacion=12,
                               generaciones=8, semilla=1)
    assert len(historial) == 8
    assert validar_integridad(igualar_longitud(mejor), originales) is True


def test_seleccion_torneo_con_torneo_completo_elige_el_mejor():
    rng = random.Random(0)
    poblacion = [["A"], ["B"], ["C"], ["D"]]
    fitnesses = [3, 9, 1, 5]
    # tam_torneo >= tamano de la poblacion -> participan todos -> gana el mejor
    elegido = seleccion_torneo(poblacion, fitnesses, rng, tam_torneo=4)
    assert elegido == ["B"]


def test_insertar_bloque_gaps_preserva_integridad():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    individuo = crear_individuo(originales, longitud, rng)
    mutado = insertar_bloque_gaps(individuo, rng)
    assert validar_integridad(mutado, originales) is True
    assert len(set(len(f) for f in mutado)) == 1


def test_mover_bloque_gaps_preserva_integridad():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    individuo = crear_individuo(originales, longitud, rng)
    mutado = mover_bloque_gaps(individuo, rng)
    assert validar_integridad(mutado, originales) is True


def test_eliminar_columnas_gaps_quita_columnas_de_solo_gaps():
    individuo = ["A-C", "G-T"]   # la columna del medio es toda gaps
    assert eliminar_columnas_gaps(individuo) == ["AC", "GT"]


def test_mutacion_bloques_preserva_integridad():
    rng = random.Random(0)
    originales = generar_secuencias(42)
    longitud = max(len(s) for s in originales) + 6
    for _ in range(30):   # cubre las tres operaciones posibles
        individuo = crear_individuo(originales, longitud, rng)
        mutado = mutacion_bloques(individuo, rng, longitud_maxima=longitud * 2)
        assert validar_integridad(mutado, originales) is True


def test_ag_mejorado_mantiene_integridad():
    originales = generar_secuencias(42)
    mejor, historial = ag_mejorado(originales, tam_poblacion=16,
                                   generaciones=10, semilla=1)
    assert len(historial) == 10
    assert validar_integridad(igualar_longitud(mejor), originales) is True


def test_ag_mejorado_historial_es_monotono_por_elitismo():
    # Con elitismo el mejor individuo nunca se pierde, asi que el mejor
    # fitness por generacion nunca debe bajar.
    originales = generar_secuencias(42)
    _, historial = ag_mejorado(originales, tam_poblacion=16,
                               generaciones=15, semilla=1)
    for i in range(len(historial) - 1):
        assert historial[i] <= historial[i + 1]


def test_graficar_crea_el_archivo(tmp_path):
    archivo = tmp_path / "grafica_prueba.png"
    graficar([1, 2, 3], [2, 3, 4], str(archivo))
    assert archivo.exists()


def test_comparar_devuelve_dos_historiales(tmp_path):
    archivo = str(tmp_path / "comparacion_prueba.png")
    hist_base, hist_mej = comparar(generaciones=8, tam_poblacion=12,
                                   archivo=archivo, mostrar=False)
    assert len(hist_base) == 8
    assert len(hist_mej) == 8


def test_diversidad_poblacion_todos_iguales():
    poblacion = [["AC", "GT"], ["AC", "GT"], ["AC", "GT"]]
    assert diversidad_poblacion(poblacion) == 1


def test_diversidad_poblacion_todos_distintos():
    poblacion = [["AC", "GT"], ["AG", "GT"], ["AC", "CT"]]
    assert diversidad_poblacion(poblacion) == 3


def test_diversidad_poblacion_vacia():
    assert diversidad_poblacion([]) == 0


def test_ag_base_invoca_callback_por_generacion():
    originales = generar_secuencias(42)
    llamadas = []
    def cb(gen, poblacion, fitnesses, tiempo_ms):
        llamadas.append((gen, len(poblacion), len(fitnesses), tiempo_ms))
    ag_base(originales, tam_poblacion=8, generaciones=5,
            semilla=1, callback=cb)
    assert len(llamadas) == 5
    for i, (gen, n_pob, n_fit, ms) in enumerate(llamadas):
        assert gen == i
        assert n_pob == 8
        assert n_fit == 8
        assert ms >= 0


def test_ag_mejorado_invoca_callback_por_generacion():
    originales = generar_secuencias(42)
    llamadas = []
    def cb(gen, poblacion, fitnesses, tiempo_ms):
        llamadas.append((gen, len(poblacion), len(fitnesses), tiempo_ms))
    ag_mejorado(originales, tam_poblacion=8, generaciones=5,
                semilla=1, callback=cb)
    assert len(llamadas) == 5
    for i, (gen, n_pob, n_fit, ms) in enumerate(llamadas):
        assert gen == i
        assert n_pob == 8
        assert n_fit == 8
        assert ms >= 0


def test_ejecutar_benchmark_devuelve_arrays_correctos():
    originales = generar_secuencias(42)
    resultado = ejecutar_benchmark(originales, n_corridas=2,
                                   tam_poblacion=8, generaciones=5)
    # resultado es un dict con claves "base" y "mejorado"
    assert set(resultado.keys()) == {"base", "mejorado"}
    for clave in ("base", "mejorado"):
        datos = resultado[clave]
        # cada uno tiene fitness (corridas x generaciones),
        # tiempo (corridas x generaciones) y diversidad (corridas x generaciones)
        assert len(datos["fitness"]) == 2
        assert len(datos["fitness"][0]) == 5
        assert len(datos["tiempo"]) == 2
        assert len(datos["tiempo"][0]) == 5
        assert len(datos["diversidad"]) == 2
        assert len(datos["diversidad"][0]) == 5


def test_graficar_convergencia_promedio_crea_archivo(tmp_path):
    archivo = tmp_path / "conv.png"
    datos = {
        "base": {"fitness": [[1, 2, 3], [1, 2, 4]]},
        "mejorado": {"fitness": [[2, 4, 6], [2, 5, 7]]},
    }
    graficar_convergencia_promedio(datos, str(archivo))
    assert archivo.exists()


def test_graficar_tiempo_crea_archivo(tmp_path):
    archivo = tmp_path / "t.png"
    datos = {
        "base": {"tiempo": [[1.0, 1.1, 1.2], [1.1, 1.0, 1.3]]},
        "mejorado": {"tiempo": [[1.5, 1.6, 1.7], [1.6, 1.5, 1.8]]},
    }
    graficar_tiempo(datos, str(archivo))
    assert archivo.exists()


def test_graficar_diversidad_crea_archivo(tmp_path):
    archivo = tmp_path / "d.png"
    datos = {
        "base": {"diversidad": [[8, 7, 5], [8, 6, 4]]},
        "mejorado": {"diversidad": [[8, 8, 7], [8, 7, 6]]},
    }
    graficar_diversidad(datos, str(archivo))
    assert archivo.exists()


def test_ejecutar_sensibilidad_un_parametro_dos_valores():
    originales = generar_secuencias(42)
    config = {"tam_torneo": [2, 5]}
    resultado = ejecutar_sensibilidad(originales, config,
                                      n_corridas=2,
                                      tam_poblacion=8,
                                      generaciones=5)
    assert set(resultado.keys()) == {"tam_torneo"}
    assert set(resultado["tam_torneo"].keys()) == {2, 5}
    for valor in (2, 5):
        celda = resultado["tam_torneo"][valor]
        assert "fitness_promedio" in celda
        assert "fitness_std" in celda
        assert isinstance(celda["fitness_promedio"], float)
        assert isinstance(celda["fitness_std"], float)


def test_graficar_sensibilidad_crea_archivo(tmp_path):
    archivo = tmp_path / "sens.png"
    resultado = {
        "tam_poblacion": {
            10: {"fitness_promedio": -10.0, "fitness_std": 2.0},
            20: {"fitness_promedio": -5.0, "fitness_std": 1.5},
        },
        "generaciones": {
            25: {"fitness_promedio": -8.0, "fitness_std": 2.5},
            50: {"fitness_promedio": -3.0, "fitness_std": 1.0},
        },
        "prob_mutacion": {
            0.1: {"fitness_promedio": -6.0, "fitness_std": 1.2},
            0.3: {"fitness_promedio": -2.0, "fitness_std": 0.8},
        },
        "tam_torneo": {
            2: {"fitness_promedio": -4.0, "fitness_std": 1.0},
            5: {"fitness_promedio": -1.0, "fitness_std": 0.7},
        },
    }
    graficar_sensibilidad(resultado, str(archivo))
    assert archivo.exists()


def test_parsear_args_default_es_demo():
    args = parsear_args([])
    assert args.benchmark is False
    assert args.sensibilidad is False


def test_parsear_args_benchmark():
    args = parsear_args(["--benchmark"])
    assert args.benchmark is True
    assert args.sensibilidad is False


def test_parsear_args_sensibilidad():
    args = parsear_args(["--sensibilidad"])
    assert args.sensibilidad is True
    assert args.benchmark is False


def test_parsear_args_flags_son_mutuamente_excluyentes(capsys):
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        parsear_args(["--benchmark", "--sensibilidad"])
