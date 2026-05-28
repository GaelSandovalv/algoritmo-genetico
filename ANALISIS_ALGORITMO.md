# Análisis del algoritmo

Descripción del problema, análisis de complejidad teórica (Big O) y
decisiones de diseño del algoritmo genético del proyecto.


## 1. Problema

**Alineamiento múltiple de secuencias de ADN.** Dadas `S = 4` secuencias
cortas sobre el alfabeto `{A, C, G, T}`, encontrar un alineamiento (matriz
de filas `S × L` con gaps `-`) que maximice una métrica de similitud
columna por columna. Restricción dura: al quitar los gaps de cada fila se
debe recuperar exactamente la secuencia original (validado en cada
generación por `validar_integridad`).

**Función de fitness (suma de pares):** por cada par de filas `(i, j)` y
cada columna `c`:

- `+1` si ambas letras coinciden y no son gap.
- `-1` si las letras difieren y no son gap.
- `-2` si exactamente una de las dos es gap.
- `0` si ambas son gap.

El fitness total es la suma sobre todos los pares y columnas. Es negativo
cuando el alineamiento es malo, y crece (puede ser positivo) cuando hay
muchas coincidencias.

## 2. Complejidad teórica (Big O)

Notación:

- **G** = número de generaciones
- **P** = tamaño de la población
- **L** = longitud del alineamiento (con gaps)
- **S** = número de secuencias

| Operador | Complejidad temporal | Notas |
|---|---|---|
| `crear_individuo` | O(S · L) | Inserta gaps en cada fila |
| `crear_poblacion` | O(P · S · L) | P individuos |
| `calcular_fitness` | O(S² · L) | Todos los pares de filas, cada columna |
| `validar_integridad` | O(S · L) | Compara fila sin gaps con original |
| `seleccion_ruleta` | O(P) | Recorre toda la población |
| `seleccion_torneo` | O(k) con k = tam_torneo | k pequeño y constante |
| `cruza_un_punto` | O(S · L) | Concatena y rellena |
| `mutacion_mover_gap` | O(S · L) | Busca gap, lo mueve |
| `mutacion_bloques` | O(S · L) | Insertar / mover / eliminar bloque |
| `elitismo` (sort + slice) | O(P · log P) | Ordenar la población |

**Por generación — AG Base:**
`O(P · S² · L)`. Domina la evaluación de `calcular_fitness` aplicada a las
`P` soluciones.

**Por generación — AG Mejorado:**
`O(P · S² · L + P · log P)`. Igual que el base más el ordenamiento que el
elitismo requiere. Cuando `S² · L >> log P` el término del sort es
despreciable.

**Total (ambos):**
`O(G · P · S² · L)`.

**Complejidad espacial:**
`O(P · S · L)` — la población completa más una nueva generación en
construcción.

