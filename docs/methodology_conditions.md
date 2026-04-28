# Metodología: 5 Condiciones Experimentales CPP
## CollabMath — Diseño e Implementación con Pipeline Multi-Agente

**Versión:** 2.0 (revisión completa)
**Referencia teórica:** docs/framework_PIE_CPS.md v3.0
**Referencia arquitectónica:** docs/c2_architecture_analysis.md

---

## Panel de Agentes Metodológicos

| Agente | Especialidad | Sesgo Principal |
|--------|-------------|-----------------|
| **Pedagogo (PED)** | Didáctica matemática, Brousseau, Vygotsky | Autenticidad pedagógica, ZPD-G real |
| **Teórico CPS (TCPS)** | PISA, ATC21S, Szewkis | Rigor en medición, validez de constructo |
| **Ingeniero IA (IA)** | LLMs, pipelines, RL, embeddings | Implementabilidad, costo, robustez técnica |
| **Matemático Computacional (MC)** | Reticulado booleano, FCA, diseño de mecanismos | Fundamentos formales, garantías teóricas |
| **Metodólogo Experimental (ME)** | Diseño experimental, validez interna, estadística | Control de confounders, replicabilidad |

---

## Parte I: Visión General Revisada

### Las 5 Condiciones — Tabla Revisada con Fundamentos CIDI

La versión 1.0 de la metodología especificaba condiciones con implementaciones de prompt engineering. La versión 2.0 incorpora el framework CIDI (Constrained Inverse Design with Iterative Validation) derivado del análisis arquitectónico y los fundamentos del BCE theorem y la tabla celda→asimetría.

| Condición | Intervención split | Intervención simulación | CDI esperado | Fundamento teórico |
|-----------|-------------------|------------------------|-------------|-------------------|
| **C1** Baseline | Prompt estándar (7 patrones) | Simulación estándar | ~0.08 | Control — corpus existente n=150 |
| **C2** CIDI-Directed | Pipeline CIDI 6 módulos (tabla asimetría + BCE) | Simulación estándar | ~0.58-0.83 | Diseño inverso algorítmico |
| **C3** Constitutional | Pipeline Constitutional iterativo (crítico+revisor) | Simulación estándar | ~0.50-0.75 | Constitutional AI + Szewkis 24-check |
| **C4** Monitor Dinámico | Prompt estándar | Monitor Szewkis post-fase PISA | ~0.25-0.50 | Facilitación dinámica, máx 2 intervenciones |
| **C5** Integrado | Pipeline C3 (constitutional) | Monitor C4 activo | ~0.75-1.00 | Situación a-didáctica (Brousseau) + facilitación |

**Hipótesis de ordenamiento revisada (H5):**
```
CDI:        C2_CIDI ≥ C5 > C3 > C4 > C1
PISA_global: C5 > C2_CIDI ≈ C3 > C4 > C1
targeting_error: C2_CIDI < C3 ≈ C5 < C4 < C1
```

**Por qué C2_CIDI puede superar a C5 en CDI pero no en PISA_global:**
- C2_CIDI maximiza la coincidencia entre el CPP objetivo y el CPP alcanzado (bajo targeting_error) — es la condición más "dirigida"
- C5 maximiza la calidad sostenida de la colaboración durante la simulación (SQS alto + monitor activo) — es la condición más "profunda" en proceso
- CDI y PISA_global miden aspectos distintos: CDI = ¿cuántas celdas se activaron? PISA_global = ¿con qué calidad se manifestó cada proceso CPS?

**MC:** El punto clave es que C2_CIDI diseña la *estructura de información* para el target exacto, mientras que C5 opera sobre el *proceso de simulación*. Ambos son necesarios para un sistema completo.

---

## Parte II: Corpus Existente y Plan de Backup

### Estado del corpus (baseline C1)

| Artefacto | Archivos | Descripción |
|-----------|---------|-------------|
| `outputs/splits/` | ~597 archivos JSON | Splits generados (n=1,2,3,4 agentes × ~150 problemas) |
| `outputs/conversations/` | 722 archivos JSON | Conversaciones simuladas (5 condiciones × ~144 problemas) |
| `outputs/scores/` | 722 archivos JSON | Scores PISA + ATC21S por conversación |
| `outputs/results/` | 5 CSV consolidados | Análisis estadístico del experimento C1 |
| `outputs/viewer.html` | 1 archivo HTML | Visor estático para evaluación humana |

**Decisión del Metodólogo:** El corpus C1 (n≈144 problemas completos con todas las condiciones) es el baseline del experimento. **No se re-genera nunca** — cualquier varianza nueva contaminaría el control.

### Plan de Backup

**Nivel 1 — Git (ya implementado):** el `.gitignore` excluye `outputs/` para no llenar el repositorio de datos. El código está versionado en GitHub.

**Nivel 2 — Backup completo a Sapelo scratch (recomendado):**
```bash
# Ejecutar desde login node de Sapelo2
rsync -av --progress /local/outputs/ /scratch/mir85108/collabmath/outputs_backup/
```

**Nivel 3 — Archivo comprimido fechado (antes de cada experimento nuevo):**
```bash
# En el repo local, antes de correr el piloto C2-C5
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/
# Guardar en ~/Documents/backups/ o en Sapelo /scratch
```

**Nivel 4 — GitHub LFS para datos grandes (opcional):**
Si se necesita compartir el corpus con colaboradores, activar `git lfs` para los JSON de `outputs/`.

---

## Parte III: Condición 1 — Baseline

### Descripción
Sin modificaciones al pipeline. Usa los outputs ya generados del corpus n=150.

### Implementación (no requiere código nuevo)
```python
from research.experiments.cpp_comparison import run_c1, _load_existing_splits
# Carga los splits y conversaciones existentes para los 4 problemas piloto
result = run_c1(problem_id="math_00042_n2", problem="...")
# result["source"] == "existing" — datos del corpus
```

### Selección de problemas piloto desde C1
El experimento piloto usa 4 problemas del corpus existente. Criterio de selección:

| Slot | Criterio | Método |
|------|---------|--------|
| P1 | algebra, L2, split más trivial | argmin(total_turns de jigsaw_2 existente) |
| P2 | geometry, L3, mayor potencial de mejora | argmin(PISA_global de jigsaw_2 existente) |
| P3 | number_theory, L4 | argmin(PISA_global de jigsaw_2 existente) |
| P4 | counting_and_probability, L5 | argmin(PISA_global de jigsaw_2 existente) |

Script de selección automática: `python3 -m research.experiments.cpp_comparison --select-only`

---

## Parte IV: Condición 2 — CIDI-Directed (Pipeline Completo)

### Fundamento revisado
La versión 1.0 describía C2 como "prompt CPP+Szewkis". La versión 2.0 implementa el pipeline completo CIDI de 6 módulos. La diferencia fundamental: **el diseño del split es algorítmico, no retórico**. El LLM no recibe instrucciones abstractas ("activa las celdas A1-C2") sino una especificación formal de la asimetría de información requerida, derivada de la tabla celda→asimetría.

### Pipeline CIDI — 6 Módulos

#### Módulo 1: Análisis Semántico del Problema

**Propósito:** Extraer la "anatomía" del problema — entidades, relaciones, sub-problemas, tipo de razonamiento. Esto alimenta todos los módulos posteriores.

**Modelo:** Llama 3.3 70B (vía Groq API, gratuito/barato) o Qwen2.5-72B (vía vLLM local en Sapelo)

```python
# research/splitting/cidi/module1_semantic.py

SEMANTIC_SYSTEM = """
Analiza el problema matemático y extrae su anatomía estructural.
Responde en JSON:
{
  "entities": [{"name": "...", "type": "variable|constant|function|set|...", "description": "..."}],
  "relations": [{"between": ["e1","e2"], "type": "equal|depends|bounds|...", "description": "..."}],
  "sub_problems": [{"id": "sp1", "description": "...", "requires": ["e1","r1"]}],
  "reasoning_type": ["algebraic", "geometric", "probabilistic", "combinatorial", "analytical"],
  "information_bottlenecks": [
    "descripción de cada punto donde se NECESITA información no derivable localmente"
  ],
  "natural_split_axes": [
    "descripción de cómo podría partirse la información de forma natural"
  ]
}
"""
```

**Output esperado para un problema de geometría:**
```json
{
  "entities": [
    {"name": "radio_r", "type": "variable", "description": "radio del semicírculo"},
    {"name": "altura_h", "type": "variable", "description": "altura del rectángulo"},
    {"name": "perimetro_total", "type": "constant", "description": "10m"}
  ],
  "information_bottlenecks": [
    "el diámetro del semicírculo = la anchura del rectángulo (conexión oculta)",
    "la contribución del perímetro requiere integrar ambos componentes"
  ],
  "natural_split_axes": [
    "componente rectangular vs componente semicircular"
  ]
}
```

#### Módulo 2: Verificación de Alcanzabilidad y Clausura de Prerrequisitos

**Propósito:** Verificar que el target CPP t es un upset del DAG de prerrequisitos y que el problema tiene la estructura semántica necesaria para activar cada celda.

```python
# research/splitting/cidi/module2_feasibility.py

PREREQ_DAG = {
    "A1": [],
    "A2": ["A1"],
    "A3": ["A1", "A2"],
    "B1": ["A1"],
    "B2": ["A2", "B1"],
    "B3": ["A3", "B2"],
    "C1": ["B1", "B2"],
    "C2": ["C1", "B2"],
    "C3": ["B3"],
    "D1": ["C1", "A1"],
    "D2": ["C2", "D1"],
    "D3": ["D2", "B3"],
}

def close_under_prerequisites(target_cells: list[str]) -> list[str]:
    """Completa target hacia el upset mínimo que lo contiene."""
    closed = set(target_cells)
    changed = True
    while changed:
        changed = False
        for cell in list(closed):
            for prereq in PREREQ_DAG[cell]:
                if prereq not in closed:
                    closed.add(prereq)
                    changed = True
    return sorted(closed)

def check_structural_feasibility(cell: str, anatomy: dict) -> tuple[bool, str]:
    """
    Verifica si el problema tiene la estructura semántica necesaria para cell.
    Returns (is_feasible, reason).
    """
    bottlenecks = anatomy.get("information_bottlenecks", [])
    # Cada celda tiene requisitos estructurales mínimos del problema
    STRUCTURAL_REQUIREMENTS = {
        "C2": lambda a: len(a.get("sub_problems", [])) >= 2,
        "D1": lambda a: any("ambig" in b.lower() or "conexión" in b.lower()
                            for b in bottlenecks),
        "D3": lambda a: len(a.get("sub_problems", [])) >= 3,
        # etc.
    }
    req = STRUCTURAL_REQUIREMENTS.get(cell)
    if req is None:
        return True, "no structural constraint"
    feasible = req(anatomy)
    return feasible, ("ok" if feasible else f"Cell {cell} requires richer problem structure")
```

#### Módulo 3: Derivación de Restricciones de Partición

**Propósito:** Para cada celda del target CPP, derivar la restricción de información específica que debe satisfacer el split. Esto es el núcleo del diseño inverso.

```python
# research/splitting/cidi/module3_constraints.py

# Tabla celda → tipo de asimetría (del framework v3.0)
CELL_TO_ASYMMETRY = {
    "A1": {
        "description": "Los agentes no conocen las capacidades del otro a priori",
        "design_rule": "Cada paquete debe contener información que revela las capacidades de ese agente "
                       "de forma que el otro agente no pueda inferirlas sin preguntar",
        "test": "¿Puede el agente inferir qué sabe el otro sin ningún intercambio? Si SÍ → falla A1"
    },
    "A2": {
        "description": "Scripts de colaboración distintos o ausentes",
        "design_rule": "No especificar protocolo de colaboración en ningún paquete; "
                       "los agentes deben negociar cómo trabajar juntos",
        "test": "¿Está pre-especificado quién lidera? Si SÍ → falla A2"
    },
    "A3": {
        "description": "Roles implicados pero no nombrados",
        "design_rule": "Los paquetes deben implicar roles complementarios sin asignarlos explícitamente",
        "test": "¿Los roles están nombrados en los paquetes? Si SÍ → falla A3"
    },
    "B1": {
        "description": "Representaciones distintas del mismo objeto",
        "design_rule": "Un paquete da representación algebraica, otro da representación geométrica "
                       "(o equivalente). Ninguno puede mapear la representación del otro sin diálogo",
        "test": "¿Ambos agentes tienen la misma representación del objeto central? Si SÍ → falla B1"
    },
    "B2": {
        "description": "Lista de sub-tareas distribuida",
        "design_rule": "Cada paquete identifica algunas sub-tareas necesarias pero no todas. "
                       "La lista completa requiere combinar ambos paquetes",
        "test": "¿Puede un agente listar todos los pasos solo? Si SÍ → falla B2"
    },
    "B3": {
        "description": "Ambigüedad genuina sobre distribución del trabajo",
        "design_rule": "El split no debe especificar quién ejecuta qué durante el cálculo; "
                       "deben negociarlo durante B",
        "test": "¿Es obvio quién calcula qué en la ejecución? Si SÍ → falla B3"
    },
    "C1": {
        "description": "Output del paso i de A es input de validación de B",
        "design_rule": "Diseñar cadena de dependencia: antes de ejecutar el paso siguiente, "
                       "el agente debe comunicar su resultado al otro para validación",
        "test": "¿Puede A ejecutar todos sus pasos sin comunicación intermedia? Si SÍ → falla C1"
    },
    "C2": {
        "description": "Output(A, paso_k) = Input(B, paso_{k+1})",
        "design_rule": "Crear cadena matemática explícita: el resultado de A es el input numérico/algebraico "
                       "que B necesita para continuar. Debe ser inquebrante",
        "test": "¿Puede B calcular su parte sin el resultado numérico de A? Si SÍ → falla C2"
    },
    "C3": {
        "description": "Mecanismo de verificación de participación equitativa",
        "design_rule": "El shared_context debe mencionar que ambos contribuyen en cada fase. "
                       "Puede incluir turnos de verificación explícitos",
        "test": "¿Un agente puede dominar toda la ejecución? Si SÍ → falla C3"
    },
    "D1": {
        "description": "Ambigüedad semántica deliberada que emerge al ejecutar",
        "design_rule": "Introducir un término, símbolo, o condición que cada paquete interpreta "
                       "ligeramente diferente. La divergencia se descubre durante C",
        "test": "¿Las interpretaciones de ambos paquetes son idénticas sobre todos los objetos? Si SÍ → falla D1"
    },
    "D2": {
        "description": "Criterios de evaluación distribuidos",
        "design_rule": "Un paquete tiene el criterio de corrección matemática; "
                       "el otro tiene el criterio de completitud/suficiencia de la solución",
        "test": "¿Puede un agente decidir solo si la solución es correcta Y suficiente? Si SÍ → falla D2"
    },
    "D3": {
        "description": "Sorpresa estructural mid-problem que requiere re-distribución",
        "design_rule": "El problema o el shared_context incluye una cláusula que se activa "
                       "durante la ejecución y requiere renegociar los roles",
        "test": "¿El plan inicial de los agentes puede completarse sin adaptación? Si SÍ → falla D3"
    },
}


def derive_partition_constraints(
    target_cells: list[str],
    anatomy: dict,
) -> dict:
    """
    Dado el target CPP y la anatomía del problema, genera las restricciones
    específicas sobre (I_A, I_B) para cada celda activa.
    Returns: dict con constraint por celda y asignación sugerida de entidades.
    """
    constraints = {}
    entity_assignment = {e["name"]: None for e in anatomy.get("entities", [])}

    for cell in target_cells:
        rule = CELL_TO_ASYMMETRY.get(cell, {})
        constraints[cell] = {
            "design_rule": rule.get("design_rule", ""),
            "test": rule.get("test", ""),
            "entity_implications": _infer_entity_implications(cell, anatomy),
        }

    return {"cell_constraints": constraints, "entity_assignment": entity_assignment}


def _infer_entity_implications(cell: str, anatomy: dict) -> list[str]:
    """Qué entidades deben ir en I_A vs I_B para activar esta celda."""
    bottlenecks = anatomy.get("information_bottlenecks", [])
    axes = anatomy.get("natural_split_axes", [])
    # Simple heuristic: use the natural split axes
    if cell in ("B1", "C2") and axes:
        return [f"Asignar entidades del eje '{axes[0]}' a I_A y del eje opuesto a I_B"]
    if cell in ("D1",) and bottlenecks:
        return [f"Introducir ambigüedad alrededor de: {bottlenecks[0]}"]
    return []
```

#### Módulo 4: Generación Lingüística con Especificación Formal

**Propósito:** Traducir la especificación formal de la partición (I_A_spec, I_B_spec + restricciones por celda) a un split JSON natural para los agentes. El LLM aquí hace traducción, no diseño.

**Modelo:** Qwen2.5-7B (local, Sapelo) o Llama 3.1 8B (Groq, rápido y barato)

```python
# research/splitting/cidi/module4_generation.py

GENERATION_SYSTEM = """
Eres un traductor de especificaciones de aprendizaje colaborativo a lenguaje natural.
Se te da una especificación formal de cómo dividir un problema matemático entre {n} agentes.

Tu tarea es traducir esta especificación a un split jigsaw concreto.
NO debes añadir ni eliminar restricciones — solo traducir fielmente.
NO asignes roles con nombres. NO des protocolo de colaboración.

La especificación garantiza que el split activará colaboración CPS genuina si se sigue.

Problema: {problem}
Anatomía del problema: {anatomy_summary}

Restricciones por celda CPP (ya calculadas algorítmicamente):
{cell_constraints_summary}

Responde SOLO con JSON válido en el formato estándar:
{
  "pattern": "SPLIT-X",
  "shared_context": "...",
  "agent_roles": [{"agent_id": 1, "role_name": "...", "role_description": "..."}],
  "packets": [{"agent_id": 1, "information": "..."}, ...],
  "split_rationale": "...",
  "cidi_metadata": {
    "target_cells": [...],
    "cells_designed_for": [...],
    "design_rules_applied": {...}
  }
}
"""
```

#### Módulo 5: Validación Predictiva — Cadena de Discriminadores CPP

**Propósito:** Predecir el CPP que el split generado activará en la conversación, antes de simular. Si el CPP predicho difiere del target por más de 2 celdas (Hamming ≤ 2), aprobar. Si no, iterar.

**Discriminadores:** 12 clasificadores ligeros entrenados en el corpus n=150, ordenados por el DAG de prerrequisitos (cadena, no independientes).

```python
# research/splitting/cidi/module5_validation.py

from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import numpy as np

CELL_ORDER = ["A1","A2","A3","B1","B2","B3","C1","C2","C3","D1","D2","D3"]

class CPPDiscriminatorChain:
    """
    12 clasificadores en cadena respetando el DAG de prerrequisitos.
    D_i(split_embedding, predictions_{<i}) → P(cell_i = 1)
    """
    def __init__(self):
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.classifiers = {}   # cell → LogisticRegression

    def train(self, splits_texts: list[str], cpp_vectors: list[list[int]]):
        """
        Entrenar sobre el corpus existente.
        splits_texts: textos de splits (JSON serializado como string)
        cpp_vectors: listas de 12 bits correspondientes
        """
        embeddings = self.encoder.encode(splits_texts)   # (N, 384)
        prev_predictions = np.zeros((len(splits_texts), 0))

        for i, cell in enumerate(CELL_ORDER):
            X = np.hstack([embeddings, prev_predictions])   # condicionado en previas
            y = [v[i] for v in cpp_vectors]
            clf = LogisticRegression(max_iter=500, class_weight='balanced')
            clf.fit(X, y)
            self.classifiers[cell] = clf
            # Añadir predicción de esta celda para el siguiente discriminador
            preds = clf.predict_proba(X)[:, 1].reshape(-1, 1)
            prev_predictions = np.hstack([prev_predictions, preds])

    def predict(self, split_text: str) -> dict:
        """
        Returns: {cell: P(cell=1)} y vector binario predicho.
        """
        emb = self.encoder.encode([split_text])   # (1, 384)
        prev = np.zeros((1, 0))
        probs = {}
        for cell in CELL_ORDER:
            X = np.hstack([emb, prev])
            prob = self.classifiers[cell].predict_proba(X)[0, 1]
            probs[cell] = prob
            prev = np.hstack([prev, [[prob]]])

        predicted_vector = [1 if probs[c] >= 0.5 else 0 for c in CELL_ORDER]
        return {
            "probabilities": probs,
            "predicted_vector": predicted_vector,
            "predicted_cdi": sum(predicted_vector) / 12,
        }

    def hamming_to_target(self, predicted_vector: list[int], target: list[int]) -> int:
        return sum(abs(p - t) for p, t in zip(predicted_vector, target))
```

**Entrenamiento de los discriminadores:** `python3 -m research.splitting.cidi.train_discriminators`
- Input: corpus `outputs/splits/` + `outputs/scores/` (CPP anotado)
- Output: modelo guardado en `outputs/models/cpp_discriminator_chain.pkl`
- Tiempo: ~5 minutos en CPU local con n=150

#### Módulo 6 (Futuro — Post-NeurIPS): GRPO Fine-tuning

**Propósito:** Reemplazar los Módulos 1-4 con un LLM pequeño (Qwen2.5-7B) fine-tuneado via GRPO que aprende directamente `(problema, target_CPP_12bit) → split`. El reward vector es el CPP match del discriminador (Módulo 5).

**Requiere:** n≥100 splits CIDI generados y anotados. Tiempo de entrenamiento: 1 GPU Sapelo × 12-24h.

**Routing del reward:**
```python
# 12-dimensional reward vector
reward_vector = [
    1 if predicted_cell == target_cell else -0.5
    for predicted_cell, target_cell in zip(predicted_vector, target_cpp)
]
# GRPO group-relative advantage (DeepSeek-R1 style)
R_group = [score_split(s) for s in sampled_splits_for_problem]
A_k = (R_group[k] - mean(R_group)) / (std(R_group) + 1e-8)
```

### Implementación completa del split C2

```python
# research/splitting/cidi/pipeline.py

def split_cidi(
    problem_id: str,
    problem: str,
    target_cpp: list[str],      # lista de celdas a activar, e.g. ["A1","A2","B1","B2","C1","C2"]
    n: int = 2,
    max_retries: int = 2,
    hamming_threshold: int = 2,  # máximo error de targeting aceptable
) -> CIDISplitResult:

    # Módulo 1: análisis semántico
    anatomy = analyze_problem_semantics(problem)

    # Módulo 2: clausura de prerrequisitos + verificación
    target_closed = close_under_prerequisites(target_cpp)
    feasibility = {c: check_structural_feasibility(c, anatomy) for c in target_closed}
    target_feasible = [c for c in target_closed if feasibility[c][0]]

    # Módulo 3: derivar restricciones de partición
    constraints = derive_partition_constraints(target_feasible, anatomy)

    best_split = None
    best_hamming = 999

    for attempt in range(max_retries + 1):
        # Módulo 4: generación lingüística
        split_result = generate_from_specification(
            problem, anatomy, constraints, target_feasible, n
        )

        # Módulo 5: validación predictiva
        chain = load_discriminator_chain()   # cargado desde outputs/models/
        prediction = chain.predict(json.dumps(split_result.raw_split))
        target_vector = [1 if c in target_feasible else 0 for c in CELL_ORDER]
        hamming = chain.hamming_to_target(prediction["predicted_vector"], target_vector)

        if hamming <= hamming_threshold:
            return CIDISplitResult(
                split=split_result,
                target_cells=target_feasible,
                predicted_cpp=prediction["predicted_vector"],
                targeting_error=hamming / 12,
                iterations=attempt + 1,
                approved=True,
            )

        if hamming < best_hamming:
            best_hamming = hamming
            best_split = (split_result, prediction)

        # Ajustar restricciones para siguiente intento
        constraints = refine_constraints(constraints, prediction, target_vector)

    # Fallback: retornar mejor split encontrado
    split_result, prediction = best_split
    return CIDISplitResult(
        split=split_result,
        target_cells=target_feasible,
        predicted_cpp=prediction["predicted_vector"],
        targeting_error=best_hamming / 12,
        iterations=max_retries + 1,
        approved=False,
    )
```

### Estructura de archivos para C2-CIDI

```
research/splitting/cidi/
  __init__.py
  pipeline.py          # split_cidi() — API principal
  module1_semantic.py  # análisis semántico del problema
  module2_feasibility.py  # DAG prerrequisitos + BCE check
  module3_constraints.py  # tabla celda→asimetría
  module4_generation.py   # generación lingüística
  module5_validation.py   # discriminador en cadena
  train_discriminators.py # entrenamiento offline de los discriminadores
  models/              # discriminadores serializados
```

---

## Parte V: Condición 3 — Constitutional Pipeline (Actualizado)

### Cambios respecto a v1.0
- El crítico ahora también evalúa si las celdas CPP son estructuralmente activadas (no solo si Szewkis se cumple), usando la tabla celda→asimetría como heurística
- Se añade un paso de verificación BCE al final de cada iteración
- El revisor recibe información sobre cuáles celdas específicas fallaron, no solo cuáles condiciones Szewkis

### Pipeline actualizado

```
ETAPA 1: GENERACIÓN
  Input: problema + restricciones Szewkis básicas
  Modelo: GPT-4.1 o Llama 3.3 70B (Groq)
  Output: split_v0

ETAPA 2: CRÍTICA DUAL (24+12 checks)
  24 checks: 4 fases PISA × 6 condiciones Szewkis
  12 checks adicionales: para cada celda CPP activa, ¿el split tiene
                         la asimetría de información correcta? (tabla celda→asimetría)
  Modelo: GPT-4.1 (crítico requiere razonamiento fuerte)
  Output: critique_matrix[4][6] + cell_violations[12]
  Score: SQS = checks_passed / 36

ETAPA 3: REVISIÓN
  Input: split_v_i + critique_matrix + cell_violations
  Modelo: Llama 3.3 70B (Groq, más económico para revisión)
  Output: split_v_{i+1} + improvements_made

CICLO: MAX_ITER = 3, APPROVAL_THRESHOLD = 0.80 (≥29/36 checks)
```

### Prompt actualizado del Crítico (C3)

```
SYSTEM PROMPT — CRITIC C3 v2.0:

Eres un evaluador experto en Collaborative Problem Solving (CPS).

EVALUACIÓN DE DOS PARTES:

PARTE A — 24 checks Szewkis × PISA (igual que v1.0):
Para cada fase PISA {A,B,C,D} × condición Szewkis {S1..S6}:
  - satisfied: true/false
  - critique: si false, descripción específica

PARTE B — 12 checks de asimetría de información (NUEVO):
Para cada celda CPP que el split pretende activar:
  - ¿El split contiene la asimetría de información correcta para esta celda?
  - Referencia: tabla de tipos de asimetría por celda
  - satisfied: true/false
  - critique: qué asimetría falta o está incorrecta

Responde en JSON:
{
  "szewkis_evaluation": {...},  // 24 checks como antes
  "cell_asymmetry_evaluation": {
    "A1": {"satisfied": bool, "critique": "..."},
    ...
  },
  "overall_sqs": float,  // (szewkis_passed + cell_passed) / 36
  "critical_failures": [...]
}
```

---

## Parte VI: Condición 4 — Monitor Dinámico (Sin Cambios Mayores)

### Estado
La implementación de v1.0 es válida. El monitor evalúa las 6 condiciones Szewkis después de las fases A y B, máximo 2 intervenciones.

### Mejora menor de v2.0
Añadir al prompt del monitor una verificación de celdas CPP activas: si la conversación hasta ahora sugiere que una celda del target CPP está en riesgo de no activarse, el monitor puede mencionar esto explícitamente en su intervención.

```python
# Adición al monitor prompt:
"""
Adicionalmente, verifica si las siguientes celdas CPP están en camino a activarse
basándote en la conversación hasta ahora:
Celdas objetivo: {target_cells}
Para cada celda, indica si hay evidencia de que se está activando o está en riesgo.
"""
```

### Máximo 2 intervenciones — justificación revisada
**PED:** Más de 2 intervenciones convierte el monitor en el protagonista del aprendizaje. En términos de Brousseau, invalida la situación a-didáctica.

**ME:** Desde el diseño experimental, el número de intervenciones es una variable dependiente de interés (H5: C5 requiere menos intervenciones que C4 porque el split CIDI es estructuralmente superior). Necesitamos que el máximo permita observar la diferencia.

**Decisión:** Mantener máximo 2. Registrar n_interventions como variable dependiente.

---

## Parte VII: Condición 5 — Integrada (C2_CIDI + C4)

### Descripción actualizada
C5 = split generado por CIDI (Módulo 1-5 de C2) + monitor dinámico activo durante simulación (C4).

**Por qué C2_CIDI + monitor (no C3 + monitor):**
- C3 optimiza para Szewkis sostenido (SQS global); CIDI optimiza para CPP targeting (celda específica)
- C5 con CIDI maximiza tanto la precisión del target CPP como la calidad sostenida del proceso
- C3 + monitor sería otra condición válida pero secundaria para el piloto

```python
def run_condition_5(problem_id, problem, target_cpp, n=2):
    # Stage C2: split via CIDI
    cidi_result = split_cidi(problem_id, problem, target_cpp, n)
    # Stage C4: simular con monitor dinámico
    conversation = simulate_with_monitor(
        cidi_result.split,
        condition=f"integrated_{n}",
        target_cells=target_cpp,   # NUEVO: pasar target al monitor
    )
    return conversation, cidi_result
```

---

## Parte VIII: Estrategia de Modelos — OpenAI vs Groq/Llama vs Sapelo vLLM

### Problema
El pipeline completo (C2_CIDI + C3 + C4 + C5 + scoring) para 600 problemas × 5 condiciones requiere:
- Módulo 1 (análisis semántico): 1 call por problema × 600 = 600 calls
- Módulo 4 (generación): 1-3 calls por problema = 600-1800 calls
- C3 (constitutional): ~10 calls por problema = 6000 calls
- Simulación: ~15 turns × n agentes = miles de calls
- CPP annotation: 1 call por conversación = miles de calls

Con GPT-4.1 ($2/$8 por M tokens), esto puede costar $200-500. Con Groq/vLLM local, el costo es $10-20.

### Tabla de routing de modelos

| Tarea | OpenAI | Groq/Llama | Sapelo vLLM | Decisión |
|-------|--------|-----------|-------------|---------|
| Análisis semántico M1 | GPT-4.1 $0.08/call | Llama 3.3 70B $0.004/call | Qwen2.5-72B $0 | **Groq** |
| Generación split M4 | GPT-4.1 $0.12/call | Llama 3.3 70B $0.006/call | Qwen2.5-7B $0 | **Sapelo vLLM** |
| Crítico C3 | GPT-4.1 $0.15/call | — | Qwen2.5-72B $0 | **Sapelo vLLM** |
| Revisor C3 | GPT-4.1 $0.10/call | Llama 3.3 70B $0.005/call | Qwen2.5-72B $0 | **Sapelo vLLM** |
| Simulación agentes | gpt-5.4-mini $0.02/call | Llama 3.3 70B $0.004/call | Qwen2.5-72B $0 | **Sapelo vLLM** |
| Monitor C4 | gpt-5.4-mini $0.02/call | — | Qwen2.5-72B $0 | **Sapelo vLLM** |
| CPP annotation | gpt-5.4-mini $0.03/call | — | Qwen2.5-72B $0 | **Sapelo vLLM** |
| PISA scoring | gpt-5.4-mini $0.03/call | — | Qwen2.5-72B $0 | **Sapelo vLLM** |
| Validación final (gold) | GPT-4.1 sample 10% | — | — | **OpenAI** (muestra) |

### Configuración de Groq en openai_utils.py

```python
# research/openai_utils.py — añadir routing Groq
import os
from openai import OpenAI

_GROQ_BASE  = "https://api.groq.com/openai/v1"
_GROQ_KEY   = os.environ.get("GROQ_API_KEY")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

_groq_client: OpenAI | None = (
    OpenAI(base_url=_GROQ_BASE, api_key=_GROQ_KEY) if _GROQ_KEY else None
)

# Variables de entorno para routing:
# GROQ_TASKS=semantic_analysis,split_generation  → usa Groq para estas tareas
# LOCAL_MODEL_BASE_URL=http://...                 → usa vLLM para el resto
```

### Costo estimado total

| Escenario | OpenAI | Groq | Sapelo vLLM | Total |
|-----------|--------|------|-------------|-------|
| Piloto 4 problemas × 5 condiciones | $3 | $0.50 | $0 | **~$3.50** |
| 30 problemas × 5 condiciones | $15 | $2 | $0 | **~$17** |
| 600 problemas × C1+C5 | $40 | $5 | $0 | **~$45** |
| 600 problemas × 5 condiciones | $150 | $20 | $0 | **~$170** |

---

## Parte IX: Diseño Completo del Experimento Piloto

### Selección de los 4 problemas piloto

Los 4 problemas son seleccionados algorítmicamente del corpus existente:

```python
# python3 -m research.experiments.cpp_comparison --select-only --verbose
```

Criterios:
- P1: algebra, L2, conversación jigsaw_2 con menor total_turns (split más trivial — mejor caso para demostrar mejora)
- P2: geometry, L3, menor PISA_global en jigsaw_2 (más potencial de mejora)
- P3: number_theory, L4, menor PISA_global en jigsaw_2
- P4: counting_and_probability, L5, menor PISA_global en jigsaw_2

### Target CPP para el piloto

Para simplificar el análisis comparativo, usar el mismo target CPP para todos los problemas:

**Target = CPP-DEEP:** `[A1, A2, A3, B1, B2, B3, C1, C2]` activadas → CDI = 8/12 = 0.667

Justificación: CPP-DEEP activa las 3 primeras filas completas + omite el monitoring (D) porque requiere estructuras de problema más complejas. Es alcanzable para problemas L2-L5 con diseño adecuado.

### Variables dependientes

| Variable | Tipo | Instrumento | Hipótesis esperada |
|----------|------|-------------|-------------------|
| CDI real | Continua [0,1] | CPP annotator (Qwen2.5-72B) | C2_CIDI > C5 > C3 > C4 > C1 |
| targeting_error | Continua [0,1] | Hamming(CPP_achieved, CPP_target)/12 | C2_CIDI < C3 ≈ C5 < C4 < C1 |
| PISA_global | Continua | Scorer Qwen2.5-72B | C5 > C2_CIDI ≈ C3 > C4 > C1 |
| ATC_SR | Continua | Scorer Qwen2.5-72B | C5 ≫ C1; C2_CIDI ≈ C3 |
| SQS | Continua [0,1] | Critic output (C3) / Monitor log (C4,C5) | C3 ≈ C5 > C2_CIDI > C4 > C1 |
| n_interventions | Entero | Monitor log | C4 > C5 (H5: CIDI reduce necesidad de monitor) |
| n_constitutional_iter | Entero | Constitutional log | indicador de dificultad por problema |
| total_turns | Entero | Conversación | C5 ≥ C3 > C2_CIDI ≈ C4 > C1 |
| accuracy | Binaria | is_correct() | Sin hipótesis clara (H3 no aplica a n=4) |

### Análisis

**Cualitativo (prioritario en el piloto — n=4 no permite inferencia):**
1. Para cada par (problema, condición): leer la conversación y mapear manualmente las primeras 5 celdas CPP activadas
2. Comparar el diseño del split CIDI vs el split C1: ¿dónde específicamente fuerza más interdependencia?
3. Verificar H8: ¿el targeting_error es menor en C2_CIDI que en C3?
4. Evaluar si el monitor en C4/C5 interfiere con la naturalidad de la conversación

**Cuantitativo:**
- Tabla 4×5 de todas las variables dependientes
- Correlación within-problem: CDI × PISA_global (4 puntos por condición)
- Comparación de medias C1 vs C5 (orientativa, sin inferencia estadística con n=4)

**Criterio de éxito del piloto:**
- Al menos 3/4 problemas: CDI(C5) > CDI(C1) + 0.40
- Al menos 3/4 problemas: targeting_error(C2_CIDI) < 0.25
- Las conversaciones C5 son cualitativamente identificables como más ricas por un evaluador ciego
- El pipeline CIDI corre sin errores en los 4 problemas

---

## Parte X: Plan de Implementación Técnica

### Estructura de archivos completa del proyecto

```
research/
  splitting/
    splitter.py                    ✅ LISTO — split estándar (C1)
    splitter.py:split_cpp_targeted ✅ LISTO — C2 prompt simple
    constitutional.py              ✅ LISTO — C3 pipeline
    cidi/                          ⬜ NUEVO — C2 CIDI completo
      __init__.py
      pipeline.py                  ⬜ split_cidi() API
      module1_semantic.py          ⬜ análisis semántico
      module2_feasibility.py       ⬜ DAG prerrequisitos
      module3_constraints.py       ⬜ tabla asimetrías
      module4_generation.py        ⬜ generación lingüística
      module5_validation.py        ⬜ discriminadores en cadena
      train_discriminators.py      ⬜ entrenamiento offline

  simulation/
    simulator.py                   ✅ LISTO — simulate_with_monitor()
    monitor.py                     ✅ LISTO — Szewkis monitor

  scoring/
    cpp_annotator.py               ✅ LISTO — CPP annotation
    pisa.py                        ✅ LISTO — PISA scoring
    atc21s.py                      ✅ LISTO — ATC21S scoring

  experiments/
    cpp_comparison.py              ✅ LISTO — runner C1-C5

  openai_utils.py                  ⬜ PENDIENTE — añadir routing Groq

scripts/
  sapelo_gpu_setup.sh              ✅ LISTO
  sapelo_gpu_job.sh                ✅ LISTO
  sapelo_pilot_job.sh              ⬜ NUEVO — job SLURM específico para piloto

docs/
  framework_PIE_CPS.md             ✅ v3.0 ACTUALIZADO
  methodology_conditions.md        ✅ v2.0 — este documento
  c2_architecture_analysis.md      ✅ COMPLETO
```

### Prioridades de implementación

**Fase 1 — Discriminadores (esta semana, ~4h):**
1. `research/splitting/cidi/module5_validation.py` — el discriminador es el enabler del pipeline
2. `research/splitting/cidi/train_discriminators.py` — entrenar con corpus existente n=150
3. Verificar que los discriminadores tienen AUC > 0.65 para al menos 8/12 celdas

**Fase 2 — Módulos 1-3 (esta semana, ~6h):**
4. `module1_semantic.py` — análisis semántico vía Groq Llama 3.3 70B
5. `module2_feasibility.py` — DAG de prerrequisitos (determinista, sin LLM)
6. `module3_constraints.py` — tabla celda→asimetría (determinista, sin LLM)

**Fase 3 — Módulo 4 + pipeline (esta semana, ~4h):**
7. `module4_generation.py` — generación via Qwen2.5-7B o Llama 3.1 8B
8. `cidi/pipeline.py` — integrar todos los módulos
9. Actualizar `cpp_comparison.py` para usar `split_cidi()` en `run_c2()`

**Fase 4 — Routing Groq + piloto (fin de semana):**
10. Añadir routing Groq a `openai_utils.py`
11. Crear `scripts/sapelo_pilot_job.sh`
12. Ejecutar piloto: 4 problemas × 5 condiciones
13. Analizar resultados, generar tabla comparativa

**Fase 5 — Escalamiento Sapelo (semana siguiente):**
14. Escalar a 30 problemas (muestra para validar tendencias)
15. Si resultados confirman H5/H8: escalar a 600 problemas (C1 ya existe, ejecutar C5)

### Script SLURM para el piloto

```bash
#!/bin/bash
# scripts/sapelo_pilot_job.sh
#SBATCH --job-name=collabmath_pilot
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:2
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/%u/collabmath/logs/pilot_%j.out

SCRATCH_JOB="/scratch/${USER}/collabmath"
export PYTHONPATH="${SCRATCH_JOB}:${PYTHONPATH:-}"

module load CUDA/12.1.1
module load Miniforge3/24.11.3-0
source activate ~/.conda/envs/collabmath_gpu

# Cargar secretos
source ~/.collabmath_secrets

# Iniciar vLLM (para simulación, scoring, constitutional, monitor)
vllm serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --tensor-parallel-size 2 \
  --port 8000 &
VLLM_PID=$!

# Esperar que vLLM esté listo
for i in $(seq 1 60); do
  sleep 5
  curl -s http://localhost:8000/health && break
  echo "[WAIT] vLLM not ready yet... $((i*5))s"
done

# Configurar routing
export LOCAL_MODEL_BASE_URL="http://localhost:8000/v1"
export LOCAL_MODEL_NAME="Qwen/Qwen2.5-72B-Instruct-AWQ"
export KEEP_REMOTE_MODELS="gpt-4.1"   # solo GPT-4.1 va a OpenAI
# Groq para análisis semántico (Módulo 1)
export GROQ_API_KEY="${GROQ_API_KEY}"
export GROQ_TASKS="semantic_analysis"

# Entrenar discriminadores sobre corpus existente (si no existen)
if [ ! -f "${SCRATCH_JOB}/outputs/models/cpp_discriminator_chain.pkl" ]; then
  python3 -m research.splitting.cidi.train_discriminators \
    --splits-dir "${SCRATCH_JOB}/outputs/splits" \
    --scores-dir "${SCRATCH_JOB}/outputs/scores" \
    --output "${SCRATCH_JOB}/outputs/models/cpp_discriminator_chain.pkl"
fi

# Correr experimento piloto
python3 -m research.experiments.cpp_comparison \
  --conditions C1 C2 C3 C4 C5 \
  --target-cpp A1 A2 A3 B1 B2 B3 C1 C2 \
  --n-problems 4 \
  --output-dir "${SCRATCH_JOB}/outputs/pilot"

# Cerrar vLLM
kill $VLLM_PID

echo "Pilot complete. Results in ${SCRATCH_JOB}/outputs/pilot/"
```

---

## Parte XI: Escalamiento y Validez Experimental

### Diseño para el paper journal (n=30)

Para el paper journal de Computers & Education (separado de NeurIPS), el diseño completo es:

```
30 problemas × 5 condiciones = 150 conversaciones nuevas
+ 144 conversaciones del corpus C1 baseline = 294 conversaciones totales

Análisis:
- Mixed-effects model: CDI ~ condición + (1|problema) + (1|subject)
- Efecto de condición: F-test dentro del modelo
- Post-hoc: Bonferroni para 10 comparaciones por pares
- Efecto de tamaño: d de Cohen para cada par C1 vs Ci
```

### Diseño para NeurIPS 2026 (n=600)

```
600 problemas × C1 (ya ejecutado) + C5 (ejecutar en Sapelo)
= 1200 conversaciones para análisis principal

Contribuciones:
1. Framework CIDI como pipeline de generación de actividades CPS
2. Demostración empírica: targeting_error(CIDI) < 0.25 en ≥70% problemas
3. Escalamiento: CDI(C5) > CDI(C1) + 0.40 con efecto de tamaño d > 0.80
```

### Amenazas a la validez interna

| Amenaza | Control |
|---------|---------|
| Varianza de sampling del LLM | C1 usa corpus existente; C2-C5 fijan temperatura 0.2 para splits |
| Contaminación entre condiciones | Cada condición corre en conversaciones independientes (no same conversation) |
| Confounding por dificultad | Controlar por level y subject en todos los análisis |
| Evaluación CPP subjetiva | CPP annotator validado con muestra de 20 conversaciones anotadas manualmente |
| Overfitting del discriminador | Holdout del 20% del corpus para validar AUC antes de usar en piloto |

---

## Parte XII: Preguntas Abiertas del Panel

**MC:** ¿Cómo validar empíricamente que el DAG de prerrequisitos es correcto? Podríamos hacer análisis FCA sobre el corpus n=150 y comparar las implicaciones descubiertas con el DAG teórico. Si el FCA dice "C2 → A1 siempre" y el DAG también, buena señal.

**PED:** ¿El target CPP-DEEP fijo para todos los problemas del piloto es correcto? Un problema L2 de álgebra podría ser demasiado simple para activar genuinamente 8 celdas. ¿Deberíamos usar targets distintos por nivel?

**IA:** La pregunta técnica más urgente: ¿los discriminadores del Módulo 5 tendrán AUC aceptable con solo n=150 training examples? Si no, el Módulo 5 de validación predictiva no funciona y el pipeline se reduce a confiar en el Módulo 4.

**ME:** ¿Debería existir una condición C0 donde ambos agentes tienen toda la información (sin split) para separar el efecto del diseño del split del efecto de la simulación colaborativa? Esto controlaría si las mejoras de C2-C5 son del split o de la fuerza del LLM para colaborar cuando se le pide.

**TCPS:** El targeting_error es una métrica nueva (H8). Para que sea creíble en NeurIPS, necesitamos inter-rater reliability del CPP annotator: que dos runs del mismo anotador sobre la misma conversación den el mismo vector. ¿Con qué frecuencia difieren? Esto debería medirse en el piloto.

---

## Referencias Adicionales

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

Bergemann, D., & Morris, S. (2019). Information Design: A Unified Perspective. *Journal of Economic Literature, 57*(1), 44–95.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving Factuality and Reasoning in Language Models through Multiagent Debate. *arXiv:2305.14325*.

Ganter, B., & Wille, R. (1999). *Formal Concept Analysis: Mathematical Foundations*. Springer.

Kamenica, E., & Gentzkow, M. (2011). Bayesian Persuasion. *American Economic Review, 101*(6), 2590–2615.

Yang, K., & Klein, D. (2021). FUDGE: Controlled Text Generation With Future Discriminators. *NAACL 2021*.
