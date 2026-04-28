# Metodología: 5 Condiciones Experimentales CPP
## CollabMath — Diseño de Implementación con Pipeline Multi-Agente

**Versión:** 1.0  
**Referencia teórica:** docs/framework_PIE_CPS.md  

---

## Panel de Agentes Metodológicos

Este documento fue construido con un panel de cuatro perspectivas expertas en tensión deliberada. Cada decisión de diseño refleja el debate entre estas perspectivas:

| Agente | Rol | Sesgo / Prioridad |
|--------|-----|-------------------|
| **Pedagogo** | Didáctica matemática, Brousseau, Vygotsky | Autenticidad pedagógica, ZPD-G real |
| **Teórico CPS** | PISA, ATC21S, Szewkis | Rigor en medición, validez de constructo |
| **Ingeniero IA** | LLMs, pipelines, Constitutional AI | Implementabilidad, costo, robustez técnica |
| **Metodólogo** | Diseño experimental, estadística | Validez interna, control de confounders |

Las posiciones en debate aparecen a lo largo del documento como `[DEBATE: ...]`.

---

## Visión General: Las 5 Condiciones

| Condición | Intervención en el split | Intervención en la simulación | CDI esperado |
|-----------|-------------------------|-------------------------------|-------------|
| **C1** Baseline | Prompt estándar | Simulación estándar | ~0.08 |
| **C2** CPP-Directed | Prompt CPP+Szewkis → perfil objetivo | Simulación estándar | ~0.50-0.75 |
| **C3** Szewkis-Sustained | Pipeline Constitutional → SQS=1 | Simulación estándar | ~0.50-0.75 |
| **C4** Dynamic Monitor | Prompt estándar | Monitor Szewkis activo en cada fase | ~0.25-0.50 |
| **C5** Integrated | Pipeline Constitutional (C3) | Monitor Szewkis (C4) | ~0.75-1.00 |

**Hipótesis de ordenamiento:** C5 > C3 ≈ C2 > C4 > C1 en PISA_global y ATC_SR/CR.

---

## Condición 1: Baseline

### Descripción
Sin modificaciones. Usa el splitter GPT-4.1 y el simulador GPT-5.4-mini actuales.

### Justificación como control
El Metodólogo insiste: C1 debe ser idéntico al corpus existente (n=150 problemas ya ejecutados), no una re-ejecución. Esto evita varianza por nuevo sampling de GPT.

**Decisión:** Para los 4 problemas del experimento piloto, usar los splits y conversaciones ya existentes en `outputs/splits/` y `outputs/conversations/`. No re-generar.

### Implementación
```python
# No requiere código nuevo — usar outputs existentes
split = load_existing_split(problem_id, n=2)
conv  = load_existing_conversation(problem_id, "jigsaw_2")
score = score_existing(problem_id, "jigsaw_2")
```

---

## Condición 2: CPP-Directed + Szewkis como piso

### Descripción
El splitter recibe como input: (a) el perfil CPP objetivo en términos de celdas PISA a activar, y (b) las 6 condiciones Szewkis como restricciones de diseño mínimas.

### Debate del Panel

**Pedagogo:** "El prompt debe describir las celdas PISA en lenguaje operacional, no en jerga de la matriz. Decir 'activa B1' no tiene sentido para el LLM si no explicas qué significa B1 en términos de la conversación."

**Ingeniero IA:** "Correcto. Traducir cada celda a un comportamiento observable esperado en la conversación. Por ejemplo: B1 activo = 'los agentes deben llegar a acuerdo explícito sobre cómo representar el problema antes de proponer ningún método de solución.'"

**Teórico CPS:** "Las 6 condiciones Szewkis como piso mínimo son necesarias pero pueden ser contradictorias con el perfil CPP objetivo si no se coordinan. Por ejemplo, pedir CPP-FULL (todas las celdas) puede violar S3 (responsabilidad individual) si un agente se vuelve dependiente en exceso."

**Metodólogo:** "Para el experimento piloto, fijar el perfil CPP objetivo en CPP-DEEP (celdas A1,A2,A3,B1,B2,B3,C1,C2 activas) para todos los problemas. Perfil único simplifica el análisis comparativo."

**Decisión:** CPP objetivo = CPP-DEEP para todos los problemas en el piloto. Szewkis como 6 restricciones explícitas en el prompt.

### Prompt del Splitter — Condición 2

```
SYSTEM PROMPT — SPLITTER C2:

Eres un experto en diseño de actividades de aprendizaje colaborativo matemático.
Tu tarea es dividir un problema matemático en {n} paquetes de información para 
que n agentes LLM puedan resolverlo conjuntamente mediante Collaborative Problem 
Solving (CPS) genuino.

## PERFIL CPP OBJETIVO
El split debe activar las siguientes celdas de la matriz PISA CPS. "Activa" 
significa que esa celda requerirá colaboración real — ningún agente puede 
completarla sin input activo del otro.

Celdas a activar:
- A1 (Explorar·Conocimiento compartido): Los agentes deben descubrir qué sabe 
  el otro y qué pueden hacer. Ninguno puede evaluar su propia información sin 
  conocer la del otro.
- A2 (Explorar·Acción): Los agentes deben establecer juntos las normas de 
  interacción (¿quién lidera cada paso? ¿cómo verifican acuerdo?).
- A3 (Explorar·Organización): Los roles deben emerger de la exploración 
  conjunta, no ser asignados a priori.
- B1 (Formular·Conocimiento compartido): Los agentes deben negociar 
  explícitamente la representación del problema — no pueden asumir que 
  comparten la misma interpretación.
- B2 (Formular·Acción): Identificar las sub-tareas requiere información de 
  ambos agentes — ninguno puede hacer la lista completo solo.
- B3 (Formular·Organización): La distribución del trabajo en la ejecución debe 
  negociarse, no derivarse automáticamente del split inicial.
- C1 (Ejecutar·Conocimiento compartido): Antes de cada paso de ejecución, los 
  agentes deben comunicar qué van a hacer y por qué — el otro debe confirmar.
- C2 (Ejecutar·Acción): Hay pasos de ejecución que literalmente requieren el 
  resultado del otro como input. No es opcional consultar.

## CONDICIONES SZEWKIS (RESTRICCIONES MÍNIMAS)
El split debe garantizar estructuralmente:
1. OBJETIVO COMÚN: La meta de la actividad debe ser compartida y explícita. 
   Ambos agentes trabajan para resolver el mismo problema completo, no "su parte".
2. INTERDEPENDENCIA POSITIVA: Ningún agente puede resolver el problema 
   completo incluso si recibe toda la información del otro en el turno 1. 
   La integración misma requiere razonamiento nuevo.
3. RESPONSABILIDAD INDIVIDUAL: Cada agente tiene una contribución única y 
   necesaria en al menos 3 momentos distintos de la resolución.
4. RECOMPENSA GRUPAL: El éxito se define como "el grupo llegó a la respuesta 
   correcta juntos" — no "cada uno resolvió su parte".
5. CONCIENCIA GRUPAL: El split debe hacer necesario que cada agente mantenga 
   un modelo mental del otro (qué sabe, qué puede hacer).
6. COORDINACIÓN Y COMUNICACIÓN: Los agentes deben coordinar activamente el 
   proceso — no pueden trabajar en paralelo y juntar al final.

## TEST DE PROFUNDIDAD
Antes de finalizar el split, verifica:
- Si Agente 1 le dice todo su paquete al Agente 2 en el turno 1, 
  ¿puede el Agente 2 resolver solo? Si la respuesta es SÍ, el split es 
  superficial — rediseña.
- ¿Requieren al menos 4 intercambios de información para llegar a la solución?
- ¿Cada agente hace razonamiento matemático propio en al menos 2 momentos?

## OUTPUT
Responde con JSON válido:
{
  "pattern": "SPLIT-X",
  "split_rationale": "...",
  "shared_context": "...",
  "packets": [{"agent_id": 1, "information": "..."}, ...],
  "agent_roles": [{"agent_id": 1, "role_name": "...", "role_description": "..."}],
  "depth_verification": {
    "agent2_can_solve_alone_after_turn1": false,
    "minimum_exchanges_needed": 4,
    "mathematical_actions_per_agent": 2,
    "szewkis_satisfied": [true, true, true, true, true, true]
  }
}
```

---

## Condición 3: Szewkis Sostenido — Pipeline Constitutional

### El problema con un prompt solo

**Ingeniero IA:** "Un único prompt con restricciones Szewkis tiene el problema de que el LLM puede generar un split que *parece* satisfacerlas pero que en la simulación degenera a CPP-T porque las restricciones no son estructuralmente verificables desde el texto del split."

**Pedagogo:** "Es correcto. En la Teoría de Situaciones Didácticas, la a-didacticidad no se declara en el enunciado — emerge de la *estructura* de la situación. Un split puede afirmar 'esto requiere colaboración en la ejecución' pero si la estructura de la información permite resolución individual, la declaración es vacía."

**Teórico CPS:** "La solución es un pipeline de verificación iterativa. Después de generar el split, un segundo agente verifica cada condición Szewkis en cada fase PISA mediante simulación parcial (razonamiento forward: ¿qué pasaría si los agentes ejecutan este split?). Si alguna condición falla, el primer agente refina."

**Decisión:** Pipeline Constitutional de 3 etapas para C3.

### Pipeline Constitutional (C3)

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: GENERACIÓN INICIAL                                │
│  Splitter estándar + restricciones Szewkis básicas          │
│  Output: split_v0                                           │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: CRÍTICA SZEWKIS × PISA (24 checks)               │
│  Para cada fase PISA {A, B, C, D}:                          │
│    Para cada condición Szewkis {S1..S6}:                    │
│      ¿Se satisface S_j en la fase X dado split_v0?          │
│      Si NO: genera critique_j_X (1-2 frases)               │
│  Output: critique_matrix[4][6] con observaciones           │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
              ¿Algún critique?
              NO ──────────────→ SPLIT APROBADO
              SÍ ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: REVISIÓN                                          │
│  Reviser recibe: split_v_i + critique_matrix                │
│  Instrucción: "Mejora el split resolviendo todas las        │
│  deficiencias identificadas sin eliminar las fortalezas     │
│  ya presentes."                                             │
│  Output: split_v_{i+1}                                      │
└─────────────────────┬───────────────────────────────────────┘
                      ↓
              ¿i < MAX_ITER (3)?
              SÍ → vuelve a STAGE 2
              NO → usar mejor split según critique score
```

### Prompt del Crítico (Stage 2)

```
SYSTEM PROMPT — CRITIC C3:

Eres un evaluador experto en Collaborative Problem Solving (CPS).
Se te proporciona un split jigsaw para {n} agentes y un problema matemático.

Evalúa si el split satisface las 6 condiciones de Szewkis en cada fase 
de resolución PISA (A: Explorar, B: Formular, C: Ejecutar, D: Monitorear).

Para CADA combinación (fase, condición), responde:
- satisfied: true/false
- critique: (si false) descripción específica de qué falla y por qué

Razona imaginando la conversación: si los agentes ejecutan este split 
fielmente, ¿qué ocurriría en cada fase?

Problema: {problem_text}
Split: {split_json}

Responde en JSON:
{
  "evaluation": {
    "A": {"S1": {"satisfied": bool, "critique": "..."}, ..., "S6": {...}},
    "B": {...},
    "C": {...},
    "D": {...}
  },
  "overall_sqs": float,  // fracción de los 24 checks que pasan
  "critical_failures": ["descripción de los fallos más importantes"]
}
```

### Prompt del Revisor (Stage 3)

```
SYSTEM PROMPT — REVISER C3:

Eres un diseñador experto de actividades CPS matemáticas.

Se te proporciona:
1. Un split jigsaw actual con deficiencias identificadas
2. Una matriz de crítica con los fallos específicos por fase y condición

Tu tarea: mejorar el split resolviendo TODOS los fallos identificados.
No puedes reducir la calidad de los aspectos ya satisfechos.
Mantén el mismo problema y el mismo n de agentes.

Criterio de éxito: el split mejorado debe pasar los 24 checks (6 Szewkis × 4 fases).

Problema: {problem_text}
Split actual: {split_json}
Críticas: {critique_matrix}

Responde con el split mejorado en el mismo formato JSON.
Incluye: "improvements_made": ["lista de cambios realizados"]
```

### Límite de Iteraciones y Fallback

```python
MAX_ITER = 3
APPROVAL_THRESHOLD = 0.80  # al menos 80% de los 24 checks

def constitutional_split(problem, n):
    split = generate_initial_split(problem, n)
    for i in range(MAX_ITER):
        critique = evaluate_szewkis_pisa(split, problem)
        if critique["overall_sqs"] >= APPROVAL_THRESHOLD:
            return split, critique
        split = revise_split(split, critique, problem)
    # Fallback: usar el split con mejor SQS del historial
    return best_split_in_history()
```

**Costo estimado:** 1 (generación) + 3 (crítica) + 2 (revisión) × MAX_ITER = ~7-10 LLM calls por problema. Para 4 problemas del piloto: ~30-40 calls ≈ $0.50-1.00.

---

## Condición 4: Monitor Dinámico Szewkis

### Descripción
El split es el estándar (C1). El cambio es en la **simulación**: después de cada fase PISA, un agente monitor evalúa si las 6 condiciones Szewkis se mantienen y, si alguna falla, inyecta una corrección en la conversación.

### Debate del Panel

**Pedagogo:** "El monitor no debe interrumpir la conversación de forma artificial. Debe actuar como un facilitador invisible que reformula la situación cuando la colaboración se desvía."

**Ingeniero IA:** "Implementado como un 'meta-turn' entre fases: después de que el sistema detecta que los agentes han concluido la fase A, inserta un turno del monitor que —si es necesario— dice a los agentes cómo redistribuir el trabajo."

**Metodólogo:** "Hay un riesgo: si el monitor interviene mucho, C4 se vuelve un experimento sobre el monitor, no sobre el split. Sugiero que el monitor pueda intervenir máximo 2 veces por conversación."

**Teórico CPS:** "El monitor también debe poder proponer una renegociación del split si la causa del problema es estructural. Esto modela lo que hacen los equipos humanos reales cuando un miembro dice 'creo que estamos distribuyendo mal el trabajo.'"

**Decisión:** Monitor activo después de fases A y B solamente (las fases de setup). Máximo 2 intervenciones. Puede proponer renegociación del split.

### Detector de Fase PISA

```python
PHASE_INDICATORS = {
    "A": ["tengo", "sé", "mi información", "me dieron", "mi parte es"],
    "B": ["entonces el plan es", "podemos hacer", "yo haré", "tú haces", 
          "propongo que", "el problema requiere"],
    "C": ["calculando", "si sustituyo", "obtengo", "el resultado es", 
          "aplicando", "resolviendo"],
    "D": ["verificando", "comprobemos", "la respuesta es", "¿estás de acuerdo?",
          "revisando", "confirmo"],
}

def detect_phase_transition(conversation_history: list[dict]) -> str:
    """Detecta en qué fase PISA está la conversación basándose en los últimos 3 turnos."""
    recent = " ".join([t["content"] for t in conversation_history[-3:]])
    scores = {phase: sum(kw in recent.lower() for kw in kws) 
              for phase, kws in PHASE_INDICATORS.items()}
    return max(scores, key=scores.get)
```

### Prompt del Monitor (C4)

```
SYSTEM PROMPT — SZEWKIS MONITOR:

Eres un facilitador experto en Collaborative Problem Solving (CPS).
Observas una conversación entre {n} agentes LLM resolviendo un problema matemático.

Se acaba de completar la fase {current_phase} de la resolución.

Evalúa si las 6 condiciones de Szewkis se mantuvieron durante esta fase:

Conversación de la fase {current_phase}:
{phase_conversation}

Evalúa cada condición:
1. Objetivo común: ¿trabajaron ambos hacia la misma meta?
2. Interdependencia positiva: ¿dependió cada uno del otro para avanzar?
3. Responsabilidad individual: ¿contribuyó activamente cada agente?
4. Recompensa grupal: ¿se orientaron al éxito del grupo, no individual?
5. Conciencia grupal: ¿se mantuvieron actualizados sobre lo que hacía el otro?
6. Coordinación: ¿coordinaron activamente o trabajaron en paralelo?

Si TODAS las condiciones se satisfacen:
→ Responde: {"intervene": false, "sqs_phase": float}

Si alguna condición falla:
→ Responde: {
    "intervene": true,
    "sqs_phase": float,
    "failing_conditions": [lista],
    "intervention": "mensaje directo a los agentes para corregir, 
                     en primera persona plural: 'Antes de continuar, 
                     necesitamos...' — máximo 3 frases"
  }
```

### Modificación del Simulador (C4)

```python
def simulate_with_monitor(split_result, condition, max_interventions=2):
    conversation = []
    interventions = 0
    current_phase = "A"
    
    for turn_idx in range(MAX_TURNS):
        # Turno normal del agente
        turn = simulate_single_turn(split_result, condition, conversation)
        conversation.append(turn)
        
        # Detectar transición de fase
        new_phase = detect_phase_transition(conversation)
        
        if new_phase != current_phase and interventions < max_interventions:
            # Evaluar fase completada con el monitor
            monitor_result = evaluate_with_monitor(
                current_phase, conversation, split_result
            )
            
            if monitor_result["intervene"]:
                # Inyectar corrección
                intervention_turn = {
                    "agent_id": "MONITOR",
                    "role": "facilitator", 
                    "content": monitor_result["intervention"]
                }
                conversation.append(intervention_turn)
                interventions += 1
            
            current_phase = new_phase
        
        # Condición de parada estándar
        if is_converged(conversation):
            break
    
    return build_conversation_object(conversation)
```

---

## Condición 5: Integrada (C3 + C4)

### Descripción
Combina el pipeline constitutional del split (C3) con el monitor dinámico de la simulación (C4).

### Debate del Panel

**Metodólogo:** "C5 debe responder: ¿son los efectos de C3 y C4 aditivos, o uno domina al otro? Esto requiere que el piloto sea lo suficientemente sensible para detectar diferencias C3 vs C5 y C4 vs C5."

**Pedagogo:** "Desde la Teoría de Situaciones Didácticas: C3 crea la situación a-didáctica (el milieu colaborativo), C4 es la intervención del profesor cuando la situación no es suficiente. Juntos modelan una clase completa."

**Teórico CPS:** "La hipótesis teórica más interesante es: ¿necesita el monitor intervenir más en C5 que en C4 (porque el split mejor diseñado ya resuelve los problemas) o menos (porque el split da menos margen a la degradación)?"

**Decisión:** Registrar número de intervenciones del monitor en C4 vs C5. Si C5 requiere menos intervenciones que C4, confirma que el split bien diseñado reduce la necesidad de corrección dinámica.

### Implementación
```python
def run_condition_5(problem_id, n=2):
    # Stage C3: generar split con pipeline constitutional
    split, critique = constitutional_split(problem_id, n)
    # Stage C4: simular con monitor dinámico
    conversation = simulate_with_monitor(split, "jigsaw_2")
    return conversation, split, critique
```

---

## Experimento Piloto: Diseño

### Selección de Problemas (4 problemas)

| # | Criterio | Problema del corpus | Por qué |
|---|----------|--------------------|-|
| P1 | L2, algebra | `math_00XXX` (seleccionar el que C1 muestre split más trivial) | Caso donde el baseline claramente falla |
| P2 | L3, geometry | — | Estructura visual → candidato natural SPLIT-A |
| P3 | L4, number_theory | — | Estructura algebraica compleja → candidato SPLIT-G |
| P4 | L5, counting_and_probability | — | Multi-fase natural → candidato CPP-DEEP |

**Método de selección:** Para P1, elegir el problema donde la conversación C1 existente tiene la secuencia más corta (menos turnos = split más trivial). Para P2-P4, elegir los problemas con PISA_global más bajo en C1 (más potencial de mejora).

### Variables y Mediciones

| Variable | Tipo | Instrumento | Valor esperado (H5, H6) |
|----------|------|-------------|------------------------|
| CDI real | Continua [0,1] | Auto-anotación LLM (12-bit vector) | C1<C2≈C3<C4<C5 |
| PISA_global | Continua | Scorer GPT-5.4-mini | C1<C2≈C3<C4<C5 |
| ATC_SR | Continua | Scorer GPT-5.4-mini | C1≪C2≈C3<C4<C5 |
| ATC_global | Continua | Scorer GPT-5.4-mini | ídem |
| accuracy | Binaria | is_correct() | Sin hipótesis clara |
| SQS | Continua [0,1] | Constitutional critic output | C1<C2<C3≈C5 |
| n_monitor_interventions | Entero | Monitor log | C4>C5 (si H cierta) |
| total_turns | Entero | Conversación | C5>C4>C2≈C3>C1 |
| n_constitutional_iter | Entero | Pipeline log | indicador de dificultad |

### Análisis

**Cualitativo (prioritario en el piloto):**
1. Leer las 20 conversaciones (4 problemas × 5 condiciones)
2. Identificar el primer turno donde la colaboración se vuelve genuina (o no lo hace)
3. Mapear manualmente 3 conversaciones al CPP real (anotar qué celdas se activan)
4. Evaluar: ¿el monitor en C4/C5 mejora o interrumpe la naturalidad?

**Cuantitativo:**
1. Tabla 4×5 de CDI, PISA_global, ATC_SR, SQS
2. Correlación CDI×PISA, CDI×ATC_SR por condición
3. Comparación de medias C1 vs C5 (aunque n=4 no permite inferencia, sí orienta escala)

**Criterio de éxito del piloto:**
- Al menos 3 de 4 problemas muestran CDI(C5) > CDI(C1) + 0.3
- Al menos 3 de 4 problemas muestran PISA_global(C5) > PISA_global(C1) + 2.0
- Las conversaciones C5 son cualitativa y claramente más ricas que C1

---

## Herramienta de Anotación CPP (Auto-Validación)

```
SYSTEM PROMPT — CPP ANNOTATOR:

Eres un experto en la matriz PISA 2015 CPS (4 procesos × 3 competencias = 12 celdas).

Lee esta conversación entre agentes LLM resolviendo un problema matemático.
Para CADA una de las 12 celdas, determina si la colaboración fue NECESARIA en esa celda 
(es decir, ningún agente pudo completar esa operación sin input del otro).

Celdas a evaluar:
A1: ¿Necesitaron descubrir perspectivas/habilidades del otro para avanzar?
A2: ¿Establecieron juntos las normas de interacción?
A3: ¿Los roles emergieron de exploración conjunta?
B1: ¿Negociaron explícitamente cómo representar el problema?
B2: ¿Identificar las sub-tareas requirió contribución de ambos?
B3: ¿Negociaron la distribución del trabajo en la ejecución?
C1: ¿Comunicaron las acciones a realizar antes de ejecutarlas?
C2: ¿Hay pasos de ejecución que requirieron input del otro?
C3: ¿Siguieron reglas de participación / se promovieron mutuamente?
D1: ¿Monitorearon y repararon el entendimiento compartido?
D2: ¿Evaluaron conjuntamente el éxito de las acciones?
D3: ¿Adaptaron roles u organización durante la conversación?

Para cada celda, responde: 1 (colaboración necesaria) o 0 (no necesaria o ausente).

Conversación: {conversation_text}

Output JSON:
{
  "cpp_vector": [A1, A2, A3, B1, B2, B3, C1, C2, C3, D1, D2, D3],
  "cdi": float,
  "rationale": {"A1": "...", ...}  // 1-2 frases por celda
}
```

---

## Escalamiento a Sapelo (600 problemas)

| Etapa | Descripción | Costo estimado | Tiempo estimado |
|-------|-------------|---------------|-----------------|
| Piloto local | 4 problemas × 5 condiciones = 20 conv | $2-5 | 1-2 horas |
| Validación | Revisar piloto, ajustar prompts | — | 2-4 horas humano |
| Escala media | 30 problemas × 5 condiciones | $15-30 | 4-6 horas |
| Sapelo full | 600 problemas × C1+C5 (2 condiciones) | $50-100 (OpenAI) | 8-12 horas |

Para Sapelo, usar solo C1 (baseline ya ejecutado) y C5 (la condición más rica) para maximizar el contraste con el menor costo. Las condiciones intermedias C2-C4 se estudian en el piloto local.

**Nota sobre vLLM:** El pipeline constitutional (C3) requiere multiple LLM calls por split. Si se usa vLLM para el simulator/scorer, los prompts del crítico y revisor deben añadirse al routing en `openai_utils.py` (también routeados al modelo local).

---

## Módulos a Implementar

| Módulo | Archivo | Estado |
|--------|---------|--------|
| Splitter C2 | `research/splitting/splitter.py` → añadir `split_cpp_targeted()` | ⬜ pendiente |
| Pipeline constitutional C3 | `research/splitting/constitutional.py` (nuevo) | ⬜ pendiente |
| Monitor Szewkis C4 | `research/simulation/monitor.py` (nuevo) | ⬜ pendiente |
| Simulador con monitor C4 | `research/simulation/simulator.py` → añadir `simulate_with_monitor()` | ⬜ pendiente |
| CPP annotator | `research/scoring/cpp_annotator.py` (nuevo) | ⬜ pendiente |
| Experimento piloto | `research/experiments/cpp_comparison.py` (nuevo) | ⬜ pendiente |

---

## Preguntas Abiertas del Panel (para próxima iteración)

1. **Pedagogo:** ¿Debería el prompt del splitter C2 incluir un ejemplo de split profundo (few-shot) o es suficiente con la descripción abstracta de las celdas?

2. **Ingeniero IA:** El detector de fase basado en keywords es frágil. ¿Usar un LLM para clasificar la fase (más robusto) o mantener keywords (más rápido y barato)?

3. **Teórico CPS:** ¿Cómo distinguir en la anotación CPP entre una celda "activa trivialmente" (el agente dice "ok, entendido" y sigue solo) y "activa genuinamente" (el agente no puede avanzar sin la respuesta del otro)?

4. **Metodólogo:** ¿Deberíamos agregar una condición C0 (colaboración sin split — ambos agentes tienen toda la información) para separar el efecto del split del efecto de la simulación colaborativa?

---

## Referencias Metodológicas Adicionales

Bai, Y., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv:2305.14325*.

Liang, T., et al. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv:2305.19118*.

Park, J. S., et al. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of UIST 2023*.

Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.

Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.
