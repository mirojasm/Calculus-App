# Análisis Arquitectónico: Condición C2 — Generación de Splits Dirigida por Perfil CPP
## CollabMath — Panel Multi-Agente de Diseño Técnico

**Versión:** 1.0  
**Referencia:** docs/framework_PIE_CPS.md, docs/methodology_conditions.md  
**Fecha:** Abril 2026

---

## Panel de Agentes

| Agente | Especialidad | Sesgo Principal |
|--------|-------------|-----------------|
| **Matemático Computacional (MC)** | Álgebra booleana, reticulados, teoría de información | Rigor formal; estructuras algebraicas exactas |
| **Ingeniero ML/IA (IA)** | Arquitecturas LLM, fine-tuning, RL, embeddings | Implementabilidad, costo, escalabilidad |
| **Teórico de Información (TI)** | Teoría de diseño de información, persuasión Bayesiana, causalidad | Fundamentos teóricos, garantías formales |
| **Científico Cognitivo CPS (CC)** | PISA, CSCL, cognición distribuida, Vygotsky | Validez cognitiva, qué significan las celdas realmente |
| **Metodólogo Experimental (ME)** | Validez interna, control de confounders, estadística | Diseño experimental, amenazas a la validez |

---

## Parte I: La Naturaleza Matemática del Problema

### El Espacio {0,1}^12 — Lo que Subestimamos

**MC:** El CPP es un vector binario de 12 dimensiones. Pero antes de hablar de generación, necesitamos entender qué tipo de espacio es {0,1}^12 en este contexto. No es un espacio plano de 4096 puntos equiprobables. Tiene estructura algebraica rica que debemos explotar.

**Estructura 1 — Álgebra booleana y reticulado:**

{0,1}^12 con el orden parcial ≤ (definido componentwise: u ≤ v iff ∀i: u_i ≤ v_i) es un **reticulado booleano** — la forma más regular posible de reticulado. Sus operaciones son:
```
join (v ∨ w)_i = max(v_i, w_i)   # unión de celdas activas
meet (v ∧ w)_i = min(v_i, w_i)   # intersección de celdas activas
complement ¬v_i = 1 - v_i
```

El reticulado CPP tiene **diamante máximo** como se muestra en el framework (CPP-Ø ⊆ CPP-IC ⊆ ... ⊆ CPP-FULL), pero esto es una cadena específica dentro del reticulado completo. El reticulado completo tiene 2^12 = **4096 nodos** y su **número de Catalan** de cadenas maximales es astronomicamente grande.

**Estructura 2 — La función de Möbius del reticulado:**

La función de Möbius μ(u, v) en el reticulado booleano cumple:
```
μ(u, v) = (-1)^|v-u|   si u ≤ v
μ(u, v) = 0             si u ≰ v
```
donde |v-u| = número de coordenadas donde v_i = 1 y u_i = 0.

**¿Para qué sirve?** La inversión de Möbius nos da la contribución *independiente* de cada celda. Si f(v) = CDI esperado al alcanzar el perfil v, entonces la inversión de Möbius da:
```
g(v) = Σ_{u ≤ v} μ(u,v) · f(u)
```
g(v) = contribución marginal de activar exactamente las celdas nuevas en v − su subperfiles. Esto identifica qué celdas *añaden* colaboración genuina versus cuáles son subproductos de otras.

**Estructura 3 — No todos los 4096 perfiles son cognitivamente coherentes:**

**CC:** El framework PISA no trata las 12 celdas como independientes. Existe una dependencia cognitiva de prerrequisito:
- No puede ocurrir C2 (Ejecutar plan) sin B2 (Identificar tareas) previa  
- No puede ocurrir D1 (Reparar entendimiento) sin A1 (Descubrir perspectivas) previa
- C3 (Seguir reglas de participación) requiere B3 (Describir roles) previa

Esto define una **relación de prerrequisito parcial**:
```
Prerrequisitos mínimos:
A1 ← (ninguno)
A2 ← A1
A3 ← A1, A2
B1 ← A1
B2 ← A2, B1
B3 ← A3, B2
C1 ← B1, B2
C2 ← C1, B2
C3 ← B3
D1 ← C1, A1
D2 ← C2, D1
D3 ← D2, B3
```

**MC:** Esto no es arbitrario — es una **relación de orden parcial** sobre {A1,...,D3} que define un DAG de prerrequisitos. Los perfiles CPP cognitivamente coherentes son exactamente los **upsets** (conjuntos superiores) de este DAG. El número de upsets de un DAG de 12 nodos es mucho menor que 4096. Con la estructura de prerrequisito arriba, estimamos ~200-400 perfiles coherentes.

**Implicación práctica:** El espacio de búsqueda efectivo es una fracción del espacio total. Esto hace el problema de generación dirigida mucho más tractable.

---

## Parte II: El Problema Central — Asimetría de Información como Causa

### La Teoría de Diseño de Información de Kamenica-Gentzkow

**TI:** Propongo reencuadrar C2 completamente. No es un problema de *prompt engineering*. Es un problema de **diseño de mecanismos** bajo información asimétrica.

El framework de Kamenica & Gentzkow (2011) y Bergemann & Morris (2019) demuestra:

> **Teorema BCE (Bergemann-Morris 2016):** Una regla de decisión (behavior rule) es alcanzable por *alguna* estructura de información si y solo si es un **Bayes Correlated Equilibrium (BCE)**.

Traduciendo a nuestro problema:
- **Sender** = diseñador del split (nosotros)
- **Receivers** = los n agentes LLM
- **State of the world** = la estructura completa del problema matemático
- **Signals** = los paquetes de información I_A, I_B (el split)
- **Actions** = los comportamientos colaborativos (discurso, pasos matemáticos)
- **Target behavior** = el perfil CPP objetivo t ∈ {0,1}^12

El teorema BCE dice: el perfil CPP t es **alcanzable** por algún split si y solo si t es un BCE de la situación de colaboración con los paquetes correctos.

**El resultado de concavificación (K&G 2011):** La utilidad máxima alcanzable por el sender es el **envelope cóncavo** de su función de utilidad evaluada en el prior. Esto da:

1. Un **test de alcanzabilidad**: ¿está t en el conjunto de BCE? Si no, ningún split puede inducirlo.
2. Una **caracterización LP**: el conjunto de BCE factibles es un politopo convexo. Las restricciones de obediencia son **lineales en t**.

**CC:** Esto tiene consecuencias pedagógicas profundas. Algunos perfiles CPP pueden ser *imposibles de inducir* para ciertos problemas matemáticos, independientemente de cómo diseñemos el split. Si un problema de álgebra L1 tiene estructura tan simple que ninguna partición crea interdependencia epistémica genuina, entonces CPP-DEEP es inalcanzable para ese problema. El BCE theorem nos dice exactamente cuándo esto ocurre.

**TI:** Y más importante: las restricciones BCE son lineales en la estructura de información. Esto significa que podemos **optimizar directamente** sobre el espacio de splits usando programación lineal — en principio, sin necesidad de LLMs.

---

## Parte III: La Teoría de Cognición Distribuida como Tabla de Correspondencias

### Celdas PISA → Tipo de Asimetría de Información

**CC:** Hay una observación que está ausente en toda la literatura y que es la pieza más valiosa para este problema.

Hutchins (1995) y Stahl (2006) establecen que los procesos cognitivos colaborativos emergen de la **propagación de estados representacionales** — no de intenciones individuales. La cognición ocurre en el *gaps* de conocimiento que obligan a la externalización.

De este principio, podemos derivar un **mapeo directo** entre cada celda PISA y el tipo de asimetría de información que la activa:

| Celda | Nombre PISA | Tipo de Asimetría Requerida | Condición Formal |
|-------|-------------|------------------------------|-----------------|
| **A1** | Descubrir perspectivas/habilidades | Los agentes NO conocen las capacidades del otro *a priori* | H(X_A \| I_A) > 0 Y H(X_B \| I_B) > 0 donde X = "qué puede hacer el otro" |
| **A2** | Descubrir normas de interacción | Los scripts de colaboración son distintos o ausentes | No existe un protocolo de comunicación compartido en los paquetes |
| **A3** | Comprender roles emergentes | Los roles no están pre-asignados en ningún paquete | Los paquetes contienen información que *implica* un rol pero no lo nombra |
| **B1** | Construir representación compartida | Los agentes tienen representaciones distintas del mismo objeto | I_A y I_B dan acceso a la misma entidad por vías distintas (algebraica vs. geométrica) |
| **B2** | Identificar tareas necesarias | La lista completa de sub-tareas requiere ambos paquetes | Ningún agente puede enumerar todos los pasos sin información del otro |
| **B3** | Describir roles en la ejecución | Existe ambigüedad genuina en quién ejecuta qué | Múltiples distribuciones de trabajo válidas son posibles dado los paquetes |
| **C1** | Comunicar acciones antes de ejecutar | El agente A debe anunciar su próximo paso para que B lo valide | El output del paso i del agente A es *input necesario* para que B valide el paso i |
| **C2** | Ejecutar el plan acordado | El cálculo de B depende del resultado intermedio de A | Output(A, step k) = Input(B, step k+1) — dependencia matemática directa |
| **C3** | Seguir reglas de participación | Hay reglas explícitas de turno o contribución | Hay un mecanismo de verificación de contribución en el diseño |
| **D1** | Reparar entendimiento compartido | Las interpretaciones divergen durante la ejecución | Ambigüedad semántica deliberada que emerge durante el proceso |
| **D2** | Monitorear éxito de acciones | Los criterios de evaluación están distribuidos | A tiene criterio de "es correcto", B tiene criterio de "es suficiente" |
| **D3** | Adaptar organización/roles | Un evento mid-problem requiere re-distribución | El problema tiene una "sorpresa estructural" que invalida el plan inicial |

**ME:** Esta tabla es el hallazgo más importante del análisis. Es el **mecanismo causal** entre diseño del split y activación de celda. Sin esta tabla, estamos haciendo prompt engineering. Con esta tabla, podemos diseñar splits algorítmicamente.

---

## Parte IV: Los 5 Enfoques Arquitectónicos

### Comparación preliminar

| Arquitectura | Garantías formales | Costo | Velocidad | Requiere datos | Generalización |
|--------------|-------------------|-------|-----------|---------------|----------------|
| A. Diseño Inverso Algorítmico | Altas (CBC/LP) | Bajo | Rápido | No | Alta |
| B. FUDGE / Steering por Discriminadores | Medias (probabilísticas) | Medio | Medio | Sí (n≥50) | Media |
| C. GRPO + LLM pequeño condicionado | Medias (RL convergencia) | Alto (entrenamiento) | Rápido (inferencia) | Sí (n≥100) | Alta |
| D. CPP-VAE Generativo | Bajas (latent space) | Alto (entrenamiento) | Rápido (inferencia) | Sí (n≥150) | Alta |
| E. FCA + Reticulado + Retrieval | Altas (FCA exacto) | Bajo | Rápido | Sí (n≥30) | Media |

---

### Arquitectura A: Diseño Inverso Algorítmico

**TI:** Esta es la arquitectura más fundamentada teóricamente y menos obvia de proponer.

**Principio:** Dado el mapeo directo celda → asimetría de información (Parte III), el problema de generación de splits *no requiere un LLM para la parte de diseño*. El LLM solo se necesita para la *traducción a lenguaje natural*.

**Pipeline:**

```
ETAPA 1: ANÁLISIS DEL PROBLEMA (LLM grande)
  Input: problema matemático P
  Output: Grafo semántico G(P) = {entidades, relaciones, operaciones, sub-problemas}
  Modelo: GPT-4.1 (solo una vez)

ETAPA 2: ASIGNACIÓN DE ASIMETRÍAS (Algoritmo)
  Input: target CPP t ∈ {0,1}^12, grafo G(P)
  For each celda i donde t_i = 1:
    Consultar tabla Celda → Asimetría (Parte III)
    Generar restricción de partición: R_i(I_A, I_B)
  Solver: SAT/ILP sobre el conjunto de restricciones {R_i}
  Output: Especificación formal de la partición (I_A_spec, I_B_spec)

ETAPA 3: RESOLUCIÓN DE CONSISTENCIA
  Verificar que la partición cumple BCE:
    Test: ¿existe u(t) tal que t es BCE del juego (I_A_spec, I_B_spec, u)?
  Si no: relajar una restricción (las de menor prioridad según Möbius)
  Si sí: continuar

ETAPA 4: TRADUCCIÓN A LENGUAJE NATURAL (LLM pequeño)
  Input: (I_A_spec, I_B_spec) en formato formal
  Output: split JSON con shared_context, packets, agent_roles
  Modelo: Qwen2.5-7B (barato, rápido)

ETAPA 5: VALIDACIÓN PREDICTIVA (clasificador CPP forward)
  Input: split JSON generado
  Output: CPP_predicted ∈ {0,1}^12
  Verificar: Hamming(CPP_predicted, t) ≤ 2
  Si no: volver a Etapa 2 con restricciones ajustadas
```

**Ventajas:**
- La parte de "inteligencia" (diseño) es algorítmica con garantías formales
- El LLM se usa solo para traducción — fácil de verificar y controlar
- Costo: 1 llamada GPT-4.1 + 1 llamada Qwen2.5-7B por split

**Limitación:**
- El solver SAT/ILP requiere que la tabla de asimetrías esté completa y sea correcta
- Funciona mejor cuando el grafo semántico del problema es rico

**IA:** El solver de la Etapa 2 puede implementarse con z3 (theorem prover) o como un problema SAT booleano con 12 variables binarias (celdas) y constraints sobre las 2n elementos de la partición. z3 puede manejar esto en milisegundos.

---

### Arquitectura B: FUDGE Adaptado — Steering por Discriminadores CPP

**IA:** Esta es la versión más directamente inspirada en la literatura de generación controlada (Yang & Klein 2021).

**Principio:** En lugar de modificar el modelo generador, entrenamos 12 clasificadores externos que predicen, dado un prefijo de split parcial, la probabilidad de que cada celda CPP sea activada en la conversación resultante. Estos clasificadores guían la decodificación token por token.

**Pipeline:**

```
FASE DE ENTRENAMIENTO (una vez):
  Dataset: {(split_text_j, cpp_vector_j)} para j = 1...N (corpus CollabMath)
  For each celda i en 1..12:
    Entrenar discriminador D_i:
      D_i(prefix) → P(cell_i = 1 | texto del split termina siendo prefix + ...)
    Arquitectura: DeBERTa-v3-base fine-tuned (multilabel) o LSTM ligero
    Loss: BCEWithLogitsLoss

FASE DE GENERACIÓN (cada split):
  Input: (problema P, target CPP t)
  LLM base: Qwen2.5-7B con prompt "Diseña un split jigsaw para..."
  
  En cada paso de decodificación:
    for each token candidato v en top-K:
      guidance_score(v) = Σ_{i: t_i=1} λ_i · log D_i(prefix + v)
                        - Σ_{i: t_i=0} μ_i · log(1 - D_i(prefix + v))
    adjusted_logits = lm_logits + α · guidance_score
    
  α = parámetro de "fuerza del steering" (calibrar en validación)
  λ_i = peso de activar celda i (= 1.0 para celdas target)
  μ_i = peso de evitar celda i (= 0.5 para celdas no-target)
```

**Por qué 12 discriminadores en lugar de un solo clasificador multi-label:**

**MC:** Hay una razón matemática. Si las 12 celdas fueran independientes, un clasificador conjunto daría lo mismo que 12 separados. Pero no lo son — tienen la relación de prerrequisito del DAG que definimos. Los discriminadores FUDGE independientes pueden entrar en conflicto cuando los prerrequisitos no se respetan. La solución: usar un **discriminador en cadena** que respeta el DAG de prerrequisitos:

```
D_chain(prefix, cell_order=[A1,A2,...,D3]):
  P(A1=1 | prefix)
  P(A2=1 | prefix, A1_hat)     # condicionado en predicción A1
  P(A3=1 | prefix, A1_hat, A2_hat)
  ...
  P(D3=1 | prefix, all_previous)
```

Este discriminador en cadena respeta la estructura causal del DAG y evita predicciones incoherentes (e.g., predecir C2=1 con A1=0, que viola el prerrequisito).

**Ventajas:**
- No requiere re-entrenar el LLM generador
- Los discriminadores pueden entrenarse con n≥50 ejemplos (eficiente)
- Control granular por celda en tiempo de inferencia

**Limitación:**
- Los discriminadores trabajan sobre texto parcial — pueden ser poco informativos en el inicio de la generación
- El parámetro α de steering requiere calibración

---

### Arquitectura C: GRPO Fine-tuning de LLM Pequeño Condicionado en CPP

**IA:** Esta es la arquitectura con mejor relación costo/beneficio a largo plazo, especialmente para el escenario de 600 problemas en Sapelo2.

**Principio:** Fine-tunear Qwen2.5-7B con GRPO (Group Relative Policy Optimization, DeepSeek 2025) donde el reward es el vector de activación CPP resultante de la conversación simulada.

**El insight de GRPO para CPP:**

GRPO elimina el critic model al usar ventaja relativa dentro del grupo:
```
A(split_i) = (R(split_i) - mean(R_group)) / std(R_group)
```
Para un grupo de K=8 splits generados para el mismo problema:
- R(split_i) = CPP match score = Σ_j t_j · cpp_j(i) - Σ_j (1-t_j) · cpp_j(i)
- donde cpp_j(i) ∈ {0,1} es si la celda j fue activada en la simulación del split i

**Ventaja clave sobre PPO:** La ventaja GRPO ya está normalizada por el grupo, por lo que no necesitamos un modelo crítico separado. Para un problema multi-objetivo (12 celdas), GRPO es mucho más estable que PPO.

**Pipeline de entrenamiento:**

```
DATOS: n=150 ejemplos del corpus CollabMath + n=N generados sintéticamente

PRE-TRAINING (SFT):
  Dataset: {(problema, target_CPP, split_óptimo)} — solo los splits con CDI > 0.5
  Formato del prompt:
    "CPP_TARGET: [0,1,1,1,1,0,1,1,0,1,0,0]\n"
    "PROBLEM: {texto del problema}\n"
    "Generate a jigsaw split that activates ONLY the cells marked 1 above."
  Output: split JSON
  Modelo: Qwen2.5-7B-Instruct, LoRA r=16

RLHF CON GRPO:
  For each problema P_i en el dataset:
    For k = 1..8:
      split_k = policy(P_i, target_CPP)
      conversation_k = simulate(split_k)
      cpp_k = annotate(conversation_k)         # el CPP annotator
      R_k = Σ_j t_j · cpp_k[j] - Σ_j (1-t_j) · cpp_k[j]
    A_k = (R_k - mean(R)) / std(R)
    Update policy by gradient ascent on Σ_k A_k · log π(split_k | P_i, target_CPP)

RESULTADO: Modelo que dado (problema, 12-bit target), genera split óptimo
```

**MC:** Hay una variante más elegante usando la estructura del reticulado CPP. En lugar de un reward escalar, podemos usar el **orden de Möbius** para ponderar la reward:

```
R_Möbius(split) = Σ_{v: v ≤ cpp_achieved} μ(0, v) · CDI(v)
```

Esto penaliza más activar celdas que son subproductos de otras y recompensa más las celdas "profundas" que requieren colaboración genuina adicional.

**Ventajas:**
- Después del fine-tuning, la generación es rápida (1 forward pass de Qwen2.5-7B)
- El modelo aprende generalización sobre el espacio de problemas matemáticos
- Costo de entrenamiento: 1 GPU × 24h en Sapelo2

**Limitación:**
- El outer loop de GRPO requiere simular las conversaciones — caro en el entrenamiento (necesita vLLM para velocidad)
- El reward requiere una simulación completa + CPP annotation por split generado

---

### Arquitectura D: CPP-VAE — Espacio Latente Generativo Condicionado

**IA:** Inspirado en los Sparse Autoencoders de Anthropic (2023/2024) pero aplicado al espacio de splits.

**Principio:** Aprender un espacio latente Z donde la geometría del espacio refleja la geometría del espacio CPP. Conversaciones con perfiles CPP similares → cercanas en Z. La generación de splits se convierte en: dado target CPP t, encontrar z* ≈ t, luego decodificar.

**Arquitectura:**

```
ENCODER: E(conversation_text) → (μ_z, σ_z) ∈ R^d

DECODER: Dec(z, problem_text) → split_text

CPP HEAD: H(z) → CPP_predicted ∈ [0,1]^12
  (Loss: BCE entre CPP_predicted y CPP_true anotado)

LOSS TOTAL:
  L = L_reconstruction + β · L_KL + γ · L_CPP + δ · L_disentanglement

L_disentanglement = penaliza correlación entre dimensiones de z 
                    que corresponden a celdas CPP no relacionadas
```

**Insight clave de los SAE de Anthropic:** Si entrenamos con suficientes datos, las dimensiones de z aprenden a representar *features monosemánticos* — cada dimensión ≈ una celda CPP. La decodificación entonces se convierte en exactamente "activar las dimensiones correspondientes a t".

**Me:** El espacio Z aprendido puede revelar algo profundo: ¿qué tan ortogonales son las 12 celdas CPP en el espacio de conversaciones reales? Si A1 y C2 son altamente correlacionadas en Z, significa que en la práctica no podemos activar una sin la otra — esto es información empírica valiosa sobre la estructura del problema.

**Ventajas:**
- El espacio latente *aprende* la estructura real del espacio CPP (vs. asumir independencia)
- Puede descubrir que algunas celdas son imposibles de separar
- Generación continua: interpolación entre perfiles CPP es natural

**Limitación:**
- Requiere n≥200 conversaciones anotadas para entrenamiento (aún no disponibles)
- La arquitectura VAE es más compleja de implementar correctamente

---

### Arquitectura E: Análisis de Conceptos Formales + Reticulado CPP + Retrieval

**MC:** Esta es la arquitectura menos dependiente de LLMs y más rigurosa matemáticamente.

**Principio:** Usar el corpus de observaciones (splits, conversaciones, CPP anotado) para construir el reticulado de conceptos formales del espacio CPP real. El reticulado revela el espacio de perfiles *alcanzables* y sus relaciones estructurales. Para generar un split con target t, recuperar el ejemplo más cercano en el reticulado y modificarlo.

**Construcción del contexto formal:**

```
Objetos: {(split_j, conversation_j)} para j = 1..N  
Atributos: {CPP_cells 1..12} + {problem_features: subject, level, n_steps, ...}

Contexto formal K = (Objects, Attributes, IncidenceRelation):
  (object_j, cell_i) ∈ K ↔ cell_i estaba activado en conversation_j
```

**El reticulado de conceptos formales** resultante tiene:
- **Extensiones** (extents): conjuntos de conversaciones que comparten todas las celdas de su concepto
- **Intenciones** (intents): conjuntos de celdas que co-ocurren en todas las conversaciones de su extensión
- **Reglas de implicación**: si celda X activa → celda Y siempre activa (en el corpus)
- **Reglas de incompatibilidad**: si celda X activa → celda Z nunca activa

**CC:** Las reglas de implicación son empíricamente las más valiosas. Si el corpus muestra "C2 → A1 siempre" (coactivación 100%), eso confirma la teoría PISA (no se puede ejecutar en colaboración sin antes compartir información). Las reglas de incompatibilidad podrían revelar conflictos teóricos no anticipados.

**Pipeline de generación:**

```
OFFLINE (una vez por versión del corpus):
  1. Construir contexto formal K
  2. Calcular reticulado L(K)
  3. Mapear cada concepto a su centroide en el embedding space de splits

ONLINE (cada split):
  1. Dado target CPP t, encontrar el concepto C* más cercano en L(K)
     (más cercano = maximiza |t ∩ intent(C*)| / |t ∪ intent(C*)|, Jaccard)
  2. Recuperar los k=3 splits del extent(C*) más similares al problema actual
  3. Few-shot: pasar los 3 splits recuperados como ejemplos al LLM generador
  4. LLM genera un nuevo split condicionado en los 3 ejemplos + target CPP descripción
```

**Ventajas:**
- Explota la estructura exacta del corpus sin suposiciones paramétricas
- Las reglas de implicación del reticulado son un prior teóricamente informado
- Few-shot con ejemplos reales es mucho más fiable que instrucciones abstractas

**Limitación:**
- Requiere corpus anotado con CPP (lo que el annotator genera)
- El reticulado puede ser pequeño si N es pequeño (n=150 da reticulado útil pero pequeño)

---

## Parte V: La Arquitectura Óptima y su Implementación

### Debate Final del Panel

**TI:** Mi posición es clara: la Arquitectura A (diseño inverso algorítmico) tiene el fundamento teórico más sólido. El BCE theorem nos garantiza que si el target CPP es alcanzable, podemos encontrar la partición óptima mediante LP. No necesitamos ML para el diseño — solo para la traducción.

**CC:** Comparto el fundamento pero tengo una objeción: el grafo semántico del problema matemático (Etapa 1) es difícil de construir correctamente con el conocimiento actual. El problema de álgebra "encuentra la ecuación" tiene una estructura semántica muy diferente de "calcula el área", y el LLM puede equivocarse en la extracción. Si el grafo es incorrecto, todo el pipeline falla.

**IA:** Mi propuesta es la Arquitectura C (GRPO fine-tuning) para el largo plazo. Pero para el piloto inmediato con n=150, usaría la Arquitectura B (FUDGE discriminadores) porque no requiere entrenamiento — los discriminadores se pueden entrenar en pocas horas con el corpus existente.

**ME:** Hay una diferencia metodológica importante entre todas estas arquitecturas. En un experimento científico, necesito saber *por qué* un split activa ciertas celdas. La Arquitectura A es la única que genera una *explicación causal* del split — porque la partición fue diseñada para satisfacer restricciones específicas que mapean a celdas específicas. Las arquitecturas B-E producen splits que tal vez activen las celdas pero sin explicabilidad del mecanismo.

**MC:** La unificación matemática es esta: todas las arquitecturas B-E son aproximaciones del problema exacto que resuelve la Arquitectura A. Son necesarias porque la Arquitectura A requiere:
1. Un grafo semántico del problema (difícil de extraer con garantías)
2. Una formalización completa de las restricciones asimétrica (la tabla de Parte III, que es una hipótesis teórica, no verificada)
3. Un solver BCE (existe pero requiere implementación)

**Decisión del panel:** Arquitectura híbrida en dos etapas.

---

### Arquitectura Propuesta: CIDI (Constrained Inverse Design with Iterative Validation)

```
┌──────────────────────────────────────────────────────────────────────┐
│  CIDI: Constrained Inverse Design with Iterative Validation          │
│  Arquitectura recomendada para Condición C2                          │
└──────────────────────────────────────────────────────────────────────┘

MÓDULO 1: ANÁLISIS SEMÁNTICO DEL PROBLEMA
  Input: texto del problema P
  Extrae via LLM (GPT-4.1):
    - Entidades matemáticas: {e_1,...,e_m}
    - Relaciones: {r_1,...,r_k}
    - Sub-problemas: {sp_1,...,sp_l}
    - Tipo de razonamiento: {algebraico, geométrico, probabilístico, ...}
  Output: JSON estructurado con la "anatomía" del problema

MÓDULO 2: SELECCIÓN DE PERFIL Y VERIFICACIÓN DE ALCANZABILIDAD
  Input: target CPP t ∈ {0,1}^12, anatomía del problema
  
  2a. VERIFICACIÓN DE PRERREQUISITOS:
      For each celda i con t_i = 1:
          Verificar que todos los prerrequisitos de i también tienen t_j = 1
          Si no: añadir celda faltante (completar hacia arriba en el DAG)
      Output: t_completo = clausura de t bajo prerrequisitos
  
  2b. VERIFICACIÓN DE ALCANZABILIDAD:
      For each celda i con t_i = 1:
          Verificar que el problema tiene la estructura semántica necesaria
          (ejemplo: C2 requiere sub-problemas con dependencia de output/input)
          Si no: marcar celda como "aspiracional" vs "estructuralmente necesaria"
      Output: t_factible = subvector de t_completo que es estructuralmente alcanzable

MÓDULO 3: DERIVACIÓN DE RESTRICCIONES DE PARTICIÓN
  Input: t_factible, anatomía del problema
  
  For each celda i con t_factible_i = 1:
      Aplicar tabla Celda → Asimetría (Parte III):
          Generar restricción formal R_i(I_A, I_B)
      Añadir restricción de consistencia con anatomía del problema:
          ¿Qué elementos semánticos van en I_A? ¿Cuáles en I_B?
  
  Resolver sistema de restricciones:
      Solver: z3 o backtracking simple sobre asignación de entidades a agentes
      Objetivo: maximizar Σ_i t_i · (satisfecha R_i)
  Output: especificación formal (I_A_spec, I_B_spec)

MÓDULO 4: GENERACIÓN LINGÜÍSTICA CON RESTRICCIONES
  Input: (I_A_spec, I_B_spec), problema P, target CPP descripto en lenguaje
  Modelo: Qwen2.5-7B (fast, cheap)
  Prompt:
    "Traduce esta especificación de split a lenguaje natural para dos agentes.
     I_A contiene: {I_A_spec}.
     I_B contiene: {I_B_spec}.
     El split debe activar estos comportamientos de colaboración: {description_t}."
  Output: split JSON

MÓDULO 5: VALIDACIÓN PREDICTIVA (CADENA DE DISCRIMINADORES)
  Input: split JSON generado
  For each celda i in orden DAG (A1 → A2 → ... → D3):
      P(cell_i = 1 | split_text, predictions_{<i}) via discriminador D_i
  Output: CPP_predicted ∈ [0,1]^12
  
  Hamming_loss = Σ_i |t_i - round(CPP_predicted_i)|
  
  Si Hamming_loss > 2:
      Identificar celdas con discrepancia
      Re-ejecutar Módulo 3 con restricciones ajustadas (max 2 iteraciones)
  
  Si Hamming_loss ≤ 2:
      APROBADO → retornar split

MÓDULO 6 (FUTURO): GRPO FINE-TUNING
  Una vez que tengamos n≥100 splits CIDI generados y anotados:
  Fine-tunear Qwen2.5-7B para aprender directamente (problema, t) → split
  Eliminar Módulos 1-4, conservar Módulo 5 como validador
```

---

## Parte VI: Matemática del Espacio CPP para NeurIPS

### Las 4096 combinaciones y su estructura

Para el paper de NeurIPS 2026, la contribución matemática es demostrar que el espacio CPP {0,1}^12 tiene las siguientes propiedades empíricamente verificables:

**Propiedad 1 — Cierre de prerrequisitos:**
El conjunto de perfiles CPP alcanzables en la práctica es un **upset** del DAG de prerrequisitos. Hipótesis: el n=150 corpus tiene ~K perfiles distintos observados, todos consistentes con el DAG.

**Propiedad 2 — Rango de correlación:**
La matriz de correlación entre celdas Σ ∈ R^{12×12} tiene rango efectivo r << 12. Esto significa que el espacio CPP efectivo es ~r-dimensional, no 12-dimensional. Hipótesis: r ≈ 4-5 factores principales (uno por fila PISA).

**Propiedad 3 — Implicaciones empíricas:**
Usando FCA sobre el corpus: si celda X activa → celda Y activa en ≥90% de los casos, esto es una **implicación empírica** del reticulado. Estas implicaciones o bien confirman el DAG teórico, o lo corrigen con evidencia empírica.

**Propiedad 4 — Distribución de CDI por condición:**
Para NeurIPS: mostrar que CDI(C2_CIDI) > CDI(C2_prompt) > CDI(C1) usando el test de Wilcoxon y la d de Cohen. La arquitectura CIDI debería producir CDI's más altos *y más específicamente targetados* que el prompt engineering simple.

**Propiedad 5 — Error de targeting:**
Definir: `targeting_error(split, target_t) = Hamming(CPP_achieved, t) / 12`

Hipótesis H8 (nueva): C2_CIDI tiene targeting_error < C2_prompt < C3_constitutional.
La razón teórica: CIDI diseña *para* el target; constitutional design solo maximiza SQS (score global, no targeting).

---

## Parte VII: Implementación Prioritaria

### Qué implementar primero

Dada la restricción de tiempo (NeurIPS abstract 4 mayo, paper 6 mayo) y los recursos disponibles (corpus n=150, Sapelo2):

**Semana 1 (piloto, ya disponible):**
- C2_prompt (lo que ya está implementado — sirve como baseline dentro de C2)

**Semana 2 (CIDI partial):**
- Implementar Módulos 1 + 3 (análisis semántico + derivación de restricciones)
- Usar la tabla de Parte III directamente como heurística en el prompt (sin solver formal)
- Esto ya es mucho mejor que C2_prompt porque el prompt está *estructurado* por la tabla

**Semana 3 (CIDI con discriminadores):**
- Entrenar discriminadores en cadena sobre el corpus n=150
- Implementar Módulo 5 (validación predictiva)
- Esto cierra el loop de C2

**Largo plazo (post-NeurIPS):**
- Módulo 3 con solver z3 formal
- GRPO fine-tuning de Qwen2.5-7B (Módulo 6) en Sapelo2
- CPP-VAE cuando corpus llegue a n≥300

---

## Referencias Cruzadas

Kamenica, E., & Gentzkow, M. (2011). Bayesian Persuasion. *American Economic Review, 101*(6), 2590–2615.

Bergemann, D., & Morris, S. (2019). Information Design: A Unified Perspective. *Journal of Economic Literature, 57*(1), 44–95.

Yang, K., & Klein, D. (2021). FUDGE: Controlled Text Generation With Future Discriminators. *NAACL 2021*.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

Ganter, B., & Wille, R. (1999). *Formal Concept Analysis: Mathematical Foundations*. Springer.

Lu, X., et al. (2022). NeuroLogic A*esque Decoding: Constrained Text Generation with Lookahead Heuristics. *NAACL 2022*.

Anthropic. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. *transformer-circuits.pub*.

Hutchins, E. (1995). *Cognition in the Wild*. MIT Press.

Stahl, G. (2006). *Group Cognition: Computer Support for Building Collaborative Knowledge*. MIT Press.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

Szewkis, E., et al. (2011). Collaboration within large groups in the classroom. *International Journal of Computer-Supported Collaborative Learning, 6*(4).
