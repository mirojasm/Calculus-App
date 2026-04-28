# Framework PIE-CPS
## Profundidad de Interdependencia Epistémica en Collaborative Problem Solving Matemático

**Proyecto:** CollabMath  
**Versión:** 3.0  
**Estado:** Revisado — incorpora fundamentos de diseño de información y estructura algebraica CPP  

---

## 1. Fundamentos Teóricos

### 1.1 PISA 2015 CPS Framework — Matriz Corregida

El marco PISA 2015 para Collaborative Problem Solving (OECD, 2013, 2017) define la competencia CPS como la capacidad de un individuo de participar efectivamente en un proceso donde dos o más agentes construyen conjuntamente una solución a un problema que ninguno puede resolver de forma independiente.

**Tres competencias de colaboración (columnas):**

| # | Competencia | Descripción |
|---|-------------|-------------|
| **Comp 1** | Establecer y mantener entendimiento compartido | Construir una representación común del problema y el proceso de solución |
| **Comp 2** | Tomar acción apropiada para resolver el problema | Identificar e implementar la solución; evaluar el resultado |
| **Comp 3** | Establecer y mantener organización del equipo | Construir y mantener roles, división del trabajo, y mecanismos de coordinación |

**Cuatro procesos de resolución de problemas (filas):**

| # | Proceso | Descripción |
|---|---------|-------------|
| **A** | Explorar y comprender | Descubrir lo que cada miembro sabe; construir representación inicial |
| **B** | Representar y formular | Construir representación compartida; identificar restricciones del problema |
| **C** | Planificar y ejecutar | Establecer metas; enactar el plan de solución |
| **D** | Monitorear y reflexionar | Monitorear el progreso; reflexionar sobre el proceso y resultados |

**Las 12 celdas (4 × 3) — definiciones oficiales OECD 2013:**

|   | **Comp 1** (Entendimiento compartido) | **Comp 2** (Tomar acción) | **Comp 3** (Organización del equipo) |
|---|--------------------------------------|--------------------------|---------------------------------------|
| **A** | **A1** Descubrir perspectivas y habilidades de los compañeros de equipo | **A2** Descubrir el tipo de interacción CPS (normas de comunicación) | **A3** Comprender los roles a desempeñar en la resolución |
| **B** | **B1** Construir representación compartida y negociar el significado del problema (*common ground*) | **B2** Identificar y describir tareas a realizar | **B3** Describir roles y organización del trabajo en equipo (protocolos de comunicación) |
| **C** | **C1** Comunicar a los compañeros las acciones a realizar | **C2** Ejecutar el plan acordado | **C3** Seguir reglas de participación (p.ej. promover que otros realicen sus tareas) |
| **D** | **D1** Monitorear y reparar el entendimiento compartido | **D2** Monitorear resultados de las acciones y evaluar el éxito | **D3** Monitorear, dar retroalimentación y adaptar la organización y roles del equipo |

> **Errores en versiones anteriores del framework:** A1 no es "construir representación conjunta" (eso es B1) sino "descubrir perspectivas y habilidades". C1 no es "ejecutar el plan" (eso es C2) sino "comunicar las acciones a realizar". D1 no es "evaluar resultados" (eso es D2) sino "reparar el entendimiento compartido". A3 no es "mantener entendimiento" (eso es D1) sino "comprender roles".

**Fuentes:** OECD (2013). *PISA 2015 Draft Collaborative Problem Solving Framework*. OECD Publishing. OECD (2017). *PISA 2015 Results (Volume V): Collaborative Problem Solving*. OECD Publishing. DOI: 10.1787/9789264285521-en

---

### 1.2 ATC21S Framework

El framework Assessment and Teaching of 21st Century Skills (Griffin & Care, 2015) define la colaboración en cinco dimensiones medibles:

| Dimensión | Código | Descripción | Relación con PISA |
|-----------|--------|-------------|------------------|
| Participation & Contribution | **PC** | Contribución activa de cada agente al proceso colectivo | Comp 2 (tomar acción) |
| Communication | **C** | Calidad y efectividad del intercambio entre agentes | Comp 1 (conocimiento compartido) |
| Collaboration | **Co** | Trabajo genuinamente conjunto hacia meta común | Comp 1+2+3 integrados |
| Co-Regulation | **CR** | Un agente regula activamente el proceso del otro | Comp 3, proceso D |
| Shared Regulation | **SR** | El grupo regula colectivamente su propio proceso | D1+D2+D3 |

**Por qué ATC21S y PISA son complementarios:** PISA mide *qué operaciones cognitivo-colaborativas ocurren* (estructura del proceso). ATC21S mide *la calidad social del proceso* (con quién, cómo, cuánto). Esta distinción está validada empíricamente en el corpus CollabMath: ATC global d=+1.51 para la condición social enriquecida; PISA global d≈0.00 para la misma comparación (n=140).

**Fuentes:** Griffin, P., & Care, E. (Eds.) (2015). *Assessment and Teaching of 21st Century Skills: Methods and Approach*. Springer. DOI: 10.1007/978-94-017-9395-7. Care, E., Griffin, P., & Wilson, M. (Eds.) (2018). *Assessment and Teaching of 21st Century Skills: Research and Applications*. Springer.

---

### 1.3 Szewkis et al. (2011) — Condiciones de Actividad Colaborativa Genuina

Szewkis et al. (2011) identificaron seis condiciones necesarias para que una actividad sea genuinamente colaborativa — no solo co-presente o co-productiva:

| # | Condición | Definición | Qué falla cuando no se cumple |
|---|-----------|-----------|-------------------------------|
| **S1** | Objetivo común | Todos persiguen la misma meta explícita | Optimización local: cada agente resuelve "su parte" sin considerar el conjunto |
| **S2** | Interdependencia positiva | El éxito de cada uno depende del éxito de todos | Split trivial: uno recibe el dato y resuelve solo |
| **S3** | Responsabilidad individual | Cada agente tiene una contribución activa, necesaria y no sustituible | Un agente domina, el otro observa o duplica |
| **S4** | Recompensa grupal | El éxito o fracaso es del grupo, no individual | Competencia o free-riding |
| **S5** | Conciencia grupal (*group awareness*) | Cada agente tiene modelo actualizado de lo que hace el otro | Duplicación de esfuerzo o vacíos no detectados |
| **S6** | Coordinación y comunicación | Los agentes coordinan activamente el proceso, no solo el resultado | Ejecución en paralelo desconectada |

**Mapeado a ATC21S:** S1↔SR, S2↔Co, S3↔PC, S4↔Co, S5↔CR, S6↔C. Esto significa que si las 6 condiciones Szewkis se satisfacen, todas las dimensiones ATC21S deberían ser altas — y viceversa.

**Fuente:** Szewkis, E., Nussbaum, M., Rosen, T., Abalos, J., Denardin, F., Caballero, D., Tagle, A., & Alcoholado, C. (2011). Collaboration within large groups in the classroom. *International Journal of Computer-Supported Collaborative Learning*, 6(4), 561–575. DOI: 10.1007/s11412-011-9123-y

---

## 2. Enriquecimientos desde Didáctica y Diseño Instruccional

### 2.1 Zona de Desarrollo Próximo — Vygotsky (1978)

La colaboración tiene valor epistémico cuando los agentes tienen **ZPDs complementarias**: lo que uno no puede hacer solo, puede hacerlo con el apoyo del otro. Un split jigsaw crea ZPDs artificialmente complementarias.

**Criterio derivado — ZPD Genuineness (ZPD-G):** ¿La contribución del otro agente activa en mí comprensión matemática que no podría alcanzar solo? Un split trivial transfiere *datos*. Un split rico transfiere *capacidad de razonamiento*. La diferencia entre ZPD-G=0 (CPP-T) y ZPD-G=1 (CPP-DEEP) es la diferencia entre "me diste el número que me faltaba" y "tu perspectiva me permitió ver la estructura del problema de una forma que no podía solo."

**Fuentes:** Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher Psychological Processes*. Harvard University Press. Rojas-Drummond, S., & Mercer, N. (2003). Scaffolding the development of effective collaboration and learning. *International Journal of Educational Research*, 39(1-2), 99–111.

---

### 2.2 Teoría de las Situaciones Didácticas — Brousseau (1997)

Una "situación a-didáctica" de calidad obliga al estudiante a construir conocimiento por la presión de la situación misma, sin intervención directa del docente. En CPS, el **milieu** (el medio que devuelve feedback) ES el colaborador.

**Mapeado al framework:** El split bien diseñado crea un milieu colaborativo donde el feedback del partner es la presión que fuerza el razonamiento conjunto:
- S2 (interdependencia positiva) ↔ adidacticidad: sin el otro no se puede avanzar
- S5 (conciencia grupal) ↔ milieu retroalimentador: el partner detecta errores que el agente solo no ve
- CPP alto ↔ situación genuinamente a-didáctica: el problema mismo obliga la colaboración

**Fuente:** Brousseau, G. (1997). *Theory of Didactical Situations in Mathematics*. Kluwer Academic Publishers. DOI: 10.1007/0-306-47211-2

---

### 2.3 Teoría de Carga Cognitiva — Kirschner, Sweller & Clark (2006)

La colaboración tiene valor neto positivo solo cuando: **beneficio cognitivo (distribución del razonamiento) > costo de coordinación (overhead comunicativo)**.

Para principiantes (problemas fáciles), el costo de coordinación supera el beneficio → la colaboración puede empeorar el rendimiento. Para expertos (problemas difíciles), el beneficio supera el costo → la colaboración mejora el rendimiento.

**Predicción derivada — CDI óptimo por nivel:**

```
CDI óptimo(nivel) ≈ f(dificultad) — función creciente

nivel 1-2: CDI_óptimo bajo  (~0.2-0.3)
nivel 3:   CDI_óptimo medio (~0.4-0.5)
nivel 4-5: CDI_óptimo alto  (~0.6-0.8)
```

Esto explica el hallazgo preliminar: ventaja colaborativa ≈ 0 para Level 1-3 con splits CPP-T (CDI~0.08). El problema no es que la colaboración no ayude — es que el CDI actual está muy por debajo del óptimo para cualquier nivel.

**Fuente:** Kirschner, P. A., Sweller, J., & Clark, R. E. (2006). Why minimal guidance during instruction does not work. *Educational Psychologist*, 41(2), 75–86. DOI: 10.1207/s15326985ep4102_1

---

### 2.4 Common Ground — Clark & Schaefer (1989)

La colaboración requiere establecer *common ground* — conocimiento mutuo verificado intersubjetivamente. Esto es costoso pero necesario. Sin common ground, los agentes operan desde representaciones distintas del mismo problema.

**Mapeado a PISA:** La celda B1 ("construir representación compartida y negociar el significado") ES el proceso de common ground establishment. La celda D1 ("monitorear y reparar el entendimiento compartido") es su mantenimiento continuo.

**Implicación crítica:** Los splits que activan B1 y D1 crean condiciones para common ground genuino. Los splits CPP-T (solo A1 activo) saltan el common ground — los agentes intercambian datos sin construir representación compartida del problema. Esto es correlato del hallazgo PISA: comp1 en jigsaw_2 es solo 41% (la mayoría es intercambio de datos, no construcción de significado).

**Fuente:** Clark, H. H., & Schaefer, E. F. (1989). Contributing to discourse. *Cognitive Science*, 13(2), 259–294. DOI: 10.1207/s15516709cog1302_7. Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In L. B. Resnick et al. (Eds.), *Perspectives on Socially Shared Cognition* (pp. 127–149). APA.

---

### 2.5 Modelo 4C/ID — van Merriënboer et al. (2002)

El aprendizaje de tareas complejas requiere: práctica de tareas completas (*whole-task*), apoyo a sub-tareas, y coordinación explícita. El jigsaw tradicional viola esto al dar a cada agente solo su parte. Los perfiles CPP altos lo restauran: ambos trabajan en la tarea completa con roles complementarios.

**Implicación:** CPP-FULL corresponde al ideal 4C/ID de whole-task collaborative practice. CPP-T corresponde a part-task division que puede inhibir la transferencia de aprendizaje.

**Fuente:** van Merriënboer, J. J. G., Clark, R. E., & de Croock, M. B. M. (2002). Blueprints for complex learning: The 4C/ID-model. *Educational Technology Research and Development*, 50(2), 39–61. DOI: 10.1007/BF02504993

---

### 2.6 Memoria Transactiva — Wegner (1987)

Los grupos desarrollan sistemas de memoria distribuida donde cada miembro se especializa en recordar ciertos tipos de información. Un split jigsaw ES un sistema de memoria transactiva artificialmente construido.

**Implicación:** La calidad del split determina si el sistema de memoria transactiva es *funcional* (los miembros saben qué sabe el otro y pueden acceder a esa información eficientemente — S5) o *disfuncional* (los miembros no saben exactamente qué sabe el otro, produciendo redundancia o vacíos).

**Fuente:** Wegner, D. M. (1987). Transactive memory: A contemporary analysis of the group mind. In B. Mullen & G. R. Goethals (Eds.), *Theories of Group Behavior* (pp. 185–208). Springer.

---

### 2.7 Cognición Distribuida — Hutchins (1995)

La cognición no ocurre solo en mentes individuales sino distribuida entre agentes y artefactos. El sistema multi-agente CollabMath es una implementación computacional de cognición distribuida: el "sistema cognitivo" es el conjunto de agentes + el split de información + el protocolo de comunicación.

**Implicación para el CDI:** CDI mide cuán distribuida es la cognición del sistema. CDI≈0: cognición individual con entrega de datos. CDI≈1: cognición genuinamente distribuida donde ningún nodo puede resolver por sí solo.

**Fuente:** Hutchins, E. (1995). *Cognition in the Wild*. MIT Press.

---

## 3. El Framework Integrado PIE-CPS

### 3.1 Definiciones Formales

**Definición 1 — Collaborative Phase Profile (CPP):**

Sea P un problema matemático y S = (I₁, ..., Iₙ) un split en n paquetes de información. El CPP es:

```
CPP(P, S) = (c_{A1}, c_{A2}, ..., c_{D3}) ∈ {0,1}^12
```

donde `c_{ij} = 1` si y solo si en la celda (fase i, competencia j) existe al menos un paso de razonamiento que ningún agente puede completar sin input activo del otro en esa misma celda.

**Definición 2 — CPS Depth Index (CDI):**

```
CDI(P, S) = Σ c_{ij} / 12  ∈ [0, 1]
```

CDI = 0: ninguna celda requiere colaboración (problema individual).
CDI = 1: todas las celdas requieren colaboración (interdependencia total).

**Definición 3 — Interdependencia Epistémica (IE) en ronda t:**

```
IE_t = 1 - max_i P(solución correcta | agente_i conoce todo lo compartido hasta ronda t)
```

IE_t = 0: un agente puede resolver solo con la información acumulada → CPS superficial.
IE_t > 0 en múltiples rondas consecutivas → CPS genuino.
IE_t > 0.5 hasta la ronda final → interdependencia profunda.

**Definición 4 — ZPD Genuineness (ZPD-G):**

```
ZPD-G = 1 si E[comprensión(agente_i | colaboración)] > E[comprensión(agente_i | solo)]
          para al menos k fases PISA
ZPD-G = 0 si la colaboración solo transfiere datos sin ampliar comprensión
```

**Definición 5 — Calidad Szewkis Sostenida (SQS):**

```
SQS(conversación) = (1/4) Σ_{fase ∈ {A,B,C,D}} (1/6) Σ_{j=1}^{6} S_j(fase)
```

donde S_j(fase) ∈ {0,1} indica si la condición Szewkis j se satisface en la fase indicada.
SQS = 1: todas las 6 condiciones se mantienen en todas las 4 fases → colaboración genuina sostenida.

---

### 3.2 El Framework de Tres Capas × Tres Dimensiones

```
                   DIM. ESTRUCTURAL      DIM. SOCIAL            DIM. COGNITIVA
                   (matemática)          (colaborativa)         (razonamiento)
               ┌─────────────────────┬────────────────────┬──────────────────────┐
CAPA 1         │ CPP(P,S) ∈ {0,1}^12 │ SQS objetivo       │ Split pattern A-G    │
DISEÑO         │ CDI objetivo         │ 6 Szewkis en diseño│ Dificultad L1-5      │
(a priori)     │ Perfil lattice       │ ZPD complementarity│ Subject domain       │
               ├─────────────────────┼────────────────────┼──────────────────────┤
CAPA 2         │ PISA codes (A1-D3)  │ SQS real (por fase)│ Common ground depth  │
PROCESO        │ CDI real             │ ATC21S (5 dims)    │ Cognitive load dist. │
(in situ)      │ IE_t por ronda      │ 6 Szewkis dinámica │ ZPD activation       │
               ├─────────────────────┼────────────────────┼──────────────────────┤
CAPA 3         │ CDI alcanzado        │ ATC_SR, ATC_CR     │ Accuracy             │
RESULTADO      │ PISA global          │ ATC global         │ Solution completeness│
(a posteriori) │ Phase coverage       │ Participación equ. │ ZPD-G score          │
               └─────────────────────┴────────────────────┴──────────────────────┘
```

**La hipótesis central del framework:**

> Un split de alta calidad — definido por CDI alto, SQS≈1, y ZPD-G=1 — produce un proceso con alto PISA_global y altos ATC_SR/CR. La accuracy no es función directa del CDI sino mediada por la dificultad del problema: existe un CDI_óptimo(nivel) que maximiza el producto accuracy × PISA_global. Por debajo del óptimo: baja calidad colaborativa. Por encima: coordination overload.

---

### 3.3 Estructura Algebraica del Espacio CPP: El Reticulado Booleano

El espacio CPP {0,1}^12 no es un conjunto plano de 4096 puntos equiprobables. Tiene estructura algebraica rica que fundamenta el diseño computacional de splits.

**El reticulado booleano:**

{0,1}^12 con el orden parcial componentwise (u ≤ v iff ∀i: u_i ≤ v_i) es un reticulado booleano completo con operaciones:
```
join: (v ∨ w)_i = max(v_i, w_i)   — unión de celdas activas
meet: (v ∧ w)_i = min(v_i, w_i)   — intersección
complemento: ¬v_i = 1 − v_i
```

**La función de Möbius del reticulado:**

En el reticulado booleano, la función de Möbius es:
```
μ(u, v) = (−1)^|v − u|   si u ≤ v
μ(u, v) = 0               si u ≰ v
```
donde |v − u| es el número de coordenadas donde v_i = 1 y u_i = 0. La inversión de Möbius da la contribución *independiente* de cada perfil: si f(v) es el CDI esperado del perfil v, entonces g(v) = Σ_{u≤v} μ(u,v)·f(u) mide cuánto CDI adicional aportan las celdas *nuevas* en v respecto a todos sus subperfiles. Esto permite identificar qué celdas añaden colaboración genuina versus cuáles son subproductos de otras.

**DAG de prerrequisitos cognitivos:**

Las 12 celdas PISA no son independientes. Existen dependencias de prerrequisito derivadas de la secuencia cognitiva de resolución de problemas:

```
A1 ←── (ninguno — punto de entrada)
A2 ←── A1
A3 ←── A1, A2
B1 ←── A1
B2 ←── A2, B1
B3 ←── A3, B2
C1 ←── B1, B2
C2 ←── C1, B2
C3 ←── B3
D1 ←── C1, A1
D2 ←── C2, D1
D3 ←── D2, B3
```

Este DAG define una relación de orden parcial. Los perfiles CPP cognitivamente coherentes son los **upsets** de este DAG: conjuntos cerrados bajo la operación "si celda i está activa, todos sus prerrequisitos también deben estar activos". El número de upsets de este DAG específico es aproximadamente 200-400, reduciendo el espacio de búsqueda efectivo desde 4096 a ~5-10% del total.

**Implicación para el diseño:** Antes de intentar generar un split que active un target CPP t, se debe verificar que t es un upset del DAG. Si no lo es, se completa hacia el upset mínimo que contiene t.

**El espacio efectivo:**

La matriz de correlación entre celdas Σ ∈ R^{12×12} calculada sobre el corpus tiene rango efectivo r << 12. Hipótesis empírica: r ≈ 4-5 factores principales, uno por fila PISA (cada fila comparte un proceso cognitivo). Esto significa que el espacio CPP efectivo es ~4-5 dimensional, no 12 dimensional, lo que hace la optimización mucho más tractable.

---

### 3.4 Diseño de Información como Fundamento Teórico de C2

**El Teorema BCE (Bergemann & Morris 2016):**

La generación de splits dirigida por perfil CPP es un problema de diseño de información en el sentido de Kamenica & Gentzkow (2011) y Bergemann & Morris (2019).

**Formalización:**
- Sender = diseñador del split (sistema CollabMath)
- Receivers = agentes LLM
- State of the world = estructura completa del problema matemático Ω
- Signals = paquetes de información (I_A, I_B) — el split
- Actions = comportamientos colaborativos en la conversación
- Target behavior rule = perfil CPP objetivo t ∈ {0,1}^12

**Teorema:** Un perfil CPP t es alcanzable por algún split si y solo si t es un Bayes Correlated Equilibrium (BCE) del juego de colaboración inducido por los paquetes. Las restricciones BCE son **lineales** en la regla de comportamiento t, lo que hace el conjunto de perfiles alcanzables un politopo convexo.

**El resultado de concavificación:** La utilidad máxima del diseñador (CDI esperado) es el envelope cóncavo de la función de utilidad evaluada en el prior. Esto da:
1. Un test de alcanzabilidad: si t no es BCE-factible, ningún split puede inducirlo
2. Un criterio de optimalidad: el split óptimo para target t se puede encontrar resolviendo un LP

**Implicación práctica:** Antes de diseñar un split para un perfil CPP objetivo, se puede verificar algorítmicamente si ese perfil es alcanzable para ese problema. Si no lo es, el algoritmo sugiere el perfil BCE-factible más cercano.

**Fuentes:** Kamenica, E., & Gentzkow, M. (2011). Bayesian Persuasion. *American Economic Review, 101*(6), 2590–2615. Bergemann, D., & Morris, S. (2019). Information Design: A Unified Perspective. *Journal of Economic Literature, 57*(1), 44–95.

---

### 3.5 Tabla de Correspondencias: Celda PISA → Tipo de Asimetría de Información

El hallazgo teórico central para el diseño computacional de splits: cada celda PISA corresponde a un **tipo específico de asimetría de información** que, cuando está presente en el split, *causa* la activación de esa celda en la conversación. Esta tabla es el mecanismo causal que conecta diseño del split con activación de celda (fundamentado en Hutchins 1995 y Stahl 2006).

| Celda | Nombre PISA | Tipo de Asimetría Requerida | Condición Formal de Información |
|-------|-------------|---------------------------|--------------------------------|
| **A1** | Descubrir perspectivas/habilidades | Los agentes no conocen las capacidades del otro *a priori* | H(X_A\|I_A) > 0 ∧ H(X_B\|I_B) > 0 donde X = "capacidades del otro" |
| **A2** | Descubrir normas de interacción | Scripts de colaboración distintos o ausentes en los paquetes | No existe protocolo de comunicación compartido pre-especificado |
| **A3** | Comprender roles emergentes | Roles implicados por la información pero no nombrados explícitamente | I_A y I_B implican roles complementarios sin asignarlos |
| **B1** | Construir representación compartida | Representaciones distintas del mismo objeto matemático | I_A da acceso algebraico; I_B da acceso geométrico (o equivalente) |
| **B2** | Identificar tareas necesarias | Lista de sub-tareas distribuida entre paquetes | Ningún agente puede enumerar todos los pasos necesarios solo |
| **B3** | Describir roles en la ejecución | Ambigüedad genuina sobre quién ejecuta qué durante el proceso | Múltiples distribuciones de trabajo válidas dado los paquetes |
| **C1** | Comunicar acciones antes de ejecutar | El agente A debe anunciar su paso para que B lo valide | Output(A, paso_i) es *necesario* para que B valide antes de continuar |
| **C2** | Ejecutar el plan acordado | Dependencia matemática directa entre outputs de agentes | Output(A, paso_k) = Input(B, paso_{k+1}) — cadena inquebrante |
| **C3** | Seguir reglas de participación | Mecanismo de verificación de contribución equitativa | Hay criterios de participación que requieren monitoreo del otro |
| **D1** | Reparar entendimiento compartido | Ambigüedad semántica deliberada que emerge durante ejecución | El problema contiene términos con interpretaciones que divergen al ejecutar |
| **D2** | Monitorear éxito de acciones | Criterios de evaluación distribuidos | A tiene criterio de "es correcto matemáticamente"; B tiene criterio de "es completo/suficiente" |
| **D3** | Adaptar organización/roles | Evento mid-problem que invalida el plan inicial | El problema tiene una "sorpresa estructural" que requiere re-distribución |

**Principio de Hutchins-Stahl:** Los procesos cognitivos colaborativos emergen de la *propagación de estados representacionales* a través de la brecha entre lo que sabe cada agente. Cada celda PISA representa un tipo distinto de brecha. Un split de alta calidad diseña brechas deliberadas que fuerzan la propagación de representaciones — esto es la colaboración genuina.

**Fuentes:** Hutchins, E. (1995). *Cognition in the Wild*. MIT Press. Stahl, G. (2006). *Group Cognition: Computer Support for Building Collaborative Knowledge*. MIT Press.

---

### 3.6 El Espacio CPP: Perfiles y Subconjuntos Relevantes

El espacio completo de perfiles binarios tiene 2¹² = 4,096 elementos. Con restricciones de coherencia (DAG de prerrequisitos + restricciones BCE), el subespacio factible es ≈200-400 perfiles. Los perfiles más relevantes para el estudio:

| Perfil | Celdas activas | CDI | Nombre | Lattice |
|--------|---------------|-----|--------|---------|
| **CPP-Ø** | ninguna | 0.00 | Individual puro | base |
| **CPP-T** | A1 | 0.08 | Trivial (actual) | L1 |
| **CPP-IC** | A1, A2 | 0.17 | Informacional | L1+ |
| **CPP-CG** | A1, A2, A3 | 0.25 | Common Ground | L2 |
| **CPP-RP** | A1-B2 (6) | 0.50 | Representacional-Formulacional | L2+ |
| **CPP-BK** | A1,A2,B1,B2,D1,D2 | 0.50 | Bookend | L2+ |
| **CPP-EX** | B1,B2,C1,C2,C3 | 0.42 | Ejecución profunda | L3 |
| **CPP-DEEP** | A1-C3 (9) | 0.75 | Profundo | L3+ |
| **CPP-FULL** | todas (12) | 1.00 | Interdependencia total | L4 |

**Propiedad de lattice:** CPP-DEEP ⊃ CPP-RP ⊃ CPP-CG ⊃ CPP-IC ⊃ CPP-T. Validar que un split alcanza CPP-DEEP implica automáticamente que alcanza todos los perfiles de menor CDI.

---

### 3.7 Relación con los Patrones de Split (SPLIT-A a G)

Los siete patrones de split empíricos se mapean al espacio CPP:

| Patrón | Descripción | CPP típico | CDI típico |
|--------|-------------|-----------|-----------|
| SPLIT-A | Figura compuesta | CPP-RP/DEEP | 0.50-0.75 |
| SPLIT-B | Representación dual | CPP-RP | 0.42-0.50 |
| SPLIT-C | Condiciones complementarias | CPP-IC a CPP-CG | 0.08-0.25 (propenso a degeneración) |
| SPLIT-D | Cadena multi-paso | CPP-EX a CPP-DEEP | 0.42-0.75 |
| SPLIT-E | Objetivo × restricciones | CPP-RP | 0.42-0.50 |
| SPLIT-F | Espacio muestral × conteo | CPP-EX | 0.42-0.58 |
| SPLIT-G | Hipótesis × lema clave | CPP-DEEP | 0.67-0.83 |

El 66% del corpus actual usa SPLIT-C, que degenera frecuentemente a CPP-T — causa directa de los hallazgos de baja interdependencia.

---

## 4. Preguntas de Investigación e Hipótesis

### 4.1 Preguntas de Investigación

**RQ1 — Validación del framework:**
¿Predice el CDI la calidad del proceso colaborativo medida por PISA_global y ATC_global?

**RQ2 — Sensibilidad diferencial:**
¿Es ATC21S más sensible que PISA_global para detectar variaciones en CDI, específicamente en las dimensiones SR y CR?

**RQ3 — CDI óptimo:**
¿Existe un CDI_óptimo que varía con la dificultad del problema, tal que CDI < CDI_óptimo produce collaboration underuse y CDI > CDI_óptimo produce coordination overload?

**RQ4 — Condiciones Szewkis:**
¿Se mapean las 6 condiciones Szewkis a dimensiones específicas de ATC21S de forma consistente con el mapeo teórico propuesto?

**RQ5 — Generatividad IA:**
¿Pueden los LLMs generar splits que alcancen perfiles CPP específicos, validado por anotación automática post-conversación?

**RQ6 — Currículo:**
¿Varía la capacidad colaborativa potencial (CDI_max alcanzable) por subject y nivel de forma predecible?

---

### 4.2 Hipótesis

| ID | Hipótesis | Método de prueba |
|----|-----------|-----------------|
| **H1** | E[PISA_global \| CDI] es monótonamente creciente | Regresión CDI → PISA, corpus 600 |
| **H2** | La pendiente de E[ATC_SR \| CDI] > pendiente de E[PISA_global \| CDI] | Comparación de slopes, test de pendientes |
| **H3** | CDI_óptimo(nivel) es función creciente del nivel (1→5) | Análisis no-lineal CDI × accuracy × level |
| **H4** | S_j ↔ ATC_dim(j) según el mapeo teórico propuesto | Correlación Szewkis_j × ATC_dim |
| **H5** | C5 > C1 en PISA_global y ATC_global (p<0.05); neutral en accuracy para L1-3 | Experimento 5 condiciones, n=20 |
| **H6** | PISA_global(CPP-DEEP) > PISA_global(CPP-T) en mismos problemas | Comparación within-problem, n=20 |
| **H7** | geometry y counting_and_probability tienen CDI_max > algebra y prealgebra | Análisis por subject, corpus 600 |
| **H8** | targeting_error(C2_CIDI) < targeting_error(C2_prompt) < targeting_error(C3_constitutional), donde targeting_error = Hamming(CPP_achieved, t) / 12 | Experimento piloto 4×5 condiciones |
| **H9** | Los perfiles CPP observados forman un subconjunto cerrado bajo la intersección (lattice closure), verificando que son upsets del DAG de prerrequisitos | Análisis FCA sobre corpus n=150 |

---

## 5. Resultados Empíricos Disponibles (Baseline n=150)

| Hallazgo | Valor | Significancia | Interpretación PIE-CPS |
|---------|-------|--------------|----------------------|
| Ventaja colaborativa | ≈0 | n.s. (p≈0.7-0.9) | Splits CPP-T (CDI~0.08) insuficientes para H1 |
| comp2 solo→jigsaw | 98.7% → 57.1% | p<0.001 | SQS bajo: agentes gastan turnos en intercambio, no en matemática |
| Phase advantage B | 0.7% → 48.6% | p<0.001 | Jigsaw activa formulación conjunta incluso con CPP-T |
| Phase advantage C | 96.7% → 38.6% | p<0.001 | Efecto más robusto del estudio |
| ATC_SR social | d=+1.51 | p<0.001 | ATC detecta S5+S6; PISA no los captura |
| PISA global social | d≈-0.04 | n.s. | Confirma RQ2: ATC > PISA para dimensión social |
| SPLIT-C dominancia | 66% corpus | — | Causa probable de CDI bajo generalizado |

Estos resultados son el **baseline L1 (CPP-T)**. Las hipótesis H1-H6 se prueban comparando este baseline contra las 5 condiciones del experimento CPP.

---

## 6. Referencias

Care, E., Griffin, P., & Wilson, M. (Eds.) (2018). *Assessment and Teaching of 21st Century Skills: Research and Applications*. Springer.

Brousseau, G. (1997). *Theory of Didactical Situations in Mathematics*. Kluwer Academic Publishers.

Clark, H. H., & Schaefer, E. F. (1989). Contributing to discourse. *Cognitive Science*, 13(2), 259–294.

Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In L. B. Resnick, J. M. Levine, & S. D. Teasley (Eds.), *Perspectives on Socially Shared Cognition* (pp. 127–149). APA.

Griffin, P., & Care, E. (Eds.) (2015). *Assessment and Teaching of 21st Century Skills: Methods and Approach*. Springer.

Hutchins, E. (1995). *Cognition in the Wild*. MIT Press.

Kirschner, P. A., Sweller, J., & Clark, R. E. (2006). Why minimal guidance during instruction does not work. *Educational Psychologist*, 41(2), 75–86.

OECD (2013). *PISA 2015 Draft Collaborative Problem Solving Framework*. OECD Publishing.

OECD (2017). *PISA 2015 Results (Volume V): Collaborative Problem Solving*. OECD Publishing.

Rojas-Drummond, S., & Mercer, N. (2003). Scaffolding the development of effective collaboration and learning. *International Journal of Educational Research*, 39(1-2), 99–111.

Roschelle, J., & Teasley, S. D. (1995). The construction of shared knowledge in collaborative problem solving. In C. E. O'Malley (Ed.), *Computer Supported Collaborative Learning* (pp. 69–97). Springer.

Szewkis, E., Nussbaum, M., Rosen, T., Abalos, J., Denardin, F., Caballero, D., Tagle, A., & Alcoholado, C. (2011). Collaboration within large groups in the classroom. *International Journal of Computer-Supported Collaborative Learning*, 6(4), 561–575.

van Merriënboer, J. J. G., Clark, R. E., & de Croock, M. B. M. (2002). Blueprints for complex learning: The 4C/ID-model. *Educational Technology Research and Development*, 50(2), 39–61.

Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher Psychological Processes*. Harvard University Press.

Wegner, D. M. (1987). Transactive memory: A contemporary analysis of the group mind. In B. Mullen & G. R. Goethals (Eds.), *Theories of Group Behavior* (pp. 185–208). Springer.

Bergemann, D., & Morris, S. (2019). Information Design: A Unified Perspective. *Journal of Economic Literature, 57*(1), 44–95. DOI: 10.1257/jel.20181489

Kamenica, E., & Gentzkow, M. (2011). Bayesian Persuasion. *American Economic Review, 101*(6), 2590–2615. DOI: 10.1257/aer.101.6.2590

Ganter, B., & Wille, R. (1999). *Formal Concept Analysis: Mathematical Foundations*. Springer. DOI: 10.1007/978-3-642-59830-2

Stahl, G. (2006). *Group Cognition: Computer Support for Building Collaborative Knowledge*. MIT Press.

Yang, K., & Klein, D. (2021). FUDGE: Controlled Text Generation With Future Discriminators. *Proceedings of NAACL 2021*. DOI: 10.18653/v1/2021.naacl-main.276

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
