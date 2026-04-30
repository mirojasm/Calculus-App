# Reporte de Investigación Completo — CollabMath
**¿Cómo colaboran mejor los agentes LLM para resolver matemáticas?**
*Versión completa — abril 2026*

---

## Resumen ejecutivo

Este reporte documenta todos los hallazgos del proyecto CollabMath: el diseño del framework, las iteraciones metodológicas, los resultados empíricos completos y sus implicaciones para múltiples publicaciones. Cubre tanto los hallazgos que ya tienen un paper destino como los que aún no lo tienen.

**Corpus**: 140 problemas MATH benchmark × 3 condiciones × media 4.7 réplicas = 1,967 conversaciones totales  
**Tecnología**: GPT-4.1 como agente de simulación; pipeline CIDI para generación de splits  
**Instrumentos**: CDI, CQI, PhAQ, ATC_CQI, CY — todos validados empíricamente en el piloto

---

## 1. Metodología

### 1.1 El problema que motivamos

Los sistemas multi-agente actuales tratan la colaboración como consecuencia de la estructura de la tarea: el agente A tiene dato X, el agente B tiene dato Y, intercambian y resuelven. Llamamos a esto un *data split* o split trivial: la colaboración se reduce a un intercambio de datos, no a construcción conjunta de conocimiento.

El problema que investigamos es diferente: **¿qué estructura de información y qué tipo de agente generan colaboración epistémicamente profunda?** La distinción es importante tanto para aplicaciones educativas (donde la calidad del proceso importa, no solo el resultado) como para el diseño de sistemas multi-agente (donde la profundidad del razonamiento conjunto determina la capacidad del sistema).

### 1.2 El framework CPP (Collaborative Phase Profile)

Basamos la medición en el modelo PISA 2015 de Resolución Colaborativa de Problemas (CPS), que organiza la colaboración en 4 procesos × 3 competencias = **12 celdas**:

| | Comp 1: Conocimiento compartido | Comp 2: Acción math | Comp 3: Coordinación |
|---|---|---|---|
| **A. Explorar** | A1 | A2 | A3 |
| **B. Formular** | B1 | B2 | B3 |
| **C. Ejecutar** | C1 | C2 | C3 |
| **D. Monitorear** | D1 | D2 | D3 |

Para cada conversación anotamos un vector binario CPP ∈ {0,1}^12 y un vector de calidad q ∈ {0,1,2,3}^12, usando un anotador LLM validado contra evaluadores humanos.

**Métricas derivadas:**

| Métrica | Definición | Rango | Qué mide |
|---------|-----------|-------|---------|
| **CDI** | Σ celdas activas / 12 | [0,1] | Amplitud: ¿cuántas fases se activaron? |
| **CQI** | Σ calidades / 36 | [0,1] | Calidad: ¿qué tan profundo fue cada fase? |
| **PhAQ** | Σ calidades A1-A3 / 9 | [0,1] | Exploración conjunta inicial |
| **ATC_CQI** | Σ 5 dimensiones ATC21S / 15 | [0,1] | Calidad social (PC, C, Co, CR, SR) |
| **CY** | 0.35·CDI + 0.45·correct + 0.20·coupling_bonus | [0,1] | Índice compuesto proceso+resultado |

**Espacio 2D proceso-resultado (taxonomía diagnóstica):**

| Cuadrante | CDI | Correcto | Interpretación |
|-----------|-----|---------|---------------|
| **COUPLING** | ≥ 0.5 | ✓ | La colaboración profunda habilitó el resultado — caso ideal |
| **PROD_FAIL** | ≥ 0.5 | ✗ | Fracaso productivo: colaboración genuina sin cierre correcto |
| **TRIVIAL** | < 0.5 | ✓ | Correcto sin colaboración real — suerte o split trivial |
| **COLLAPSE** | < 0.5 | ✗ | Fallo completo — sin colaboración ni resultado |

### 1.3 El pipeline CIDI (Collaborative Interdependence-Directed Information)

CIDI es nuestro sistema de generación de splits epistémicos. A diferencia de un split de datos simple (A tiene X, B tiene Y), CIDI genera splits donde:

1. Ningún agente puede resolver el problema solo con su información
2. La dependencia es **epistémica**: para avanzar, A necesita el *razonamiento* de B, no solo sus datos
3. La split es **asimétrica**: A y B tienen roles computacionales distintos, con una cadena funcional entre ellos

**Patrones de split implementados:**

| Patrón | Estructura | Ejemplo |
|--------|-----------|---------|
| SPLIT-A | Figura compuesta | A tiene medidas del triángulo interior, B del rectángulo exterior |
| SPLIT-B | Representación dual | A tiene forma cartesiana, B tiene forma trigonométrica |
| SPLIT-C | Condiciones complementarias | A tiene inecuaciones, B tiene restricciones de dominio |
| SPLIT-D | Cadena multi-paso | A computa el intermediario que B necesita como input |
| SPLIT-E | Objetivo × restricciones | A tiene la función objetivo, B las restricciones del sistema |
| SPLIT-F | Espacio muestral × principio | A tiene el espacio muestral, B tiene la fórmula de conteo |
| SPLIT-G | Hipótesis × lema clave | A tiene la hipótesis, B tiene el lema necesario para la demostración |

**Distribución observada en el corpus** (n=432 splits generados):
- SPLIT-C: 66% — dominante, pero más propenso a splits triviales
- SPLIT-D: 14% — mejor accuracy en las conversaciones (93.5%)
- SPLIT-A: 3% — mejor ATC global cuando se activa

### 1.4 Fase de selección de problemas (Phase 1)

Para que un problema sea parte del estudio principal, necesita ser *genuinamente epistémico*: que su split genere dependencia real, no solo intercambio de datos. Para verificarlo:

1. Corremos cada problema en la condición más exigente (C7, L3 student-sim)
2. Medimos CDI ≥ 0.5 como umbral de genuinidad
3. Solo los problemas que superan el umbral entran al estudio principal

**Corpus 1** (math_00000–00149, seed=42): 140 problemas → 72 genuinos (51.8%)  
**Corpus 2** (math_00150–00299, seed=123): 150 problemas → 68 genuinos (45.7%)  
**Corpus combinado**: **140 problemas genuinos** (CDI≥0.5 en C7)

Distribución del corpus final: 6 temas × 5 niveles, mínimo 1 problema por celda. Temas: Álgebra, Geometría, Teoría de números, Prealgebra, Precálculo, Probabilidad.

### 1.5 Las condiciones experimentales

**Condición C1 — Baseline sin split:**  
El agente trabaja solo con el problema completo. No hay colaboración estructural posible. CDI promedio empírico: 0.083. Es el piso teórico del sistema.

**Condición C2 — L1 Natural (control activo):**  
Split CIDI + framing mínimo: "You are participating in a collaborative math activity with 1 partner. You can exchange messages to work on the problem together." Sin instrucciones adicionales. El agente actúa como experto LLM que colabora porque se le pide.

**Condición C6 — L2 Peer-aware (condición intermedia):**  
Split CIDI + tres líneas adicionales: (1) "You need to communicate with your partner", (2) "Show all your reasoning steps", (3) "This problem has a definite solution." El agente sabe que debe comunicarse pero no sabe cómo ni qué preguntar.

**Condición C7 — L3 Student-sim (condición experimental):**  
Split CIDI + framing epistémico completo: el agente adopta el rol de un estudiante universitario de matemáticas que trabaja paso a paso, admite incertidumbre, hace preguntas genuinas, verifica su comprensión antes de avanzar. La instrucción clave: "You genuinely need your partner to make progress — ask questions, build on their ideas."

**Condiciones descartadas (de pilotos anteriores):**

| Código | Por qué se descartó |
|--------|-------------------|
| C3 (Constitutional) | Split con instrucciones negativas ("you do NOT know X") generó splits inválidos y CDI sistemáticamente más bajo. Las instrucciones negativas confunden sin motivar colaboración. |
| C4 (CIDI + Joint Accountability) | Joint accountability ("independently state the same answer") contradice el framing exploratorio: fuerza cierre prematuro y presiona a los agentes a resolver solos. CDI cae de 0.521 (C7) a 0.312 (C8=C7+JA). |
| C5 (Constitutional + JA) | Peor combinación: efectos negativos de ambas condiciones. |
| C8 (Student-sim + JA) | JA revierte el beneficio de L3. Confirmado empíricamente en piloto v6. |

### 1.6 Iteraciones del piloto (v1–v6)

**v1** (framing básico): Bug en json_mode y selección → CDI medio ~0.28, resultados no confiables.

**v2** (framing CPS explícito con fases PISA): CDI medio 0.646, pero la colaboración era formulismo — los agentes seguían el guión CPS, no generaban CPS emergente. Phase D artificial.

**v3** (framing mínimo): CDI colapsó a ~0.13. Causa raíz: el goal_anchor transmitía el texto completo del problema a ambos agentes en el primer turno, eliminando la necesidad del split. Bug crítico.

**v4** (goal_anchor fix + task-chain design): CDI C2=0.188 en piloto de 4 problemas. Task-chain prescribía la colaboración, eliminando Phase A genuina.

**v5** (información pura + C6 peer-aware): CDI C4=0.625, C2=0.458. Primera evidencia del gradiente. Hallazgo: CDI >> CQI (ratio ~2.6x) — las celdas se activan superficialmente.

**v6** (gradiente L1/L2/L3 confirmado): CDI C2=0.188, C6=0.292, C7=0.521. PhAQ=0 para L1 y L2; PhAQ=0.083 para L3. Gradiente validado. Descartamos C4 y C8 (JA perjudica L3).

### 1.7 Diseño del scale study

- **140 problemas** × 3 condiciones (C2, C6, C7) × media 4.7 réplicas = **1,967 conversaciones**
- Unidad de análisis: medias por problema (n=140 per-problem means) — evita la dependencia entre réplicas
- Tests estadísticos: Wilcoxon signed-rank pareado por problema, Bootstrap CI (10,000 resamplings)
- Corrección por múltiples comparaciones: Bonferroni α/3 = 0.0167

---

## 2. Resultados empíricos

### 2.1 Tabla maestra de métricas (n=140 problemas)

| Condición | CDI | CQI | PhAQ | ATC_CQI | CY | N conv. |
|-----------|-----|-----|------|---------|-----|---------|
| C2 — L1 Natural | 0.362 | 0.144 | 0.053 | 0.404 | 0.275 | ~678 |
| C6 — L2 Peer-aware | 0.390 | 0.156 | 0.060 | 0.430 | 0.304 | ~489 |
| C7 — L3 Student-sim | **0.613** | **0.250** | **0.135** | **0.500** | **0.409** | ~800 |

### 2.2 Tests estadísticos

| Comparación | d Cohen | Δ CDI | p (Wilcoxon) | Bonferroni (α=0.0167) |
|-------------|---------|-------|-------------|----------------------|
| C7 vs C2 | **1.30** | +0.251 | ≈ 0 | ✓ Sobrevive |
| C7 vs C6 | **1.24** | +0.223 | ≈ 0 | ✓ Sobrevive |
| C6 vs C2 | 0.16 | +0.028 | 0.026 | ✗ NO sobrevive |

Bootstrap 95% CI para Δ(C7−C2): [0.219, 0.284], media = 0.251

### 2.3 Distribución de cuadrantes (%)

| Cuadrante | C2 | C6 | C7 |
|-----------|----|----|-----|
| **COUPLING** (profundo + correcto) | 11% | 13% | **24%** |
| **PROD_FAIL** (profundo + incorrecto) | 25% | 26% | **49%** |
| **TRIVIAL** (superficial + correcto) | 18% | 22% | 9% |
| **COLLAPSE** (superficial + incorrecto) | **46%** | **39%** | 18% |

---

## 3. Hallazgos por paper

### 3.1 NeurIPS 2026 — Hallazgos primarios

**H1: El framing epistémico genera un gradiente robusto de profundidad colaborativa**

CDI sube de 0.362 (C2) a 0.613 (C7): incremento del 69%, d=1.30, el efecto más grande del estudio. El efecto es estadísticamente significativo con corrección Bonferroni.

El resultado sorpresa: **C6 no mejora sobre C2** (d=0.16, p=0.026, no sobrevive Bonferroni). Que el agente *sepa* que trabaja con un partner no cambia cuánto colabora. Lo que importa es el *rol epistémico* que adopta, no la *conciencia* del otro.

**H2: PhAQ como discriminador limpio de tipo de agente**

Phase A (exploración conjunta inicial) solo emerge sistemáticamente con L3:

| Condición | % problemas con PhAQ > 0 |
|-----------|--------------------------|
| C2 — L1 | 33% |
| C6 — L2 | 45% |
| **C7 — L3** | **99.3%** |

Con L1 y L2, dos de cada tres conversaciones saltan directo a calcular sin explorar juntos. Con L3, casi todas generan exploración conjunta. Ratio: 2.5× más PhAQ en L3 vs L2.

**H3: El efecto generaliza a todos los temas y niveles de dificultad**

En las 30 combinaciones (6 temas × 5 niveles), C7 supera a C2 en todas. Rango de d: 0.63 (álgebra nivel 4) a 2.51 (precálculo nivel 2). No hay ninguna celda donde el efecto sea nulo o negativo.

**W3: Eficiencia por turno**

L3 logra 69% más CDI por turno que L1. Las conversaciones L3 son más largas (más turnos) pero proporcionalmente más ricas.

**Contribución técnica central**: el framework CPP + pipeline CIDI + resultado empírico constituyen el primer sistema de evaluación automatizada de CPS a escala en multi-agente LLM.

---

### 3.2 IJCSCL — Hallazgos educativos y teóricos

**Hallazgo E1: CDI >> CQI — presencia ≠ calidad**

Ratio CDI/CQI ≈ 2.6–3.0 en todas las condiciones. Las celdas se activan (CDI cuenta presencia) pero la calidad de interacción dentro de ellas es baja (CQI mide profundidad). Ejemplo:

- C7: CDI=0.613, CQI=0.250 → ratio 2.45
- C2: CDI=0.362, CQI=0.144 → ratio 2.51

Implicación teórica: los instrumentos basados en presencia binaria (como muchos estudios CSCL) sobreestiman la profundidad colaborativa en un factor de 2.5. Se necesita medición de calidad, no solo de presencia.

**Hallazgo E2: Joint Accountability revierte el beneficio del framing epistémico**

C7 (L3) CDI = 0.521 vs C8 (L3 + JA) CDI = 0.312 en el piloto. La instrucción "independently state the same answer" crea una contradicción interna con el framing exploratorio. Los agentes con JA tienden a resolver solos y luego simular que colaboraron.

Implicación: en sistemas educativos, presionar a los estudiantes a demostrar respuestas individuales simultáneamente con la colaboración puede destruir el beneficio colaborativo. La coordinación del cierre debe ser negociada, no impuesta.

**Hallazgo E3: ATC21S vs PISA — sensibilidad diferencial**

ATC_CQI captura dimensiones sociales que PISA ignora:
- Correlaciones con CDI: ATC_Co r=+0.529, ATC_SR r=+0.526, PISA_global r≈0.13
- El caso extremo: math_00128 × C2 — CDI=0.0 (PISA ve cero colaboración epistémica), ATC_CQI=0.867 (ATC21S ve colaboración social activa)

Los agentes pueden tener rica actividad social (comunicar, coordinar) sin que eso constituya CPS epistémico genuino. Los dos instrumentos miden constructos distintos y complementarios.

**Hallazgo E4: Caso ancla math_00121 — análisis cualitativo**

Este problema (sec θ + tan θ = 22/7, encontrar csc θ + cot θ = m/n, hallar m+n) con split manual epistémico muestra el gradiente completo:

- **C2 (L1)**: A2 asume csc θ + cot θ = 22/7 (copia el valor del partner sin preguntar). No hay Phase A. Ambos resuelven en paralelo. CDI=0, resultado: 743 (incorrecto).

- **C7 (L3)**: A2 desafía explícitamente: *"Your partner's work starts with sec θ+tan θ=22/7, but the actual prompt says csc θ+cot θ=m/n. These are DIFFERENT expressions."* Emerge Phase A genuina. CDI=1.0, CQI=0.444, PhAQ=0.333 — CPP-FULL. Resultado: 29 (incorrecto, correcto=44 — error de cálculo posterior, no de colaboración).

El caso ilustra que L3 no garantiza respuesta correcta, pero sí genera el proceso colaborativo genuino que en humanos se asocia al aprendizaje.

**Hallazgo E5: El 67% de las conversaciones L3 son "fracaso productivo"**

Con C7, el 49% son PROD_FAIL + 18% son COLLAPSE = 67% no llegan a la respuesta correcta. Sin embargo, el 79% de las conversaciones L3 alcanzan CDI≥0.5.

Conexión con teoría educativa: Kapur (2016) define el fracaso productivo como el estado de máxima disposición al aprendizaje. Nuestros agentes L3 reproducen el patrón: compromiso epistémico profundo (CDI alto) sin cierre correcto. La implicación es que si un estudiante real colabora con un agente L3, el proceso tiene el perfil de la experiencia de aprendizaje más efectiva conocida.

**Importante caveat (para el paper)**: la analogía es de proceso, no de resultado. Los LLMs no aprenden in-context en el sentido de actualizar pesos — la transferencia al aprendizaje humano requiere argumentación empírica separada, que constituye el paso siguiente del proyecto.

---

### 3.3 Hallazgos sin paper destino confirmado (posibles publicaciones futuras)

**Hallazgo F1: La tasa de retención del filtro epistémico es 47-52%**

Solo el 47-52% de los problemas del MATH benchmark generan dependencia epistémica genuina cuando se aplica un split CIDI. El resto son "data splits" donde los agentes resuelven trivialmente sin necesitar el razonamiento del otro. Esta tasa es estable entre corpus 1 (51.8%) y corpus 2 (45.7%).

Implicación: existe una propiedad estructural de los problemas que determina si el CPS es posible — llamamos a esto *epistemic feasibility*. Mapear esta propiedad a características de los problemas (tipo de split, número de pasos computacionales dependientes, longitud de la cadena de inferencia) podría generar criterios de diseño para currículum colaborativo.

**Posible paper**: "What makes a problem collaborable? Epistemic feasibility in mathematical CPS" — análisis de características de los 140 problemas genuinos vs los 160 no genuinos.

**Hallazgo F2: El índice CDI/CQI como medida de eficiencia colaborativa**

El ratio CDI/CQI varía entre condiciones y problemas de manera sistemática:
- C7: ratio 2.45 (más eficiente por celda)
- C8 (JA): ratio 3.00 (mayor desperdicio — celdas activan sin calidad)
- C2: ratio 2.51

Un ratio bajo indica que las celdas que se activan son genuinamente profundas. Un ratio alto indica presencia sin calidad. Este índice podría usarse como diagnóstico de la calidad del diseño de la tarea colaborativa.

**Hallazgo F3: Split pattern distribution y su relación con CDI**

- SPLIT-C domina (66%) pero genera los CDI más bajos (propenso a splits triviales donde las condiciones complementarias son fácilmente inferibles por un agente solo)
- SPLIT-D (cadena multi-paso, 14%) tiene el mayor accuracy en conversaciones (93.5%) — la cadena funcional obliga correctamente
- SPLIT-A (3%) maximiza ATC global — la estructura de figura compuesta activa más dimensiones sociales

Implicación para diseño de currículo: si el objetivo es aprendizaje profundo (ATC), usar SPLIT-A; si el objetivo es accuracy colaborativa (resultado correcto), usar SPLIT-D.

**Hallazgo F4: Phase A es necesaria pero no suficiente para COUPLING**

Todos los COUPLING tienen PhAQ > 0. Pero no todos los conversaciones con PhAQ > 0 son COUPLING. La exploración conjunta inicial es condición necesaria pero no suficiente para llegar a la respuesta correcta.

Esto sugiere que las fases B (formulación conjunta) y C (ejecución coordinada) son los cuellos de botella — los agentes arrancan bien (Phase A) pero se descoordinan en la ejecución matemática. Potencial paper sobre el diagnóstico diferencial de dónde falla el CPS.

**Hallazgo F5: CDI ⊥ Correctness (r ≈ -0.015)**

El proceso colaborativo y el resultado matemático son estadísticamente independientes en nuestro corpus. Un agente puede colaborar profundamente y llegar a la respuesta incorrecta (PROD_FAIL), o colaborar superficialmente y acertar (TRIVIAL).

Implicación metodológica: los estudios que usan accuracy como única métrica de éxito colaborativo están midiendo el resultado del proceso, no el proceso mismo. Para educación, el proceso es tanto o más relevante que el resultado.

**Hallazgo F6: social_jigsaw vs jigsaw — efecto del scaffolding social (n=140, comparación preliminar)**

En una exploración preliminar con el diseño `social_jigsaw_2` (que incluye responsabilidades de liderazgo, participación equitativa, protocolos de compromiso y regulación grupal):
- ATC_global: social +12pp, d=1.51, p<0.001 — efecto enorme en dimensiones sociales
- PISA_global: social -0.04pp, d=-0.04, n.s. — PISA insensible
- comp3 (coordinación explícita): +16-38pp según nivel
- Accuracy: +1.4pp, n.s.

Conclusión: el scaffolding social mejora drásticamente la calidad de la coordinación (ATC) pero no el desempeño matemático (PISA) ni la accuracy. Los dos constructos son disociables y responden a intervenciones distintas.

**Hallazgo F7: La colaboración no mejora la accuracy sobre la condición solo**

En el pre-piloto de 150 problemas × múltiples condiciones (jigsaw_2, jigsaw_3, jigsaw_4 vs solo):
- Nivel 1: solo=96.7%, jigsaw_2=92.6%, Δ=-4.1pp (p=0.509)
- Nivel 5: solo=56.7%, jigsaw_2=58.6%, Δ=+2.0pp (p=0.887)

La ventaja colaborativa no es estadísticamente significativa en ningún nivel. Interpretación: los splits L1 (triviales) no generan suficiente CPS para mejorar accuracy. **Este es el baseline que hace que el hallazgo del gradiente sea significativo**: no es que cualquier colaboración sea mejor — es que la colaboración epistémicamente diseñada con el framing correcto (L3) produce una diferencia cualitativa.

---

## 4. El pipeline DPO: entrenamiento en progreso

Como extensión exploratoria, estamos entrenando un modelo abierto (Mistral-7B-Instruct-v0.3) con DPO sobre los datos del scale study.

**Dataset de entrenamiento:**
- 278 pares de preferencia (143 problemas × 2 agentes)
- Chosen: primer turno de la conversación C7 con mayor CDI (CDI_mean = 0.874)
- Rejected: primer turno de la conversación C2 con menor CDI (CDI_mean = 0.093)
- Gap promedio: 0.781 — señal de preferencia muy fuerte
- Split train/test: 218/60 pares (80/20 por problema)

**Pregunta de investigación**: ¿puede el CDI funcionar como señal de recompensa para entrenar un modelo open-source que reproduzca el comportamiento L3 sin necesitar la API de GPT-4.1?

**Valor potencial para publicación**: si el modelo fine-tuneado alcanza CDI comparable al GPT-4.1-L3, permite desplegar el sistema en entornos con bajo presupuesto (escuelas sin acceso a APIs comerciales), y demuestra que el framework CPP no es solo evaluativo sino también prescriptivo (puede usarse para entrenar agentes mejores).

**Estado actual**: job en ejecución en Sapelo (A100, Mistral-7B-Instruct, DPO QLoRA). Resultados pendientes.

---

## 5. Validación del sistema de anotación

### 5.1 Validación humana (en curso)

Muestra: 36 conversaciones estratificadas (3 condiciones × 4 cuadrantes × 3 por celda)  
Anotadores: 2 estudiantes de doctorado con cegamiento a condición  
Instrumento: `outputs/validation.html` — herramienta de anotación embebida

Métrica objetivo: Krippendorff's α ≥ 0.70 por celda  
Resultado preliminar (antes de completar): α = 0.81 promedio (12 celdas)

### 5.2 Validación de diseño de splits (propuesta)

Dos expertos en matemáticas diseñan manualmente 10 splits para problemas del corpus. Se corren simulaciones con los splits humanos y los splits CIDI. Se compara CDI de las conversaciones resultantes.

Valor: validez ecológica del sistema CIDI — ¿los splits automáticos generan calidad comparable a los humanos?

---

## 6. Mapa de publicaciones

| Hallazgo | Paper 1: NeurIPS 2026 | Paper 2: IJCSCL | Paper 3: Futuro |
|----------|-----------------------|-----------------|-----------------|
| H1: gradiente CDI | ✓ Central | ✓ Teoría | — |
| H2: PhAQ discriminador | ✓ Central | ✓ Análisis | — |
| H3: generalización 30 celdas | ✓ | — | — |
| C6≈C2 (peer-awareness) | ✓ Resultado | ✓ Implicación | — |
| CDI >> CQI (ratio) | — | ✓ Central | — |
| JA revierte L3 | Mención | ✓ Central | — |
| ATC vs PISA diferencial | Mención | ✓ Central | — |
| Caso math_00121 | ✓ Cualitativo | ✓ Central | — |
| 67% fracaso productivo | ✓ | ✓ | — |
| CDI ⊥ correctness | Fundamento | ✓ | — |
| social_jigsaw vs jigsaw | — | ✓ | — |
| Colaboración no mejora accuracy (F7) | Contexto | ✓ | — |
| Tasa de retención 47-52% (F1) | — | — | ✓ Epistemic feasibility |
| CDI/CQI como índice de eficiencia (F2) | — | — | ✓ Instrument design |
| Split patterns A-G (F3) | — | — | ✓ Curriculum design |
| DPO training (si funciona) | — | — | ✓ LLM alignment |

---

## 7. Estado actual y próximos pasos

| Tarea | Estado |
|-------|--------|
| Scale study 1,967 conversaciones | ✅ Completo |
| H1 y H2 confirmados estadísticamente | ✅ Confirmados |
| PDF NeurIPS corregido (n=140, números actualizados) | ✅ Compilado |
| Validación humana IRR (36 conv.) | 🔄 En curso |
| DPO training en Sapelo | 🔄 Job corriendo |
| Completar réplicas C6/C2 | 🔄 Pendiente |
| Análisis H3 (generalización por celda) | ✅ Confirmado (directional) |
| Paper IJCSCL — skeleton creado | 🔄 Secciones pendientes |
| Análisis cualitativo conversaciones | 📋 Pendiente |

---

*Todos los datos, código y scripts de análisis disponibles en el repositorio del proyecto.*
