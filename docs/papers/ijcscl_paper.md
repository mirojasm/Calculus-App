# Epistemic Interdependence as a Design Principle for AI-Generated Collaborative Mathematical Tasks: A Framework, Simulation Method, and Pilot Study

**Target**: International Journal of Computer-Supported Collaborative Learning (IJCSCL), Springer  
**Status**: Skeleton — awaiting pilot v3 results  
**Estimated length**: 8,000–10,000 words

---

## Abstract (draft)

Collaborative mathematical problem solving requires more than distributing information across participants — it demands that no agent can formulate a solvable approach without engaging with another's knowledge. We distinguish *epistemic splits*, where information distribution creates genuine reasoning interdependence, from *data splits*, which merely partition facts. Drawing on the PISA 2015 Collaborative Problem Solving framework, we introduce the Collaborative Phase Profile (CPP), a formal binary vector over twelve cognitive cells (four PISA processes × three competencies), and the CPS Depth Index (CDI), a scalar measure of collaborative engagement depth. We propose five experimental conditions for inducing and measuring collaborative depth in LLM agent dyads: C1 (baseline), C2 (CIDI-directed epistemic split), C3 (constitutional critic), C4 (Szewkis process monitor), and C5 (integrated). We additionally introduce Collaborative Yield (CY), a composite metric that couples process quality (CDI) with outcome correctness in a way grounded in productive failure theory. A three-iteration pilot study (n=4 problems × 5 conditions = 20 conversations per iteration) validates the framework, reveals CDI inflation when splits are non-epistemic, and demonstrates that minimal collaborative framing with strong epistemic splits outperforms over-specified CPS instruction. Implications for the design of collaborative mathematical activities are discussed.

---

## 1. Introduction

Collaborative problem solving (CPS) has emerged as a core twenty-first-century competency (OECD, 2017), yet the design of tasks that genuinely require collaboration — rather than merely permitting it — remains an open challenge. In mathematics education, jigsaw-style activities divide information across learners, but the division is typically informational: once each learner shares their piece, a single agent could complete the solution. This creates *data splits* rather than *epistemic splits*, and the collaboration it elicits is shallow: participants exchange data but do not jointly reason.

The advent of large language model (LLM) agents as proxies for learners opens new methodological possibilities. LLM agents can be simulated at scale, their interactions logged and scored automatically, and controlled conditions can be tested before deployment with real students. However, the design of these simulations conceals significant choices: how to frame the collaborative context, what information to provide each agent, and how to measure whether genuine collaborative problem solving occurred.

This paper addresses three interrelated problems:

1. **Theoretical**: What constitutes a genuinely epistemic split, and how can it be formally characterized?
2. **Methodological**: How can LLM agent simulation validly measure collaborative problem-solving depth?
3. **Empirical**: Which conditions most reliably produce deep collaboration, and under what circumstances does process quality predict correct outcomes?

We contribute:
- The CPP framework and CDI metric as formal tools for characterizing CPS task design
- A five-condition experimental design with theoretical grounding for each condition
- The Collaborative Yield (CY) metric integrating process and outcome within a productive failure framework
- Evidence from a three-iteration pilot study of methodological design decisions

---

## 2. Theoretical Background

### 2.1 Collaborative Problem Solving: The PISA 2015 Framework

The PISA 2015 assessment framework defines collaborative problem solving as "the capacity of an individual to engage effectively in a process whereby two or more agents attempt to solve a problem by sharing the understanding and effort required to come to a solution" (OECD, 2017, p. 134). The framework decomposes CPS into a 4 × 3 matrix: four problem-solving processes (A: Exploring and Understanding; B: Representing and Formulating; C: Planning and Executing; D: Monitoring and Reflecting) crossed with three collaborative competencies (1: Establishing and maintaining shared understanding; 2: Taking appropriate action to solve the problem; 3: Establishing and maintaining team organization).

This yields twelve cells (A1–D3) that operationalize where and how collaboration occurs across the problem-solving trajectory. Prior research has shown that jigsaw conditions elevate cells A1 (shared exploration) and B1 (joint formulation) relative to solo conditions, but frequently suppress C2 (mathematical execution action) — suggesting that conventional splits trade procedural performance for collaborative engagement (see Results, this paper).

### 2.2 Epistemic Splits vs. Data Splits

**Data splits** partition known quantities across agents (Agent A knows value x, Agent B knows value y; sharing x and y allows either agent to complete the solution independently). **Epistemic splits** partition *reasoning roles*: Agent A must derive intermediate values that Agent B needs to apply a different mathematical principle; neither can formulate a solvable equation without the other's contribution.

The distinction is operationalized through the *epistemic fidelity criterion*: a split has high epistemic fidelity if the point-biserial correlation r_pb(CDI, correctness) across conditions is positive and significant for that problem. A data split produces r_pb ≈ 0 because CDI and correctness are statistically independent — any agent can stumble toward the answer without genuine collaboration.

**Design example** (math_00121, sec θ + tan θ = 22/7): An epistemic split assigns Agent 1 the given equation and the role of deriving sin θ and cos θ; Agent 2 receives the identity csc θ + cot θ = (1 + cos θ)/sin θ and the task of computing m + n from the reduced fraction. Neither can proceed without the other's output. A data split would give Agent 1 the equation and Agent 2 only "find m + n" — Agent 1 can derive the entire solution independently.

### 2.3 Productive Failure and the Process-Outcome Relationship

Kapur (2012) demonstrated that students who engage deeply with a problem and fail produce better subsequent learning than students who succeed with shallow engagement. This *productive failure* finding reframes the interpretation of high-process/incorrect-outcome pairs: they represent maximal engagement within the ZPD, not educational failure.

For our framework, this implies that CDI and correctness cannot be collapsed into a single metric without theoretical loss. The Collaborative Yield (CY) metric (introduced in Section 3.6) preserves this distinction while providing a practical composite for condition ranking.

### 2.4 The ICAP Framework

Chi and Wylie's (2014) ICAP framework predicts that interactive engagement produces greater learning than constructive, active, or passive engagement. In our context, interactive engagement corresponds to cells requiring *bidirectional knowledge construction* — both agents jointly building representations neither held alone. ICAP predicts that conditions producing higher proportions of interactive (rather than constructive or active) turns will show both higher CDI and higher learning outcomes.

### 2.5 Szewkis Collaborative Conditions

Szewkis et al. (2011) identified six conditions for productive collaborative learning: (1) common goal, (2) positive interdependence, (3) individual accountability, (4) group reward orientation, (5) group awareness, and (6) coordination. Condition C4 in our design implements a Szewkis monitor that evaluates these conditions after PISA Phase A and B transitions and injects corrective interventions when conditions are violated.

---

## 3. Methodology

### 3.1 The CPP Framework and CDI

The Collaborative Phase Profile (CPP) is a binary vector **CPP ∈ {0,1}^12** over the twelve PISA CPS cells. For a given problem P and split strategy S with n agents, CPP(P, S, n) indicates in which cells collaboration is structurally necessary for problem resolution. 

The CPS Depth Index is defined as:
$$CDI(P, S) = \frac{\sum_{i=1}^{12} cpp_i}{12} \in [0, 1]$$

CDI = 0 indicates no collaborative engagement is required or observed. CDI = 1 indicates activation across all twelve cells. In practice, five empirical profiles emerge: CPP-IC (<0.25), CPP-CG (0.25–0.42), CPP-RP (0.42–0.58), CPP-DEEP (0.58–0.83), CPP-FULL (1.0).

### 3.2 The CIDI Pipeline (Module 1–5)

*[Figure 2 here: CIDI pipeline flow]*

The Collaborative Interaction Design Infrastructure (CIDI) is a six-module pipeline for generating CPP-targeted epistemic splits:

- **M1 — Semantic Analysis** (LLM, Groq): Produces a problem anatomy including sub-problems, reasoning type, information bottlenecks, and natural split axes.
- **M2 — Feasibility DAG** (deterministic): Builds a directed acyclic graph of logical dependencies between sub-problems; computes DAG closure to derive reachable CPP profiles.
- **M3 — Constraint Derivation** (deterministic): Maps a target CPP profile to per-cell design instructions and entity assignment hints.
- **M4 — Linguistic Generation** (LLM): Translates the formal specification into natural-language information packets for each agent. Critically, M4 must generate *epistemic* splits: the prompt forbids informational splits (neither agent should be able to formulate a solvable equation without the other).
- **M5 — CPP Discriminator Chain** (trained): 12 logistic regression classifiers in DAG order, each predicting one PISA cell's activation probability from TF-IDF features of simulated conversation turns. Trained on 140 annotated examples; mean Hamming error 1.11/12 on validation set.
- **M6 — GRPO Feedback** (future): Closed-loop reinforcement learning from CPP scoring to M4 generation.

### 3.3 Five Experimental Conditions

| Condition | Name | Split Design | Simulation | Theoretical Basis |
|-----------|------|-------------|------------|-------------------|
| C1 | Baseline | Standard (pipeline v1) | Jigsaw simulation | Reference condition |
| C2 | CIDI-directed | CIDI M1-M5 targeting CPP-DEEP | Standard simulation | Epistemic split theory |
| C3 | Constitutional | 36-check constitutional critic | Standard simulation | Constitutional AI critic cycle |
| C4 | Szewkis monitor | Standard split | With Szewkis monitor | Szewkis (2011) conditions |
| C5 | CIDI + Monitor | CIDI M1-M5 | With Szewkis monitor | Full integration |

All conditions use identical simulation infrastructure (same model, same max_turns, same goal-anchor). Differences are in split generation and/or dynamic monitoring only.

### 3.4 Collaborative Instruction Design: The Minimal Framing Principle

A critical methodological decision concerns the system prompt given to agents. Pilots v1 and v2 used increasingly detailed collaborative instructions (v2: explicit phase labels A–D with turn specifications, "you cannot solve this alone", "YOUR PRIVATE INFORMATION"). Analysis of v2 results revealed two validity threats:

1. **Metatextual activation**: Phrase sequences like "PHASE A — Exploration: state what you know, what you DON'T know" are verbatim descriptions of PISA Phase A from training data. Agents followed the CPS script rather than generating CPS behavior from the task structure.
2. **Discovery elimination**: Labeling information as "private" eliminates the exploration sub-phase (PISA A1) in which agents discover the information asymmetry through conversation — replacing a genuine Phase A with a scripted one.

Pilot v3 adopts the **minimal framing principle**: the system prompt states only that the activity is collaborative and that the agent can exchange messages with partners. No phase structure, no epistemic constraints stated explicitly, no information labels. The research question becomes: *does epistemic split quality determine whether collaboration emerges with minimal framing?* This is a stronger claim than *does explicit instruction produce higher CDI?*

Specifically, v3 system prompt = "You are participating in a collaborative math activity with N partner(s). You can exchange messages to work on the problem together. [context + packet]"

### 3.5 LLM Agent Simulation

Each conversation involves n=2 agents alternating turns for a maximum of max_turns=8 turns (configurable). A goal-anchor message opens the conversation with the original problem statement and the specification "Your final answer must be a single integer" — this prevents goal-anchor dilution (reporting an intermediate value instead of the required final integer).

A Phase D verification turn is injected by the simulator at the penultimate turn position: "[GROUP VERIFICATION] Re-read the original question. Both of you: do you agree on a single integer answer? State FINAL ANSWER: <integer> only when both partners agree." This turn is structural (enforced by the protocol) rather than instructional (not prescribed in the agent system prompt), preserving the minimal framing principle while ensuring Phase D activation is tested.

The Szewkis monitor (conditions C4, C5) evaluates the six collaborative conditions after Phase A and B transitions (detected via keyword scoring), injecting corrective interventions when conditions fail. A turn-delay guard (no intervention before turn 3) prevents monitor disruption of the initial information-sharing phase.

### 3.6 Metrics

**CDI (CPS Depth Index)**: Primary process metric. Σ active CPP cells / 12.

**Correctness**: Binary outcome metric. Extracted via regex matching against known answers for pilot problems.

**Collaborative Yield (CY)**: Composite process-outcome metric grounded in productive failure theory:
$$CY = 0.35 \cdot CDI + 0.45 \cdot corr_{bin} + 0.20 \cdot \delta_{coupling}$$
where $\delta_{coupling} = +1$ if CDI ≥ 0.5 AND correct (process enabled outcome), $\delta_{coupling} = -0.5$ if CDI < 0.3 AND correct (trivial or lucky), 0 otherwise.

**Process-Outcome Quadrants**: Qualitative classification of each conversation:
- **COUPLING** (CDI ≥ 0.5, correct): deep collaboration produced correct answer — target state
- **PRODUCTIVE FAILURE** (CDI ≥ 0.5, incorrect): deep collaboration without closure — valuable struggle
- **TRIVIAL** (CDI < 0.5, correct): correct answer without deep process — split too easy or lucky
- **COLLAPSE** (CDI < 0.5, incorrect): both process and outcome failed

**Epistemic fidelity indicator**: r_pb(CDI, correctness) per problem across conditions — positive significant correlation indicates the split is genuinely epistemic.

### 3.7 Pilot Problems

Four problems were selected from the corpus by ascending PISA global index (most room for improvement first), subject to having a baseline conversation available:

| ID | Problem type | Known answer | Split override? |
|----|-------------|-------------|-----------------|
| math_00014 | Quadratic roots (m+n+p form) | 26 | No |
| math_00050 | Divisibility | 8 | No |
| math_00121 | Trigonometric identities (sec/tan → csc/cot, m+n) | 44 | Yes |
| math_00128 | Factorials | 48 | No |

math_00121 requires a manually designed epistemic split (override applied for C2-C5) because the CIDI pipeline generates a data split for this problem. The override is documented and noted in results.

---

## 4. Pilot Study Results

*[Results from pilot v3 to be inserted here]*

### 4.1 Quantitative Results

#### 4.1.1 CDI and CY by condition (v1 → v2 → v3 comparison)

*[Table: mean CDI, mean CY, % correct by condition and iteration]*
*[Figure 5: bar chart CDI/CY by condition, three iterations]*

Key findings from v1 and v2 (v3 pending):
- v1 → v2: CDI improved d=1.539 (p<0.0001) with explicit CPS instruction
- v1 → v2: CDI ⊥ correctness (r=-0.015) — CDI inflation confirmed
- v2: Phase D activation 0% genuine (artifact of explicit instruction)
- v3 hypothesis: CDI gains should be smaller but more genuine; r_pb(CDI, correctness) should be positive

#### 4.1.2 Quadrant distribution

*[Table: COUPLING / PROD_FAIL / TRIVIAL / COLLAPSE per condition]*

#### 4.1.3 Epistemic fidelity by problem

*[r_pb(CDI, correctness) per problem — expected: math_00121 with override shows positive r_pb]*

### 4.2 Qualitative Analysis

#### 4.2.1 COUPLING case: math_00128 × C3
*[Conversation excerpt — what genuine coupling looks like]*
*[How the constitutional critic produced an epistemic split for this problem]*

#### 4.2.2 PRODUCTIVE FAILURE case: math_00121 (pre-override)
*[Conversation excerpt — CDI=1.000, answer "29" instead of "44"]*
*[Goal-anchor dilution: agent reports numerator of csc+cot rather than m+n]*
*[How the override resolves this]*

#### 4.2.3 Formulism vs genuine collaboration
*[Analysis of "I know X. I don't know Y. Can you tell me Z?" pattern in v2 vs v3]*
*[Evidence that minimal framing reduces formulism]*

#### 4.2.4 Phase D activation
*[Comparison: v2 Phase D from instruction vs v3 Phase D from structural turn]*

---

## 5. Discussion

### 5.1 Epistemic Interdependence as a Design Principle
The central finding across all pilot iterations is that CDI is a function of split quality, not instruction quality. No amount of collaborative framing can compensate for a data split: when the information distribution allows either agent to formulate a complete solution, CDI remains low regardless of whether agents are instructed to collaborate deeply...

### 5.2 The Minimal Framing Principle: Methodological and Practical Implications
The failure of v2's explicit CPS instruction (phases A-D with turn labels) reveals a fundamental validity threat in LLM-based CPS simulation: agents trained on educational text will follow the description of CPS phases when these are placed in their system prompt. This is instruction-following, not collaborative problem solving. The minimal framing principle — state the collaborative context, provide the problem, nothing more — makes the epistemic structure the primary driver of collaborative behavior...

### 5.3 CY and the Process-Outcome Relationship
The observed independence of CDI and correctness (r=-0.015 in pilot v2) has two interpretations. First, it may indicate that the conditions do not yet produce epistemic splits of sufficient quality for collaboration to be necessary for correct outcomes. Second, it may reflect genuine productive failure dynamics: agents that collaborate deeply on difficult problems engage in a form of mathematical struggle that is educationally valuable even without immediate success (Kapur, 2012)...

### 5.4 Implications for Collaborative Activity Design
The framework suggests a design workflow for practitioners: (1) analyze the problem for natural split axes and reasoning bottlenecks (M1); (2) verify that no single agent can formulate a solvable equation with their assigned information (epistemic fidelity test); (3) provide minimal framing without prescribing CPS phases; (4) use the Phase D verification turn as a structural closure mechanism rather than an instructional prescription...

### 5.5 Limitations
- **LLM agents as learner proxies**: LLM agents have broad mathematical knowledge not constrained to any developmental level. The study demonstrates the framework's measurement properties with LLM agents; validity with human learners requires separate investigation.
- **Pilot scale**: n=4 problems × 5 conditions = 20 conversations per iteration. Statistical power is limited; conclusions are primarily methodological.
- **Domain specificity**: Problems are competition-style algebra/trigonometry. Generalizability to other mathematical domains requires testing.

---

## 6. Conclusions and Future Work

This paper has introduced the CPP framework, the CDI metric, the CY composite, and the minimal framing principle as theoretical and methodological contributions to the design and study of AI-generated collaborative mathematical tasks. The pilot study demonstrates that genuine CPS measurement requires epistemic splits, not data splits, and that collaborative behavior emerges from task structure when framing is minimal.

Future work includes:
- Validation with human learners at defined curriculum levels (applying domain restriction, not persona, to the agent if used as a peer)
- Scale-up to n=30 problems across multiple mathematical domains
- Extension to n=3 and n=4 agent groups to test phase dynamics at higher n
- Automated epistemic fidelity scoring (r_pb estimation from M5 output)
- Curriculum mapping: applying the CPP analyzer to identify epistemic split opportunities across K-12 mathematics

---

## References

*[To be completed — key sources:]*
- OECD (2017). PISA 2015 Results (Volume V): Collaborative Problem Solving.
- Greiff, S. et al. (2014). Domain-general problem solving skills and education.
- Kapur, M. (2012). Productive failure in learning the concept of variance.
- Chi, M. T. H., & Wylie, R. (2014). The ICAP framework.
- Dillenbourg, P. (1999). What do you mean by collaborative learning?
- Szewkis, E. et al. (2011). Collaboration within large groups in the classroom.
- Kirschner, F. et al. (2018). From cognitive load theory to collaborative cognitive load theory.
- Webb, N. M. (1991). Task-related verbal interaction and mathematics learning.
- Sfard, A. (1998). On two metaphors for learning.
- Vygotsky, L. S. (1978). Mind in society.
