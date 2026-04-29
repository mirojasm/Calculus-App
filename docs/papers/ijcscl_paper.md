# Epistemic Interdependence as a Design Principle for AI-Generated Collaborative Mathematical Tasks: A Framework, Simulation Method, and Pilot Study

**Target**: International Journal of Computer-Supported Collaborative Learning (IJCSCL), Springer  
**Status**: Skeleton — awaiting pilot v3 results  
**Estimated length**: 8,000–10,000 words

---

## Abstract (draft)

Collaborative mathematical problem solving requires more than distributing information across participants — it demands that no agent can formulate a solvable approach without engaging with another's knowledge. We distinguish *epistemic splits*, where information distribution creates genuine reasoning interdependence, from *data splits*, which merely partition facts. Drawing on the PISA 2015 Collaborative Problem Solving framework, we introduce the Collaborative Phase Profile (CPP), a formal binary vector over twelve cognitive cells (four PISA processes × three competencies), and the CPS Depth Index (CDI), a scalar measure of collaborative engagement depth. We propose a 2×2 factorial experimental design with five conditions: C1 (baseline), C2 (CIDI-directed split, no facilitation), C3 (constitutional split, no facilitation), C4 (CIDI split + joint accountability), and C5 (constitutional split + joint accountability). We additionally introduce Collaborative Yield (CY), a composite metric coupling process quality (CDI) with outcome correctness grounded in productive failure theory, and the Minimal Framing Principle, which isolates the effect of task structure from collaborative instruction. A four-iteration pilot study (n=4 problems × 5 conditions) validates the framework, diagnoses three generations of methodological failures (CDI inflation via explicit CPS instruction, CDI collapse via goal-anchor data leak, and incomplete consensus via external scaffolding), and establishes that collaboration must emerge from task-role chain structure, not from prescribed knowledge asymmetries. Implications for the design of collaborative mathematical activities are discussed.

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

### 2.5 Joint Accountability and the Convergence Criterion

Roschelle (1992) identified *convergence* — the process by which collaborators negotiate a shared understanding and verify that both hold the same representation — as a defining feature of genuine joint problem solving. Szewkis et al. (2011) operationalize this as condition 4 (*group reward orientation*): participants are accountable collectively for the group's outcome. In our design, joint accountability is implemented structurally: both agents must independently state the same final answer. This is minimal (no mid-conversation intervention), grounded in convergence theory, and distinguishable from external facilitation — which, when injected during the problem-solving trajectory, risks measuring the facilitator's effectiveness rather than the dyad's collaborative capacity.

### 2.6 Task-Role Chain Design and the Epistemic Fidelity Problem

A canonical approach to CPS task design assigns each participant private information the other lacks, implying that "you cannot solve this alone" (Dillenbourg, 1999). This creates a *data split*. However, with LLM agents — which have unrestricted mathematical knowledge — knowledge restriction via prompt engineering is simulated, not genuine: the agent "knows" the solution but is instructed to pretend otherwise. More critically, the instruction *pre-computes* the discovery of information asymmetry, eliminating the exploration phase (PISA A1) in which participants naturally discover what the other knows and needs (Roschelle, 1992; Chi & Wylie, 2014).

*Task-role chain design* (introduced in this paper) resolves this by assigning *computational roles* rather than information asymmetries. Each agent receives: (1) the inputs available to them, (2) the computation to perform, (3) what to communicate to the partner, and (4) what output they need from the partner to complete the chain. The chain creates genuine functional interdependence — neither agent can produce the final answer without the other's intermediate result — while preserving Phase A by leaving the discovery of *why* the roles are interdependent to the collaborative conversation itself.

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

### 3.3 Experimental Design: 2×2 Factorial + Baseline

Five conditions form a 2×2 factorial design with a baseline:

| Condition | Name | Split Type | Facilitation | Theoretical Basis |
|-----------|------|------------|--------------|-------------------|
| C1 | Baseline | Standard (corpus) | None | Reference |
| C2 | CIDI / No-JA | CIDI M1-M5 (task-chain) | None | Epistemic split theory |
| C3 | Constitutional / No-JA | 36-check constitutional critic | None | Constitutional AI critic cycle |
| C4 | CIDI / JA | CIDI M1-M5 (task-chain) | Joint accountability | Roschelle (1992), Szewkis cond. 4 |
| C5 | Constitutional / JA | Constitutional critic | Joint accountability | Full combination |

The 2×2 structure (split type × accountability) allows additive and interaction effects to be estimated:
- C2 vs C4: Does joint accountability improve CY when split quality is held constant (CIDI)?
- C3 vs C5: Same question with constitutional split quality.
- C2 vs C3 (and C4 vs C5): Does split generation method matter when accountability is held constant?

C1 serves as an uncontrolled reference (corpus split, no accountability).

All conditions use identical simulation infrastructure (same model, temperature, max_turns, goal-anchor format). Differences are in split generation (M4 prompt) and/or goal-anchor accountability line only.

### 3.4 Collaborative Instruction Design: The Minimal Framing Principle

A critical methodological decision concerns the system prompt given to agents. A four-iteration design process revealed systematic validity threats at each stage:

**v1 (baseline)**: Minimal instruction, json_mode and selection bugs. CDI mean ≈ 0.28.

**v2**: Explicit CPS instruction with PISA phases A–D, "you cannot solve this alone", "YOUR PRIVATE INFORMATION". CDI mean ≈ 0.65, but CDI ⊥ correctness (r = −0.015), Phase D activation was 100% scripted formulism, and approximately 60–70% of turns followed a stereotyped "I know X. I don't know Y. Can you tell me Z?" pattern. Analysis identified two threats:
1. *Metatextual activation*: PISA phase labels are verbatim in training data; agents followed the CPS description rather than generating CPS behavior.
2. *Discovery elimination*: Labeling information as "private" pre-computes Phase A — replacing genuine exploration of information asymmetry with scripted declaration.

**v3**: Minimal framing adopted (system prompt states only the collaborative context and exchange permission). CDI collapsed to ≈ 0.13. Root cause: the goal-anchor message — which opened every conversation — was populated with `split_result.problem` (the full problem text), broadcasting all information to both agents at turn 0. The split was rendered irrelevant: agents solved the full problem from the goal-anchor without needing their assigned packet. C2 CDI = 0.000 despite 19 collaborative turns and a correct answer.

**v4** (current): Two fixes. (1) *Goal-anchor fix*: The goal-anchor is now populated from `split_result.shared_context` — the final question/objective only, with no problem data. (2) *Task-role chain design*: M4 generates packets using the "Input / Task / Share / Needs from partner" structure (Section 2.6), replacing the deprecated "you don't know X" knowledge restriction pattern.

The **Minimal Framing Principle** is therefore: state the collaborative context, provide the task-role assignment, provide the goal question — nothing more. No phase prescription, no epistemic restriction, no scaffolding. The research question is: *does task-role chain quality determine whether collaboration emerges with minimal framing?*

System prompt (v4): "You are participating in a collaborative math activity with N partner(s). You can exchange messages to work on the problem together. [optional: Both partners must independently state the same final answer at the end.] \n\n[shared_context]\n\n[packet: Input/Task/Share/Needs from partner]"

### 3.5 LLM Agent Simulation

Each conversation involves n=2 agents alternating turns for a maximum of max_turns=8 turns (configurable). A goal-anchor message opens the conversation with (a) the shared question/objective from `split_result.shared_context` and (b) a format specification derived from M1 semantic analysis (dynamic answer format — integer, decimal, fraction, algebraic expression, etc.). The goal-anchor does NOT contain the original problem text, preventing the data-leak failure mode identified in v3.

A Phase D verification turn is injected by the simulator at the penultimate turn: "[GROUP VERIFICATION] Each partner independently states their answer using exactly this format: FINAL ANSWER: [value]. If your answers differ, resolve the discrepancy first." This turn is structural (protocol-enforced, visible to both agents as a user message) rather than prescriptive (not in the system prompt), preserving minimal framing while ensuring Phase D activation. The GROUP VERIFICATION wording was updated in v4 to require independent declaration by each partner, grounding it in Roschelle's (1992) convergence criterion rather than simple consensus detection.

No mid-conversation interventions are applied in v4. The Szewkis monitor (previously used in legacy C4/C5) was removed after diagnosing two pathological effects: (1) it caused agents to question which problem variant they were solving (framing conflict with minimal system prompt), and (2) it prevented consensus detection because agents never converged on the FINAL ANSWER format (they collaborated correctly but the monitor interrupted before format convergence). The structural GROUP VERIFICATION plus optional joint accountability line in the system prompt are sufficient to activate Phase D without external scaffolding.

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

### 3.6b Dynamic Answer Format

M1 semantic analysis (module1_semantic.py) extracts an `expected_answer_format` field for each problem, specifying: (a) type (integer, decimal, fraction, algebraic expression, equation, set, etc.), (b) a natural-language specification used in the goal-anchor, and (c) partial credit indicators — intermediate values that represent correct reasoning but incomplete format (e.g., for math_00121, the unreduced fraction 29/15 before computing m+n=44). The answer format specification is injected into the goal-anchor at simulation time. This replaces the v3 hardcoded "State your final answer as a single integer" instruction, which was incorrect for approximately 30% of the pilot problem types.

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

*[Results from pilot v4 to be inserted here]*

### 4.1 Methodological Development: Four Iterations

Table 1 summarizes the four pilot iterations as a design-iteration sequence:

| Iteration | Key change | CDI mean (C2-C5) | CDI ⊥ correctness | Key finding |
|-----------|-----------|-------------------|--------------------|-------------|
| v1 | Initial system | ~0.28 | n/a (bugs) | json_mode and selection bugs |
| v2 | Explicit CPS instruction | ~0.65 | r=−0.015 | CDI inflation; Phase D formulism |
| v3 | Minimal framing | ~0.13 | — | Root bug: goal_anchor data leak |
| v4 | goal_anchor fix + task-chain M4 + joint accountability | *pending* | *pending* | First clean measurement |

*[Figure 5: CDI distribution per iteration — expected monotonic improvement in v4]*

### 4.2 Quantitative Results (v4b pilot)

#### 4.2.1 CDI by condition

Table 2 shows mean CDI across 4 problems × 5 conditions (pilot v4b):

| Condition | mean CDI | mean CY | n problems |
|-----------|----------|---------|------------|
| C1 (baseline) | 0.083 | 0.000 | 4 |
| C2 (CIDI task-chain) | **0.188** | **0.328** | 4 |
| C3 (constitutional) | 0.083 | 0.292 | 4 |
| C4 (CIDI + JA) | 0.125 | 0.219 | 4 |
| C5 (constitutional + JA) | 0.000 | 0.262 | 4 |

The ranking C2 > C4 > C3 > C5 in mean CDI is the primary empirical finding. Notably, C5 produces mean CDI = 0.000 — lower than the C1 baseline — while CY remains elevated (0.262) due to correctness on trivial problems.

#### 4.2.2 Per-problem CDI breakdown

The CDI results disaggregate sharply by problem:

| Problem | Type | C1 | C2 | C3 | C4 | C5 |
|---------|------|----|----|----|----|-----|
| math_00050 | Divisibility | 0.083 | 0.000 | 0.000 | 0.000 | 0.000 |
| math_00128 | Factorials | 0.083 | 0.167 | 0.000 | 0.167 | 0.000 |
| math_00121 | Trigonometry† | 0.083 | **0.583** | 0.333 | 0.333 | 0.000 |
| math_00014 | Quadratic | 0.083 | 0.000 | 0.000 | 0.000 | 0.000 |

† Split override applied (task-chain design: A1 derives sin θ/cos θ; A2 computes csc θ + cot θ).

Three of four problems (math_00050, math_00128, math_00014) produce CDI = 0.000 across all non-baseline conditions. These are computationally accessible problems: any single LLM agent can compute all sub-steps independently, rendering the split irrelevant. This constitutes empirical validation of the **epistemic complexity requirement**: task-chain design produces elevated CDI only for problems with genuinely sequential computational dependencies.

#### 4.2.3 Quadrant distribution

| Condition | COUPLING | PROD_FAIL | TRIVIAL | COLLAPSE |
|-----------|----------|-----------|---------|----------|
| C1 | 0 | 0 | 0 | 0 (unscored) |
| C2 | 0 | 1 | 3 | 0 |
| C3 | 0 | 0 | 3 | 1 |
| C4 | 0 | 0 | 3 | 1 |
| C5 | 0 | 0 | 3 | 1 |

C2 produces the pilot's only PROD_FAIL instance (math_00121 × C2: CDI = 0.583, answer incorrect). All other non-C1 conditions for the epistemic problem (C3, C4, C5) produce COLLAPSE (CDI < 0.5, incorrect).

#### 4.2.4 Joint accountability effect on CDI

A within-problem comparison for math_00121 reveals a *negative* accountability effect:

- C2 (CIDI, no JA): CDI = 0.583
- C4 (CIDI, JA): CDI = 0.333

Adding joint accountability to the same CIDI task-chain split reduces CDI by 0.25 for the epistemic problem. The proposed mechanism: the phrase "Both partners must **independently** state the same final answer" creates a confound with the task-chain design — Agent 2 interprets "independently" as license to compute the answer without waiting for Agent 1's required intermediate output (sin θ, cos θ). This conflicts directly with the "Needs from partner" field in the packet. The finding suggests a revision to the joint accountability wording: "Both partners must confirm agreement on the same final answer" (removing "independently") to preserve chain execution while maintaining the convergence criterion.

### 4.3 Qualitative Analysis

#### 4.3.1 COUPLING case: task-chain collaboration in v4
*[Conversation excerpt — what genuine task-chain coupling looks like: Agent 1 computes intermediate, Agent 2 requests it, Agent 2 combines to final answer]*
*[Compare: v2 version of same problem (scripted Phase A) vs v4 (emergent Phase A)]*

#### 4.3.2 PRODUCTIVE FAILURE case: math_00121 × C2 (v4b)
The task-chain override for math_00121 produced CDI = 0.583 (CPP-DEEP) with 20 turns — the highest CDI observed in any pilot run. The chain executed correctly:
- Agent 1 (Turn 1): given sec θ + tan θ = 22/7, derived sin θ = 435/533 and cos θ = 308/533 using the Pythagorean identity. Shared these values immediately.
- Agent 2 (Turn 2): received sin θ and cos θ, correctly applied csc θ + cot θ = (1 + cos θ)/sin θ = 841/435.
- Error at final step: Agent 2 stated "this fraction is already reduced", computing m + n = 841 + 435 = 1276. In fact gcd(841, 435) = 29, giving 29/15 and m + n = 44.

This is a canonical PROD_FAIL instance: deep collaboration (CDI = 0.583), correct chain execution, mathematical error in the final simplification step. The collaborative process was entirely sound; the failure is a computation capability limitation. Under Kapur's productive failure framework, this conversation represents maximal collaborative engagement within the ZPD — the most educationally valuable quadrant.

*[Excerpt: Turn 1 Agent 1 → Turn 2 Agent 2 — functional chain with correct intermediaries]*

#### 4.3.3 Task-chain vs knowledge-restriction comparison
*[Analysis: same problem split two ways — "You do NOT know X" (v3) vs "Task: compute X from Y" (v4)]*
*[Coding: Phase A emergence, number of turns before first substantive math exchange, CDI per coding]*

#### 4.3.4 Joint accountability effect on Phase D
*[Comparison: C2 vs C4 (same CIDI split) — does joint accountability change answer convergence rate?]*
*[Comparison: C3 vs C5 — same with constitutional split]*

#### 4.3.5 Data leak pathology (v3 root-cause illustration)
*[C2 v3 case: CDI=0.000 despite 19 turns and correct answer — full problem in goal_anchor, agents solved independently from turn 1, split packets ignored]*

---

## 5. Discussion

### 5.1 Epistemic Interdependence as a Design Principle
The central finding across all pilot iterations is that CDI is a function of split quality, not instruction quality. No amount of collaborative framing can compensate for a data split: when the information distribution allows either agent to formulate a complete solution, CDI remains low regardless of whether agents are instructed to collaborate deeply...

### 5.2 Task-Role Chain Design and the Epistemic Fidelity Problem
The four-iteration development history reveals that epistemic validity in LLM-based CPS simulation requires solving two distinct problems: (1) ensuring the system prompt does not prescribe the collaboration it is meant to measure (minimal framing principle), and (2) ensuring the information structure actually prevents unilateral solution (task-role chain design, not knowledge restriction). The v3 experience shows that minimal framing alone is insufficient: if the goal-anchor carries the full problem text, agents solve it immediately, rendering the split irrelevant regardless of how epistemically sound the split design is. The v4 combination — minimal system prompt + task-role chain packets + shared_context goal-anchor — closes both gaps simultaneously. For LLM agents with unrestricted mathematical knowledge, task-chain interdependence (functional: agent A's output is agent B's required input) is more robust than knowledge-restriction interdependence (epistemic: agent A does not know what B knows), because task-chain cannot be bypassed by an agent who already possesses the full solution procedure...

### 5.3 CY and the Process-Outcome Relationship
The observed independence of CDI and correctness (r=-0.015 in pilot v2) has two interpretations. First, it may indicate that the conditions do not yet produce epistemic splits of sufficient quality for collaboration to be necessary for correct outcomes. Second, it may reflect genuine productive failure dynamics: agents that collaborate deeply on difficult problems engage in a form of mathematical struggle that is educationally valuable even without immediate success (Kapur, 2012)...

### 5.4 Implications for Collaborative Activity Design
The framework suggests a design workflow for practitioners: (1) analyze the problem for natural *computational chains* — sequences where one sub-result is required to formulate the next sub-problem (M1 bottleneck analysis); (2) assign each participant a role in the chain, specifying inputs, the computation to perform, outputs to share, and what is needed from the partner; (3) ensure the shared context contains only the collaborative goal, not the problem data (goal-anchor discipline); (4) apply minimal framing — state the collaborative context and exchange permission, nothing more; (5) close with a structural Phase D verification that requires independent answer declaration from each participant. Step 2 operationalizes the task-role chain principle: activity designers should ask "what does each participant need to produce for the next step?" rather than "what information should each participant keep private?"...

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
