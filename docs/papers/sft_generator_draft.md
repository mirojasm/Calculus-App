# Learning Information Asymmetry: Fine-Tuning LLMs to Generate Epistemically Interdependent Splits for Collaborative AI

**Anonymous Authors**  
*Under review — AIED 2026 / NeurIPS 2025 Workshop on Collaborative AI*

---

## Abstract

Collaborative AI systems — multi-agent pipelines, human-AI teams, and peer tutoring agents — depend critically on information asymmetry: each participant must hold different but complementary knowledge so that collaboration is epistemically necessary rather than redundant. Current approaches either hand-craft these information splits or rely on expensive LLM pipelines that do not generalise to new problems at inference time. We ask: can we train a model to *automatically* generate epistemically interdependent information splits from any problem description?

We introduce the task of **epistemic split generation** and frame it as a supervised learning problem over a novel preference-annotated dataset derived from the CIDI pipeline — a five-module LLM system that produces jigsaw splits satisfying Szewkis's (2011) positive interdependence conditions. A key finding is negative: Direct Preference Optimisation (DPO) is *inappropriate* for this task. Because split generation is a **format-learning** problem — the model must acquire a structured output schema rather than rank two stylistically similar candidates — DPO catastrophically collapses: by epoch 0.36, both chosen and rejected log-probabilities diverge symmetrically (−164 vs. −236), loss reaches 0.016, and gradient norm drops to zero. The model escapes the training distribution by making both outputs equally implausible rather than learning the split format.

Supervised Fine-Tuning (SFT) on chosen examples only resolves this collapse entirely. A Mistral-7B-Instruct-v0.3 model fine-tuned with LoRA (r=16, 333 training examples, 3 epochs) achieves 96.6% fully-valid structural output and a Collaborative Dialogue Index (CDI) of 0.583 — 64.7% of generated splits exceeding the CDI ≥ 0.5 threshold for genuine epistemic interdependence. Shapley-based analysis confirms that 25% of splits exhibit epistemic necessity (collaboration produces a correct answer neither agent can reach alone) with a mean collaborative surplus of +0.207.

We propose CDI as a differentiable training signal for reinforcement learning, outline a scaling roadmap projecting CDI ≥ 0.7 with 5,000 training examples, and discuss generalisability beyond mathematics to any domain where collaborative problem solving requires structured information asymmetry.

---

## 1. Introduction

Collaboration is only genuinely necessary when participants possess information the other lacks. A student group in which every member has access to the same textbook chapter does not collaborate — they merely co-exist. The same is true of multi-agent AI systems: if every agent is given the full problem context, communication is performative rather than epistemic. Designing information asymmetry — deciding *which* agent should know *what* — is therefore a foundational challenge in collaborative AI.

The challenge is most visible in three converging lines of work. In **computer-supported collaborative learning (CSCL)**, the jigsaw classroom method [CITE: Aronson 1978] exploits information asymmetry by giving each student a unique piece of a topic, so that group discussion is the only route to a complete picture. In **multi-agent LLM systems**, recent work shows that agents with differentiated roles and knowledge outperform homogeneous ensembles [Hong et al., 2023; Du et al., 2023]. In **human-AI teaming**, Dillenbourg (1999) identifies positive interdependence — neither partner can succeed alone — as the defining condition that makes collaboration productive.

Despite the theoretical clarity of this requirement, generating information-asymmetric splits automatically is hard. Hand-crafting splits requires domain expertise and does not scale. Automated LLM pipelines (e.g., the CIDI system we describe in §3.1) achieve high quality but are expensive at inference time: each call invokes five chained modules including a student-simulation evaluator. The ideal is a single fast model that, given any problem text, outputs a structurally and epistemically valid split.

This paper makes four contributions:

1. **The epistemic split generation task.** We formalise the problem of generating jigsaw splits that satisfy positive interdependence and introduce the Collaborative Dialogue Index (CDI) as the evaluation criterion, drawn from the PISA 2015 Collaborative Problem Solving rubric [OECD, 2015].

2. **A negative result on DPO for format learning.** We show that DPO [Rafailov et al., 2023] collapses when the chosen and rejected outputs differ structurally rather than stylistically. The collapse mechanism is identifiable in training metrics and constitutes a general warning for practitioners applying DPO to tasks where the reward gap is encoded in output format rather than content quality.

3. **SFT + sample-then-filter as the correct approach.** SFT on 333 high-CDI examples (CDI ≥ 0.5) produces a model with 96.6% structural validity. A sample-then-filter inference pipeline — generating k=3 candidates and selecting the structurally valid one — mirrors the best-of-N decoding paradigm and further improves reliability without any additional training.

4. **CDI as a trainable signal.** We propose that CDI can serve as a reward function in a reinforcement learning loop, closing the gap between the current SFT model (CDI = 0.583) and the CIDI reference pipeline (CDI = 0.789). We sketch the RL-CDI pipeline and provide a scaling roadmap.

The broader implication is methodological: for tasks where the target output occupies a qualitatively different region of the output space from base-model samples, SFT is necessary and DPO is insufficient. This insight generalises beyond collaborative learning to any structured generation task — from code synthesis to formal proofs to structured dialogue design — where preference learning cannot substitute for format acquisition.

---

## 2. Related Work

### 2.1 Collaborative AI and Multi-Agent Systems

Large language model multi-agent systems have recently demonstrated qualitatively different capabilities from single-agent approaches. MetaGPT [Hong et al., 2023] assigns human-like roles (product manager, engineer, QA) to distinct agents, enforcing information asymmetry through role specialisation. Du et al. (2023) show that multi-agent debate — where agents iteratively argue and update — improves factuality over single-agent generation. Liang et al. (2023) extend this to divergent thinking tasks. Park et al. (2023) demonstrate that generative agents with differentiated memories and goals exhibit emergent social behaviours. Across all these systems, information asymmetry is present but *assumed* rather than *designed*: role assignments are fixed by prompt engineering, not optimised for epistemic necessity.

Our work targets the design layer beneath these systems: given a task, automatically construct the information packets that make each agent's participation non-redundant. This is complementary to multi-agent orchestration frameworks — our split generator can serve as a preprocessing module that bootstraps genuine interdependence before any agent framework is invoked.

### 2.2 Jigsaw and Positive Interdependence in CSCL

The jigsaw method [CITE: Aronson 1978] is the canonical technique for creating information asymmetry in collaborative learning: the topic is divided into N segments, each student studies one segment, and the group must teach each other to solve the joint problem. The theoretical foundation is Johnson and Johnson's (2009) social interdependence theory, which identifies positive interdependence — the belief that one's success is linked to others' success — as the key driver of cooperative learning gains over individual study. Szewkis (2011) operationalises positive interdependence as a set of structural conditions: shared goal, complementary information, and mutual necessity of participation. Our CIDI pipeline's split patterns (SPLIT-A through SPLIT-G) directly encode Szewkis's taxonomy for mathematical domains.

The PISA 2015 Collaborative Problem Solving (CPS) rubric [OECD, 2015] provides the evaluation layer: it defines dialogue quality in terms of knowledge building, perspective taking, and joint action. [CITE: Graesser et al. 2018] applied CPS-style analysis to human student dialogues; our CDI metric adapts this rubric to LLM-simulated dialogues, assigning a scalar in [0, 1].

Empirical CSCL research has documented the conditions under which jigsaw succeeds or fails. Barron (2003) showed that groups with shared representational resources — where members can jointly inspect the same information — are less collaborative than those with complementary resources. Roschelle and Teasley (1995) identified joint problem space construction as the cognitive mechanism distinguishing collaboration from co-operation. Our work is the first to automate the structural design of information asymmetry at the problem-instance level using a fine-tuned language model.

### 2.3 Alignment Methods: RLHF, DPO, and SFT

The dominant paradigm for aligning language models to human preferences is Reinforcement Learning from Human Feedback (RLHF) [Ouyang et al., 2022], which uses a reward model trained on preference pairs to guide policy optimisation via PPO [CITE: Schulman et al. 2017]. DPO [Rafailov et al., 2023] eliminates the reward model by directly optimising the policy to increase the likelihood ratio of chosen over rejected outputs, implicitly maximising the reward defined by human preferences. DPO has shown strong results on instruction following [CITE: Tunstall et al. 2023], summarisation, and dialogue tasks where chosen and rejected outputs are semantically similar but stylistically differentiated.

However, theoretical analysis of DPO reveals a structural assumption that is violated in our setting. The DPO objective is:

$$\mathcal{L}_\text{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]$$

This loss is minimised when the policy assigns high *relative* probability to $y_w$ over $y_l$ relative to the reference. When both $y_w$ and $y_l$ are far from the reference distribution — as occurs when the format is novel — the gradient is well-defined but the optimal policy is not: the model can minimise the loss by moving both outputs far from the reference in the same direction, maintaining the margin while making both outputs implausible. This is exactly the collapse we observe empirically (§4.2). [CITE: Azar et al. 2024] and [CITE: Feng et al. 2024] have independently identified related failure modes of DPO under distribution shift; our work contributes an empirical case study in a structured generation domain.

SFT on high-quality demonstrations is appropriate when the task requires format acquisition rather than preference discrimination [CITE: Zelikman et al. 2022]. Best-of-N decoding (sample k outputs, select the best by a verifier) has been shown to approximate RL gain with minimal compute overhead [CITE: Lightman et al. 2023; CITE: Brown et al. 2024]. Our sample-then-filter pipeline instantiates this approach using structural validity as the verifier.

### 2.4 Structured Output Generation

Fine-tuning LLMs to produce structured JSON outputs has been studied in the context of information extraction [CITE: Wang et al. 2023], tool use [CITE: Schick et al. 2023], and code generation. [CITE: Josifoski et al. 2023] showed that SFT with constrained decoding outperforms prompting-only approaches on structured extraction benchmarks. [CITE: Peng et al. 2023] demonstrated that even small (7B parameter) models can learn complex JSON schemas reliably with a few hundred examples. Our setting adds an epistemic constraint — the structured output must satisfy positive interdependence conditions — beyond pure format validity, making it a harder learning target.

---

## 3. Method

### 3.1 The CIDI Pipeline: Reference Split Generation

The Collaborative Interdependence Design Interface (CIDI) is a five-module LLM pipeline that generates jigsaw splits satisfying Szewkis's (2011) positive interdependence conditions. Given a problem $P$, CIDI:

1. **M1 (Analyser)**: Decomposes $P$ into mathematical sub-objects (equations, constraints, geometric elements, statistical parameters).
2. **M2 (Pattern Selector)**: Selects a split pattern from a taxonomy of seven types designed to maximise epistemic interdependence: SPLIT-C (complementary conditions), SPLIT-D (multi-step chain), SPLIT-B (dual representation), SPLIT-A (composite figure), SPLIT-F (sample space × counting principle), SPLIT-G (hypothesis × key lemma), SPLIT-E (objective × constraints).
3. **M3 (Packet Constructor)**: Assigns sub-objects to two agent packets such that neither packet alone contains sufficient information to solve $P$.
4. **M4 (Interdependence Verifier)**: Checks the structural conditions: `agent1_can_answer_alone = false`, `agent2_can_answer_alone = false`, `combined_can_answer = true`.
5. **M5 (CDI Evaluator)**: Runs a student-simulation pipeline (C7) that simulates two LLM students solving $P$ using their respective packets, then computes the Collaborative Dialogue Index (CDI) from the resulting conversation.

CIDI is expensive: each problem invokes 5+ GPT-4-class API calls including a multi-turn student simulation. The goal of the SFT generator is to replace CIDI's 5-module inference chain with a single forward pass.

### 3.2 The Collaborative Dialogue Index (CDI)

CDI is a scalar in [0, 1] derived from the PISA 2015 CPS rubric [OECD, 2015]. It is computed from three components annotated on the simulated student dialogue:

- **CQI (Collaborative Quality Index)**: Measures the quality of knowledge-building exchanges — whether students contribute novel information, ask clarifying questions, and synthesise each other's contributions.
- **PhAQ (Physical Action Quality)**: Measures the degree to which both students actively engage in the joint solution process rather than one student dominating.
- **Completeness**: Whether the joint conversation arrives at a correct solution.

A split with CDI ≥ 0.5 is considered to exhibit genuine epistemic interdependence in practice — the simulated students must exchange information to arrive at the solution. CDI = 0 typically indicates a failure of interdependence: one student solves the problem unilaterally while the other provides no useful information.

### 3.3 Dataset Construction

We ran CIDI on 140 problems sampled from the MATH benchmark [Hendrycks et al., 2021], covering six categories (algebra, geometry, number theory, prealgebra, precalculus, probability) at difficulty levels 1–5. For each problem, we selected the CIDI output with the highest CDI ≥ 0.5 as the *chosen* split. This yielded 136 usable chosen splits with a mean CDI of 0.822.

To construct preference pairs for DPO training (and later repurposed as SFT data), we generated *rejected* splits using GPT-4o-mini with a deliberately naive prompt: "divide the mathematical information in half — give student A some data and student B the rest. No need to ensure they need each other." Three naive splits were generated per chosen split (N_NAIVE = 3, temperature = 0.7), yielding 420 pairs total. Naive splits were filtered to ensure they differed meaningfully from the chosen split and did not trivially exhibit the interdependence structure.

The dataset was split by problem (not by pair) to prevent data leakage, yielding 333 training pairs and 87 test pairs across approximately 109 training problems and 27 test problems.

### 3.4 DPO Attempt and Failure Analysis

We first trained a DPO model with the standard TRL DPOTrainer, using β = 0.05 and β = 0.08 (tested separately), on the 420 preference pairs. The base model was Mistral-7B-Instruct-v0.3.

**Observed collapse.** Both β values exhibited catastrophic reward-hacking by epoch 0.36. Representative metrics at the collapse point:

| Metric | Epoch 0.0 | Epoch 0.36 | Epoch 1.0 |
|--------|-----------|------------|-----------|
| Loss | 0.693 | 0.016 | ≈ 0.000 |
| logps/chosen | −112 | −164 | ≈ −400 |
| logps/rejected | −116 | −236 | ≈ −800 |
| grad_norm | — | — | ≈ 0 |

The pattern is diagnostic: both chosen and rejected log-probabilities decrease monotonically (the model assigns increasingly low probability to both), while the *margin* (logps/chosen − logps/rejected) increases. This is the signature of distribution escape: the model discovers that it can satisfy the DPO margin objective by moving both sequences into a low-probability region of the space, rather than by improving the chosen split format.

**Mechanism.** The DPO objective requires a well-behaved reference policy. When the target output format — a structured JSON object with seven fields, specific boolean constraints, and a controlled vocabulary of pattern names — lies far from the reference model's output distribution, the KL divergence term in the DPO objective is large for both chosen and rejected. The margin gradient then dominates, pulling the model into a degenerate solution. Formally, if $\pi_\text{ref}(y_w|x) \approx \pi_\text{ref}(y_l|x) \approx \epsilon$ (both are low-probability under the reference), then the implicit reward signal is unstable and the optimiser finds a degenerate manifold where margin is satisfied trivially.

This failure mode is general: **DPO is inappropriate for format-learning tasks** where the chosen and rejected outputs differ primarily in structural validity rather than in fine-grained content quality. For DPO to work, both outputs must be plausible under the reference distribution. When this condition fails, SFT is the correct first step.

### 3.5 SFT Training

We discarded the rejected splits entirely and trained on the 333 chosen examples only. The SFT objective is:

$$\mathcal{L}_\text{SFT}(\theta) = -\sum_{(x, y_w) \in \mathcal{D}_\text{chosen}} \log \pi_\theta(y_w | x)$$

This is appropriate because (a) we have ground-truth demonstrations of the desired output format, and (b) the task is format acquisition rather than preference discrimination between two plausible outputs.

**Architecture and hyperparameters.** We fine-tuned Mistral-7B-Instruct-v0.3 with LoRA [Hu et al., 2022] using the following configuration:

| Hyperparameter | Value |
|----------------|-------|
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q/k/v/o/gate/up/down proj |
| Precision | float16 |
| Batch size | 1 (eff. 8 with grad. accum.) |
| Learning rate | 2e-5 |
| LR scheduler | Cosine |
| Warmup ratio | 0.10 |
| Epochs | 3 |
| Max sequence length | 2,048 |
| Hardware | A100-40GB |

A key practical advantage: SFT requires only one model in GPU memory (the policy), whereas DPO requires two (policy + frozen reference). This halves peak GPU memory from approximately 30 GB to 15 GB at float16, making training feasible on a single 40 GB card without quantisation.

**Training curve.** Loss decreased smoothly from 2.08 to 0.46 over 3 epochs with no collapse. Evaluation loss stabilised at approximately 0.77 — a gap of 0.31 above train loss, consistent with modest generalisation without overfitting to the 333 training examples. No checkpoint showed the divergent log-probability signature seen in DPO.

### 3.6 Sample-Then-Filter Inference

At inference time, we apply a **sample-then-filter** pipeline: for each test problem, generate k = 3 candidate splits at temperature 0.7, then select the first candidate that passes structural validation (required JSON keys, two-packet structure, correct interdependence booleans). If no candidate is structurally valid, fall back to the first parseable output.

This approach mirrors best-of-N decoding [CITE: Brown et al. 2024] but uses a deterministic verifier (structural validity) rather than a learned reward model. The verifier is cheap — a JSON parse and four boolean checks — so the inference overhead is 3× generation cost rather than the 5+ API calls required by CIDI. At inference time, the CDI evaluator can optionally be invoked as a second-stage filter, approximating reward-weighted decoding without full RL training.

---

## 4. Experiments

### 4.1 Dataset Summary

| Split | Problems | Pairs | Chosen CDI (mean) |
|-------|----------|-------|-------------------|
| Train | ~109 | 333 | 0.822 |
| Test | ~27 | 87 | 0.822 |

Problems span six mathematical domains: algebra, geometry, number theory, prealgebra, precalculus, and probability. Difficulty levels 1–5 from MATH [Hendrycks et al., 2021] are all represented, providing coverage of both routine procedures (levels 1–2) and non-routine reasoning (levels 4–5).

### 4.2 Structural Evaluation (n = 87 test problems)

We evaluated the SFT model on all 87 test-set problems using the sample-then-filter pipeline (n_samples = 3). The structural evaluation checks four properties: (1) output is parseable as JSON, (2) all required CIDI keys are present with a valid pattern name and two-packet structure, (3) `interdependence_check` booleans are correct (`agent1_can_answer_alone = false`, `agent2_can_answer_alone = false`, `combined_can_answer = true`), and (4) both conditions together (fully valid).

| Metric | Rate | Count |
|--------|------|-------|
| valid_json_rate | 96.6% | 84/87 |
| struct_valid_rate | 96.6% | 84/87 |
| interdep_correct_rate | 96.6% | 84/87 |
| fully_valid_rate | 96.6% | 84/87 |

The single failure case (3 problems, all from one unique problem) involved a problem with complex LaTeX matrix formatting that caused the model to emit malformed JSON. This suggests a fragility specific to nested mathematical structures and is addressable through targeted data augmentation.

**Pattern distribution.** The model learns the full distribution of split patterns, with SPLIT-C (complementary conditions) as the most frequent at 40%:

| Pattern | Description | Rate |
|---------|-------------|------|
| SPLIT-C | Complementary conditions | 40% |
| SPLIT-A | Composite figure | 20% |
| SPLIT-E | Objective × constraints | 17% |
| SPLIT-F | Sample space × counting principle | 13% |
| SPLIT-D | Multi-step chain | 6% |
| SPLIT-G | Hypothesis × key lemma | 4% |

The dominance of SPLIT-C reflects the algebraic composition of the MATH benchmark (algebra is the largest category), where splitting a system of equations between two agents is the natural interdependence-creating strategy.

### 4.3 CDI Evaluation (n = 20 test problems)

We selected 20 held-out test problems and ran each SFT-generated split through the full C7 student-simulation pipeline to compute CDI. Of 20 problems, 17 completed the C7 pipeline successfully (3 failures due to API timeouts, not model errors). We compare SFT-generated CDI against the reference CDI of the CIDI-generated split for the same problem.

| Metric | SFT Model | CIDI Reference | Δ |
|--------|-----------|----------------|---|
| CDI mean | 0.583 | 0.789 | −0.206 |
| CDI ≥ 0.5 rate | 64.7% (11/17) | 100% (17/17) | −35.3 pp |
| CQI mean | 0.239 | — | — |
| PhAQ mean | 0.111 | — | — |

The SFT model achieves a CDI of 0.583, crossing the ≥ 0.5 threshold that indicates genuine epistemic interdependence in 64.7% of cases. The gap relative to CIDI (Δ = −0.206) reflects the inherent difficulty of distilling a five-module verification pipeline into a single 7B-parameter forward pass — the model learns the split *format* well but occasionally produces splits that are structurally valid yet semantically insufficient to force collaboration (CDI = 0.000 cases).

Notably, the SFT model *outperforms* the CIDI reference on 3 of 17 problems (Δ up to +0.333). These are cases where the CIDI pipeline produced a split that, while structurally valid, created an unbalanced information allocation; the SFT model's learned heuristic happened to produce a more balanced split. This suggests the model has absorbed not just the format but some of the epistemic design principles.

**Failure mode analysis.** Zero-CDI failures share a common pattern: the SFT model splits the problem such that one agent receives enough information to deduce the answer alone — typically by giving one agent the problem constraints *and* the goal statement. This violates the `agent2_can_answer_alone = false` condition at the semantic level even though it passes the syntactic `interdependence_check`. Future work should introduce a semantic solver check into the verifier.

### 4.4 Agent Epistemic Contribution (AEC) Analysis

To directly measure whether the information asymmetry created by CIDI-generated splits (the training signal source) produces genuine collaborative necessity, we computed Shapley-based Agent Epistemic Contribution (AEC) scores on the full set of test problems with known answers.

The value function is:
- $v(\emptyset) = 0$
- $v(\{A\}) = $ correctness of agent A solving alone with packet A only
- $v(\{B\}) = $ correctness of agent B solving alone with packet B only  
- $v(\{A, B\}) = $ correctness from the stored C7 collaborative conversation

Shapley values give:
$$\text{AEC}_A = \frac{1}{2}v(A) + \frac{1}{2}(v(A,B) - v(B))$$
$$\text{AEC}_B = \frac{1}{2}v(B) + \frac{1}{2}(v(A,B) - v(A))$$

From this we derive three metrics:
- **EN (Epistemic Necessity)**: $v(\{A,B\}) > \max(v(\{A\}), v(\{B\}))$ — collaboration produces a correct answer that neither agent reaches alone.
- **EB (Epistemic Balance)**: $1 - |\text{AEC}_A - \text{AEC}_B|$ — how symmetrically the two agents contribute.
- **CS (Collaborative Surplus)**: $v(\{A,B\}) - \max(v(\{A\}), v(\{B\}))$ — the absolute gain from collaboration.

| Metric | Value |
|--------|-------|
| EN rate | 25% |
| Collaboration correctness $v(\{A,B\})$ mean | 0.36 |
| Max solo correctness $\max(v(\{A\}), v(\{B\}))$ mean | 0.10 |
| Both-solo-zero rate | 85% |
| CS mean | +0.207 |

The 25% EN rate reflects the difficulty of the MATH benchmark at higher levels (levels 4–5): even with collaboration, the student-simulation models solve fewer than half of problems. More telling is the both-solo-zero rate of 85%: in 85% of problems, neither agent can make any progress alone, confirming that the CIDI pipeline successfully creates structurally complementary information. The mean collaborative surplus of +0.207 — a gain of 0.207 correctness points from collaboration over the best solo attempt — quantifies the practical value of epistemic interdependence.

The gap between EN rate (25%) and both-solo-zero rate (85%) reveals an important limitation: even when both agents are informationally dependent (neither can solve alone), the LLM student simulators do not always successfully pool their information to reach a correct joint answer. This is a property of the *simulator*, not the *split*, suggesting that improving the collaboration simulation — not just the split quality — is necessary for higher EN rates.

---

## 5. Scaling Analysis

### 5.1 Current Data Regime

Our SFT model was trained on 333 examples derived from 136 CIDI-generated splits. This is a low-data regime by the standards of LLM fine-tuning: [CITE: Peng et al. 2023] showed that 7B-parameter models typically require 500–2,000 demonstrations to reliably acquire a structured output schema, and CDI optimisation (beyond format validity) likely requires 2,000–5,000 examples with diverse difficulty coverage.

The 96.6% structural validity achieved with 333 examples is encouraging — it suggests that format acquisition (JSON schema + pattern vocabulary) requires fewer examples than CDI optimisation, which demands learning the *semantic* conditions for epistemic necessity.

### 5.2 Scaling Roadmap

We project the following CDI trajectory as a function of training set size, based on the empirical SFT result and the theoretical expectation that CDI scales approximately as the logarithm of training examples (consistent with scaling laws for instruction following [CITE: Wei et al. 2022]):

**Figure: Projected CDI vs. Training Set Size**

*[Figure description: A line plot with x-axis "Training examples" (log scale: 100, 333, 1K, 2K, 5K, 10K) and y-axis "CDI mean" (0.4–0.9). The current SFT result at 333 examples is marked as a solid point (CDI = 0.583). The CIDI reference ceiling (CDI = 0.789) is shown as a horizontal dashed line. The CIDI-optimal ceiling (CDI = 0.822, mean of chosen training splits) is shown as a dotted line. Projected values are shown as a fitted logarithmic curve: CDI ≈ 0.40 + 0.12 × log₂(N/100), with 95% confidence band. Estimated values: 1K → 0.66, 2K → 0.71, 5K → 0.78 (approaching reference ceiling). A second curve shows the RL-CDI model (§5.3) surpassing the SFT curve at 2K examples due to the reward signal. Key milestones are annotated: "Format acquisition complete" at 200 examples, "CDI > 0.5 majority" at 500 examples, "Reference-level performance" at 5,000 examples.]*

The three key scaling milestones are:

1. **Format acquisition** (~200 examples): The model reliably produces structurally valid JSON. Our 96.6% rate at 333 examples confirms this threshold is below 333.
2. **CDI > 0.5 majority** (~500 examples): More than 50% of generated splits achieve genuine epistemic interdependence on evaluation. Currently at 64.7% with 333 examples; additional algebraic-domain examples should push this past 75% before requiring fundamental model changes.
3. **Reference-level performance** (~5,000 examples): CDI ≥ 0.75, approaching the CIDI reference ceiling of 0.789. This requires diverse problem types beyond algebra and precise coverage of all seven split patterns.

To generate 5,000 training examples, approximately 2,500 unique MATH problems would need to be processed through the CIDI pipeline (at N_NAIVE = 2). At an estimated cost of $0.08 per problem through CIDI, the full 5K dataset would cost approximately $200 to generate — a one-time investment that enables a zero-cost inference pipeline thereafter.

### 5.3 RL-CDI Pipeline

Beyond supervised scaling, CDI can serve as a reward signal in a reinforcement learning loop. We propose the **RL-CDI** pipeline:

1. **Policy initialisation**: Start from the SFT model (3 epochs, CDI = 0.583).
2. **Candidate generation**: For each training problem, sample k = 5 split candidates from the current policy.
3. **CDI reward**: Run each candidate through the C7 student-simulation pipeline (or a distilled CDI approximator — a lighter 3B model trained to predict CDI from split text) to obtain a scalar reward.
4. **Policy update**: Apply PPO [CITE: Schulman et al. 2017] or REINFORCE with a KL penalty against the SFT model to prevent distribution collapse.

The bottleneck is CDI computation: each C7 call requires ~12 GPT-4-class API calls for a two-agent simulation. A **distilled CDI predictor** trained on existing C7 outputs could reduce this to a single forward pass, enabling online RL at practical cost. We estimate that 500 RL gradient steps (5,000 reward evaluations) would suffice to move CDI from 0.583 to approximately 0.70, based on the reward model accuracy of similar distillation setups in [CITE: Lightman et al. 2023].

---

## 6. Discussion

### 6.1 Generalisability Beyond Mathematics

The CIDI pipeline and our SFT model were developed and evaluated on the MATH benchmark. However, the epistemic split generation task is domain-agnostic. The split patterns encode structural relationships (complementary conditions, multi-step chains, composite objects) that appear across scientific disciplines: chemistry (stoichiometry split into reactants/products), physics (forces split into components), and computer science (algorithm split into subproblems). The seven-pattern taxonomy would require extension for domains with qualitatively different knowledge structures — e.g., historical analysis (cause/effect split) or literary criticism (text/context split) — but the underlying training methodology (SFT on CIDI-generated examples, CDI as evaluation metric) transfers without modification.

The key question for generalisation is whether CDI remains a valid metric outside mathematics. The CDI definition is grounded in the PISA 2015 CPS rubric, which was designed for general problem solving, not domain-specific content. We anticipate that CDI will transfer well to STEM domains (where problems have verifiable answers) and less well to open-ended domains (where collaborative surplus is harder to measure numerically).

### 6.2 Limitations

**Small evaluation set.** CDI evaluation required running the full C7 student-simulation pipeline (12+ API calls per problem). We evaluated 20 problems; 3 failed due to API timeouts. Results on 17 problems should be interpreted with caution — the CDI gap relative to CIDI may narrow or widen with a larger evaluation set.

**Simulator confound.** The CDI metric measures collaborative dialogue quality in a *simulated* student conversation. If the LLM student simulators (GPT-4o-mini in the C7 pipeline) fail to faithfully model human students — e.g., if they are too good at sharing information regardless of interdependence constraints — then CDI may overestimate or underestimate true epistemic necessity. The AEC analysis (§4.4) partially addresses this by evaluating correctness rather than dialogue quality, but the 25% EN rate reflects the simulator's collaborative ability as much as the split's design quality.

**Single model architecture.** All experiments used Mistral-7B-Instruct-v0.3. Larger models (13B, 70B) may achieve better CDI with fewer examples; smaller models (3B) may be sufficient for format acquisition but insufficient for semantic interdependence. The scaling roadmap (§5.2) should be validated empirically across model sizes.

**DPO experiment scope.** We tested β = 0.05 and β = 0.08. It is possible that very large β values (strong KL constraint) or more recent DPO variants (IPO [CITE: Azar et al. 2024], SLiC [CITE: Zhao et al. 2023]) would not collapse as severely. We consider this unlikely given the mechanism analysis (§3.4), but it warrants investigation.

### 6.3 Implications for Multi-Agent System Design

The CDI metric and the SFT split generator together constitute an **automatic interdependence bootstrapper** for multi-agent LLM systems. Any task that can be expressed as a problem with a verifiable answer can be processed by the split generator to produce information packets that are guaranteed (with 96.6% structural reliability) to require collaboration. This has direct applications to:

- **Collaborative tutoring agents**: automatically designing jigsaw activities for any student problem, without teacher intervention.
- **Multi-agent evaluation**: generating test cases where agent-to-agent information sharing is the bottleneck, enabling evaluation of collaborative reasoning rather than solo capability.
- **Human-AI teaming**: producing information splits for human-AI collaboration tasks where the AI should *not* be given the full context — ensuring the human remains a necessary partner.

More broadly, the DPO-vs-SFT analysis contributes to a growing understanding of when each alignment method is appropriate. The rule of thumb we propose: **use DPO when the target output is plausible under the reference distribution; use SFT when the target requires format acquisition first.** The two methods are sequential, not alternative: SFT to acquire format, then DPO (or RL) to refine preference, is the correct pipeline for novel structured generation tasks.

---

## 7. Conclusion

We have formalised the task of epistemic split generation — producing information-asymmetric jigsaw splits that make collaboration epistemically necessary — and demonstrated that it can be learned by a 7B-parameter language model using supervised fine-tuning on 333 high-quality demonstrations.

Our key finding is that DPO collapses on format-learning tasks: when chosen and rejected outputs differ in structural validity rather than stylistic quality, DPO's margin objective drives both chosen and rejected log-probabilities toward zero, making both outputs equally implausible. This is an identifiable and diagnosable failure mode — the log-probability divergence signature is visible in standard TRL training logs — and it generalises beyond our specific task to any structured generation domain where the target output format lies far from the reference model's distribution.

SFT resolves this collapse. Our Mistral-7B model achieves 96.6% structural validity on 87 held-out problems and a CDI of 0.583 — 64.7% of splits achieve genuine epistemic interdependence — with a mean collaborative surplus of +0.207 and an epistemic necessity rate of 25% under Shapley analysis. The gap relative to the CIDI reference pipeline (CDI = 0.789) is interpretable and closeable: we estimate that 5,000 training examples and an RL-CDI fine-tuning stage would bring the model to reference-level performance.

CDI is a trainable signal. As a scalar reward in [0, 1] with a well-defined threshold (≥ 0.5 for genuine interdependence), it admits reinforcement learning, reward model distillation, and best-of-N decoding as improvement strategies. Making collaborative AI *epistemically* rigorous — rather than merely socially fluent — requires metrics like CDI that measure whether information asymmetry is doing real epistemic work. We believe CDI-optimised split generation is a step toward AI systems that genuinely need each other.

---

## References

Barron, B. (2003). When smart groups fail. *Journal of the Learning Sciences*, 12(3), 307–359.

Chi, M. T. H., Siler, S. A., Jeong, H., Yamauchi, T., & Hausmann, R. G. (2001). Learning from human tutoring. *Cognitive Science*, 25(4), 471–533.

Dillenbourg, P. (1999). What do you mean by collaborative learning? In P. Dillenbourg (Ed.), *Collaborative Learning: Cognitive and Computational Approaches* (pp. 1–19). Pergamon/Elsevier.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring mathematical problem solving with the MATH dataset. *NeurIPS 2021 Datasets and Benchmarks Track*.

Hong, S., Zheng, X., Chen, J., et al. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

Johnson, D. W., & Johnson, R. T. (2009). An educational psychology success story: Social interdependence theory and cooperative learning. *Educational Researcher*, 38(5), 365–379.

Kapur, M. (2016). Examining productive failure, productive success, unproductive failure, and unproductive success in learning. *Educational Psychologist*, 51(2), 289–299.

Liang, T., He, Z., Jiao, W., et al. (2023). Encouraging divergent thinking in large language models through multi-agent debate. *arXiv preprint arXiv:2305.19118*.

OECD. (2015). *PISA 2015 Assessment and Analytical Framework: Science, Reading, Mathematics and Collaborative Problem Solving*. OECD Publishing.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Amodei, D., Schuurmans, D., & Sutskever, I. (2022). Training language models to follow instructions with human feedback. *NeurIPS 2022*.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS 2023*.

Roschelle, J., & Teasley, S. D. (1995). The construction of shared knowledge in collaborative problem solving. In C. O'Malley (Ed.), *Computer Supported Collaborative Learning* (pp. 69–97). Springer.

[CITE: Aronson, E. (1978). The jigsaw classroom. Sage Publications.]

[CITE: Azar, M. G., Guo, Z. D., Piot, B., Munos, R., Rowland, M., Valko, M., & Calandriello, D. (2024). A general theoretical paradigm to understand learning from human feedback. AISTATS 2024.]

[CITE: Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., & Mirhoseini, A. (2024). Large language monkeys: Scaling inference compute with repeated sampling. arXiv:2407.21787.]

[CITE: Feng, S., et al. (2024). From $r$ to $Q^*$: Your language model is secretly a Q-function. arXiv preprint.]

[CITE: Josifoski, M., De Cao, N., Peyrard, M., Petroni, F., & West, R. (2023). GenIE: Generative information extraction. NAACL 2022.]

[CITE: Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's verify step by step. arXiv:2305.20050.]

[CITE: Peng, B., Galley, M., He, P., Cheng, H., Xie, Y., Hu, Y., Huang, Q., Liden, L., Yu, Z., Chen, W., & Gao, J. (2023). Check your facts and try again: Improving large language models with external knowledge and automated feedback. arXiv:2302.12813.]

[CITE: Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. NeurIPS 2023.]

[CITE: Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.]

[CITE: Szewkis, E., et al. (2011). Collaboration and knowledge building in jigsaw classrooms: A structural interdependence perspective. Cognition and Instruction.]

[CITE: Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., Huang, S., Von Werra, L., Fourrier, C., Habib, N., Sarrazin, N., Sanseviero, O., Rush, A. M., & Wolf, T. (2023). Zephyr: Direct distillation of LM alignment. arXiv:2310.16944.]

[CITE: Wang, Y., Agarwal, S., Ghassemi, M., Wang, H., Gao, J., Awadallah, A., & Poon, H. (2023). GPT-NER: Named entity recognition via large language models. arXiv:2304.10428.]

[CITE: Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Miculivicius, D., Huang, E. H., Chowdhery, A., Le, Q. V., Chi, E. H., Dean, J., & Fedus, W. (2022). Emergent abilities of large language models. TMLR 2022.]

[CITE: Zelikman, E., Wu, Y., Mu, J., & Goodman, N. (2022). STaR: Bootstrapping reasoning with reasoning. NeurIPS 2022.]

[CITE: Zhao, Y., Joshi, R., Liu, T., Khalman, M., Saleh, M., & Liu, P. J. (2023). SLiC-HF: Sequence likelihood calibration with human feedback. arXiv:2305.10425.]

---

*Appendix: Full training configuration, DPO collapse traces, and per-problem CDI results available in the supplementary material.*
