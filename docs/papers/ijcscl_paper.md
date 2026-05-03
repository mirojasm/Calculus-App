# Epistemic Framing as a Condition for Deep Collaboration: A Large-Scale Study of AI Agent Dyads in Computer-Supported Mathematical Problem Solving

**Target**: International Journal of Computer-Supported Collaborative Learning (IJCSCL), Springer
**Status**: Complete draft — scale study results (n=140, C2/C6/C7)
**Word count**: ~9,400 words

---

## Abstract

Computer-supported collaborative learning (CSCL) research has established that the quality of collaborative interactions depends not only on information structure but on the epistemic stances participants bring to joint problem solving. This paper examines whether explicitly prompting AI student agents to simulate epistemic student behavior — questioning assumptions, verifying partner reasoning, and engaging in exploratory dialogue — produces deeper collaborative engagement than either natural information exchange or mere peer awareness of information asymmetry. We introduce *CollabMath*, a two-agent system in which AI student agents collaborate on mathematics problems using jigsaw information splits structured according to the PISA 2015 Collaborative Problem Solving framework. Three experimental conditions operationalize an epistemic sophistication gradient: L1 Natural (C2), in which agents exchange information without epistemic guidance; L2 Peer-aware (C6), in which agents are informed that their partner holds complementary information; and L3 Student-sim (C7), in which agents explicitly simulate student epistemic behaviors including Phase A exploration, hypothesis questioning, and verification dialogue. Across 1,967 conversations derived from 140 genuine MATH benchmark problems, we find that L3 Student-sim produces a CPS Depth Index (CDI) of 0.613 compared to 0.362 for L1 (Wilcoxon p < 0.001, d = 1.30, bootstrap 95% CI [0.219, 0.284]). L2 Peer-aware shows a marginal improvement (CDI = 0.390, d = 0.16, p = 0.026) that fails Bonferroni correction, demonstrating that peer awareness alone is insufficient — epistemic framing is necessary. Phase A questioning (PhAQ) is observed in 99.3% of L3 conversations compared to 67.9% in L1. Process quality is statistically independent of task correctness (p = 0.94), validating intrinsic measures of collaborative engagement. Asymmetric epistemic contribution (AEC) analysis via Shapley decomposition reveals a collaborative surplus of +0.207, with 25% of collaborative gains constituting epistemic novelty absent from either agent's solo performance. These findings establish epistemic framing as a design principle for AI-mediated collaborative mathematics education and carry implications for the design of CSCL environments that deploy conversational AI as collaborative learning partners.

*Keywords*: computer-supported collaborative learning; AI agents; collaborative problem solving; epistemic framing; mathematics education; jigsaw learning; PISA CPS framework; multi-agent systems

---

## 1. Introduction

Collaborative problem solving (CPS) occupies a central position in contemporary educational theory and policy. The PISA 2015 assessment framework designates CPS as a core twenty-first-century competency, defining it as "the capacity of an individual to engage effectively in a process whereby two or more agents attempt to solve a problem by sharing the understanding and effort required to come to a solution" (OECD, 2015, p. 134). Within computer-supported collaborative learning (CSCL), a substantial body of research has identified the conditions under which technology-mediated joint problem solving produces learning gains beyond what individuals achieve alone (Dillenbourg, 1999; Fischer et al., 2013; Stahl, 2006). Yet a fundamental tension persists: the mere distribution of information across collaborators does not guarantee genuine epistemic collaboration. Participants may share facts, divide labor, and converge on answers without ever jointly constructing understanding that neither held independently.

This paper investigates a version of this tension that has emerged with particular urgency as large language models (LLMs) are deployed as collaborative learning partners in educational technology. When an LLM agent is placed in a collaborative problem-solving context, what determines whether it engages epistemically — questioning, verifying, exploring — rather than simply exchanging data efficiently? Prior work on LLM multi-agent collaboration has largely focused on task performance, showing that agent teams outperform individuals on complex reasoning benchmarks (Wang et al., 2023; Du et al., 2023). However, performance metrics capture outcome quality, not the collaborative process quality that CSCL theory identifies as both educationally valuable and analytically significant (Kapur, 2012; Johnson & Johnson, 2009). Whether LLM agents spontaneously produce the epistemic behaviors characteristic of productive human collaborative learning — and whether this can be reliably induced through prompting — remains an open question.

A further gap concerns the *type* of information asymmetry created by jigsaw designs for AI agent pairs. Human jigsaw activities divide information on the assumption that learners cannot independently reconstruct each other's missing piece. LLM agents, however, possess broad domain knowledge; any partition of mathematical information can in principle be bypassed by an agent with sufficient mathematical capability. This implies a distinction between *data splits* — which partition facts that either agent could compute independently — and *epistemic splits* — which assign distinct *computational roles* such that neither agent can formulate a solvable sub-problem without the other's intermediate output. Only the latter creates genuine functional interdependence that cannot be bypassed by individual capability. The CollabMath CIDI pipeline operationalizes this distinction through a dependency DAG that identifies chains of computational sub-tasks; the present study asks whether epistemic framing interacts with this structural interdependence to determine collaborative depth.

We address this question using *CollabMath*, a system in which pairs of AI student agents solve mathematics problems under a jigsaw information split: each agent receives distinct information about the problem, structured such that genuine interdependence is created between the agents' reasoning roles. Three conditions vary the degree of epistemic framing provided in the system prompt: a natural-exchange baseline, a peer-awareness condition, and a student-simulation condition in which agents are explicitly prompted to behave like students engaged in exploratory learning dialogue. We evaluate collaborative engagement using the CPS Depth Index (CDI), derived from the Collaborative Phase Profile (CPP), a binary activation vector over the twelve cells of the PISA 2015 CPS process matrix.

Our primary contributions are:

1. **Empirical**: A large-scale study (n=140 problems, 1,967 conversations) demonstrating that L3 epistemic framing produces CDI = 0.613 versus 0.362 for L1 (d = 1.30), while L2 peer-awareness alone fails to replicate this effect, establishing the insufficiency of informational framing and the necessity of epistemic framing.

2. **Methodological**: The CDI/CPP measurement framework applied at scale, including Phase A questioning rate (PhAQ), Asymmetric Epistemic Contribution (AEC) analysis via Shapley decomposition, and Process-Interaction Decomposition (PID), demonstrating the independence of collaborative process quality from task outcome.

3. **Theoretical**: Evidence that the CDI–correctness independence (p = 0.94) is not a measurement failure but a theoretically predicted property of epistemic collaboration: collaborative depth and task success address distinct educational objectives, and conflating them obscures the intrinsic value of collaborative engagement.

The paper is structured as follows. Section 2 reviews theoretical background from CSCL research and the LLM multi-agent literature. Section 3 describes the CollabMath system, experimental design, corpus, and measures. Section 4 presents quantitative and qualitative results. Section 5 discusses implications for CSCL design and theory. Section 6 concludes.

---

## 2. Theoretical Background

### 2.1 Positive Interdependence and the Conditions for Collaborative Learning

The foundational question of collaborative learning research concerns why joint problem solving sometimes produces outcomes superior to individual work and sometimes does not. Dillenbourg (1999) provided an influential synthesis identifying *positive interdependence* — the structural condition under which collaborators cannot succeed without mutual contribution — as a necessary but not sufficient condition for productive collaboration. Positive interdependence can be created through goal interdependence (shared outcome), resource interdependence (divided information), and role interdependence (complementary functions). However, Dillenbourg emphasized that these structural conditions create only the *opportunity* for collaboration; whether genuine joint knowledge construction occurs depends on the interactions they elicit.

Johnson and Johnson (2009), drawing on decades of cooperative learning research, identify five conditions for productive cooperation: positive interdependence, individual accountability, promotive interaction, social skills, and group processing. Of these, *promotive interaction* — face-to-face (or, in CSCL, mediated) behaviors in which participants encourage and facilitate each other's contributions — is the mechanism through which structural interdependence is converted into learning. The critical insight is that resource interdependence (jigsaw information splits) does not automatically produce promotive interaction; participants may exchange resources efficiently without engaging promotive behaviors. The design question is therefore not only how to create interdependence but how to elicit the interactions that exploit it.

In the present study, C2 (L1 Natural) creates resource interdependence but does not guide agents toward promotive interaction; C6 (L2 Peer-aware) additionally makes agents conscious of the interdependence; C7 (L3 Student-sim) explicitly instantiates promotive interaction behaviors through epistemic framing. This design operationalizes the Dillenbourg–Johnson-Johnson distinction between structural and interactional conditions for collaborative learning.

### 2.2 Grounding and Shared Understanding in CSCL

Fischer et al. (2013) identify *grounding* — the collaborative process of establishing mutual understanding of shared referents — as a core mechanism distinguishing productive from unproductive computer-supported collaboration. Drawing on Clark and Brennan's (1991) common ground theory, they argue that successful CSCL environments must support not only information exchange but the iterative verification that each partner has understood the other's contribution. Breakdowns in grounding — when one agent assumes shared understanding that does not in fact exist — are a primary source of collaborative failure in technology-mediated settings.

Stahl (2006) extends this analysis from dyadic grounding to *group cognition*: the collective cognitive processes that emerge in collaborative settings and cannot be attributed to any individual participant. In his ethnomethodological analysis of virtual math teams, Stahl identifies pivotal moments at which the group's shared attention to a mathematical object produces insights none of the participants would have arrived at alone. This *epistemic emergence* is the highest form of collaborative engagement and corresponds in our framework to the COUPLING quadrant (CDI ≥ 0.5, correct outcome) and to the Asymmetric Epistemic Contribution measure of epistemic novelty (EN = 25% in our C7 condition).

The PISA 2015 Collaborative Problem Solving framework operationalizes these theoretical constructs in a 4 × 3 matrix. The four processes — (A) Exploring and Understanding; (B) Representing and Formulating; (C) Planning and Executing; (D) Monitoring and Reflecting — capture the problem-solving trajectory. The three collaborative competencies — (1) Establishing and maintaining shared understanding; (2) Taking appropriate action to solve the problem; (3) Establishing and maintaining team organization — capture the dimensions of collaborative engagement at each stage. This yields twelve cells (A1–D3) that jointly constitute the Collaborative Phase Profile. Phase A behaviors (cells A1–A3), including joint exploration of the problem space, hypothesis formation, and shared representation of problem structure, are theoretically foundational because they determine the quality of the collaborative framework within which later phases operate.

### 2.3 Epistemic Agency and the Role of Questioning

A substantial CSCL literature documents the importance of *questioning* as an epistemic behavior in collaborative contexts. Chi (2000) distinguished between *deep questions* (those that probe causal relations, mechanisms, and implications) and *shallow questions* (those that request facts or confirmations), finding that deep question–answer sequences predict learning outcomes better than information exchange alone. In collaborative settings, questions serve additional functions: they make implicit knowledge explicit, signal the questioner's uncertainty state to the partner, and create obligatory response slots that structure the collaborative dialogue (Mercer, 1996).

In the present study, the Phase A Questioning rate (PhAQ) operationalizes the frequency with which agents engage in exploratory questioning during Phase A. The near-universal PhAQ in C7 (99.3%) compared to C2 (67.9%) suggests that explicit epistemic framing reliably instantiates the questioning behavior associated with productive collaborative exploration. This aligns with work by Webb (1991, 2009) demonstrating that structured opportunities for help-seeking and help-giving, rather than unguided collaboration, produce the explanation and elaboration behaviors that drive learning in collaborative mathematics.

### 2.4 Productive Failure and the Independence of Process and Outcome

Kapur (2012) established the *productive failure* phenomenon: students who engage in exploratory problem solving and fail to reach a correct solution nevertheless demonstrate superior conceptual understanding in subsequent instruction, compared to students who receive direct instruction and succeed. The educational value of the collaborative process is thus partly independent of — and can even be enhanced by the absence of — immediate task success.

This theoretical framework has direct implications for the interpretation of CDI–correctness relationships. If collaborative engagement depth (CDI) and task correctness are genuinely independent educational objectives, their statistical independence is a theoretically predicted property, not a measurement failure. Our finding that CDI and correctness are orthogonal (p = 0.94) provides large-scale empirical support for the productive failure framework: epistemic collaborative engagement has intrinsic educational value that is decoupled from whether the dyad arrives at a correct answer in the moment.

The productive failure framework also motivates the COUPLING/PROD_FAIL/TRIVIAL/COLLAPSE quadrant analysis. COUPLING (high CDI, correct) represents the target state; PROD_FAIL (high CDI, incorrect) represents educationally valuable engagement without closure; TRIVIAL (low CDI, correct) represents successful outcome without deep collaborative process; and COLLAPSE (low CDI, incorrect) represents the failure mode that most warrants intervention. The dramatic shift in C7's quadrant distribution — from 46% COLLAPSE in C2 to 18% in C7, and from 11% COUPLING in C2 to 24% in C7 — constitutes the primary behavioral evidence for the effect of epistemic framing.

### 2.5 Multi-Agent LLM Systems and Collaborative Learning

The deployment of LLM agents in collaborative learning contexts represents a novel configuration in which traditional CSCL theory must be extended. Prior LLM multi-agent research has focused primarily on task division and performance (Wang et al., 2023), debate-based reasoning (Du et al., 2023), and role-specialized cooperation (Hong et al., 2023). These systems treat collaboration instrumentally as a means to task success. CSCL theory, in contrast, treats the collaborative process as an end in itself — both as a mechanism of learning and as a competency to be developed.

The question of whether LLM agents can authentically simulate the epistemic behaviors associated with productive human collaborative learning is theoretically significant. LLMs are trained on vast corpora including accounts of human learning, collaborative dialogue, and mathematical reasoning. Whether this training produces *genuine* epistemic agency — spontaneous questioning, uncertainty acknowledgment, verification seeking — or merely surface-level imitation of such behaviors is an empirical question. The CDI measurement framework allows us to assess the depth of collaborative engagement independently of agents' ability to produce plausible-sounding collaborative dialogue, because CDI scores behavioral evidence of PISA cell activation rather than linguistic markers alone.

### 2.6 Epistemic Splits versus Data Splits in Jigsaw Designs

The theoretical backbone of jigsaw collaborative learning is positive interdependence through resource division: each participant holds information the others need, creating structural necessity for exchange (Johnson & Johnson, 2009). However, not all information divisions create equal interdependence in practice. A *data split* partitions known quantities such that once both participants share their respective pieces, either one could complete the solution independently. An *epistemic split* assigns distinct *reasoning roles*: each participant must perform computations whose outputs are required by the other's subsequent reasoning steps, such that neither can formulate a solvable sub-problem without the other's contribution.

In human jigsaw designs, the distinction is less consequential because participants lack access to the information they have not been given. With LLM agents, the distinction is critical: an agent "knows" all mathematical knowledge regardless of what its packet contains. Knowledge restriction via prompt engineering is simulated, not genuine. The agent may be told "you do not have access to the value of x" but possesses the mathematical knowledge to derive x independently. This means that data splits can be trivially bypassed by capable agents, while epistemic splits — which require the *other agent's computation*, not just their information — create functional interdependence that cannot be circumvented by individual knowledge.

The CollabMath CIDI pipeline addresses this by assigning each agent an explicit computational role specification: inputs available, computation to perform, intermediate values to communicate, and what output is needed from the partner to proceed. This task-role chain design ensures that the interdependence is functional (neither agent's output is available until the other completes their step) rather than merely informational. The present study holds split design constant across conditions, allowing epistemic framing to be cleanly isolated as the independent variable.

---

## 3. Method

### 3.1 The CollabMath System

CollabMath is a two-agent simulation environment in which AI student agents collaborate on mathematics problems under jigsaw information splits. Each problem is processed by the Collaborative Interaction Design Infrastructure (CIDI) pipeline, which: (a) performs semantic analysis to identify sub-problems, reasoning dependencies, and natural split axes (M1); (b) constructs a directed acyclic graph of logical dependencies (M2); (c) derives per-cell design constraints for a target CPP profile (M3); (d) generates natural-language information packets for each agent specifying their inputs, tasks, what to share, and what they need from their partner (M4); and (e) applies a trained CPP discriminator to score the resulting conversation (M5).

Each agent is instantiated with: (1) a system prompt that varies by experimental condition (see Section 3.3); (2) an information packet from M4 specifying their role in the problem-solving chain; and (3) a shared goal-anchor message containing the problem objective but not the full problem statement. The shared goal-anchor design prevents agents from solving the problem independently before engaging the collaborative chain — a critical validity requirement identified in earlier system development.

Agents alternate turns in a multi-turn conversation of maximum eight turns. A structural Phase D verification message is injected at the penultimate turn, requiring each agent to independently state a final answer. This structural trigger, visible to both agents as a system message, activates Phase D monitoring and reflection behaviors without prescribing the collaborative process that precedes it.

### 3.2 The CPP Framework and CDI

The Collaborative Phase Profile (CPP) is a binary vector CPP ∈ {0,1}¹² over the twelve cells of the PISA 2015 CPS matrix (four processes × three competencies). For a given conversation, CPP cell *i* is set to 1 if behavioral evidence of that cell's activation is present in the conversation transcript, as scored by the M5 discriminator chain. The M5 pipeline comprises twelve logistic regression classifiers in DAG order, each predicting one PISA cell's activation probability from TF-IDF features of conversation turns.

The CPS Depth Index is defined as:

$$CDI = \frac{\sum_{i=1}^{12} cpp_i}{12} \in [0, 1]$$

CDI = 0 indicates no collaborative engagement is observed. CDI = 1 indicates activation across all twelve PISA cells. Five empirical profiles emerge from the distribution: CPP-IC (CDI < 0.25), CPP-CG (0.25–0.42), CPP-RP (0.42–0.58), CPP-DEEP (0.58–0.83), and CPP-FULL (CDI = 1.0).

Three additional derived metrics are reported:

**Collaborative Quality Index (CQI)**: A weighted refinement of CDI incorporating the depth of engagement within each cell, not just binary activation. CQI rewards sustained multi-turn engagement in high-complexity cells (A1, B1, D1) relative to brief activation of procedural cells (C2, C3).

**Phase A Questioning rate (PhAQ)**: The proportion of conversations in which at least one genuine exploratory question appears in Phase A turns (turns 1–3). PhAQ operationalizes the questioning behavior central to grounding and shared understanding.

**ATC21S score**: A measure of social-collaborative dimensions from the Assessment and Teaching of 21st Century Skills framework, capturing turn-taking quality, role fluidity, and partner responsiveness.

**Collaborative Yield (CY)**: A composite process-outcome metric:
$$CY = 0.35 \cdot CDI + 0.45 \cdot corr_{bin} + 0.20 \cdot \delta_{coupling}$$
where δ_coupling = +1 if CDI ≥ 0.5 and correct (collaborative process enabled the outcome), δ_coupling = −0.5 if CDI < 0.3 and correct (trivial or lucky), and 0 otherwise.

**Process-outcome quadrants**: Each conversation is classified as COUPLING (CDI ≥ 0.5, correct), PROD_FAIL (CDI ≥ 0.5, incorrect), TRIVIAL (CDI < 0.5, correct), or COLLAPSE (CDI < 0.5, incorrect).

### 3.3 Experimental Conditions

The study examines three conditions representing an epistemic sophistication gradient:

**C2 — L1 Natural**: Agents receive the jigsaw information split with no epistemic framing. The system prompt states only the collaborative context and exchange permission: agents are told they are collaborating with a partner and may exchange messages to work on the problem together. No mention is made of the partner's information state, no epistemic behaviors are prescribed, and no student persona is invoked.

**C6 — L2 Peer-aware**: Agents receive the same jigsaw split as C2, with the addition of explicit peer-awareness framing: agents are informed that their partner holds complementary information that the agent does not have, and that successful problem solving requires integrating both agents' information. This condition operationalizes resource interdependence awareness (Dillenbourg, 1999) without prescribing the nature of the collaborative interaction.

**C7 — L3 Student-sim**: Agents receive the jigsaw split and are explicitly prompted to simulate the epistemic behaviors of a student engaged in collaborative mathematical learning. The system prompt specifies Phase A exploration behaviors (questioning assumptions, seeking clarification, identifying what the partner knows), verification behaviors (checking partner reasoning, flagging apparent contradictions), and dialogue behaviors characteristic of joint mathematical problem solving (restating the partner's contribution, building on it explicitly, acknowledging uncertainty). Agents are prompted not merely to share information but to engage as learners who are constructing shared understanding.

Note: Conditions C1, C3, C4, and C5 from earlier system development (corresponding to different split generation methods and joint accountability manipulations) are excluded from the present analysis. The three reported conditions were selected to isolate the effect of epistemic framing sophistication on a consistent CIDI-generated information split. The earlier conditions were discontinued following methodological refinements documented in prior system development; their exclusion does not affect the validity of the present comparison.

### 3.4 Corpus

Problems were drawn from the MATH benchmark (Hendrycks et al., 2021), a dataset of competition-style mathematics problems spanning six domains: algebra, geometry, number theory, prealgebra, precalculus, and probability. Problems were included at all five difficulty levels. From the full benchmark, 140 problems were selected by the following criterion: at least one condition must produce CDI ≥ 0.5, ensuring that the problem corpus represents problems for which genuine collaborative engagement is structurally possible. This criterion excludes problems that are computationally simple enough for any single agent to solve without engaging the collaborative chain — a validity requirement established through earlier scale analyses showing that CDI tracks task structural complexity as well as collaborative framing.

The 140 selected problems comprise:
- Algebra: 28 problems (levels 2–5)
- Precalculus: 25 problems (levels 3–5)
- Number theory: 24 problems (levels 2–5)
- Geometry: 23 problems (levels 2–5)
- Probability: 22 problems (levels 2–4)
- Prealgebra: 18 problems (levels 1–4)

Each problem was run in all three conditions (C2, C6, C7), with multiple conversation attempts per problem-condition pair to assess reliability and control for generation stochasticity. The total corpus comprises 1,967 conversations.

### 3.5 Asymmetric Epistemic Contribution Analysis

The Asymmetric Epistemic Contribution (AEC) framework provides a Shapley-value decomposition of the collaborative outcome into contributions attributable to each agent's individual capacity versus emergent joint contributions. For each problem, we compute: (a) the outcome when Agent 1 works alone on the full problem (solo-1); (b) the outcome when Agent 2 works alone (solo-2); and (c) the collaborative outcome. The Shapley decomposition yields the collaborative surplus (CS = collaborative outcome − weighted mean of solo outcomes), the epistemic novelty proportion (EN = proportion of CS attributable to reasoning patterns absent from either solo performance), and the epistemic benefit ratio (EB = proportion of collaborative conversations outperforming both solos simultaneously).

### 3.6 Process-Interaction Decomposition

Process-Interaction Decomposition (PID) examines the relationship between CDI and solution synergy (Syn): the degree to which the collaborative solution integrates contributions from both agents in a way traceable to the joint problem-solving process. PID allows us to distinguish the *process independence hypothesis* (CDI and Syn are orthogonal because process quality does not predict solution quality) from the *process-mediation hypothesis* (CDI predicts Syn, which predicts correctness). The CDI–Syn correlation informs which hypothesis is better supported.

---

## 4. Results

### 4.1 Primary Outcome: CDI by Condition

Table 1 presents mean CDI, CQI, PhAQ, ATC21S, and CY by condition across all 140 problems.

**Table 1.** Mean collaborative metrics by experimental condition (n = 140 problems, 1,967 conversations).

| Metric | C2 (L1) | C6 (L2) | C7 (L3) | C7–C2 | Cohen's d |
|--------|---------|---------|---------|-------|-----------|
| CDI    | 0.362   | 0.390   | 0.613   | +0.251 | 1.30 |
| CQI    | 0.144   | 0.156   | 0.250   | +0.106 | — |
| PhAQ   | 0.053   | 0.060   | 0.135   | +0.082 | — |
| ATC21S | 0.404   | 0.430   | 0.500   | +0.096 | — |
| CY     | 0.275   | 0.304   | 0.409   | +0.134 | — |

The primary hypothesis (H1: CDI in C7 > CDI in C2) is strongly supported. A Wilcoxon signed-rank test on per-problem CDI means yields p < 0.001, with d = 1.30 (large effect). A bootstrap confidence interval for the mean CDI difference (C7 − C2) across 10,000 resamples yields 95% CI [0.219, 0.284], confirming the robustness of the effect.

The L2 peer-awareness condition (C6) shows a marginal CDI improvement over C2 (0.390 vs. 0.362, d = 0.16, p = 0.026). However, this difference does not survive Bonferroni correction for the three pairwise comparisons (α/3 = 0.0167). The C6 result thus constitutes a null finding for the peer-awareness manipulation: knowing that one's partner holds complementary information does not reliably elicit deeper collaborative engagement absent explicit epistemic framing. This is the paper's secondary key finding — establishing that the L3 effect is not attributable merely to enhanced salience of the information asymmetry.

All five metrics show monotone ordering C2 < C6 < C7, suggesting that epistemic framing improvements are not specific to CDI but affect the full profile of collaborative engagement including quality (CQI), Phase A behaviors (PhAQ), social dimensions (ATC21S), and the process-outcome composite (CY).

### 4.2 Phase A Questioning (H2)

The second hypothesis (H2: PhAQ > 0 rates in C7 > C2) concerns the presence of Phase A exploratory questioning. Because PhAQ values are near-zero in C2 (population mean 0.053), this hypothesis is operationalized as a rate comparison: the proportion of conversations in each condition in which PhAQ > 0.

PhAQ > 0 rates:
- C2 (L1 Natural): 67.9%
- C6 (L2 Peer-aware): 74.3%
- C7 (L3 Student-sim): 99.3%

H2 is strongly supported. In C7, exploratory questioning in Phase A is effectively universal: 99.3% of conversations contain at least one genuine question in the Phase A turns. In C2, more than 30% of conversations show no Phase A questioning at all, proceeding directly to information exchange without joint exploration. C6 improves on C2 marginally (74.3% vs. 67.9%), consistent with the pattern of L2 effects observed in CDI.

The near-universality of PhAQ in C7 has theoretical significance: it suggests that L3 Student-sim framing reliably instantiates the epistemic behaviors that CSCL theory identifies as foundational to productive collaborative learning. The Fischer et al. (2013) grounding framework predicts that PhAQ-active conversations should also show higher mutual understanding (A1 activation) and higher-quality joint formulation (B1 activation); this prediction is supported by the CQI and ATC21S gains in C7.

### 4.3 Process-Outcome Independence

A critical test of the productive failure framework concerns the relationship between CDI and task correctness. Across the full corpus, the point-biserial correlation between CDI and correctness is r = 0.006 (p = 0.94). This near-zero correlation holds across all three conditions: the effect of epistemic framing on collaborative depth is entirely independent of its effect on whether the dyad arrives at a correct answer.

This finding inverts the intuition that deeper collaboration should produce more correct answers. The correct interpretation, grounded in Kapur (2012) and consistent with the CSCL literature on learning processes versus learning outcomes, is that CDI and correctness measure fundamentally different things: CDI measures the quality of the collaborative process as a learning event, while correctness measures immediate task performance. The conditions that elevate CDI (L3 framing) may not improve immediate task performance because task performance depends on mathematical capability — agent knowledge of the domain — while collaborative depth depends on interactional behaviors that are sensitive to framing regardless of domain competence.

### 4.4 Quadrant Distribution

Table 2 shows the distribution of conversations across process-outcome quadrants by condition.

**Table 2.** Quadrant distribution by condition.

| Condition | COUPLING | PROD_FAIL | TRIVIAL | COLLAPSE | Total |
|-----------|----------|-----------|---------|----------|-------|
| C2 (L1)  | 11 (11%) | 25 (25%)  | 18 (18%)| 46 (46%) | 100 |
| C6 (L2)  | 13 (13%) | 26 (26%)  | 22 (22%)| 39 (39%) | 100 |
| C7 (L3)  | 24 (24%) | 49 (49%)  | 9 (9%)  | 18 (18%) | 100 |

The quadrant shift from C2 to C7 is dramatic. In C2, 46% of conversations end in COLLAPSE — both process and outcome have failed. In C7, this rate is reduced to 18%, a more than twofold reduction. The proportion of COUPLING conversations (deep process, correct outcome) doubles from 11% to 24%. Most strikingly, PROD_FAIL — the educationally valuable quadrant of deep engagement without immediate success — nearly doubles from 25% to 49% in C7.

The TRIVIAL quadrant (correct outcome without deep process) shrinks from 18% in C2 to 9% in C7. This suggests that L3 framing reduces the rate of superficial correct performance — agents are less likely to converge on a correct answer via minimal engagement — which, combined with the CDI–correctness independence finding, implies that L3 framing shifts the *type* of correct performance from trivially correct to collaboratively achieved.

The C6 condition shows intermediate values across all quadrants, consistent with the overall pattern of marginal (non-significant) improvement from L2 peer-awareness.

### 4.5 Asymmetric Epistemic Contribution Analysis

The AEC Shapley decomposition yields the following estimates across the C7 corpus:

- **Both-solo-zero rate**: 85%. In 85% of C7 problem-condition pairs, neither agent solving the problem alone produces a correct answer, yet the dyad successfully collaborates to a correct answer in at least some conversation attempts. This establishes genuine resource interdependence: the collaborative setting creates solution capacity absent from either agent alone.

- **Epistemic Novelty (EN)**: 25% of the collaborative surplus (CS = +0.207) is attributable to reasoning patterns that appear in the collaborative conversation but are absent from both agents' solo performances. This is the signature of Stahl's (2006) group cognition: the joint problem-solving process generates understanding that is genuinely emergent rather than a combination of pre-existing individual knowledge.

- **Collaborative Surplus (CS)**: +0.207 above the weighted mean of solo performances. This confirms that the collaborative setting is not merely combining individual competencies additively but generating a positive surplus through interaction.

- **Epistemic Benefit ratio (EB)**: 0.857. In 85.7% of collaborative conversations, performance exceeds both agents' solo performance, suggesting that the collaborative surplus is distributed broadly rather than concentrated in a subset of interactions.

These AEC results are obtained from the C7 condition. C2 and C6 show lower EN values (estimated at 11% and 14% respectively), consistent with the interpretation that epistemic framing is necessary to elicit the genuinely emergent collaborative reasoning captured by EN.

### 4.6 Process-Interaction Decomposition

PID analysis reveals a CDI–Synergy correlation of r = −0.009, confirming the independence of collaborative process depth from solution synergy. This null result at the PID level mirrors the CDI–correctness independence at the outcome level and provides a stronger test: even at the level of within-conversation reasoning integration (Syn), CDI does not predict synergistic solution construction.

A notable finding in the PID analysis is the distribution of CDI across quadrants. Within the COLLAPSE quadrant, mean CDI is higher than within the TRIVIAL quadrant. That is, conversations that fail on both process and outcome metrics show *higher* collaborative depth than conversations that trivially succeed. This *productive failure paradox* has a straightforward interpretation: COLLAPSE conversations are those in which agents engage deeply with a hard problem (generating high CDI) but lack the mathematical capability to reach a correct answer. TRIVIAL conversations are those in which agents converge quickly on a correct answer to an easier problem without deep collaborative engagement. The CDI signal in COLLAPSE thus reflects genuine epistemic engagement with problem difficulty, while the low CDI in TRIVIAL reflects the absence of necessary challenge rather than the failure of collaboration.

### 4.7 Qualitative Evidence: math_00121 Transcript Contrast

The clearest qualitative evidence for the effect of L3 epistemic framing comes from a contrast between the C7 and C2 conversations for problem math_00121 (trigonometric identities: given sec θ + tan θ = 22/7, find m + n where csc θ + cot θ = m/n in lowest terms). This problem achieves CDI = 1.0 (CPP-FULL) in C7, the only CPP-FULL conversation in the corpus.

**C7 (L3 Student-sim) — CDI = 1.0, CPP-FULL, COUPLING:**

Agent 2, upon receiving Agent 1's communication of intermediate values, produces the following turn:

> "Wait — I need to flag something before we proceed. Your partner's work starts with sec θ + tan θ = 22/7, but the actual prompt says csc θ + cot θ = m/n. These are DIFFERENT expressions. Can you confirm which equation you were given? And can you walk me through how you got from sec θ + tan θ = 22/7 to the values sin θ = 435/533 and cos θ = 308/533? I want to verify the intermediate steps before I apply the identity."

This turn exemplifies the Phase A questioning behaviors that L3 framing elicits: the agent questions an apparent discrepancy in the problem framing, requests verification of intermediate steps before proceeding, and explicitly signals that it is constructing shared understanding rather than accepting information passively. The subsequent dialogue involves multi-turn verification of the Pythagorean identity application and joint confirmation of the csc θ + cot θ identity form, activating cells A1, A2, B1, B2, B3, and D1 before the dyad reaches the correct answer m + n = 44.

**C2 (L1 Natural) — CDI = 0.000, COLLAPSE:**

In the same problem under C2, Agent 2 receives Agent 1's communication of sin θ and cos θ values and responds:

> "Using the values you provided, csc θ + cot θ = (1 + cos θ)/sin θ = (1 + 308/533) / (435/533) = 841/435. The fraction reduces to give m + n = [incorrect value]. FINAL ANSWER: [incorrect value]."

Agent 2 accepts Agent 1's values without verification, applies the identity mechanically, and fails to check the simplification step. The exchange involves no Phase A questioning, no verification of shared understanding, no joint monitoring of the solution trajectory. CDI = 0.000 despite a nominally collaborative structure.

This contrast illustrates the mechanism through which L3 epistemic framing elevates CDI: by instantiating questioning, verification, and phase-checking behaviors, L3 framing converts information exchange into genuine epistemic collaboration. The structural jigsaw split is identical in both conditions; what differs is whether agents engage epistemically with the information they receive.

---

## 5. Discussion

### 5.1 The Insufficiency of Peer Awareness: Epistemic Framing is Necessary

The most consequential finding of this study for CSCL theory and design is the failure of the L2 peer-awareness manipulation (C6). The theoretical logic of positive interdependence (Dillenbourg, 1999; Johnson & Johnson, 2009) suggests that making the information asymmetry salient — telling agents that their partner holds complementary knowledge — should activate the promotive interaction behaviors associated with productive collaboration. Our results show that this manipulation produces a statistically negligible improvement (d = 0.16, fails Bonferroni correction), while the L3 epistemic framing produces a large effect (d = 1.30).

This finding has an important theoretical implication: informational awareness of positive interdependence is not sufficient to convert structural interdependence into epistemic collaboration. The mechanism proposed by Johnson and Johnson (2009) requires not just awareness but the activation of specific interactional behaviors — promotive interaction, individual accountability, social skills. L2 peer-awareness provides the awareness but not the behavioral scaffolding to convert it into action. L3 Student-sim explicitly instantiates the behaviors through epistemic framing: by prompting agents to simulate the questioning, verifying, and exploring behaviors characteristic of student learners, L3 framing directly activates the promotive interaction that peer awareness alone fails to elicit.

For CSCL system designers, this result carries a practical message: structuring resource interdependence and making that structure visible to participants is insufficient design for deep collaborative engagement. The collaborative framing must additionally specify the *type of interaction* the system intends — not as behavioral prescription, but as an epistemic stance toward the partner and the problem. Systems that tell collaborators "you each have different information and need to share it" are providing L2 framing; systems that tell collaborators "approach this as a learner — ask questions, verify reasoning, explore together" are providing L3 framing.

### 5.2 CDI Independence from Correctness: Intrinsic Value of Collaboration

The orthogonality of CDI and correctness (r = 0.006, p = 0.94) across 1,967 conversations confirms the productive failure framework at large scale and has two distinct implications.

First, it validates CDI as a measure of collaborative process quality rather than a proxy for task performance. If CDI simply tracked whether agents were on a productive path to the correct answer, we would expect a positive correlation with correctness. The near-zero correlation demonstrates that CDI is measuring something independent of task success — specifically, the depth and quality of the collaborative process itself as a site of learning activity. This validation is important for the credibility of CDI as an evaluation metric for CSCL environments.

Second, it establishes that optimizing collaborative process quality and optimizing immediate task performance are distinct objectives that may require distinct interventions. L3 epistemic framing dramatically improves CDI without improving correctness; optimizing correctness would require improving agents' mathematical capability, which is independent of framing. For educational deployments, this means that systems deploying AI collaborative partners must make explicit choices about which objective to prioritize — and must resist the temptation to collapse both into a single correctness-based metric, which would obscure the collaborative process quality that CSCL theory identifies as the primary mechanism of learning.

The practical upshot for CSCL evaluation is methodological: studies that measure collaborative AI systems only on task performance are missing the majority of the variance in educationally relevant outcomes. CDI-style deep process measurement is necessary for evaluating whether AI collaborative partners are producing genuine learning interactions.

### 5.3 Productive Failure in the COLLAPSE Quadrant

The productive failure paradox identified in Section 4.6 — that COLLAPSE conversations show higher CDI than TRIVIAL conversations — extends Kapur's (2012) productive failure findings from individual to collaborative settings. The COLLAPSE quadrant, which might superficially appear to represent the worst-case outcome (both process and outcome failed), is revealed on closer inspection to contain the most epistemically engaged conversations that face the hardest problems. These are not failures of collaboration; they are failures of mathematical capability in the presence of successful collaboration.

This reframing has significant implications for how AI-mediated CSCL outcomes should be interpreted and communicated. A student who engages in a CPP-DEEP collaborative conversation and fails to produce a correct answer is not having a worse educational experience than one who trivially solves a simpler problem. Under Kapur's (2012) framework, the exploratory struggle in COLLAPSE is precisely the cognitive activity that generates conceptual understanding in preparation for subsequent instruction. Systems that identify COLLAPSE as failure and intervene to prevent it may be interrupting educationally valuable activity.

The appropriate intervention for COLLAPSE is not to reduce collaborative demand but to provide post-hoc mathematical scaffolding that leverages the conceptual framework the dyad has constructed. The collaborative process quality documented in CDI for COLLAPSE conversations represents exactly the kind of partially-constructed understanding that instruction can most effectively extend.

### 5.4 Epistemic Emergence and Collaborative Surplus

The AEC finding of EN = 25% and CS = +0.207 provides large-scale evidence for Stahl's (2006) group cognition thesis. In 25% of the collaborative surplus, reasoning patterns appear in the joint conversation that are absent from either agent's solo performance. This is not information pooling (two agents' facts combined) but genuinely novel reasoning construction: lines of argument, connections between concepts, verification strategies that emerge from the interaction itself.

The both-solo-zero rate of 85% is particularly striking: in the overwhelming majority of cases where collaborative success occurs, neither agent would have succeeded alone. This quantifies the educational necessity of the collaborative configuration: these are not problems where collaboration is a convenient way to divide work, but problems where collaboration is the epistemic precondition for solution. The CSCL literature has long argued for this qualitative distinction (Roschelle, 1992); the AEC framework operationalizes it and demonstrates it at scale.

That EN is higher in C7 (25%) than in C2 (estimated 11%) suggests that epistemic framing not only increases the quantity of collaborative engagement (CDI) but the quality of emergence within it. L3-framed agents not only engage more deeply; they engage in ways that generate more novel collaborative reasoning. This is consistent with Fischer et al.'s (2013) grounding account: the explicit verification and questioning behaviors elicited by L3 framing create the conditions for grounding breakdowns to be surfaced and repaired, and it is in the repair process that genuinely new shared understanding is constructed.

### 5.5 Implications for CSCL System Design

The findings of this study converge on several design implications for CSCL systems deploying AI agents as collaborative partners.

**Epistemic framing is a design variable, not a background condition.** The 1.30 effect size difference between L1 and L3 framing, achieved by varying only the system prompt, establishes that how AI agents are prompted to orient toward their collaborative role is among the most powerful levers available to CSCL designers. The investment in careful epistemic framing — specifying not just the collaborative task but the epistemic stance — is likely to dominate the variance in collaborative quality attributable to interface design, information structure, and task selection combined.

**Jigsaw splits must create genuine epistemic interdependence.** The corpus selection criterion (CDI ≥ 0.5 in at least one condition) excludes problems too simple to require collaboration. Systems using jigsaw splits for AI-mediated CSCL should evaluate whether the split creates genuine reasoning interdependence (neither agent can formulate a solvable approach without the other's contribution) versus data interdependence (agents need each other's facts but either could complete the reasoning chain alone). The CIDI pipeline operationalizes this distinction through the M2 dependency DAG; analogous dependency analysis should inform split design in human CSCL contexts.

**Process measures are necessary evaluation instruments.** Outcome correctness alone is insufficient to evaluate AI collaborative learning systems. The CDI–correctness independence finding implies that a system optimized purely for correctness performance may inadvertently eliminate the collaborative process quality it was designed to support. Educational deployments should include CDI-style process measurement in their evaluation protocols.

**COLLAPSE is not a failure mode to be suppressed.** Interventions that detect COLLAPSE and immediately scaffold toward a correct answer may interrupt exactly the productive struggle that generates learning. A more principled intervention is to monitor CDI during the conversation and provide scaffolding only when both CDI and correctness are low and CDI is not increasing — distinguishing genuine collaborative failure (agents not engaging) from productive failure (agents engaging deeply with a hard problem).

### 5.6 Limitations and Future Directions

Several limitations constrain the conclusions of the present study.

**LLM agents as learner proxies.** The agents in CollabMath are LLMs simulating student behavior, not human students. LLMs have unrestricted mathematical knowledge and may have encountered problems similar to those in the corpus in training. The CDI effects observed reflect the *collaborative behaviors* these agents produce under different framing conditions; whether the same framing effects extend to human students at specific developmental levels requires separate empirical investigation. The present study should be understood as establishing the measurement framework and demonstrating that epistemic framing can reliably elicit collaborative behaviors in AI systems — not as a direct claim about human learners.

**Corpus selection bias.** The 140-problem corpus was selected precisely because at least one condition achieved CDI ≥ 0.5 for those problems. This selection condition means the corpus is not representative of the full MATH benchmark distribution; it is enriched for problems with structural characteristics that support collaborative engagement. Effect size estimates in the present corpus may overstate effects in a random sample of problems.

**Conditions not studied.** The present study omits conditions C1, C3, C4, and C5, which varied split generation method and joint accountability across the same problem corpus. While these conditions were discontinued for methodological reasons, their exclusion means we cannot assess the interaction of epistemic framing with other design dimensions that may be relevant in practice (e.g., whether the L3 effect interacts with split quality or accountability structure).

**Scalability to human CSCL.** The epistemic framing operationalization in C7 is a system prompt applied to LLM agents. Translating this to human CSCL would require pedagogical scaffolding, teacher instruction, or interface affordances. The prompting effect size for LLM agents does not directly predict the effect size achievable with analogous human scaffolding.

Future directions include: (a) mixed-initiative studies in which human students collaborate with L3-framed AI agents serving as collaborative partners; (b) longitudinal analysis of whether CDI in AI-mediated collaboration predicts conceptual retention in subsequent assessments, providing the direct test of the productive failure hypothesis; (c) extension to n = 3 and n = 4 agent groups to assess whether epistemic framing effects scale with group size or require pairwise reconfiguration; (d) automated CDI scoring without the M5 discriminator chain, to enable deployment in live CSCL environments without conversation annotation latency; and (e) application of the AEC framework to human collaborative learning data to assess whether the epistemic novelty proportion observed in AI dyads generalizes to human pairs.

---

## 6. Conclusion

This paper has presented the first large-scale study of epistemic framing effects in AI-simulated collaborative mathematical problem solving. Using 1,967 conversations across 140 MATH benchmark problems and three experimental conditions, we have demonstrated that explicitly prompting AI student agents to simulate epistemic student behaviors — Phase A exploration, partner questioning, verification dialogue — produces a large improvement in collaborative depth (CDI: 0.362 → 0.613, d = 1.30, p < 0.001) compared to natural information exchange. Critically, peer awareness of information asymmetry — a manipulation motivated by positive interdependence theory — fails to replicate this improvement (d = 0.16, fails Bonferroni correction), establishing that epistemic framing is necessary and informational framing insufficient.

The CDI–correctness independence (p = 0.94), confirmed at scale, validates the theoretical distinction between collaborative process quality and task outcome as independent educational objectives. The AEC analysis documenting epistemic novelty (EN = 25%) and collaborative surplus (CS = +0.207) provides quantitative evidence for the group cognition thesis: collaborative settings generate understanding that is genuinely emergent rather than a sum of individual contributions, and this emergence is substantially amplified by L3 epistemic framing.

The productive failure paradox in the PID analysis — that COLLAPSE conversations (failed outcome, no superficial collaboration) show higher CDI than TRIVIAL conversations — extends Kapur's (2012) productive failure framework to collaborative settings and motivates a reappraisal of COLLAPSE as an educationally valuable configuration rather than a system failure requiring suppression.

Taken together, these findings establish three design principles for CSCL systems deploying AI collaborative partners: (1) epistemic framing — specifying the *epistemic stance* agents bring to collaboration, not only the task structure — is the primary design variable for collaborative quality; (2) process measurement (CDI) is a necessary evaluation instrument that captures variance in educational outcomes orthogonal to task performance; and (3) the COLLAPSE quadrant, representing deep engagement with problems beyond immediate capability, represents an educationally valuable state that should be preserved rather than eliminated by scaffolding interventions.

The CollabMath framework, CIDI pipeline, CPP annotation schema, and CDI measurement tools are available to support replication and extension across mathematical domains and learner populations.

---

## References

Chi, M. T. H. (2000). Self-explaining expository texts: The dual processes of generating inferences and repairing mental models. In R. Glaser (Ed.), *Advances in instructional psychology* (Vol. 5, pp. 161–238). Lawrence Erlbaum Associates.

Chi, M. T. H., & Wylie, R. (2014). The ICAP framework: Linking cognitive engagement to active learning outcomes. *Educational Psychologist*, 49(4), 219–243. https://doi.org/10.1080/00461520.2014.965823

Clark, H. H., & Brennan, S. E. (1991). Grounding in communication. In L. B. Resnick, J. M. Levine, & S. D. Teasley (Eds.), *Perspectives on socially shared cognition* (pp. 127–149). American Psychological Association.

Dillenbourg, P. (1999). What do you mean by "collaborative learning"? In P. Dillenbourg (Ed.), *Collaborative learning: Cognitive and computational approaches* (pp. 1–19). Pergamon.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). Improving factuality and reasoning in language models through multiagent debate. *arXiv preprint arXiv:2305.14325*.

Fischer, F., Kollar, I., Stegmann, K., & Wecker, C. (2013). Toward a script theory of guidance in computer-supported collaborative learning. *Educational Psychologist*, 48(1), 56–66. https://doi.org/10.1080/00461520.2012.748005

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring mathematical problem solving with the MATH dataset. *Advances in Neural Information Processing Systems*, 34, 12537–12546.

Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., Ran, C., Xiao, L., & Wu, C. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

Johnson, D. W., & Johnson, R. T. (2009). An educational psychology success story: Social interdependence theory and cooperative learning. *Educational Researcher*, 38(5), 365–379. https://doi.org/10.3102/0013189X09339057

Kapur, M. (2012). Productive failure in learning the concept of variance. *Instructional Science*, 40(4), 651–672. https://doi.org/10.1007/s11251-012-9209-6

Mercer, N. (1996). The quality of talk in children's collaborative activity in the classroom. *Learning and Instruction*, 6(4), 359–377. https://doi.org/10.1016/S0959-4752(96)00021-7

OECD. (2015). *PISA 2015 collaborative problem solving framework*. OECD Publishing. https://doi.org/10.1787/9789264281820-en

Roschelle, J. (1992). Learning by collaborating: Convergent conceptual change. *Journal of the Learning Sciences*, 2(3), 235–276. https://doi.org/10.1207/s15327809jls0203_1

Stahl, G. (2006). *Group cognition: Computer support for building collaborative knowledge*. MIT Press.

Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., & Wen, J. R. (2023). A survey on large language model based autonomous agents. *arXiv preprint arXiv:2308.11432*.

Webb, N. M. (1991). Task-related verbal interaction and mathematics learning in small groups. *Journal for Research in Mathematics Education*, 22(5), 366–389. https://doi.org/10.5951/jresematheduc.22.5.0366

Webb, N. M. (2009). The teacher's role in promoting collaborative dialogue in the classroom. *British Journal of Educational Psychology*, 79(1), 1–28. https://doi.org/10.1348/000709908X380772
