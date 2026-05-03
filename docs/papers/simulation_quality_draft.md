# Beyond Information Exchange: Improving Epistemic Collaboration Quality in LLM Multi-Agent Systems

**Target venue:** AIED 2027 / EDM 2027 / NeurIPS 2027 Workshop on AI for Education  
**Status:** Skeleton — pending experiments  
**Depends on:** CollabMath NeurIPS 2026 paper (framework + C7 baseline)

---

## Abstract [PLACEHOLDER]

Large language model (LLM) agents can be prompted to collaborate on structured problem-solving tasks, yet even with optimal information splits and epistemic framing (C7 student simulation), agents fail to reach genuine collaborative outcomes in [X]% of cases. We analyze the failure modes of LLM collaborative agents — COLLAPSE, PROD_FAIL, TRIVIAL — and propose [N] interventions targeting Phase A epistemic behavior. [Results TBD]

**Key numbers to fill:**
- C7 baseline: CDI=0.613, 24% COUPLING, 49% PROD_FAIL, 18% TRIVIAL, 9% COLLAPSE
- PhAQ=0.135 (Phase A activated in 99.3% of C7 conversations but low intensity)
- Goal: CDI≥0.75, COUPLING rate ≥40%

---

## 1. Introduction

### 1.1 The Collaboration Quality Gap

The CollabMath framework demonstrated that L3 student-simulation agents (C7) significantly outperform natural agents (C2) in epistemic collaboration quality (CDI: 0.613 vs 0.362, d=1.30). Yet even in the best condition, only 24% of agent pairs reach genuine collaborative success (COUPLING quadrant). The majority fail:

- **49% PROD_FAIL**: agents engage epistemically but do not converge on a correct answer
- **18% TRIVIAL**: agents solve in parallel without genuine collaboration
- **9% COLLAPSE**: agents fail to engage meaningfully

This paper asks: what prevents LLM agents from collaborating better, even with ideal information structure and epistemic framing?

### 1.2 Research Questions

1. **RQ1 — Failure mode analysis**: What are the proximate causes of COLLAPSE, PROD_FAIL, and TRIVIAL in LLM agent pairs?
2. **RQ2 — Phase A activation**: Why does PhAQ remain low (mean=0.135) even when Phase A is technically activated (99.3% of conversations)?
3. **RQ3 — Intervention efficacy**: Which prompting or protocol interventions improve CDI and COUPLING rate without invalidating the existing CPP framework?
4. **RQ4 — Model scale**: Does using a more capable base LLM (e.g., GPT-4.1 vs GPT-4o-mini for simulation) substantially change collaboration quality?

### 1.3 Contributions [PLACEHOLDER — depends on experiments]

1. A taxonomy of LLM collaboration failure modes grounded in the PISA CPS rubric
2. [N] prompting interventions targeting Phase A epistemic behavior
3. Evaluation protocol compatible with the CPP framework (results comparable to NeurIPS baseline)
4. Empirical evidence on the role of model scale in collaborative quality

---

## 2. Background

### 2.1 Collaborative Learning Theory

**Positive interdependence** (Johnson & Johnson, 1989): collaboration is only genuine when group members cannot succeed independently. The CollabMath framework enforces this via information splits (jigsaw).

**Grounding and Phase A** (Roschelle & Teasley, 1995; Fischer et al., 2013): effective collaboration requires an initial phase of shared understanding construction before problem-solving. This is Phase A in the CPP framework — agents must establish a joint problem representation before executing.

**Transactive discourse** (Berkowitz & Gibbs, 1983): high-quality collaboration involves agents building directly on each other's reasoning, not just turn-taking. The CQI metric captures this.

**Productive failure** (Kapur, 2016): incorrect collaborative attempts that activate deep epistemic engagement may produce better learning outcomes than correct solo attempts. The PROD_FAIL quadrant may be more valuable than its label suggests.

### 2.2 LLM Multi-Agent Collaboration

[CITE: relevant work on LLM multi-agent systems — MetaGPT Hong 2023, Camel Li 2023, AutoGen Wu 2023, Society of Mind Park 2023]

Key gap: existing multi-agent frameworks optimize for task completion, not epistemic collaboration quality. None of them measure or target Phase A behavior.

### 2.3 Why LLM Agents Fail at Epistemic Collaboration

Hypotheses (to be tested):

**H-A (Competence escape)**: When an agent can partially solve the problem from its own information, it bypasses Phase A. The jigsaw split forces information-level dependence but not reasoning-level dependence.

**H-B (Authority acceptance)**: LLM agents trained on human text are biased toward accepting partner statements rather than challenging them (sycophancy from RLHF). This suppresses Phase A.

**H-C (Protocol gap)**: Without explicit turn structure, agents default to efficient information exchange rather than epistemic exploration. The C7 prompt allows but does not force Phase A.

**H-D (Scale effects)**: GPT-4o-mini (used for simulation) may lack the metacognitive capacity to sustain Phase A behavior under cognitive load.

---

## 3. Failure Mode Analysis

### 3.1 Taxonomy of Collaboration Failures

[Analysis of CPP quadrant patterns from n=1,967 conversations]

| Quadrant | Rate (C7) | Proximate cause | CDI range |
|----------|-----------|-----------------|-----------|
| COUPLING | 24% | — success — | ≥0.5, CY≥0.5 |
| PROD_FAIL | 49% | Epistemic engagement without convergence | ≥0.5, CY<0.5 |
| TRIVIAL | 18% | Parallel solving, no genuine exchange | <0.5, CY≥0.5 |
| COLLAPSE | 9% | No meaningful engagement | <0.5, CY<0.5 |

**Key insight from existing data**: COLLAPSE quadrant paradoxically has the highest CDI mean (productive failure paradox). TRIVIAL has the lowest CDI despite correct answers — agents solve independently.

### 3.2 Phase A Quality Analysis

[PLACEHOLDER — needs deeper analysis of PhAQ=0 vs PhAQ>0 conversations]

- PhAQ>0 in 99.3% of C7 conversations: Phase A is triggered
- But PhAQ mean=0.135: Phase A is shallow in most cases
- What does shallow Phase A look like? Agent asks one clarifying question, accepts answer, proceeds to solve.
- What does deep Phase A look like? math_00121×C7 — agent explicitly challenges partner's claim, refuses to accept, forces re-evaluation.

### 3.3 Qualitative Failure Case Analysis

[PLACEHOLDER — needs 3-4 annotated transcript examples]

**Case 1 — COLLAPSE (C7)**: [transcript excerpt where agent accepts incorrect partner info without challenge]  
**Case 2 — PROD_FAIL**: [transcript excerpt where agents engage genuinely but computation fails]  
**Case 3 — TRIVIAL**: [transcript excerpt where agents solve in parallel, exchange is superficial]  
**Case 4 — COUPLING (anchor)**: math_00121×C7 — Agent 2 challenges "sec θ+tan θ=22/7 vs csc θ+cot θ=m/n — DIFFERENT expressions."

---

## 4. Proposed Interventions

### 4.1 Structured Turn Protocol (STP)

**Hypothesis**: Explicitly requiring Phase A before Phase B prevents premature solution attempts.

**Implementation**: Add a mandatory turn structure to the C7 system prompt:
```
Turn 1: Share your information packet only. Do NOT attempt to solve.
Turn 2: Ask your partner at least one verification question about their information.
Turn 3: Answer your partner's question. Challenge any inconsistency.
Turn 4+: Collaborate on the solution.
```

**Expected effect**: Higher PhAQ (forced Phase A), potentially lower CDI if it introduces rigidity.

**Risk**: May reduce naturalistic collaboration, inflate PhAQ without improving epistemic quality.

### 4.2 Epistemic Skepticism Prompting (ESP)

**Hypothesis**: LLM sycophancy suppresses Phase A. An explicit anti-sycophancy instruction ("assume your partner may have made an error") activates challenge behavior.

**Implementation**: Add to C7 system prompt:
```
IMPORTANT: Do not assume your partner's information or calculations are correct.
Before using any value they provide, verify it against your own packet.
If something seems inconsistent, say so explicitly.
```

**Expected effect**: Higher Phase A quality, more genuine challenges, potentially more PROD_FAIL (agents challenge even correct info).

### 4.3 Socratic Prompting

**Hypothesis**: Replacing direct information exchange with question-based interaction forces deeper engagement.

**Implementation**: Agents must ask questions to elicit partner's information rather than receive it directly. Partner answers questions instead of sharing packets.

### 4.4 Model Scale Ablation

**Hypothesis**: GPT-4o-mini lacks metacognitive capacity for sustained Phase A; GPT-4.1 would produce substantially better collaboration.

**Design**: Replicate C7 condition with GPT-4.1 as simulation model on n=30 problems (cost: ~$15).

**Expected effect**: Higher PhAQ, more COUPLING. If model scale is the main bottleneck, this is evidence for using stronger models in future work.

---

## 5. Experimental Design

### 5.1 Conditions

| Condition | Description | Baseline |
|-----------|-------------|---------|
| C7 | L3 student-sim (current best) | ✓ existing data |
| C7-STP | C7 + structured turn protocol | new |
| C7-ESP | C7 + epistemic skepticism prompt | new |
| C7-SOC | C7 + Socratic prompting | new |
| C7-GPT4 | C7 with GPT-4.1 simulator | new |

### 5.2 Problems

Use the n=140 problems from the CollabMath scale study. CDI scores for C7 already available — new conditions directly comparable.

Run 3 repetitions per condition (standard in CollabMath pipeline).

### 5.3 Metrics

Same CPP metrics: CDI, CQI, PhAQ, ATC, CY, quadrant distribution.
Primary outcome: CDI (same as NeurIPS paper, directly comparable).
Secondary: COUPLING rate (% problems reaching COUPLING quadrant).

### 5.4 Cost Estimate

- C7-STP, C7-ESP, C7-SOC: n=140 × 3 reps × 3 conditions ≈ 1,260 conversations × ~$0.04 = ~$50
- C7-GPT4: n=140 × 3 reps × ~$0.15 = ~$63
- Total: ~$115

### 5.5 Evaluation Timeline [PLACEHOLDER]

- Data collection: [TBD]
- Analysis: 1 week
- Writing: 2 weeks

---

## 6. Expected Results and Significance

### 6.1 Best-case scenario

One or more interventions significantly improves CDI (d>0.5) and COUPLING rate (>35%). This establishes that collaboration quality is malleable through prompting — it is not purely a property of the split quality or model capability.

### 6.2 Null result scenario

No intervention significantly improves CDI. This would indicate that model scale (H-D) or the fundamental structure of the task is the bottleneck, not prompting. This is also a publishable result — it bounds what prompting can achieve.

### 6.3 Significance for the field

- First systematic analysis of LLM collaboration failure modes at the PISA CPS rubric level
- Establishes what aspects of epistemic collaboration can be improved through prompting vs. require architectural changes
- Provides a protocol for evaluating future multi-agent systems on collaboration quality (not just task accuracy)

---

## 7. Discussion [PLACEHOLDER]

### 7.1 Implications for Collaborative AI Design

### 7.2 Relationship to Human CPS Literature

### 7.3 Limitations

- LLM simulation ≠ human student behavior
- CDI measures epistemic engagement, not learning outcome
- Results may be model-family specific

---

## 8. Conclusion [PLACEHOLDER]

---

## References [PLACEHOLDER]

- Johnson, D. W., & Johnson, R. T. (1989). *Cooperation and Competition: Theory and Research.*
- Roschelle, J., & Teasley, S. D. (1995). The construction of shared knowledge in collaborative problem solving.
- Fischer, F., et al. (2013). *The International Handbook of Collaborative Learning.* Routledge.
- Kapur, M. (2016). Examining productive failure, productive success, unproductive failure, and unproductive success in learning. *Educational Psychologist*.
- Berkowitz, M. W., & Gibbs, J. C. (1983). Measuring the developmental features of moral discussion. *Merrill-Palmer Quarterly*.
- [CITE: MetaGPT, CAMEL, AutoGen, relevant LLM multi-agent papers]

---

## Appendix: Experiment Tracking

| Condition | Job ID | Status | CDI mean | COUPLING% |
|-----------|--------|--------|----------|-----------|
| C7 (baseline) | — | Done (NeurIPS) | 0.613 | 24% |
| C7-STP | TBD | Pending | — | — |
| C7-ESP | TBD | Pending | — | — |
| C7-SOC | TBD | Pending | — | — |
| C7-GPT4 | TBD | Pending | — | — |
