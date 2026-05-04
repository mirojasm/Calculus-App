# Benchmark Card: CIDI-140

**Associated paper**: "Epistemic Role Framing Elicits Deep Collaboration in LLM Multi-Agent Systems"
**Track**: NeurIPS 2026 Evaluations & Datasets
**Version**: 1.0 (2026-05-04)

---

## Dataset Summary

CIDI-140 is a benchmark of 140 MATH problems equipped with Collaborative
Interdependence-Directed Information (CIDI) splits designed to measure how
deeply LLM agents collaborate when solving mathematical problems jointly.

Each problem is decomposed into:
- **Shared context**: problem elements visible to both agents
- **Packet A**: information exclusive to Agent A (Agent A cannot solve without Agent B's deductions)
- **Packet B**: symmetric complement for Agent B

The benchmark enables controlled comparison of agent-framing conditions (C2: natural,
C6: peer-aware, C7: active-learner) and ablation conditions (CFULL: learner framing
without split, CEXP: expert framing with split).

---

## Corpus Construction

### Source
Problems drawn from the MATH benchmark (Hendrycks et al., 2021), spanning:
- 6 subject areas: Algebra, Geometry, Number Theory, Prealgebra, Precalculus, Counting & Probability
- 5 difficulty levels (MATH levels 1–5)
- 30 subject × difficulty cells

### Filtering (Epistemic Filter)
From an initial pool of 300 problems, we ran all problems under condition C7 and
retained those with CDI ≥ 0.5 in at least 2 of 3 replications. This filter
identifies problems for which the CIDI split produces genuine reasoning-level
interdependence.

- **Initial pool**: 300 problems
- **Retained**: 140 problems (47% retention rate)
- **Cell coverage**: all 30 cells, minimum 1 problem per cell, mean 4.7

### Critical: Selection Conditionality

**All estimates derived from this corpus are conditional on CIDI-amenable problems.**

Because the filter uses C7 performance to select the corpus:
1. Absolute CDI values reflect performance on pre-selected problems, not a random MATH sample
2. Within-corpus effect sizes (C7 vs. C2) may overestimate framing benefits on unfiltered samples
3. Paired comparisons remain internally valid within this selected regime

This conditionality is the most important caveat for users of this benchmark.
Users should NOT interpret CIDI-140 results as estimates of framing effects on
arbitrary MATH problems.

---

## Intended Use

**Appropriate uses:**
- Comparing multi-agent framing conditions under controlled epistemic interdependence
- Evaluating collaborative discourse quality using the CPP framework
- Benchmarking LLM multi-agent systems on information-split collaborative tasks
- Studying the structural vs. activated interdependence distinction via AEC

**Not appropriate for:**
- Estimating absolute collaborative problem-solving performance on a random sample
- Benchmarking single-agent performance (problems require two-agent collaboration)
- Generalizing framing effects beyond the CIDI-amenable regime without validation

---

## Metrics

### CDI (Collaborative Discourse Index)
Fraction of 12 CPP cells with score ≥ 1. Range [0, 1].
High CDI indicates broad engagement across collaboration phases and competencies.

### CQI (Collaborative Quality Index)
Quality-weighted sum: Σ scores / 36. Range [0, 1].
Penalizes superficial engagement more than CDI.

### PhAQ (Phase A Quality)
Quality of the joint exploration phase (Process A, cells A1–A3): Σ A-scores / 9.
PhAQ = 0 indicates agents proceeded to solution without joint exploration.

### Quadrant Taxonomy
Conversations are classified into 4 quadrants based on CDI (≥0.5 vs. <0.5) and correctness:
- **COUPLING**: high CDI + correct (ideal)
- **PROD_FAIL**: high CDI + incorrect (productive failure)
- **TRIVIAL**: low CDI + correct (solved without collaboration)
- **COLLAPSE**: low CDI + incorrect (complete failure)

### AEC (Agent Epistemic Contribution)
Shapley-value decomposition of v({A}), v({B}), v({A,B}) measuring:
- **EN (Epistemic Necessity)**: pair surpasses better solo
- **CS (Collaborative Surplus)**: gain magnitude
- **EB (Epistemic Balance)**: symmetry of agent contributions

Note: v({A,B}) uses best C7 replication; EN and CS are upper-bound estimates
under C7, not directly comparable across conditions.

---

## AEC Validation Results

| Metric                    | Value  |
|---------------------------|--------|
| v_A (Agent A solo)        | 0.100  |
| v_B (Agent B solo)        | 0.057  |
| v_AB (C7 pair)            | 0.357  |
| Both-solo-zero rate       | 85.0%  |
| EN rate (epistemic need)  | 25.0%  |
| CS mean (collab. surplus) | +0.207 |
| EB mean (balance)         | 0.857  |

85% of problems exhibit both-solo-zero, confirming genuine epistemic interdependence
at the correctness level. Even under the strongest framing (C7), only 25% achieve
activated epistemic necessity.

---

## Conversation Corpus Statistics

| Condition   | n_problems | n_conversations | Mean CDI       |
|-------------|-----------|-----------------|----------------|
| C2 (natural)| 140       | 656             | 0.362 ± 0.015  |
| C6 (peer)   | 140       | 654             | 0.390 ± 0.015  |
| C7 (learner)| 140       | 657             | 0.613 ± 0.014  |
| CFULL       | 140       | 412*            | 0.296 ± 0.018  |
| CEXP        | 140       | 420             | 0.267 ± 0.017  |

*8 CFULL conversations excluded due to content-policy refusals (840 planned, 832 completed)

---

## Known Limitations

1. **Corpus conditionality**: See "Critical" note above.
2. **Single model**: All experiments use GPT-4.1; results may not generalize to other models.
3. **Reliability sample**: Inter-rater reliability assessed on n=36 conversations (preliminary).
4. **AEC scope**: EN and CS are upper-bound estimates; per-condition AEC not yet run.
5. **Subcomponent ablations**: Ignorance-only and debate framing conditions not yet run.
6. **Collaborative register**: Cannot rule out discourse mimicry without micro-level trace analysis.

---

## Licensing

- **CPP rubric, CIDI prompt, annotations, scripts**: CC BY 4.0
- **MATH problem IDs and derived splits**: CC BY 4.0 (derived work)
- **Original MATH problem text**: NOT redistributed (per MATH benchmark license)
- **Conversation transcripts**: Terms of applicable API provider (OpenAI)

---

## Citation

```bibtex
@inproceedings{anonymous2026epistemic,
  title={Epistemic Role Framing Elicits Deep Collaboration in {LLM} Multi-Agent Systems},
  booktitle={Advances in Neural Information Processing Systems --- Evaluations \& Datasets Track},
  year={2026},
  note={Anonymous submission}
}
```

---

## Maintenance

This benchmark card reflects the state of the dataset as of 2026-05-04.
Known gaps to be addressed in camera-ready: per-cell κ expansion (n≥100),
Croissant metadata file, per-condition AEC runs.
