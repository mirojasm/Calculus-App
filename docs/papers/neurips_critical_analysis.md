# CollabMath — Critical Analysis for NeurIPS 2026

**Prepared:** 2026-04-29  
**Role:** Senior NeurIPS reviewer + CSCL researcher  
**Scope:** Brutally honest pre-submission audit. Prioritized by impact on acceptance probability.

---

## Executive Summary

The paper has a genuinely interesting empirical finding: role framing dramatically changes agent
collaboration depth in a way that is measurable via a new metric (CDI). The effect size (d=1.45
between C2 and C7 in the full-scale run, d=0.95 in Phase 2 partial) is large and robust enough
to survive scrutiny. However, the submission has five methodological wounds that will appear in
every NeurIPS review, two novelty vulnerabilities that need pre-emptive citation armor, and a
framing identity crisis that will kill it if unresolved before the May 6 deadline. None of these
are fatal — all are fixable — but they require specific, concrete work in the next 7 days.

---

## Section 1: Top 5 Methodological Weaknesses a Reviewer WILL Flag

### W1 — The Annotator Is the Metric: LLM Self-Evaluation Circularity (CRITICAL)

**The problem.** CDI is computed by an LLM annotator (CPP annotator, likely GPT-4o-mini or
Qwen2.5-72B-AWQ) that reads conversations produced by the same model family under different
prompt framings. The annotator was not validated against independent human judgment for the
CDI/CQI scale. The kappa values reported (H-H=0.80, H-LLM=0.78) come from the C&E paper on a
different system (n=30 Calculus sessions) with a different scoring instrument (PISA 12 codes,
not the CPP/CDI framework). There is no inter-rater reliability number for CDI specifically.

**Why reviewers will flag it.** NeurIPS 2024-2025 saw an explosion of papers claiming LLM
evaluation of LLM behavior. The community is specifically sensitized to this. Reviewer 2 will
ask: "How do we know CDI=0.643 in C7 is real collaboration and not the annotator recognizing
the student-sim framing and scoring it higher because it sounds more collaborative?"

**Concrete fix.**
1. Sample 50 conversations (balanced: 20 C2, 20 C7, 10 C6) from the actual pilot corpus.
2. Have two human coders blind to condition annotate the CPP 12-cell binary vector (not CDI
   directly — cell by cell is more reliable).
3. Compute: (a) Krippendorff's alpha for the CPP vector, (b) Cohen's kappa per cell, (c)
   Pearson r between human-derived CDI and LLM-derived CDI.
4. Report this as Table 1 in the paper. Target: alpha > 0.70, r > 0.80.
5. If you cannot recruit two human coders in 7 days, do it yourself twice with a 2-week gap —
   this is methodologically defensible for a pilot-scale study and reviewers accept it for novel
   metrics.

**Expected result.** Based on the PhAQ=0 for C2/C6 and PhAQ>0 for C7 finding, the cells that
discriminate conditions (A1-A3, C1-C2) should have high reliability because they are anchored
to observable conversational acts (explicit questions, sharing of sub-results). The social cells
(D1-D3) will be noisier. Report both figures.

---

### W2 — The Student-Sim Condition Is Not an Ablation: It's a Confounder (HIGH)

**The problem.** C7 (L3, student-sim) differs from C2 (L1, natural) on at least three
dimensions simultaneously:
1. **Role framing**: "simulate a Calc-1 student" vs. neutral
2. **Epistemic disposition**: tentativeness, question-asking behavior
3. **Domain knowledge restriction**: implicit (student sim implies limited knowledge)

When CDI(C7) > CDI(C2) with d=1.45, the paper cannot claim the effect is due to "collaborative
framing depth." It could be entirely explained by (3): the student-sim agent genuinely does not
confidently execute mathematical steps alone, so it must ask. That is not CPS — it is simulated
ignorance. This is W1 of the CSCL literature (Dillenbourg 1999: "the asymmetry must be
epistemic, not merely behavioral").

**Why reviewers will flag it.** AC3 (collaborative AI) and any CSCL reviewer will immediately
say: "Your C7 effect could be entirely explained by the agent performing ignorance rather than
generating epistemic interdependence. Your C6 (peer-aware) vs. C7 (student-sim) comparison is
the relevant test, and C6≈C2 in CDI is your most important result, not C7>C2."

**Concrete fix.**
1. Add a C7' condition: same student-sim framing, but problem given to a single agent (no
   partner). If CDI_solo(C7') is high — impossible by definition since CDI requires multi-agent
   — use CQI_solo as proxy. What matters: does the student-sim agent attempt to reason through
   all phases A-D alone, or does it genuinely wait for input?
2. Measure **turn-by-turn information dependency**: for each turn in C7 conversations, annotate
   whether Agent 2's next computation step could have been executed without Agent 1's previous
   message. This distinguishes "asking because of framing" from "asking because of genuine
   information need."
3. Add the C6≈C2 finding prominently. The fact that adding peer-awareness alone does nothing
   (d=-0.04, p=0.60 in Phase 2 partial) is your strongest argument that C7's effect is NOT
   trivially due to "being told to communicate." Frame C6 as a crucial null result, not a
   confounding condition to minimize.
4. In the paper: explicitly name the alternative hypothesis ("H_alt: effect is due to simulated
   ignorance, not collaborative framing") and show evidence against it.

---

### W3 — 0% Correct Answers in the Pilot (0 COUPLING quadrant) Is a Red Flag (HIGH)

**The problem.** Phase 1 full corpus (n=139 C7 simulations): 20 COUPLING (correct + CDI≥0.5),
119 not COUPLING. The Phase 2 partial shows partial data. But across the pilot v5/v6 runs: 0
COUPLING cases across 4 problems × 4 conditions. More critically, the paper's main claim is
about collaboration quality, but if the collaborations never produce correct answers under the
conditions with highest CDI, reviewers will question whether CDI measures anything meaningful
educationally.

**The deeper problem.** CDI⊥correctness (r=−0.015 reported in v2). If higher collaboration
depth (CDI) does not predict better outcomes, a NeurIPS reviewer will ask: "What is this
metric for? You've built an elaborate scoring system for a process that doesn't predict
results. This is a framework in search of an application."

**Why reviewers will flag it.** NeurIPS is an ML conference. The community expects metrics
to predict something. Kapur's "productive failure" framing is CSCL theory that most NeurIPS
reviewers will not be familiar with, and the paper cannot assume it.

**Concrete fix.**
1. Do NOT collapse CDI and correctness. Instead, test: does **problem-level** CDI variance
   predict correctness? That is: for problems where C7 generates CDI≥0.5, is the accuracy
   higher than for problems where C7 generates CDI<0.5 on the same problem set? This is a
   within-subject comparison that controls for problem difficulty.
2. Compute Kendall's tau between CDI(C7) and P(correct | C7) at the problem level. Expected:
   positive. Report this.
3. Reframe the outcome: correctness for competition math problems (MATH benchmark) is near
   impossible for GPT-4o-mini-class agents in any condition. Use **answer quality** instead:
   did the conversation get closer to the right approach, even if the final numerical answer
   is wrong? This is a softer measure but more defensible.
4. Alternatively: include answer quality (partial credit) scoring. A conversation where the
   agents set up the correct integral but make an arithmetic error should score higher than one
   where they set up the wrong approach. This is implementable with one additional LLM scoring
   call.

---

### W4 — The CDI Metric Has No Theoretical Derivation: Why 12 Cells Equally Weighted? (MEDIUM)

**The problem.** CDI = Σ cells_active / 12. This weights all 12 PISA cells equally. But:
- Phase A cells (exploration) are preconditions for Phase B/C/D cells by the DAG in your own
  framework (PREREQ_DAG in module2_feasibility.py).
- Activating D1 is qualitatively harder than activating A1 — yet both count as 1/12.
- The CDI formula has no empirical or theoretical derivation for the equal weighting.
- Reviewer 3 will ask: "Why is CDI not information-theoretically derived? Why not weight
  cells by their conditional entropy given the DAG?"

**Concrete fix.**
1. Add a DAG-weighted CDI variant: CDI_w = Σ depth(cell_i) × active(cell_i) / Σ depth(cell_i),
   where depth = topological depth in the PREREQ_DAG. This is a 30-line computation.
2. Show that CDI and CDI_w give the same condition ordering. If they do, report both and note
   convergent validity. If they don't, discuss why and choose the more theoretically motivated.
3. Alternatively, derive CDI from information theory: CDI = I(A1;A2) where A1 and A2 are the
   information sets of the two agents over the conversation. This requires a different
   measurement approach but is theoretically elegant and would be genuinely novel.

---

### W5 — Replication Variance Is Unknown: n=3 Reps May Be Insufficient (MEDIUM)

**The problem.** The design calls for 3 replications per condition × problem cell. But the
pilot data shows enormous variance: CDI ranges from 0.0 to 1.0 for the same problem under
the same condition across different runs (e.g., math_00014 × C2: different reps show
CDI=0.667, 0.500, 0.000 in different pilot versions). The variance of CDI within-condition
within-problem appears to be 0.20-0.35 (SD). With n=3, the 95% CI on a cell mean is ±0.20.
The effect size between C2 and C7 is ~0.25 in CDI units — barely distinguishable from noise
at n=3 per cell.

**Concrete fix.**
1. Compute the within-condition within-problem standard deviation of CDI from your existing
   Phase 2 data. You have multiple C7 reps for the same problems — use them.
2. Run a power analysis: given observed SD, what n per cell gives 80% power at alpha=0.05
   for the observed effect size? If n=5 is needed, you have 7 days and the data is already
   being generated.
3. Report confidence intervals on all CDI means in the paper, not just p-values.

---

## Section 2: Additional Analyses That Would Substantially Strengthen the Paper

### A1 — Problem Difficulty Interaction: Does the Agent-Type Effect Scale with Problem Complexity?

**What to compute.** Stratify the 72 genuinely collaborative problems (CDI(C7)≥0.5) by MATH
difficulty level (L1-L5) and compute CDI(C7) - CDI(C2) by level. Run a 2-way interaction:
condition × level. Test: does the L1→L3 gradient become steeper for harder problems?

**Expected result.** Based on your Phase 1 data: L3-L5 problems likely show larger C7>C2 gaps
because they have more multi-step epistemic chains. L1-L2 problems are solvable by either
agent type. If confirmed, this is a prediction about when student-sim framing matters — a
practical finding with direct design implications.

**Why it matters for reviewers.** NeurIPS reviewers want to know when a finding generalizes
and when it doesn't. An interaction effect transforms "C7 is better" into "C7 is better for
hard problems, irrelevant for easy ones" — a much more useful and publishable claim.

---

### A2 — Conversation Structure Analysis: Turn-Level CDI Trajectories

**What to compute.** For 20 conversations (10 C2, 10 C7), compute CDI at each turn (running
CDI = cells activated up to turn t / 12). Plot the trajectory for both conditions. Quantify:
(a) average turn at which each condition reaches CDI=0.5; (b) whether C7 activates Phase A
cells (A1-A3) earlier in the conversation than C2.

**Expected result.** C7 should show rapid Phase A activation in turns 1-4, followed by
sustained Phase B/C. C2 should show late or absent Phase A, with most CDI coming from Phase C
(execution sharing). This would confirm the qualitative finding from math_00121 × C7 and
make it quantitative.

**Why it matters for reviewers.** This analysis directly addresses W2 (the confounding
hypothesis). If C7 activates Phase A earlier because of genuine epistemic uncertainty (not
just framing compliance), you can argue the effect is not purely behavioral imitation.

---

### A3 — CDI Reliability and Annotator Consistency: Split-Half Reliability

**What to compute.** For 30 conversations, run the CPP annotator twice (two independent LLM
calls with temperature > 0). Compute: (a) agreement rate per cell; (b) CDI correlation between
the two runs; (c) which cells have the lowest reliability.

**Expected result.** A1, C1, C2 (structurally observable cells) will have high reliability
(>0.85). D1-D3 (monitoring/verification cells) will have lower reliability (0.60-0.75).
Report this as the annotator reliability section. This directly addresses W1.

**Why it matters for reviewers.** Providing annotator reliability is the minimum bar for a
new metric paper at NeurIPS. Without it, the paper will not clear review regardless of the
empirical findings.

---

### A4 — What Percentage of MATH Is "Genuinely Collaboratable"? A Corpus Characterization

**What to compute.** Use your full Phase 1 result (139 problems): fit a logistic regression
predicting CDI(C7)≥0.5 from: (a) MATH level, (b) MATH subject, (c) number of distinct
mathematical objects in the problem (proxy for structural complexity). Report the ROC-AUC.

**Expected result.** Based on your 51.8% genuine rate and the observation that data splits
dominate in simpler problems: MATH level will be a significant predictor (higher level →
more likely genuine epistemic split). Subject may also matter (number theory and probability
vs. algebra and pre-algebra).

**Why it matters for reviewers.** This analysis transforms the paper from "we ran an
experiment" to "we characterized the space of problems where collaborative AI is beneficial."
That is a dataset contribution independent of the CDI finding, and NeurIPS appreciates
contributions that provide reusable characterizations of benchmark datasets.

---

## Section 3: Missing Baselines NeurIPS Reviewers Will Expect

### B1 — Single-Agent Baseline with Deliberate Self-Reflection

The paper compares multi-agent collaboration (C7) against single-agent (C1 baseline) and
multi-agent without student-sim framing (C2). Missing: a single-agent condition where the
agent is prompted to think step-by-step, debate with itself, and self-verify — essentially
a "chain-of-thought with critique" baseline. This is directly comparable to C7 without the
multi-agent setup.

If single-agent CoT+critique achieves similar correctness to C7 with lower cost, the
multi-agent architecture's value is questionable. The paper should run this baseline (one
LLM call per problem) and show that the CDI metric requires genuine two-agent interaction —
by construction, it should be zero for a single agent, but the correctness comparison matters.

**Existing literature to cite.** Du et al. (2023, arXiv:2305.14325) "Improving Factuality
through Multiagent Debate" is the most direct comparison and is likely already in the
reviewer's mind. You must cite it and explicitly compare.

### B2 — Baseline from Existing Multi-Agent Math Literature

ChatDev (Qian et al., 2023), MetaGPT (Hong et al., 2023), and MathChat (Wu et al., 2023)
all use multi-agent setups for problem solving. The paper needs to differentiate explicitly:
"Unlike ChatDev/MetaGPT where agents have fixed role assignments (developer/tester), our C7
condition generates emergent role negotiation measured by CDI cells A1-A3." One sentence of
comparison is not enough — reviewers will want a quantitative comparison on overlapping
problems.

### B3 — Oracle Upper Bound on CDI

What is the maximum achievable CDI for your 72 genuine problems? The framework allows CDI=1.0
(all 12 cells active). You have 14 problems achieving CDI=1.0 in Phase 1. Use these as an
empirical oracle. Ask: what fraction of the potential CDI gain does C7 capture vs. C2?
CDI_gain_fraction = (CDI(C7) - CDI(C2)) / (1.0 - CDI(C2)). If this is 0.60 on average,
that is a concrete, interpretable result: "student-sim framing captures 60% of the achievable
collaboration gain."

### B4 — Human-Human Collaboration Baseline (Even if Approximate)

The paper is about CPS, which is defined relative to human learning behavior. Without any
human-human comparison, reviewers in the CSCL sub-area (and NeurIPS has CSCL-adjacent
reviewers in the education track) will ask: "How does this compare to what real students do?"

The n=30 Calculus sessions from the C&E paper provide this. You have human kappa data and
PISA scores from those sessions. Even as an appendix figure: "Human students in jigsaw CPS
conditions showed PISA_global of X, ATC_SR of Y. Our LLM agents under C7 show CDI of Z" —
this grounds the metric in human behavior.

---

## Section 4: Novelty Evaluation — CPP/CDI vs. Prior Art

### What Is Genuinely Novel

1. **CDI as a binary coverage metric over the PISA CPS cell space.** The PISA 2015 framework
   has been used for classification of individual utterances but not as a coverage metric over
   the 12-cell space. The reduction CDI = cells_active/12 is simple but the conceptual
   contribution — treating CPS as a set-cover problem over a theoretical space — is defensible
   as novel. The key claim: CDI operationalizes the LATENT STRUCTURE of collaboration (which
   phase-competency pairs were activated) rather than the manifest frequency of behaviors.

2. **Agent-type gradient (L1→L2→L3) as a design variable.** Existing multi-agent LLM work
   treats role assignments as fixed (e.g., MetaGPT: programmer/reviewer/tester). The discovery
   that the EPISTEMIC DISPOSITION of agents — not just their information content — determines
   collaboration depth is new. C6≈C2 (peer-aware adds nothing without epistemic framing) is
   the key discriminating result.

3. **CIDI (problem → epistemic split design).** Algorithmically deriving the information
   partition from the problem structure using the PREREQ_DAG is new. Prior jigsaw work
   (Aronson 1978, Kapur 2012 productive failure) designs splits manually. Automated split
   design from a formal epistemic analysis is a genuine algorithmic contribution.

### Dangerous Prior Art Citations (May Show Prior Art)

**These papers could kill your novelty claim if reviewers know them:**

1. **Shirley Wang et al. (2024), "Can LLMs Collaborate?" (EMNLP 2024 or similar).** There is
   growing work on evaluating when LLMs collaborate genuinely vs. superficially. Search for
   papers on "LLM collaboration depth" or "multi-agent epistemic reasoning" published in 2024.
   If any paper measures qualitatively different collaboration behaviors under different prompt
   conditions, you must compare explicitly.

2. **Aher et al. (2023), "Using Large Language Models to Simulate Multiple Humans and
   Replicate Human Subject Studies" (ICML 2023).** This paper uses LLMs to simulate human
   behavioral experiments. Your C7 (student-sim) is a specific instance of this and must
   cite it — but do so offensively: "Unlike Aher et al. who simulate human responses to
   surveys, we show that the cognitive disposition embedded in role framing changes the
   multi-agent interaction structure, measured by CDI."

3. **Pedagogical Agent literature.** Graesser et al. (2005) "AutoTutor," VanLehn (2011)
   "The Relative Effectiveness of Human Tutoring, Intelligent Tutoring Systems, and Other
   Tutoring Systems." These show the learning gains from peer vs. tutor agents. If reviewers
   know this literature, they will ask: "How is C7 different from existing pedagogical agent
   systems that already simulate peer cognition?" Your answer: "We don't simulate a peer agent
   interacting with a human — we measure the emergent CPS structure when two LLM agents with
   different epistemic framings interact with each other."

4. **Dillenbourg 1999, "What Do You Mean By 'Collaborative Learning'?"** This is the
   canonical paper that defines epistemic interdependence and warns against the confounding
   between behavioral and epistemic collaboration (directly relevant to W2). You MUST cite it.
   Citing it preemptively shows methodological awareness. Failing to cite it and having a
   reviewer know it is catastrophic.

5. **Okita & Schwartz (2013), "Learning Cascade."** Showed that explaining to a partner
   improves learning because of the epistemic demand. C7 (student-sim) creates this demand
   artificially. You should cite this as theoretical grounding for why student-sim disposition
   changes behavior.

### What to Explicitly Claim As Novel (Carefully)

Safe claims:
- "First formal operationalization of CPS depth as a coverage metric over the PISA 12-cell
  matrix (CDI)."
- "First empirical demonstration that agent epistemic disposition, not information asymmetry
  alone, determines collaboration depth in multi-agent LLM systems."
- "First algorithmic framework for generating information splits that target specific CPP
  profiles (CIDI pipeline)."

Avoid claiming:
- "First multi-agent LLM collaboration study" — clearly not true.
- "First automated CPS assessment" — CSCL community has been doing this since the 1990s.
- "First jigsaw design with LLMs" — prior work exists.

---

## Section 5: Ideal Framing for NeurIPS 2026

### The Identity Crisis

The paper currently has three competing framings:
1. **Systems paper**: "We built a pipeline (CIDI) that generates epistemic splits."
2. **Metric paper**: "We propose CDI as a new measure of CPS depth."
3. **Behavioral finding paper**: "We discovered that student-sim framing generates dramatically
   deeper collaboration (CDI=0.643 vs 0.362, d=1.45)."

At NeurIPS, these are three different papers. Submitting all three in one paper reads as
a shotgun approach and will be penalized by reviewers who expect a crisp contribution.

### Recommended Framing: Behavioral Finding + Metric Validation

**Lead with the behavioral finding.** "We discover that LLM agent epistemic disposition —
specifically, whether an agent simulates the epistemic state of a learner — is the primary
determinant of collaboration depth in multi-agent mathematical problem solving, independent
of information asymmetry structure." This is surprising, counter-intuitive (peer-awareness
alone doesn't work — C6≈C2), and directly actionable for practitioners building multi-agent
educational systems.

**Use CDI as the instrument.** Frame CDI as "the measurement tool we developed to make this
finding precise" — not the primary contribution. This is more defensible because:
1. CDI is simple (cells/12) and doesn't require a complex justification.
2. The behavioral finding is what NeurIPS readers will find interesting.
3. You avoid the "why equal weights?" attack (W4) by not claiming CDI as the main contribution.

**Move CIDI to future work or a 1-paragraph system description.** The full CIDI pipeline (6
modules) is overengineered for the current data scale and will distract from the main finding.
The paper should describe how splits are generated (CIDI), but not make the pipeline itself
a contribution. Save that for the full journal paper.

### Why This Beats Alternative Framings at NeurIPS

- **vs. Systems paper framing**: NeurIPS systems papers require clear engineering novelty and
  ablations showing each component matters. The CIDI pipeline has unvalidated modules (Module 5
  discriminators were trained on n=150 with unverified AUC). This will not survive review.
- **vs. Metric paper framing**: Metric papers require validation against a gold standard.
  Without human annotation of CDI (W1), a pure metric paper is rejected. As a secondary
  contribution (the instrument), CDI needs less validation.
- **vs. IJCSCL framing**: The CSCL community cares about learning outcomes, qualitative
  analysis, and theoretical grounding. NeurIPS cares about what the finding reveals about
  LLM behavior. These are different audiences with different priors.

### Title Recommendation

Current: "From Information Splits to Epistemic Partnerships" (too abstract, sounds like IJCSCL)

Better: "Epistemic Disposition Determines Collaboration Depth in Multi-Agent LLM Systems"
or: "Student-Simulator Framing Elicits Genuine Collaborative Problem Solving in LLM Dyads"
or: "When Does Multi-Agent LLM Collaboration Become Genuine? A Formal Framework and Empirical Study"

The title must include "multi-agent LLM" or an equivalent for NeurIPS discoverability.

---

## Section 6: What Can Be Done in the Next 7 Days (Before May 6)

Sorted by impact-to-effort ratio:

### Day 1-2: Data Collection (Running in Background)

**Action**: Ensure Phase 2 scale study reaches full 3 reps × all conditions for all 72 (or
the subset with partial data). You need: n=72 problems × 3 conditions × 3 reps = 648
conversations minimum for the main table. Your Phase 2 jobs are partially complete — focus
computing resources on completing them, especially C2 reps 2-3 (which are the baseline
needed for the effect size calculation).

**What not to do**: Do not run new conditions or expand the corpus. Finish what's running.

---

### Day 2-3: Human Annotation of CDI (Addresses W1)

**Action**: Extract 50 conversations from the pilot corpus covering C2 (15), C6 (10), C7 (15),
and C1 (10). For each conversation, create an annotation form with the 12 CPP cells and a
binary checkmark. You annotate independently from memory of the results (use the raw
conversation JSON, not the score files). Do it twice for 20 conversations (self test-retest).
Compute Krippendorff's alpha. This takes 4-6 hours.

If you can recruit one collaborator (co-advisor, labmate), have them annotate the same 20.
Cohen's kappa with a second annotator is much stronger than self test-retest.

---

### Day 3: Turn-Level CDI Trajectory Analysis (Addresses W2, provides A2)

**Action**: Write a 40-line Python script that computes running CDI at each conversation turn
for 20 conversations. Plot mean trajectories by condition. This will likely show the Phase A
activation pattern predicted above and provides Figure 2 of the paper.

---

### Day 3-4: Within-Problem Correctness Correlation (Addresses W3)

**Action**: From Phase 2 data, compute P(correct | CDI≥0.5) vs. P(correct | CDI<0.5) using
Fisher's exact test. Separately: compute Kendall's tau between problem-level CDI(C7) and
P(correct | C7) across the 72 problems. This is a 20-line pandas/scipy computation.

---

### Day 4-5: CDI Annotator Reliability (Addresses W1, provides A3)

**Action**: Re-run the CPP annotator on 30 conversations a second time with a different
random seed / temperature=0.7. Compute cell-level agreement and CDI correlation between the
two runs. This is an automated job requiring 60 LLM calls (~$0.60, ~20 minutes).

---

### Day 5: DAG-Weighted CDI (Addresses W4)

**Action**: Implement CDI_w using topological depth from the PREREQ_DAG. Compute it for the
full Phase 2 dataset. Show that CDI and CDI_w give the same condition ordering. This is a
deterministic computation, no new LLM calls. Takes 2 hours to implement and run.

---

### Day 5-6: Problem Difficulty Stratification (Provides A1)

**Action**: Run the logistic regression predicting CDI(C7)≥0.5 from MATH level and subject
using Phase 1 data (n=139 problems). Compute ROC-AUC. This is a 15-line sklearn script.
Use it as the basis for the corpus characterization analysis (A4).

---

### Day 6-7: Write the Paper

With the above data in hand:
- Section 1 (Introduction): Lead with the behavioral finding. 1 page.
- Section 2 (Framework): CDI definition, CIDI description (brief), agent typology L1/L2/L3. 1.5 pages.
- Section 3 (Experiment): Design, corpus characterization, annotator reliability. 1.5 pages.
- Section 4 (Results): Main effect table, CDI trajectories, correctness analysis, stratification. 2 pages.
- Section 5 (Discussion): C6≈C2 null result as key finding, limitations, implications. 1 page.

Total target: 8 pages + references, NeurIPS format.

---

## Section 7: Risk Assessment and Honest Acceptance Probability

### Factors Working For The Paper

- Effect size d=0.95-1.45 is large and will survive power concerns.
- The C6≈C2 null result is a strong conceptual contribution — it rules out a simple
  "more framing = more collaboration" interpretation.
- The MATH benchmark is a well-known, accepted dataset in the NeurIPS community.
- The PhAQ=0 finding (Phase A only activates with L3) is a clean discriminating result.
- The scale (1,215+ conversations) is respectable for this type of study.

### Factors Working Against

- LLM-evaluated LLM behavior without strong human validation (W1) is a known NeurIPS
  rejection criterion in 2025-2026.
- The paper's contribution is primarily behavioral/empirical with a simple metric, not a new
  algorithm or model — NeurIPS main track prefers ML methods contributions.
- No learning outcome or downstream task validation — reviewers will question educational
  validity.
- The CSCL theoretical framing (PISA, ATC21S, Szewkis) is unfamiliar to most NeurIPS
  reviewers and risks being dismissed as domain-specific jargon.

### Realistic Acceptance Probability

Without changes: ~10-15% (strong rejection risk on W1 and W2 alone).

With fixes W1 (human annotation) + W2 (ablation analysis + C6 prominence) + correct framing
(behavioral finding, not systems/metric paper): ~30-40%.

This is achievable. NeurIPS 2026 accept rate is ~25-30%. The paper is in the right zone if
the methodological vulnerabilities are addressed before submission.

### Honest Recommendation

If the deadline cannot be met with the fixes above: submit to **NeurIPS workshop** (Learning
on Graphs, or the new AI for Education workshop if available) rather than main track. The
workshop allows a 4-page version without the scrutiny of the full review. Use the workshop
feedback to strengthen the paper for IJCSCL or the next NeurIPS cycle. The C&E paper (already
complete) is the right outlet for the current evidence quality.

If you are committed to main track: the 7-day plan above is aggressive but achievable. The
most important single thing is W1 (human annotation of CDI). Everything else is secondary.

---

## References Reviewers Will Expect (That May Currently Be Missing)

- Du, Y., et al. (2023). Improving Factuality and Reasoning in Language Models through
  Multiagent Debate. arXiv:2305.14325. [Required — direct comparison baseline]
- Dillenbourg, P. (1999). What do you mean by 'collaborative learning'? In P. Dillenbourg
  (Ed.), Collaborative-Learning: Cognitive and Computational Approaches. [Required — W2]
- Kapur, M. (2012). Productive Failure in Learning the Concept of Variance. Instructional
  Science. [Required — PROD_FAIL framing]
- Aher, G., et al. (2023). Using Large Language Models to Simulate Multiple Humans and
  Replicate Human Subject Studies. ICML 2023. [Required — C7 precedent]
- Okita, S., & Schwartz, D. (2013). Learning Cascade. Journal of Educational Psychology.
  [Recommended — C7 theoretical grounding]
- Qian, C., et al. (2023). Communicative Agents for Software Development (ChatDev).
  arXiv:2307.07924. [Required — system comparison baseline B2]
- PISA (2015). Collaborative Problem Solving Framework. OECD. [Already cited, verify version]
- Roschelle, J., & Teasley, S. (1995). The construction of shared knowledge in collaborative
  problem solving. [Required — convergence / joint accountability grounding]
- Szewkis, E., et al. (2011). Collaboration within large groups in the classroom. IJCSCL.
  [Already in paper, verify citation completeness]
