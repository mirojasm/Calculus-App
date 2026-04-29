"""
Multi-agent conversation simulator.

Conditions:
  solo              — 1 agent, full problem, solves alone
  unrestricted_pair — 2 agents, both see full problem (no split); tests
                      whether collaboration happens naturally without necessity
  jigsaw_N          — N agents, each with only their packet (forced collaboration)

Each conversation: agents alternate turns until consensus or max_turns.
Returns a Conversation object with full transcript + extracted final answer.
"""
import json, re, textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from research.config import CFG
from research.splitting.splitter import SplitResult, Packet
from research.openai_utils import chat


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class Turn:
    agent_id:  int
    role:      str   # "user" | "assistant" (OpenAI convention)
    content:   str

@dataclass
class Conversation:
    problem_id:    str
    condition:     str   # e.g. "jigsaw_2", "solo", "unrestricted_pair"
    n:             int
    turns:         List[Turn] = field(default_factory=list)
    final_answer:  Optional[str] = None
    consensus:     bool = False
    total_turns:   int  = 0

    def to_dict(self) -> dict:
        return {
            "problem_id":   self.problem_id,
            "condition":    self.condition,
            "n":            self.n,
            "turns":        [{"agent_id": t.agent_id, "content": t.content} for t in self.turns],
            "final_answer": self.final_answer,
            "consensus":    self.consensus,
            "total_turns":  self.total_turns,
        }


# ── system prompts ────────────────────────────────────────────────────────────

def _build_jigsaw_system(shared: str, packet: Packet, packets: List[Packet],
                         n: int, agent_id: int) -> str:
    """
    Minimal collaborative framing (v3).
    Provides context and the agent's view of the problem without prescribing
    CPS phases, labeling information as private, or asserting what the agent
    can or cannot do alone — lets the epistemic structure drive collaboration.
    """
    context = f"{shared}\n\n{packet.information}".strip() if shared else packet.information
    return textwrap.dedent(f"""
    You are participating in a collaborative math activity with {n - 1} partner(s).
    You can exchange messages to work on the problem together.

    {context}
    """).strip()


def _build_social_jigsaw_system(shared: str, packet: Packet, packets: List[Packet],
                                n: int, agent_id: int) -> str:
    """
    Jigsaw prompt enriched with explicit social coordination responsibilities:
    leadership, participation equity, commitment protocol, group regulation,
    and acknowledgment — the elements absent from the baseline jigsaw condition.
    """
    role_header = f"You are the **{packet.role_name}** (Agent {agent_id})." \
                  if packet.role_name else f"You are Agent {agent_id}."

    partner_lines = []
    for p in packets:
        if p.agent_id != agent_id:
            name = p.role_name or f"Agent {p.agent_id}"
            partner_lines.append(f"  - Agent {p.agent_id}: {name}")
    partners_block = "\n".join(partner_lines) if partner_lines else "  (none)"
    role_desc = f"\nYOUR ROLE: {packet.role_description}" if packet.role_description else ""

    return textwrap.dedent(f"""
    {role_header}{role_desc}

    SHARED GOAL (all agents see this):
    {shared}

    YOUR PARTNERS AND THEIR EXPERTISE:
{partners_block}

    YOUR PRIVATE INFORMATION (only you know this — do not reveal all at once):
    {packet.information}

    ── MATHEMATICAL COLLABORATION ──
    - You CANNOT solve the problem alone. Share your expertise and ask partners.
    - Build the joint solution step by step.
    - When all agents agree on a final answer, state: FINAL ANSWER: <answer>

    ── SOCIAL COORDINATION RESPONSIBILITIES ──
    You are responsible for BOTH the mathematical content AND the group process.
    The transcript shows a participation counter [Participation: ...] each turn — use it.

    1. LEADERSHIP: If the group is stuck, repeating itself, or missing a key step,
       take the initiative: "I think we should refocus on [X] because..."
       Do not wait for someone else to redirect — lead when needed.

    2. PARTICIPATION EQUITY: If a partner has fewer turns in the counter,
       explicitly invite them: "Agent X, I'd like your perspective on [Y]."
       Do not let any agent be passive.

    3. COMMITMENT PROTOCOL: When you agree to do something specific, declare it:
       "I COMMIT TO: [action]"
       When following up on a partner's commitment:
       "You committed to [X] — can you share that now?"

    4. GROUP REGULATION: Every 3–4 turns, briefly assess collective progress:
       "GROUP CHECK: We have [what]. We still need [what]. Next step: [who does what]."

    5. ACKNOWLEDGMENT: Before building on a partner's contribution, confirm:
       "I understand you found that [X]. Using this, I can now..."

    Be concise: 3–5 sentences per turn. Do NOT invent information you were not given.
    """).strip()


def _build_unrestricted_system(problem: str, agent_id: int, n: int) -> str:
    peers = [i for i in range(1, n + 1) if i != agent_id]
    peer_str = ", ".join(f"Agent {i}" for i in peers)
    return textwrap.dedent(f"""
    You are Agent {agent_id} collaborating with {peer_str} to solve a mathematics problem.
    You both have full access to the problem.

    PROBLEM:
    {problem}

    Work together. Share your reasoning. When you agree on an answer, state:
    FINAL ANSWER: <answer>
    Be concise: 2–4 sentences per turn.
    """).strip()


def _build_solo_system(problem: str) -> str:
    return textwrap.dedent(f"""
    Solve the following mathematics problem step by step.
    Show your reasoning. End your response with: FINAL ANSWER: <answer>

    PROBLEM:
    {problem}
    """).strip()


# ── answer extraction ─────────────────────────────────────────────────────────

_ANSWER_PATTERN = re.compile(r"FINAL ANSWER\s*:\s*(.+)", re.IGNORECASE)

def _extract_answer(text: str) -> Optional[str]:
    m = _ANSWER_PATTERN.search(text)
    return m.group(1).strip() if m else None


def _consensus_reached(turns: List[Turn], n: int) -> bool:
    """True when at least n distinct agents have stated a FINAL ANSWER."""
    agents_with_answer = {
        t.agent_id for t in turns if _extract_answer(t.content)
    }
    return len(agents_with_answer) >= max(1, n - 1)   # majority agreement


# ── core call ─────────────────────────────────────────────────────────────────

def _chat(system: str, history: List[dict], max_tokens: int = 300) -> str:
    messages = [{"role": "system", "content": system}] + history
    return chat(
        messages=messages,
        model=CFG.model_simulator,
        temperature=CFG.temperature_sim,
        max_tokens=max_tokens,
    )


# ── simulators ────────────────────────────────────────────────────────────────

def simulate_solo(split_result: SplitResult) -> Conversation:
    """N=1: single agent solves alone."""
    system = _build_solo_system(split_result.problem)
    history = [{"role": "user",
                "content": "Please solve the problem. You MUST end with: FINAL ANSWER: <answer>"}]
    response = _chat(system, history, max_tokens=600)

    conv = Conversation(
        problem_id=split_result.problem_id,
        condition="solo",
        n=1,
        turns=[Turn(agent_id=1, role="assistant", content=response)],
        final_answer=_extract_answer(response),
        consensus=bool(_extract_answer(response)),
        total_turns=1,
    )
    return conv


def simulate_pair(split_result: SplitResult, condition: str) -> Conversation:
    """N≥2: jigsaw or unrestricted pair/group conversation."""
    n = split_result.n
    packets = split_result.packets

    # Build per-agent system prompts
    systems = {}
    if condition == "unrestricted_pair":
        for pkt in packets:
            systems[pkt.agent_id] = _build_unrestricted_system(
                split_result.problem, pkt.agent_id, n
            )
    else:   # jigsaw
        for pkt in packets:
            systems[pkt.agent_id] = _build_jigsaw_system(
                split_result.shared_context, pkt, packets, n, pkt.agent_id
            )

    # Conversation histories per agent (each sees only their own + shared transcript)
    shared_transcript: List[dict] = []   # what all agents see
    turns: List[Turn] = []

    # Goal-anchor: minimal framing with dynamic answer format specification from M1
    fmt = getattr(split_result, "answer_format", {}) or {}
    format_spec = fmt.get("specification", "").strip()
    if not format_spec:
        format_spec = "State your final answer clearly when both partners agree."
    goal_anchor = (
        f"COLLABORATIVE TASK:\n\n"
        f"{split_result.problem}\n\n"
        f"{format_spec}"
    )

    agent_order = [((i % n) + 1) for i in range(CFG.max_turns)]

    for turn_idx, agent_id in enumerate(agent_order):
        history = shared_transcript.copy()
        if not history:
            history.append({"role": "user", "content": goal_anchor})
        elif turn_idx == len(agent_order) - 2:
            # Phase D: force group verification before final turn
            history.append({
                "role": "user",
                "content": (
                    "[GROUP VERIFICATION] Re-read the original question. "
                    "Both of you: do you agree on a single integer answer? "
                    "State FINAL ANSWER: <integer> only when both partners agree."
                ),
            })

        response = _chat(systems[agent_id], history)
        turn = Turn(agent_id=agent_id, role="assistant", content=response)
        turns.append(turn)

        shared_transcript.append({"role": "user" if turn_idx % 2 == 0 else "assistant",
                                   "content": f"[Agent {agent_id}]: {response}"})

        if _consensus_reached(turns, n):
            break

    final = None
    for t in reversed(turns):
        a = _extract_answer(t.content)
        if a:
            final = a
            break

    return Conversation(
        problem_id=split_result.problem_id,
        condition=condition,
        n=n,
        turns=turns,
        final_answer=final,
        consensus=_consensus_reached(turns, n),
        total_turns=len(turns),
    )


def simulate_social_pair(split_result: SplitResult, condition: str) -> Conversation:
    """
    social_jigsaw_N — same information structure as jigsaw_N but agents carry
    explicit social coordination responsibilities (leadership, participation equity,
    commitment protocol, group regulation, acknowledgment).
    Participation counts are injected into the transcript each turn.
    """
    n = split_result.n
    packets = split_result.packets

    systems = {
        pkt.agent_id: _build_social_jigsaw_system(
            split_result.shared_context, pkt, packets, n, pkt.agent_id
        )
        for pkt in packets
    }

    shared_transcript: List[dict] = []
    turns: List[Turn] = []
    participation: dict = {pkt.agent_id: 0 for pkt in packets}

    agent_order = [((i % n) + 1) for i in range(CFG.max_turns)]

    for turn_idx, agent_id in enumerate(agent_order):
        part_str = " | ".join(f"Agent {a}: {c} turns" for a, c in participation.items())
        part_header = f"[Participation: {part_str}]"

        history = shared_transcript.copy()
        if not history:
            history.append({"role": "user",
                            "content": f"{part_header}\n"
                                       "Let's start. Introduce your role, share what you know, "
                                       "and establish how we'll work together."})
        else:
            history.append({"role": "user", "content": part_header})

        response = _chat(systems[agent_id], history)
        turn = Turn(agent_id=agent_id, role="assistant", content=response)
        turns.append(turn)
        participation[agent_id] += 1

        shared_transcript.append({
            "role": "user" if turn_idx % 2 == 0 else "assistant",
            "content": f"[Agent {agent_id} | Turn {turn_idx + 1}]: {response}",
        })

        if _consensus_reached(turns, n):
            break

    final = None
    for t in reversed(turns):
        a = _extract_answer(t.content)
        if a:
            final = a
            break

    return Conversation(
        problem_id=split_result.problem_id,
        condition=condition,
        n=n,
        turns=turns,
        final_answer=final,
        consensus=_consensus_reached(turns, n),
        total_turns=len(turns),
    )


# ── public API ────────────────────────────────────────────────────────────────

def simulate_with_monitor(
    split_result: SplitResult,
    condition: str,
    max_interventions: int = 2,
) -> Conversation:
    """
    Condition C4/C5: jigsaw simulation with a Szewkis monitor injecting
    corrective interventions after PISA phase transitions (A and B only).

    Injects at most max_interventions monitor turns into the conversation.
    Monitor turns appear as Agent 0 with role 'facilitator'.
    """
    from research.simulation.monitor import detect_phase, evaluate_phase, MONITOR_PHASES

    n       = split_result.n
    packets = split_result.packets

    systems = {
        pkt.agent_id: _build_jigsaw_system(
            split_result.shared_context, pkt, packets, n, pkt.agent_id
        )
        for pkt in packets
    }

    shared_transcript: List[dict] = []
    turns:             List[Turn] = []
    interventions:     int = 0
    current_phase:     str = "A"
    phase_start_idx:   int = 0

    agent_order = [((i % n) + 1) for i in range(CFG.max_turns)]

    for turn_idx, agent_id in enumerate(agent_order):
        history = shared_transcript.copy()
        if not history:
            history.append({"role": "user", "content": "Let's start. Share what you know."})

        response = _chat(systems[agent_id], history)
        turn = Turn(agent_id=agent_id, role="assistant", content=response)
        turns.append(turn)

        shared_transcript.append({
            "role": "user" if turn_idx % 2 == 0 else "assistant",
            "content": f"[Agent {agent_id}]: {response}",
        })

        # Detect phase transition — never intervene in first 3 turns (phase A build-up)
        history_dicts = [{"content": t.content} for t in turns]
        new_phase = detect_phase(history_dicts)

        if new_phase != current_phase and interventions < max_interventions \
                and len(turns) > 3:
            if current_phase in MONITOR_PHASES:
                phase_turns = [
                    {"agent_id": t.agent_id, "content": t.content}
                    for t in turns[phase_start_idx:]
                ]
                monitor_result = evaluate_phase(
                    current_phase, phase_turns, n, split_result.problem
                )
                if monitor_result.intervene and monitor_result.intervention:
                    intervention_text = monitor_result.intervention
                    monitor_turn = Turn(
                        agent_id=0,
                        role="user",
                        content=f"[FACILITATOR]: {intervention_text}",
                    )
                    turns.append(monitor_turn)
                    shared_transcript.append({
                        "role": "user",
                        "content": f"[FACILITATOR]: {intervention_text}",
                    })
                    interventions += 1

            current_phase   = new_phase
            phase_start_idx = len(turns)

        if _consensus_reached(turns, n):
            break

    final = None
    for t in reversed(turns):
        a = _extract_answer(t.content)
        if a:
            final = a
            break

    return Conversation(
        problem_id=split_result.problem_id,
        condition=condition,
        n=n,
        turns=turns,
        final_answer=final,
        consensus=_consensus_reached(turns, n),
        total_turns=len(turns),
    )


def simulate(split_result: SplitResult, condition: str) -> Conversation:
    """
    condition ∈ {"solo", "unrestricted_pair", "jigsaw_2", "jigsaw_3", "jigsaw_4",
                 "social_jigsaw_2", "social_jigsaw_3", "social_jigsaw_4",
                 "cpp_directed_2", "monitored_jigsaw_2", "integrated_2"}
    """
    if not split_result.valid and condition != "solo":
        raise ValueError(
            f"Split for {split_result.problem_id} is invalid — cannot simulate {condition}"
        )

    if condition == "solo":
        return simulate_solo(split_result)

    if condition.startswith("social_jigsaw"):
        return simulate_social_pair(split_result, condition)

    if condition in ("monitored_jigsaw_2", "integrated_2"):
        return simulate_with_monitor(split_result, condition)

    return simulate_pair(split_result, condition)
