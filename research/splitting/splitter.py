"""
N-way jigsaw splitter grounded in Szewkis et al. (2011) and jigsaw CPS literature.

Taxonomy of 7 split patterns:
  SPLIT-A  Composite Figure Components     → Geometry
  SPLIT-B  Dual Representation             → Precalculus, Algebra
  SPLIT-C  Complementary Conditions        → Algebra, Number Theory, Systems
  SPLIT-D  Multi-Step Chain                → All domains
  SPLIT-E  Objective × Constraints         → Optimization
  SPLIT-F  Sample Space × Counting Principle → Probability, Combinatorics
  SPLIT-G  Hypothesis × Key Lemma          → Proofs, Number Theory

split(problem_id, problem, n) -> SplitResult
  - n=1  → solo condition (no split)
  - n≥2  → jigsaw with explicit roles, group awareness, Szewkis conditions

split_cpp_targeted(problem_id, problem, n) -> SplitResult
  - Condition C2: CPP-DEEP objective + Szewkis as hard constraints
"""
import json, textwrap
from dataclasses import dataclass, field
from typing import List, Optional

from research.config import CFG
from research.openai_utils import chat


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class Packet:
    agent_id:         int
    information:      str    # private information for this agent
    role_name:        str = ""   # e.g. "Rectangle Expert"
    role_description: str = ""   # what this agent knows + who partners are

@dataclass
class SplitResult:
    problem_id:     str
    problem:        str
    n:              int
    pattern:        str          # which of the 7 patterns was applied
    shared_context: str          # visible to ALL agents
    packets:        List[Packet]
    valid:          bool
    validation_log: dict = field(default_factory=dict)
    raw_split:      dict = field(default_factory=dict)


# ── system prompt ─────────────────────────────────────────────────────────────

_SPLIT_SYSTEM = textwrap.dedent("""
You are an expert collaborative learning designer specializing in mathematics education.
Split the given problem into exactly {n} jigsaw packets grounded in CPS theory.

══════════════════════════════════════════════
SZEWKIS ET AL. (2011) — 6 REQUIRED CONDITIONS
══════════════════════════════════════════════
Your split MUST satisfy ALL of these:
1. COMMON GOAL        — All agents share an explicit, stated objective.
2. POSITIVE INTERDEPENDENCE — No agent (or strict subset) can solve the problem alone.
   The split creates genuine mutual dependency on domain knowledge, not just data.
3. INDIVIDUAL ACCOUNTABILITY — Each agent has a distinct named ROLE with clear expertise.
4. GROUP REWARD        — Success is the joint final answer only.
5. GROUP AWARENESS     — Each agent explicitly knows their partners' roles (not just
   that partners exist, but WHAT expertise each partner holds).
6. COORDINATION & COMMUNICATION — Information exchange must be NON-OPTIONAL:
   the split makes it mathematically impossible to proceed without sharing.

══════════════════════════════════════════════
7 JIGSAW SPLIT PATTERNS — choose the best fit
══════════════════════════════════════════════

SPLIT-A: Composite Figure Components [Geometry]
  Each agent sees one geometric component of a composite figure with its own formulas.
  Hidden connection: the shared boundary/relationship between components must be
  discovered through dialogue (e.g., diameter = base width in a Norman window).
  Strong interdependence: neither component alone determines the combined quantity.

SPLIT-B: Dual Representation [Precalculus, Algebra]
  Each agent holds a different mathematical representation of the SAME object
  (e.g., graph of f vs graph of f', symbolic vs geometric, table vs formula).
  Neither representation alone identifies or fully analyzes the object.
  Forces exchange of complementary observations to reconstruct the complete picture.

SPLIT-C: Complementary Conditions [Algebra, Number Theory, Systems]
  The problem requires N conditions/equations/constraints, each strictly necessary.
  Each agent receives a disjoint subset. No subset yields the solution alone.
  Strongest when each condition involves a different variable or modulus.

SPLIT-D: Multi-Step Chain [All domains]
  The problem has sequential steps: Step k's output feeds Step k+1.
  Agent 1 has tools for stage 1 but not stage 2's tools.
  Agent 2 has tools for stage 2 but needs stage 1's output.
  The handoff between stages IS the collaboration moment.

SPLIT-E: Objective × Constraints [Optimization]
  Agent 1 knows WHAT to optimize (objective function, quantity to extremize).
  Agent 2 knows the CONSTRAINTS (domain, boundary conditions, restriction equations).
  Neither can set up the full optimization problem without the other.

SPLIT-F: Sample Space × Counting Principle [Probability, Combinatorics]
  Agent 1 has the event structure (what is being counted, the target condition).
  Agent 2 has the sample space structure and the applicable counting rule/principle.
  Neither can compute the probability or count without combining both.

SPLIT-G: Hypothesis × Key Lemma [Proofs, Number Theory]
  Agent 1 has the starting hypothesis and the initial algebraic/logical transformation.
  Agent 2 has a key intermediate lemma, identity, or theorem to complete the proof.
  Together they chain: hypothesis → lemma → conclusion.

══════════════════════════════════════════════
FEW-SHOT EXAMPLES
══════════════════════════════════════════════

--- EXAMPLE 1 (SPLIT-A, n=2) ---
PROBLEM:
A Norman window has the shape of a rectangle surmounted by a semicircle. The total
perimeter is 10 m (including the base). Find the dimensions that maximize total area.

OUTPUT:
{{
  "pattern": "SPLIT-A",
  "shared_context": "A Norman window consists of a rectangle topped by a semicircle. The total perimeter is 10 m. Two experts — a Rectangle Expert and a Semicircle Expert — must collaborate to find the width and height that maximize the total area. Each expert knows only their own component.",
  "agent_roles": [
    {{
      "agent_id": 1,
      "role_name": "Rectangle Expert",
      "role_description": "You see only the rectangular base of the Norman window. Your partner is the Semicircle Expert who knows the curved top part."
    }},
    {{
      "agent_id": 2,
      "role_name": "Semicircle Expert",
      "role_description": "You see only the semicircular top of the Norman window. Your partner is the Rectangle Expert who knows the rectangular base."
    }}
  ],
  "packets": [
    {{
      "agent_id": 1,
      "information": "You see the rectangular base. Let w = width and h = height. Area of rectangle: A_rect = w·h. Perimeter contribution: two vertical sides (2h) plus the base (w). The top side of the rectangle is NOT added to the perimeter because it is shared with the semicircle — you must establish with your partner how the two parts connect."
    }},
    {{
      "agent_id": 2,
      "information": "You see the semicircular top. Its diameter equals the width of the rectangle (this is the key connection you must establish with your partner). Radius r = w/2. Area of semicircle: A_semi = π·r²/2. Curved perimeter contribution: π·r."
    }}
  ],
  "split_rationale": "Composite figure split: each agent owns one geometric component; the diameter=width connection is the hidden link that forces genuine dialogue before any equation can be written."
}}

--- EXAMPLE 2 (SPLIT-B, n=2) ---
PROBLEM:
One student has the graph of an unknown function f(x); another has the graph of its
derivative f'(x). Neither has the algebraic expression. Collaborate to identify f(x).

OUTPUT:
{{
  "pattern": "SPLIT-B",
  "shared_context": "An unknown polynomial function f(x) exists. A Function Expert and a Derivative Expert must collaborate to identify f(x) as precisely as possible. Neither has the algebraic expression.",
  "agent_roles": [
    {{
      "agent_id": 1,
      "role_name": "Function Graph Expert",
      "role_description": "You see the graph of f(x): its shape, zeros, maxima, minima, and concavity. Your partner is the Derivative Expert who sees f'(x)."
    }},
    {{
      "agent_id": 2,
      "role_name": "Derivative Graph Expert",
      "role_description": "You see the graph of f'(x): where it is positive/negative and where it is zero. Your partner is the Function Graph Expert who sees f(x)."
    }}
  ],
  "packets": [
    {{
      "agent_id": 1,
      "information": "Graph of f(x) observations: local maximum near x=0, local minimum near x=2; f(0) ≈ 4; f(x) → +∞ as x → +∞ and f(x) → -∞ as x → -∞, suggesting odd degree; inflection point near x=1."
    }},
    {{
      "agent_id": 2,
      "information": "Graph of f'(x) observations: upward-opening parabola; f'(x)=0 at x=0 and x=2; f'(x)>0 for x<0 and x>2; f'(x)<0 for 0<x<2. So f'(x) = k·x·(x−2) = k(x²−2x) for some positive constant k."
    }}
  ],
  "split_rationale": "Dual-representation split: f-graph anchors shape and scale, f'-graph gives structural form; integrating both is necessary and sufficient to reconstruct f(x) up to a constant."
}}

══════════════════════════════════════════════
CRITICAL RULES
══════════════════════════════════════════════
1. Choose the pattern that creates the STRONGEST positive interdependence.
2. NEVER drop an operator or relation (if the problem involves subtraction of two
   expressions, the minus sign must appear in shared_context or in a packet explicitly).
3. Shared_context states: (a) the common goal, (b) the number of agents and that each
   has a different named expertise — WITHOUT revealing what the other agents know.
4. Each packet must be self-consistent and internally meaningful but insufficient alone.
5. Balance: packets should be roughly equal in informational weight.
6. For n≥3: assign a genuinely distinct role to each agent; avoid redundant packets.

Return ONLY valid JSON with this exact schema:
{{
  "pattern": "<SPLIT-A|SPLIT-B|SPLIT-C|SPLIT-D|SPLIT-E|SPLIT-F|SPLIT-G>",
  "shared_context": "<goal + group awareness: N agents each with different named expertise>",
  "agent_roles": [
    {{"agent_id": 1, "role_name": "<Expert Name>", "role_description": "<what you know + who your partners are>"}},
    ...
  ],
  "packets": [
    {{"agent_id": 1, "information": "<complete private information for agent 1>"}},
    ...
  ],
  "split_rationale": "<one sentence: which fault line was chosen and why it enforces interdependence>"
}}
""")

_VALIDATE_SYSTEM = textwrap.dedent("""
You are a mathematician attempting to solve a problem. You have access ONLY to the
information below — you may NOT use any knowledge beyond what is explicitly given.

Respond with JSON:
{{
  "can_solve": true | false,
  "confidence": 0.0–1.0,
  "reasoning": "<what you can and cannot determine from the given information alone>",
  "missing_info": "<what specific information is absent that you would need>"
}}
""")

_VALIDATE_COMBINED_SYSTEM = textwrap.dedent("""
You are a mathematician with access to all the information below. Attempt a full solution.

Respond with JSON:
{{
  "can_solve": true | false,
  "confidence": 0.0–1.0,
  "answer": "<final answer or expression>",
  "reasoning": "<brief solution sketch>"
}}
""")


# ── core functions ─────────────────────────────────────────────────────────────

def _call(system: str, user: str, model: str, temperature: float = 0.0,
          max_tokens: int = 2000) -> str:
    return chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model,
        temperature=temperature,
        json_mode=True,
        max_tokens=max_tokens,
    )


def _generate_split(problem: str, n: int) -> dict:
    system = _SPLIT_SYSTEM.format(n=n)
    user   = f"PROBLEM:\n{problem}"
    raw    = _call(system, user, model=CFG.model_splitter, temperature=0.3, max_tokens=3500)
    return json.loads(raw)


def _validate_solo(shared: str, packet: Packet, problem: str) -> dict:
    """Check that a single packet CANNOT solve the problem using only given info."""
    role_note = f"Your role: {packet.role_name}. " if packet.role_name else ""
    user = (
        f"GOAL: {shared}\n\n"
        f"{role_note}"
        f"YOUR PRIVATE INFORMATION (this is ALL you may use — no outside knowledge):\n"
        f"{packet.information}\n\n"
        f"Using ONLY the information above (not general mathematical knowledge), "
        f"can you fully solve the problem?"
    )
    raw = _call(_VALIDATE_SYSTEM, user, model=CFG.model_validator)
    return json.loads(raw)


def _validate_combined(shared: str, packets: List[Packet], problem: str) -> dict:
    """Check that ALL packets together CAN solve the problem."""
    combined = "\n\n".join(
        f"{p.role_name or f'Agent {p.agent_id}'} information:\n{p.information}"
        for p in packets
    )
    user = (
        f"GOAL: {shared}\n\n"
        f"COMBINED INFORMATION FROM ALL AGENTS:\n{combined}\n\n"
        f"Solve the problem completely."
    )
    raw = _call(_VALIDATE_COMBINED_SYSTEM, user, model=CFG.model_validator)
    return json.loads(raw)


def _is_valid_split(validation_log: dict, n: int) -> bool:
    """
    Valid when:
    - Each solo agent cannot solve (confidence < 0.5 or can_solve == False)
    - Combined agents can solve (confidence >= 0.6 and can_solve == True)
    """
    for i in range(1, n + 1):
        solo = validation_log.get(f"solo_{i}", {})
        if solo.get("can_solve") and solo.get("confidence", 0) >= 0.6:
            return False   # necessity violated

    combined = validation_log.get("combined", {})
    if not combined.get("can_solve") or combined.get("confidence", 0) < 0.6:
        return False   # completeness violated

    return True


# ── public API ────────────────────────────────────────────────────────────────

_CPP_TARGETED_SYSTEM = textwrap.dedent("""
Eres un experto en diseño de actividades de aprendizaje colaborativo matemático.
Tu tarea es dividir un problema matemático en {n} paquetes de información para
que {n} agentes LLM puedan resolverlo conjuntamente mediante Collaborative Problem
Solving (CPS) genuino.

## PERFIL CPP OBJETIVO (CPP-DEEP)
El split debe activar las siguientes celdas de la matriz PISA CPS.
"Activa" significa que esa celda requerirá colaboración real — ningún agente puede
completarla sin input activo del otro.

Celdas a activar:
- A1 (Explorar·Conocimiento compartido): Los agentes deben descubrir qué sabe el
  otro y qué pueden hacer. Ninguno puede evaluar su propia información sin conocer
  la del otro.
- A2 (Explorar·Acción): Los agentes deben establecer juntos las normas de
  interacción (¿quién lidera cada paso? ¿cómo verifican acuerdo?).
- A3 (Explorar·Organización): Los roles deben emerger de la exploración conjunta,
  no ser asignados a priori.
- B1 (Formular·Conocimiento compartido): Los agentes deben negociar explícitamente
  la representación del problema — no pueden asumir que comparten la misma
  interpretación.
- B2 (Formular·Acción): Identificar las sub-tareas requiere información de ambos
  agentes — ninguno puede hacer la lista completo solo.
- B3 (Formular·Organización): La distribución del trabajo en la ejecución debe
  negociarse, no derivarse automáticamente del split inicial.
- C1 (Ejecutar·Conocimiento compartido): Antes de cada paso de ejecución, los
  agentes deben comunicar qué van a hacer y por qué — el otro debe confirmar.
- C2 (Ejecutar·Acción): Hay pasos de ejecución que literalmente requieren el
  resultado del otro como input. No es opcional consultar.

## CONDICIONES SZEWKIS (RESTRICCIONES MÍNIMAS)
El split debe garantizar estructuralmente:
1. OBJETIVO COMÚN: La meta debe ser compartida y explícita. Ambos agentes trabajan
   para resolver el mismo problema completo, no "su parte".
2. INTERDEPENDENCIA POSITIVA: Ningún agente puede resolver el problema completo
   incluso si recibe toda la información del otro en el turno 1. La integración
   misma requiere razonamiento nuevo.
3. RESPONSABILIDAD INDIVIDUAL: Cada agente tiene una contribución única y
   necesaria en al menos 3 momentos distintos de la resolución.
4. RECOMPENSA GRUPAL: El éxito se define como "el grupo llegó a la respuesta
   correcta juntos" — no "cada uno resolvió su parte".
5. CONCIENCIA GRUPAL: El split debe hacer necesario que cada agente mantenga un
   modelo mental del otro (qué sabe, qué puede hacer, qué ya hizo).
6. COORDINACIÓN Y COMUNICACIÓN: Los agentes deben coordinar activamente el proceso
   — no pueden trabajar en paralelo y juntar al final.

## TEST DE PROFUNDIDAD
Antes de finalizar el split, verifica internamente:
- Si Agente 1 le dice todo su paquete al Agente 2 en el turno 1, ¿puede Agente 2
  resolver solo? Si SÍ, el split es superficial — rediseña.
- ¿Requieren al menos 4 intercambios de información para llegar a la solución?
- ¿Cada agente hace razonamiento matemático propio en al menos 2 momentos?

## OUTPUT
Responde SOLO con JSON válido:
{{
  "pattern": "SPLIT-X",
  "shared_context": "...",
  "agent_roles": [
    {{"agent_id": 1, "role_name": "...", "role_description": "..."}},
    ...
  ],
  "packets": [
    {{"agent_id": 1, "information": "..."}},
    ...
  ],
  "split_rationale": "...",
  "depth_verification": {{
    "agent2_can_solve_alone_after_turn1": false,
    "minimum_exchanges_needed": 4,
    "mathematical_actions_per_agent": 2,
    "szewkis_satisfied": [true, true, true, true, true, true]
  }}
}}
""")


def split_cpp_targeted(
    problem_id: str,
    problem:    str,
    n:          int = 2,
    validate:   bool = True,
    max_retries: int = 2,
) -> SplitResult:
    """
    Condition C2: generate a split explicitly targeting CPP-DEEP with
    Szewkis conditions as hard constraints in the prompt.
    """
    system = _CPP_TARGETED_SYSTEM.format(n=n)
    raw_split: dict = {}
    packets:   List[Packet] = []
    shared:    str = ""
    pattern:   str = ""
    log:       dict = {}

    for attempt in range(max_retries + 1):
        try:
            raw    = _call(system, f"PROBLEM:\n{problem}", model=CFG.model_splitter,
                           temperature=0.3, max_tokens=4000)
            raw_split = json.loads(raw)
            shared    = raw_split.get("shared_context", "")
            pattern   = raw_split.get("pattern", "SPLIT-C")

            roles = {r["agent_id"]: r for r in raw_split.get("agent_roles", [])}
            packets = [
                Packet(
                    agent_id=p["agent_id"],
                    information=p["information"],
                    role_name=roles.get(p["agent_id"], {}).get("role_name",
                              f"Agent {p['agent_id']}"),
                    role_description=roles.get(p["agent_id"], {}).get("role_description", ""),
                )
                for p in raw_split.get("packets", [])
            ]
            if len(packets) != n:
                continue
        except (json.JSONDecodeError, KeyError):
            continue

        if not validate:
            return SplitResult(
                problem_id=problem_id, problem=problem, n=n,
                pattern=pattern, shared_context=shared, packets=packets,
                valid=True, raw_split=raw_split,
            )

        log = {}
        for pkt in packets:
            log[f"solo_{pkt.agent_id}"] = _validate_solo(shared, pkt, problem)
        log["combined"] = _validate_combined(shared, packets, problem)

        if _is_valid_split(log, n):
            return SplitResult(
                problem_id=problem_id, problem=problem, n=n,
                pattern=pattern, shared_context=shared, packets=packets,
                valid=True, validation_log=log, raw_split=raw_split,
            )

    return SplitResult(
        problem_id=problem_id, problem=problem, n=n,
        pattern=pattern, shared_context=shared, packets=packets,
        valid=False, validation_log=log, raw_split=raw_split,
    )


def split(problem_id: str, problem: str, n: int,
          validate: bool = True, max_retries: int = 2) -> SplitResult:
    """
    Generate and validate an n-way jigsaw split using the 7-pattern taxonomy.
    For n=1, returns a trivial solo result (no split needed).
    """
    if n == 1:
        return SplitResult(
            problem_id=problem_id,
            problem=problem,
            n=1,
            pattern="solo",
            shared_context=problem,
            packets=[Packet(agent_id=1, information=problem,
                            role_name="Solo Solver", role_description="Solve alone.")],
            valid=True,
            validation_log={"note": "solo condition — no split"},
        )

    raw_split: dict = {}
    packets:   List[Packet] = []
    shared:    str = ""
    pattern:   str = ""
    log:       dict = {}

    for attempt in range(max_retries + 1):
        try:
            raw_split = _generate_split(problem, n)
            shared    = raw_split.get("shared_context", "")
            pattern   = raw_split.get("pattern", "SPLIT-C")

            # Merge agent_roles into packets
            roles = {r["agent_id"]: r for r in raw_split.get("agent_roles", [])}
            packets = [
                Packet(
                    agent_id=p["agent_id"],
                    information=p["information"],
                    role_name=roles.get(p["agent_id"], {}).get("role_name",
                              f"Agent {p['agent_id']}"),
                    role_description=roles.get(p["agent_id"], {}).get("role_description", ""),
                )
                for p in raw_split.get("packets", [])
            ]
            if len(packets) != n:
                continue
        except (json.JSONDecodeError, KeyError):
            continue

        if not validate:
            return SplitResult(
                problem_id=problem_id, problem=problem, n=n,
                pattern=pattern, shared_context=shared, packets=packets,
                valid=True, raw_split=raw_split,
            )

        log = {}
        for pkt in packets:
            log[f"solo_{pkt.agent_id}"] = _validate_solo(shared, pkt, problem)
        log["combined"] = _validate_combined(shared, packets, problem)

        if _is_valid_split(log, n):
            return SplitResult(
                problem_id=problem_id, problem=problem, n=n,
                pattern=pattern, shared_context=shared, packets=packets,
                valid=True, validation_log=log, raw_split=raw_split,
            )

    return SplitResult(
        problem_id=problem_id, problem=problem, n=n,
        pattern=pattern, shared_context=shared, packets=packets,
        valid=False, validation_log=log, raw_split=raw_split,
    )
