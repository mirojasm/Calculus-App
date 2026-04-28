"""
Genera outputs/viewer.html — visor estático autocontenido para evaluación humana.

Lee outputs/data/, outputs/splits/, outputs/conversations/, outputs/scores/
y embebe todos los datos como JSON en el HTML.  No requiere servidor.

Uso:
  python -m research.viz.generate_viewer
  # Luego abrir outputs/viewer.html en cualquier navegador
"""
import json, sys
from pathlib import Path

OUTPUTS   = Path("outputs")
DATA_DIR  = OUTPUTS / "data"
SPLITS    = OUTPUTS / "splits"
CONVS     = OUTPUTS / "conversations"
SCORES    = OUTPUTS / "scores"
OUT_HTML  = OUTPUTS / "viewer.html"

CONDITIONS = ["solo", "unrestricted_pair", "jigsaw_2", "jigsaw_3", "jigsaw_4"]
COND_LABEL = {
    "solo":             "Solo",
    "unrestricted_pair":"Par libre",
    "jigsaw_2":         "Jigsaw 2",
    "jigsaw_3":         "Jigsaw 3",
    "jigsaw_4":         "Jigsaw 4",
}

def _load(path: Path):
    with open(path) as f:
        return json.load(f)

def build_data() -> list[dict]:
    problems_path = DATA_DIR / "math_sample.json"
    if not problems_path.exists():
        sys.exit(f"[ERROR] No encontré {problems_path}. Ejecuta stage_load primero.")
    problems = _load(problems_path)
    print(f"[INFO] {len(problems)} problemas encontrados")

    records = []
    for p in problems:
        pid = p["id"]
        rec = {
            "id":       pid,
            "subject":  p.get("subject",""),
            "level":    p.get("level", 0),
            "openness": p.get("openness","closed"),
            "problem":  p.get("problem",""),
            "answer":   p.get("answer",""),
            "splits":   {},
            "convs":    {},
            "scores":   {},
        }

        # splits n=1..4
        for n in [1,2,3,4]:
            sp = SPLITS / f"{pid}_n{n}.json"
            if sp.exists():
                d = _load(sp)
                rec["splits"][str(n)] = {
                    "pattern":         d.get("pattern",""),
                    "split_rationale": d.get("split_rationale",""),
                    "shared_context":  d.get("shared_context",""),
                    "packets":         d.get("packets",[]),
                    "agent_roles":     d.get("agent_roles",[]),
                    "valid":           d.get("valid", False),
                }

        # conversations & scores
        for cond in CONDITIONS:
            cp = CONVS / f"{pid}_{cond}.json"
            if cp.exists():
                d = _load(cp)
                rec["convs"][cond] = {
                    "turns":        d.get("turns",[]),
                    "final_answer": d.get("final_answer",""),
                    "consensus":    d.get("consensus", False),
                    "total_turns":  d.get("total_turns", 0),
                }
            sp = SCORES / f"{pid}_{cond}_scores.json"
            if sp.exists():
                d = _load(sp)
                rec["scores"][cond] = {
                    "correct":     d.get("correct", False),
                    "pisa_global": round(d["pisa"]["global_index"],2),
                    "pisa_proc":   d["pisa"]["process_share"],
                    "pisa_comp":   d["pisa"]["competence_share"],
                    "atc_global":  round(d["atc21s"]["global_index"],2),
                    "atc_dims":    d["atc21s"]["dim_means"],
                }

        records.append(rec)
        if len(records) % 30 == 0:
            print(f"  procesados {len(records)}/{len(problems)} ...")

    return records

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CollabMath — Evaluación Humana</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  .agent-0{background:#dbeafe;border-color:#3b82f6}
  .agent-1{background:#dcfce7;border-color:#22c55e}
  .agent-2{background:#fef9c3;border-color:#eab308}
  .agent-3{background:#fae8ff;border-color:#a855f7}
  .shared{background:#f3f4f6;border-color:#9ca3af}
  .turn-A1{background:#eff6ff} .turn-A2{background:#f0fdf4}
  .turn-A3{background:#fefce8} .turn-A4{background:#fdf4ff}
  .turn-solo{background:#f9fafb}
  .star{cursor:pointer;font-size:1.5rem;color:#d1d5db;transition:color .15s}
  .star.on,.star:hover{color:#f59e0b}
  .pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.7rem;font-weight:600}
  .correct{background:#dcfce7;color:#166534} .wrong{background:#fee2e2;color:#991b1b}
  ::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
</style>
</head>
<body class="bg-gray-100 text-gray-800 h-screen flex flex-col overflow-hidden">

<!-- Header -->
<header class="bg-indigo-700 text-white px-4 py-3 flex items-center justify-between shrink-0 shadow">
  <div>
    <span class="font-bold text-lg">CollabMath</span>
    <span class="ml-2 text-indigo-200 text-sm">Evaluación Humana de Splits Jigsaw</span>
  </div>
  <div class="flex gap-3 items-center text-sm">
    <span id="evalCount" class="bg-indigo-500 px-3 py-1 rounded-full">0 evaluados</span>
    <button onclick="exportRatings()" class="bg-white text-indigo-700 px-3 py-1 rounded font-semibold hover:bg-indigo-50">Exportar JSON</button>
  </div>
</header>

<!-- Body -->
<div class="flex flex-1 overflow-hidden">

  <!-- Sidebar -->
  <aside class="w-72 bg-white border-r flex flex-col shrink-0 overflow-hidden">
    <!-- Filters -->
    <div class="p-3 border-b space-y-2 shrink-0">
      <input id="searchBox" oninput="applyFilters()" placeholder="Buscar problema..."
        class="w-full text-sm border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-indigo-400">
      <div class="flex gap-2">
        <select id="filterSubject" onchange="applyFilters()" class="flex-1 text-xs border rounded px-1 py-1">
          <option value="">Todos los temas</option>
          <option>algebra</option><option>geometry</option><option>precalculus</option>
          <option>prealgebra</option><option>number_theory</option><option>counting_and_probability</option>
        </select>
        <select id="filterLevel" onchange="applyFilters()" class="w-20 text-xs border rounded px-1 py-1">
          <option value="">Nivel</option>
          <option>1</option><option>2</option><option>3</option><option>4</option><option>5</option>
        </select>
      </div>
      <select id="filterPattern" onchange="applyFilters()" class="w-full text-xs border rounded px-1 py-1">
        <option value="">Todos los patrones</option>
        <option>SPLIT-A</option><option>SPLIT-B</option><option>SPLIT-C</option>
        <option>SPLIT-D</option><option>SPLIT-E</option><option>SPLIT-F</option><option>SPLIT-G</option>
      </select>
    </div>
    <!-- Problem list -->
    <div id="problemList" class="flex-1 overflow-y-auto text-sm"></div>
    <div class="p-2 border-t text-xs text-gray-400 text-center" id="listCount"></div>
  </aside>

  <!-- Main -->
  <main class="flex-1 overflow-y-auto p-4 space-y-4" id="main">
    <div class="flex items-center justify-center h-full text-gray-400">
      <div class="text-center">
        <div class="text-5xl mb-3">🧩</div>
        <p class="text-lg font-medium">Selecciona un problema de la lista</p>
        <p class="text-sm mt-1">Navega por los splits jigsaw y evalúa su calidad</p>
      </div>
    </div>
  </main>
</div>

<script>
const DATA = /*DATA_PLACEHOLDER*/null/*END*/;
let currentId = null;
let ratings = JSON.parse(localStorage.getItem('cm_ratings')||'{}');

// ── Problem list ──────────────────────────────────────────────────────────────
function applyFilters(){
  const q   = document.getElementById('searchBox').value.toLowerCase();
  const sub = document.getElementById('filterSubject').value;
  const lvl = document.getElementById('filterLevel').value;
  const pat = document.getElementById('filterPattern').value;
  const filtered = DATA.filter(p=>{
    if(sub && p.subject !== sub) return false;
    if(lvl && String(p.level) !== lvl) return false;
    if(pat){
      const patterns = Object.values(p.splits).map(s=>s.pattern);
      if(!patterns.includes(pat)) return false;
    }
    if(q && !p.problem.toLowerCase().includes(q) && !p.id.includes(q)) return false;
    return true;
  });
  renderList(filtered);
}

function renderList(items){
  const el = document.getElementById('problemList');
  el.innerHTML = items.map(p=>{
    const rated = ratings[p.id] ? '✓ ' : '';
    const pat   = p.splits['2']?.pattern || p.splits['3']?.pattern || '';
    const badge = pat ? `<span class="text-indigo-500">${pat}</span>` : '';
    return `<div onclick="loadProblem('${p.id}')"
      class="px-3 py-2 border-b cursor-pointer hover:bg-indigo-50 ${currentId===p.id?'bg-indigo-100 font-semibold':''}"
      id="li_${p.id}">
      <div class="flex justify-between items-center">
        <span class="text-gray-500 text-xs">${p.subject} · L${p.level}</span>
        ${badge}
      </div>
      <div class="truncate text-xs mt-0.5 text-gray-700">${rated}${p.problem.slice(0,70)}...</div>
    </div>`;
  }).join('');
  document.getElementById('listCount').textContent = `${items.length} problemas`;
  updateEvalCount();
}

// ── Problem detail ────────────────────────────────────────────────────────────
function loadProblem(id){
  currentId = id;
  document.querySelectorAll('[id^=li_]').forEach(el=>el.classList.remove('bg-indigo-100','font-semibold'));
  const li = document.getElementById('li_'+id);
  if(li) li.classList.add('bg-indigo-100','font-semibold');

  const p = DATA.find(x=>x.id===id);
  if(!p) return;

  const nTabs = [2,3,4].filter(n=>p.splits[n]);
  const condOpts = Object.keys(p.convs).map(c=>`<option value="${c}">${{
    solo:'Solo',unrestricted_pair:'Par libre',jigsaw_2:'Jigsaw 2',
    jigsaw_3:'Jigsaw 3',jigsaw_4:'Jigsaw 4'}[c]||c}</option>`).join('');

  document.getElementById('main').innerHTML = `
  <!-- Problem header -->
  <div class="bg-white rounded-xl shadow p-4">
    <div class="flex items-start justify-between gap-3">
      <div>
        <div class="flex gap-2 items-center mb-1 flex-wrap">
          <span class="pill bg-indigo-100 text-indigo-700">${p.subject}</span>
          <span class="pill bg-gray-100 text-gray-600">Nivel ${p.level}</span>
          <span class="pill ${p.openness==='open'?'bg-green-100 text-green-700':'bg-gray-100 text-gray-600'}">${p.openness}</span>
          <span class="text-xs text-gray-400">${p.id}</span>
        </div>
        <p class="text-sm leading-relaxed font-mono bg-gray-50 rounded p-3 mt-2 whitespace-pre-wrap">${esc(p.problem)}</p>
      </div>
    </div>
    <div class="mt-2 text-xs text-gray-500">
      <strong>Respuesta correcta:</strong> <code class="bg-yellow-50 px-1 rounded">${esc(p.answer)}</code>
    </div>
  </div>

  <!-- Splits -->
  <div class="bg-white rounded-xl shadow p-4">
    <h2 class="font-semibold text-gray-700 mb-3 flex items-center gap-2">🧩 Split Jigsaw
      <span class="text-xs font-normal text-gray-400">— ¿cómo se distribuye la información?</span>
    </h2>
    <div class="flex gap-2 mb-3" id="splitTabs">
      ${nTabs.map((n,i)=>`<button onclick="showSplit(${n})" id="stab${n}"
        class="text-sm px-3 py-1 rounded-full border ${i===0?'bg-indigo-600 text-white border-indigo-600':'bg-white text-gray-600'} hover:border-indigo-400">
        n = ${n} agentes</button>`).join('')}
    </div>
    <div id="splitContent"></div>
  </div>

  <!-- Conversation -->
  <div class="bg-white rounded-xl shadow p-4">
    <div class="flex items-center justify-between mb-3">
      <h2 class="font-semibold text-gray-700">💬 Conversación simulada</h2>
      <select onchange="showConv(this.value)" class="text-sm border rounded px-2 py-1 focus:outline-none">${condOpts}</select>
    </div>
    <div id="convContent"></div>
  </div>

  <!-- Scores -->
  <div class="bg-white rounded-xl shadow p-4" id="scoresSection">
    <h2 class="font-semibold text-gray-700 mb-3">📊 Puntajes</h2>
    ${renderScores(p)}
  </div>

  <!-- Evaluation -->
  <div class="bg-white rounded-xl shadow p-4 border-2 border-indigo-200">
    <h2 class="font-semibold text-indigo-700 mb-4">⭐ Tu evaluación</h2>
    ${renderEvalForm(id)}
  </div>
  `;

  if(nTabs.length>0) showSplit(nTabs[0]);
  const firstCond = Object.keys(p.convs)[0];
  if(firstCond) showConv(firstCond);
  loadSavedRating(id);
}

// ── Split view ────────────────────────────────────────────────────────────────
function showSplit(n){
  document.querySelectorAll('[id^=stab]').forEach(b=>{
    b.classList.remove('bg-indigo-600','text-white','border-indigo-600');
    b.classList.add('bg-white','text-gray-600');
  });
  const tab = document.getElementById('stab'+n);
  if(tab){ tab.classList.add('bg-indigo-600','text-white','border-indigo-600'); tab.classList.remove('bg-white','text-gray-600'); }
  const p = DATA.find(x=>x.id===currentId);
  const s = p?.splits[n];
  if(!s){ document.getElementById('splitContent').innerHTML='<p class="text-gray-400 text-sm">Split no disponible</p>'; return; }

  const agentColors = ['agent-0','agent-1','agent-2','agent-3'];
  const packets = s.packets.map((pk,i)=>{
    const role = s.agent_roles?.find(r=>r.agent_id===pk.agent_id)||{};
    return `<div class="border-2 rounded-lg p-3 ${agentColors[i%4]}">
      <div class="font-semibold text-sm mb-1">Agente ${pk.agent_id}
        ${role.role_name?`<span class="font-normal text-gray-500">— ${esc(role.role_name)}</span>`:''}
      </div>
      ${role.role_description?`<p class="text-xs text-gray-500 mb-2 italic">${esc(role.role_description)}</p>`:''}
      <p class="text-sm whitespace-pre-wrap">${esc(pk.information)}</p>
    </div>`;
  }).join('');

  document.getElementById('splitContent').innerHTML = `
    <div class="flex items-center gap-2 mb-3">
      <span class="pill bg-violet-100 text-violet-700 text-sm">${s.pattern||'Sin patrón'}</span>
      ${s.split_rationale?`<span class="text-xs text-gray-500 italic">${esc(s.split_rationale)}</span>`:''}
      ${s.valid?'<span class="pill correct">✓ válido</span>':'<span class="pill wrong">✗ inválido</span>'}
    </div>
    ${s.shared_context?`<div class="border-2 rounded-lg p-3 shared mb-3">
      <div class="font-semibold text-sm mb-1 text-gray-600">Contexto compartido (todos los agentes)</div>
      <p class="text-sm whitespace-pre-wrap">${esc(s.shared_context)}</p>
    </div>`:''}
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">${packets}</div>
  `;
}

// ── Conversation view ─────────────────────────────────────────────────────────
const turnColors = {A1:'turn-A1',A2:'turn-A2',A3:'turn-A3',A4:'turn-A4',solo:'turn-solo'};
function showConv(cond){
  const p = DATA.find(x=>x.id===currentId);
  const c = p?.convs[cond];
  if(!c){ document.getElementById('convContent').innerHTML='<p class="text-gray-400 text-sm">Conversación no disponible</p>'; return; }

  const turns = c.turns.map((t,i)=>{
    const agent = t.agent_id||'?';
    const cls   = turnColors[agent]||'bg-gray-50';
    return `<div class="${cls} rounded p-2 mb-2 text-sm">
      <span class="font-semibold text-xs text-gray-500">${agent}</span>
      <p class="mt-0.5 whitespace-pre-wrap text-gray-800">${esc(t.content)}</p>
    </div>`;
  }).join('');

  const score = p.scores[cond];
  const correctBadge = score ? `<span class="pill ${score.correct?'correct':'wrong'}">${score.correct?'✓ Correcto':'✗ Incorrecto'}</span>` : '';

  document.getElementById('convContent').innerHTML = `
    <div class="flex gap-2 items-center mb-2 text-xs text-gray-500">
      <span>${c.total_turns} turnos</span>
      <span>·</span>
      <span>Respuesta: <code class="bg-yellow-50 px-1">${esc(c.final_answer||'—')}</code></span>
      ${correctBadge}
    </div>
    <div class="max-h-72 overflow-y-auto pr-1">${turns||'<p class="text-gray-400">Sin turnos</p>'}</div>
  `;
}

// ── Scores ────────────────────────────────────────────────────────────────────
function bar(val, max, color){
  const pct = Math.min(100, (val/max)*100).toFixed(0);
  return `<div class="flex items-center gap-2 text-xs">
    <div class="w-32 bg-gray-100 rounded-full h-2">
      <div class="h-2 rounded-full ${color}" style="width:${pct}%"></div>
    </div>
    <span class="text-gray-600 w-8">${typeof val==='number'?val.toFixed(1):val}</span>
  </div>`;
}
function renderScores(p){
  const rows = Object.entries(p.scores).map(([cond,s])=>`
    <tr class="border-b text-sm">
      <td class="py-1 pr-3 font-medium text-gray-600 whitespace-nowrap">${{solo:'Solo',unrestricted_pair:'Par libre',jigsaw_2:'Jigsaw 2',jigsaw_3:'Jigsaw 3',jigsaw_4:'Jigsaw 4'}[cond]||cond}</td>
      <td class="pr-3"><span class="pill ${s.correct?'correct':'wrong'}">${s.correct?'✓':'✗'}</span></td>
      <td class="pr-3">${bar(s.pisa_global,15,'bg-indigo-400')}</td>
      <td>${bar(s.atc_global,100,'bg-emerald-400')}</td>
    </tr>`).join('');
  return `<table class="w-full">
    <thead><tr class="text-xs text-gray-400 border-b"><th class="text-left pb-1">Condición</th><th>Correcto</th><th>PISA global</th><th>ATC global</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
}

// ── Evaluation form ────────────────────────────────────────────────────────────
function renderEvalForm(id){
  return `
  <div class="grid md:grid-cols-2 gap-4">
    <div>
      <label class="text-sm font-medium text-gray-700 block mb-1">Calidad del split jigsaw</label>
      <p class="text-xs text-gray-400 mb-1">¿Qué tan bien está dividido el problema?</p>
      <div class="flex gap-1" id="stars_split">
        ${[1,2,3,4,5].map(i=>`<span class="star" data-v="${i}" onclick="setStar('split',${i})">★</span>`).join('')}
      </div>
    </div>
    <div>
      <label class="text-sm font-medium text-gray-700 block mb-1">Calidad de la conversación</label>
      <p class="text-xs text-gray-400 mb-1">¿Qué tan realista y útil es la simulación?</p>
      <div class="flex gap-1" id="stars_conv">
        ${[1,2,3,4,5].map(i=>`<span class="star" data-v="${i}" onclick="setStar('conv',${i})">★</span>`).join('')}
      </div>
    </div>
    <div>
      <label class="text-sm font-medium text-gray-700 block mb-1">¿La colaboración es realmente necesaria?</label>
      <div class="flex gap-2 mt-1" id="radios_interdep">
        ${['Sí','Parcial','No'].map(v=>`<label class="flex items-center gap-1 text-sm cursor-pointer">
          <input type="radio" name="interdep_${id}" value="${v}" onchange="saveRating()"> ${v}</label>`).join('')}
      </div>
    </div>
    <div>
      <label class="text-sm font-medium text-gray-700 block mb-1">¿El split parece pedagógicamente natural?</label>
      <div class="flex gap-2 mt-1">
        ${['Sí','Parcial','No'].map(v=>`<label class="flex items-center gap-1 text-sm cursor-pointer">
          <input type="radio" name="natural_${id}" value="${v}" onchange="saveRating()"> ${v}</label>`).join('')}
      </div>
    </div>
  </div>
  <div class="mt-3">
    <label class="text-sm font-medium text-gray-700 block mb-1">Observaciones</label>
    <textarea id="comments_field" oninput="saveRating()" rows="2" placeholder="Comentario opcional..."
      class="w-full text-sm border rounded p-2 focus:outline-none focus:ring-1 focus:ring-indigo-300"></textarea>
  </div>
  <div class="mt-2 flex items-center gap-3">
    <button onclick="saveRating(true)" class="bg-indigo-600 text-white text-sm px-4 py-1.5 rounded hover:bg-indigo-700">Guardar</button>
    <span id="saveMsg" class="text-xs text-green-600 hidden">✓ Guardado</span>
  </div>`;
}

function setStar(type, val){
  document.querySelectorAll(`#stars_${type} .star`).forEach(s=>{
    s.classList.toggle('on', parseInt(s.dataset.v)<=val);
  });
  saveRating();
}

function saveRating(showMsg=false){
  if(!currentId) return;
  const starVal = type => {
    let v=0;
    document.querySelectorAll(`#stars_${type} .star.on`).forEach(s=>v=Math.max(v,parseInt(s.dataset.v)));
    return v;
  };
  const radio = name => {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? el.value : '';
  };
  ratings[currentId] = {
    split_quality:    starVal('split'),
    conv_quality:     starVal('conv'),
    interdependencia: radio(`interdep_${currentId}`),
    naturalidad:      radio(`natural_${currentId}`),
    comments:         document.getElementById('comments_field')?.value||'',
    ts: new Date().toISOString(),
  };
  localStorage.setItem('cm_ratings', JSON.stringify(ratings));
  updateEvalCount();
  if(showMsg){
    const msg = document.getElementById('saveMsg');
    msg.classList.remove('hidden');
    setTimeout(()=>msg.classList.add('hidden'), 2000);
  }
  // update sidebar indicator
  const li = document.getElementById('li_'+currentId);
  if(li && ratings[currentId].split_quality > 0){
    const txt = li.querySelector('.text-xs.mt-0\\.5');
    if(txt && !txt.textContent.startsWith('✓ ')){
      txt.textContent = '✓ ' + txt.textContent;
    }
  }
}

function loadSavedRating(id){
  const r = ratings[id];
  if(!r) return;
  if(r.split_quality) setStar('split', r.split_quality);
  if(r.conv_quality)  setStar('conv', r.conv_quality);
  if(r.interdependencia){
    const el = document.querySelector(`input[name="interdep_${id}"][value="${r.interdependencia}"]`);
    if(el) el.checked=true;
  }
  if(r.naturalidad){
    const el = document.querySelector(`input[name="natural_${id}"][value="${r.naturalidad}"]`);
    if(el) el.checked=true;
  }
  const cf = document.getElementById('comments_field');
  if(cf) cf.value = r.comments||'';
}

function updateEvalCount(){
  const n = Object.keys(ratings).filter(k=>ratings[k].split_quality>0).length;
  document.getElementById('evalCount').textContent = `${n} evaluados`;
}

function exportRatings(){
  const out = {
    exported: new Date().toISOString(),
    total_problems: DATA.length,
    evaluated: Object.keys(ratings).length,
    ratings,
  };
  const blob = new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `collabmath_ratings_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
}

function esc(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ── Init ──────────────────────────────────────────────────────────────────────
applyFilters();
</script>
</body>
</html>
"""

def generate():
    print("[INFO] Leyendo datos de outputs/ ...")
    data = build_data()
    print(f"[INFO] {len(data)} problemas procesados. Generando HTML ...")

    data_json = json.dumps(data, ensure_ascii=False)
    html = HTML.replace("/*DATA_PLACEHOLDER*/null/*END*/", data_json)

    OUTPUTS.mkdir(exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUT_HTML.stat().st_size / 1e6
    print(f"[OK] Visor generado: {OUT_HTML}  ({size_mb:.1f} MB)")
    print(f"     Abre en tu navegador: file://{OUT_HTML.resolve()}")

if __name__ == "__main__":
    generate()
