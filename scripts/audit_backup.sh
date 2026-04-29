#!/usr/bin/env bash
# audit_backup.sh — Backup completo con manifest de auditoría para el paper.
#
# Genera:
#   backups/audit_YYYYMMDD_HHMMSS/
#     manifest.json         — qué se corrió, cuándo, con qué versión del código
#     pilot_canonical.json  — resultados definitivos (merge de runs parciales)
#     scripts/              — copia exacta de todo el código fuente
#     outputs/              — todos los resultados (splits, convs, scores, pilot)
#     git_log.txt           — log de commits hasta este punto
#     requirements.txt      — versiones de dependencias
#
# Uso: bash scripts/audit_backup.sh

set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="backups/audit_${TIMESTAMP}"
mkdir -p "${BACKUP_ROOT}"

echo "[1/6] Creando manifest..."
GIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "no-git")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
PYTHON_VER=$(python3 --version 2>&1)

cat > "${BACKUP_ROOT}/manifest.json" <<EOF
{
  "backup_timestamp":  "${TIMESTAMP}",
  "git_commit":        "${GIT_HASH}",
  "git_branch":        "${GIT_BRANCH}",
  "python_version":    "${PYTHON_VER}",
  "working_directory": "$(pwd)",
  "hostname":          "$(hostname)",
  "contents": {
    "pilot_canonical.json": "Resultados definitivos del piloto (4 problemas × 5 condiciones)",
    "scripts/":             "Código fuente completo (research/, scripts/)",
    "outputs/":             "Splits, conversaciones, scores, resultados piloto",
    "git_log.txt":          "Historial de commits",
    "requirements.txt":     "Versiones de dependencias Python"
  }
}
EOF

echo "[2/6] Guardando git log..."
git log --oneline --no-walk=unsorted HEAD~20..HEAD 2>/dev/null \
    > "${BACKUP_ROOT}/git_log.txt" || git log --oneline -20 \
    > "${BACKUP_ROOT}/git_log.txt"
git log --oneline > "${BACKUP_ROOT}/git_log_full.txt" 2>/dev/null || true

echo "[3/6] Generando pilot_canonical.json (merge de runs)..."
python3 - <<'PYEOF'
import json, sys
from pathlib import Path

pilot_dir = Path("outputs/pilot")
# Collect all individual result files, newest last
result_files = sorted(pilot_dir.glob("*_C*_*.json"))

canonical = {}   # key: (problem_id, condition) → best result
for f in result_files:
    try:
        data = json.loads(f.read_text())
        pid   = data.get("problem_id")
        cond  = data.get("condition")
        if not pid or not cond:
            continue
        key = (pid, cond)
        # Skip error records when a good one exists
        if key in canonical and canonical[key].get("error"):
            canonical[key] = data
        elif key not in canonical:
            canonical[key] = data
        else:
            # Prefer newer non-error result
            if not data.get("error"):
                canonical[key] = data
    except Exception as e:
        print(f"  [WARN] {f.name}: {e}", file=sys.stderr)

results = sorted(canonical.values(), key=lambda r: (r.get("problem_id",""), r.get("condition","")))
out = Path("outputs/pilot/pilot_canonical.json")
out.write_text(json.dumps({"results": results, "n_total": len(results)}, indent=2, ensure_ascii=False))
print(f"  Canonical: {len(results)} resultados → {out}")
PYEOF

echo "[4/6] Copiando código fuente..."
mkdir -p "${BACKUP_ROOT}/scripts"
cp -r research/  "${BACKUP_ROOT}/scripts/research/"
cp -r scripts/   "${BACKUP_ROOT}/scripts/scripts/"
cp -r docs/      "${BACKUP_ROOT}/scripts/docs/" 2>/dev/null || true

echo "[5/6] Copiando outputs..."
mkdir -p "${BACKUP_ROOT}/outputs"
cp -r outputs/splits/        "${BACKUP_ROOT}/outputs/splits/"
cp -r outputs/conversations/ "${BACKUP_ROOT}/outputs/conversations/"
cp -r outputs/scores/        "${BACKUP_ROOT}/outputs/scores/"
cp -r outputs/pilot/         "${BACKUP_ROOT}/outputs/pilot/"
cp -r outputs/models/        "${BACKUP_ROOT}/outputs/models/" 2>/dev/null || true

echo "[6/6] Guardando dependencias..."
pip freeze > "${BACKUP_ROOT}/requirements.txt" 2>/dev/null || true

# Comprimir
echo ""
echo "Comprimiendo → backups/audit_${TIMESTAMP}.tar.gz ..."
tar -czf "backups/audit_${TIMESTAMP}.tar.gz" -C backups "audit_${TIMESTAMP}/"
rm -rf "${BACKUP_ROOT}"

SIZE=$(du -sh "backups/audit_${TIMESTAMP}.tar.gz" | cut -f1)
echo ""
echo "✓ Backup completo: backups/audit_${TIMESTAMP}.tar.gz (${SIZE})"
echo "  Incluye: código fuente, outputs, pilot_canonical.json, manifest, git log"
