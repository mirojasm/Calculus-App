#!/bin/bash
# =============================================================================
# CollabMath — Backup script for the experimental corpus
#
# Backs up outputs/ (splits, conversations, scores, results) to:
#   - A timestamped tar.gz locally
#   - Sapelo2 /scratch (if SAPELO_USER is set)
#
# Usage:
#   bash scripts/backup_corpus.sh                    # local backup only
#   SAPELO_USER=mir85108 bash scripts/backup_corpus.sh  # + Sapelo rsync
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP_DIR="${REPO_DIR}/backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${BACKUP_DIR}/corpus_${TIMESTAMP}.tar.gz"

mkdir -p "${BACKUP_DIR}"

echo "============================================================"
echo "  CollabMath Corpus Backup"
echo "  Timestamp : ${TIMESTAMP}"
echo "  Archive   : ${ARCHIVE}"
echo "============================================================"

# ── Count files ──────────────────────────────────────────────────────────────
N_SPLITS=$(ls "${REPO_DIR}/outputs/splits/"*.json 2>/dev/null | wc -l || echo 0)
N_CONVS=$(ls "${REPO_DIR}/outputs/conversations/"*.json 2>/dev/null | wc -l || echo 0)
N_SCORES=$(ls "${REPO_DIR}/outputs/scores/"*.json 2>/dev/null | wc -l || echo 0)
echo "[INFO] Splits: ${N_SPLITS}, Conversations: ${N_CONVS}, Scores: ${N_SCORES}"

# ── Create local archive ──────────────────────────────────────────────────────
echo "[INFO] Creating archive..."
tar -czf "${ARCHIVE}" \
  -C "${REPO_DIR}" \
  outputs/splits \
  outputs/conversations \
  outputs/scores \
  outputs/results \
  $([ -d "${REPO_DIR}/outputs/pilot" ] && echo "outputs/pilot" || echo "") \
  2>/dev/null || true

ARCHIVE_SIZE=$(du -sh "${ARCHIVE}" | cut -f1)
echo "[OK]  Archive created: ${ARCHIVE} (${ARCHIVE_SIZE})"

# ── Optionally sync to Sapelo ─────────────────────────────────────────────────
if [ -n "${SAPELO_USER:-}" ]; then
  SAPELO_BACKUP="/scratch/${SAPELO_USER}/collabmath/corpus_backups/"
  echo "[INFO] Syncing to Sapelo2 ${SAPELO_BACKUP}..."
  ssh "${SAPELO_USER}@sapelo2.gacrc.uga.edu" "mkdir -p ${SAPELO_BACKUP}"
  scp "${ARCHIVE}" "${SAPELO_USER}@sapelo2.gacrc.uga.edu:${SAPELO_BACKUP}"
  echo "[OK]  Synced to Sapelo2."
fi

# ── Keep only last 5 local archives ──────────────────────────────────────────
echo "[INFO] Keeping last 5 archives in ${BACKUP_DIR}..."
ls -t "${BACKUP_DIR}"/corpus_*.tar.gz 2>/dev/null | tail -n +6 | xargs -r rm --
echo "[OK]  Cleanup done."

echo ""
echo "============================================================"
echo "  Backup complete."
echo "  Archive: ${ARCHIVE} (${ARCHIVE_SIZE})"
echo "============================================================"
