#!/usr/bin/env bash
# Build the FCVAEOnnxScorer Open Processor and install it into Striim's UploadedFiles.
#
# All Striim dependencies are system-scoped; ONNX Runtime + Gson are bundled by
# shade. Maven shade produces target/FCVAEOnnxScorer.jar; we copy it to
# $STRIIM_HOME/UploadedFiles/FCVAEOnnxScorer.scm, then load it in the console:
#
#   LOAD OPEN PROCESSOR 'UploadedFiles/FCVAEOnnxScorer.scm';
#
# This is a WAEvent pass-through OP: no CREATE TYPE, no types JAR, no
# JAR-removal trick. Wire it directly in TQL (see striim/FCVAE_ONNX.tql).
set -euo pipefail
cd "$(dirname "$0")"

STRIIM_HOME="${STRIIM_HOME:-/opt/Striim}"
MODULE="FCVAEOnnxScorer"

mvn clean package

# Drop any stale copy from Striim's OpenProcessor cache (see CLAUDE.md).
rm -f "${STRIIM_HOME}/.striim/OpenProcessor/${MODULE}.scm"

cp "target/${MODULE}.jar" "${STRIIM_HOME}/UploadedFiles/${MODULE}.scm"
# Also refresh the committed copy alongside the source so the repo always
# carries a deployable artifact for this OP.
cp "target/${MODULE}.jar" "${MODULE}.scm"
echo ""
echo "Installed ${STRIIM_HOME}/UploadedFiles/${MODULE}.scm"
echo "Repo copy:  $(pwd)/${MODULE}.scm"
echo "Load it with:  LOAD OPEN PROCESSOR 'UploadedFiles/${MODULE}.scm';"
echo "Reminder: do a full Striim restart before re-loading an updated .scm (ZLIB cache)."
