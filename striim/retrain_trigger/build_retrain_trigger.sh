#!/usr/bin/env bash
# ==============================================================================
# build_retrain_trigger.sh
#
# Compiles and packages the RetrainTriggerCaller Open Processor and its
# RetrainTrigger_Type_1_0 user type for deployment to Striim 5.2.0.4.
#
# Prerequisites:
#   - Java 8+ JDK (javac, jar)
#   - Striim installed at /opt/Striim (or set STRIIM_HOME)
#
# Usage:
#   cd <repo>/striim
#   ./retrain_trigger/build_retrain_trigger.sh
# ==============================================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────

STRIIM_HOME="${STRIIM_HOME:-/opt/Striim}"
STRIIM_LIB="${STRIIM_HOME}/lib"
STRIIM_MODULES="${STRIIM_HOME}/modules"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"

TYPE_JAR="retrain_trigger_types.jar"
OP_SCM="RetrainTriggerCaller.scm"

# ──────────────────────────────────────────────────────────────────
# Preflight checks
# ──────────────────────────────────────────────────────────────────

echo "=== Preflight checks ==="

if ! command -v javac &>/dev/null; then
    echo "ERROR: javac not found. Install Java 8+ JDK."
    exit 1
fi

JAVA_VERSION=$(javac -version 2>&1 | awk '{print $2}')
echo "  javac version: ${JAVA_VERSION}"

if [ ! -d "${STRIIM_LIB}" ]; then
    echo "ERROR: Striim lib directory not found at ${STRIIM_LIB}"
    exit 1
fi
echo "  Striim lib:    ${STRIIM_LIB}"

if [ ! -f "${SRC_DIR}/wa/fcvae/RetrainTrigger_Type_1_0.java" ]; then
    echo "ERROR: Source file not found: ${SRC_DIR}/wa/fcvae/RetrainTrigger_Type_1_0.java"
    exit 1
fi

if [ ! -f "${SRC_DIR}/com/striim/fcvae/RetrainTriggerCaller.java" ]; then
    echo "ERROR: Source file not found: ${SRC_DIR}/com/striim/fcvae/RetrainTriggerCaller.java"
    exit 1
fi

echo "  Sources found. Proceeding."
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 1: Clean
# ──────────────────────────────────────────────────────────────────

echo "=== Step 1: Clean build directory ==="
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"
echo "  Done."
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 2: Build classpath
# ──────────────────────────────────────────────────────────────────

echo "=== Step 2: Build classpath ==="
CP=""
for jar in "${STRIIM_LIB}"/*.jar; do
    if [ -n "${CP}" ]; then
        CP="${CP}:${jar}"
    else
        CP="${jar}"
    fi
done
echo "  Classpath includes $(echo "${CP}" | tr ':' '\n' | wc -l | tr -d ' ') JARs from ${STRIIM_LIB}"
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 3: Compile the user type
# ──────────────────────────────────────────────────────────────────

echo "=== Step 3: Compile RetrainTrigger_Type_1_0 ==="
javac \
    -cp "${CP}" \
    -d "${BUILD_DIR}" \
    -source 1.8 -target 1.8 \
    "${SRC_DIR}/wa/fcvae/RetrainTrigger_Type_1_0.java"
echo "  Compiled: ${BUILD_DIR}/wa/fcvae/RetrainTrigger_Type_1_0.class"
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 4: Package the type JAR
# ──────────────────────────────────────────────────────────────────

echo "=== Step 4: Package ${TYPE_JAR} ==="
(cd "${BUILD_DIR}" && jar cf "${DIST_DIR}/${TYPE_JAR}" \
    wa/fcvae/RetrainTrigger_Type_1_0.class)
echo "  Created: ${DIST_DIR}/${TYPE_JAR}"
jar tf "${DIST_DIR}/${TYPE_JAR}"
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 5: Compile the Open Processor
# ──────────────────────────────────────────────────────────────────

echo "=== Step 5: Compile RetrainTriggerCaller ==="
javac \
    -cp "${CP}:${DIST_DIR}/${TYPE_JAR}" \
    -d "${BUILD_DIR}" \
    -source 1.8 -target 1.8 \
    "${SRC_DIR}/com/striim/fcvae/RetrainTriggerCaller.java"
echo "  Compiled: ${BUILD_DIR}/com/striim/fcvae/RetrainTriggerCaller.class"
echo ""

# ──────────────────────────────────────────────────────────────────
# Step 6: Package the OP as an SCM
# ──────────────────────────────────────────────────────────────────

echo "=== Step 6: Package ${OP_SCM} ==="

# Create MANIFEST.MF with Striim's required entries.
# Format matches the working FCVAEScoreCaller.scm manifest:
#   Striim-Module-Name: <module name used in LOAD OPEN PROCESSOR>
#   Striim-Service-Implementation: <fully qualified OP class>
#   Striim-Service-Interface: <base class>
mkdir -p "${BUILD_DIR}/META-INF"
cat > "${BUILD_DIR}/META-INF/MANIFEST.MF" << EOF
Manifest-Version: 1.0
Striim-Module-Name: RetrainTriggerCaller
Striim-Service-Implementation: com.striim.fcvae.RetrainTriggerCaller
Striim-Service-Interface: com.webaction.runtime.components.openprocessor.StriimOpenProcessor
EOF

echo "  Created META-INF/MANIFEST.MF with Striim module entries"

# Build the SCM JAR with the custom manifest (jar cfm = create + manifest)
(cd "${BUILD_DIR}" && jar cfm "${DIST_DIR}/${OP_SCM}" \
    META-INF/MANIFEST.MF \
    com/striim/fcvae/RetrainTriggerCaller.class)

echo "  Created: ${DIST_DIR}/${OP_SCM}"
jar tf "${DIST_DIR}/${OP_SCM}"

# Verify the manifest was written correctly
echo ""
echo "  Verifying manifest:"
unzip -p "${DIST_DIR}/${OP_SCM}" META-INF/MANIFEST.MF
echo ""

# ──────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────

echo "=========================================="
echo "BUILD COMPLETE"
echo "=========================================="
echo ""
echo "Artifacts:"
echo "  ${DIST_DIR}/${TYPE_JAR}"
echo "  ${DIST_DIR}/${OP_SCM}"
echo ""
echo "=========================================="
echo "DEPLOY STEPS"
echo "=========================================="
echo ""
echo "1. Stop Striim:"
echo "   ${STRIIM_HOME}/bin/server.sh stop"
echo ""
echo "2. Copy the type JAR to Striim's lib/:"
echo "   cp ${DIST_DIR}/${TYPE_JAR} ${STRIIM_LIB}/"
echo ""
echo "3. Copy the OP SCM to Striim's modules/:"
echo "   cp ${DIST_DIR}/${OP_SCM} ${STRIIM_MODULES}/"
echo ""
echo "4. Start Striim:"
echo "   ${STRIIM_HOME}/bin/server.sh start"
echo ""
echo "5. In Striim console:"
echo "   USE fcvae;"
echo "   LOAD OPEN PROCESSOR \"${STRIIM_MODULES}/${OP_SCM}\";"
echo ""
echo "6. Stop and undeploy the FCVAE app:"
echo "   STOP APPLICATION fcvae.FCVAE;"
echo "   UNDEPLOY APPLICATION fcvae.FCVAE;"
echo ""
echo "7. Add the TQL from retrain_trigger_additions.tql"
echo ""
echo "8. Wire the OP in Flow Designer:"
echo "   - Open fcvae.FCVAE in Flow Designer"
echo "   - Drag 'Open Processor' from Base Components"
echo "   - Name: RetrainOP"
echo "   - Module: RetrainTriggerCaller"
echo "   - Input Stream: RetrainTriggerStream"
echo "   - Output Stream: RetrainTriggerAckStream"
echo "   - Save"
echo ""
echo "9. Deploy and start:"
echo "   DEPLOY APPLICATION fcvae.FCVAE;"
echo "   START APPLICATION fcvae.FCVAE;"
echo ""
