#!/bin/bash
# =============================================================
# RetrainEventParser - Build & Install Script
# =============================================================
# Run this from the retrain-event-parser/ directory.
#
# Prerequisites:
#   - Maven installed (brew install maven)
#   - Striim installed at /opt/Striim (or set STRIIM_HOME)
#   - Java 11 (javac) available
# =============================================================

set -e

STRIIM_HOME="${STRIIM_HOME:-/opt/Striim}"
SDK_JAR="$STRIIM_HOME/StriimSDK/StriimOpenProcessor-SDK.jar"
COMMON_JAR="$STRIIM_HOME/lib/Common-5.2.0.4.jar"

echo "=== Step 1: Install Striim SDK into local Maven repo ==="
echo ""

if [ ! -f "$SDK_JAR" ]; then
    echo "ERROR: SDK jar not found at $SDK_JAR"
    echo "Set STRIIM_HOME or update SDK_JAR in this script."
    exit 1
fi

if [ ! -f "$COMMON_JAR" ]; then
    echo "ERROR: Common jar not found at $COMMON_JAR"
    echo "Look for it: find $STRIIM_HOME/lib -name 'Common-*.jar'"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim \
    -DartifactId=OpenProcessorSDK \
    -Dversion=1.0.0-SNAPSHOT \
    -Dpackaging=jar \
    -Dfile="$SDK_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 2: Build Striim-compatible type classes ==="
echo ""

TYPES_DIR="$(pwd)/types-src"
mkdir -p "$TYPES_DIR/wa/fcvae"

if [ ! -f "$TYPES_DIR/wa/fcvae/RetrainEvent_1_0.java" ]; then
    echo "ERROR: RetrainEvent_1_0.java not found in $TYPES_DIR/wa/fcvae/"
    exit 1
fi

if [ ! -f "$TYPES_DIR/wa/fcvae/TypedRetrainInput_Type_1_0.java" ]; then
    echo "ERROR: TypedRetrainInput_Type_1_0.java not found in $TYPES_DIR/wa/fcvae/"
    exit 1
fi

# Compile type classes against SDK + Common jars
javac -source 11 -target 11 \
    -cp "$SDK_JAR:$COMMON_JAR" \
    "$TYPES_DIR/wa/fcvae/RetrainEvent_1_0.java" \
    "$TYPES_DIR/wa/fcvae/TypedRetrainInput_Type_1_0.java"

jar cf "$TYPES_DIR/retrain_types.jar" \
    -C "$TYPES_DIR" wa/fcvae/RetrainEvent_1_0.class \
    -C "$TYPES_DIR" wa/fcvae/TypedRetrainInput_Type_1_0.class

echo "Built retrain_types.jar with:"
jar tf "$TYPES_DIR/retrain_types.jar"
echo ""

mvn install:install-file \
    -DgroupId=com.striim.retrain \
    -DartifactId=retrain-types \
    -Dversion=1.0.0-SNAPSHOT \
    -Dpackaging=jar \
    -Dfile="$TYPES_DIR/retrain_types.jar" \
    -DgeneratePom=true

echo "Installed retrain_types.jar into Maven repo"

#cp "$TYPES_DIR/retrain_types.jar" "$STRIIM_HOME/lib/retrain_types.jar"
echo "Copied retrain_types.jar to $STRIIM_HOME/lib/"

echo ""
echo "=== Step 3: Build the Open Processor ==="
echo ""

mvn clean package

echo ""
echo "=== Step 4: Copy .scm to Striim modules directory ==="
echo ""

cp target/RetrainEventParser.jar "$STRIIM_HOME/modules/RetrainEventParser.scm"

echo "Copied to $STRIIM_HOME/modules/RetrainEventParser.scm"
echo ""
echo "=== Build complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Restart Striim:"
echo "     $STRIIM_HOME/bin/server.sh stop"
echo "     $STRIIM_HOME/bin/server.sh start"
echo ""
echo "  2. Load the OP in Striim Console:"
echo '     UNLOAD OPEN PROCESSOR "'"$STRIIM_HOME"'/modules/RetrainEventParser.scm";'
echo '     LOAD OPEN PROCESSOR "'"$STRIIM_HOME"'/modules/RetrainEventParser.scm";'
echo "     LIST OPENPROCESSORS;"
echo ""
echo "  3. Wire up the pipeline (see TQL in README)."
echo ""
echo "=== Done ==="