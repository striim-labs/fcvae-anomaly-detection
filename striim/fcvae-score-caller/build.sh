#!/bin/bash
# =============================================================
# FCVAE Score Caller - Build & Install Script
# =============================================================
# Run this from the fcvae-score-caller/ directory.
#
# Prerequisites:
#   - Maven installed (brew install maven)
#   - Striim installed at /opt/Striim
# =============================================================

set -e

echo "=== Step 1: Install Striim SDK into local Maven repo ==="
echo ""
echo "The Open Processor SDK jar ships with your Striim install."
echo "We need to register it with Maven so the compiler can find it."
echo ""

SDK_JAR="/opt/Striim/StriimSDK/StriimOpenProcessor-SDK.jar"

if [ ! -f "$SDK_JAR" ]; then
    echo "ERROR: SDK jar not found at $SDK_JAR"
    echo ""
    echo "Look for it with:  find /opt/Striim -name '*OpenProcessor*' -o -name '*SDK*' 2>/dev/null"
    echo "Then update the SDK_JAR variable in this script."
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
echo "=== Step 1b: Install FCVAE types JAR into local Maven repo ==="
echo ""

TYPES_JAR="$(cd "$(dirname "$0")" && pwd)/../lib/fcvae_types.jar"

if [ ! -f "$TYPES_JAR" ]; then
    echo "ERROR: Types jar not found at $TYPES_JAR"
    exit 1
fi

mvn install:install-file \
    -DgroupId=com.striim.fcvae \
    -DartifactId=fcvae-types \
    -Dversion=1.0.0-SNAPSHOT \
    -Dpackaging=jar \
    -Dfile="$TYPES_JAR" \
    -DgeneratePom=true

mvn install:install-file \
    -DgroupId=com.striim.fcvae \
    -DartifactId=FCVAETypes \
    -Dversion=1.0.0-SNAPSHOT \
    -Dpackaging=jar \
    -Dfile="$TYPES_JAR" \
    -DgeneratePom=true

echo ""
echo "=== Step 2: Build the Open Processor ==="
echo ""

mvn clean package

echo ""
echo "=== Step 3: Copy .scm to Striim modules directory ==="
echo ""

# The shade plugin produces target/FCVAEScoreCaller.jar
# Striim expects .scm extension
cp target/FCVAEScoreCaller.jar /opt/Striim/modules/FCVAEScoreCaller.scm

echo "Copied to /opt/Striim/modules/FCVAEScoreCaller.scm"
echo ""
echo "=== Step 4: Load into Striim ==="
echo ""
echo "Go to the Striim Console and run:"
echo ""
echo '  LOAD OPEN PROCESSOR "/opt/Striim/modules/FCVAEScoreCaller.scm";'
echo ""
echo "Then verify with:"
echo ""
echo "  LIST OPENPROCESSORS;"
echo ""
echo "You should see FCVAEScoreCaller in the list."
echo ""
echo "=== Done ==="
