#!/bin/sh
# Compile report/main.tex inside the TeX Live container.
#
# The image is run with --rm, so IEEEtran and pgfplots have to be installed by
# tlmgr in the *same* invocation as the compile; installs are therefore
# ephemeral by design (they take ~5s and keep this script self-contained).
set -e
cd "$(dirname "$0")"
docker run --rm \
  -v "$(cd .. && pwd)":/repo -w /repo/report \
  -v genai-texmf:/usr/local/texlive/texmf-local \
  texlive/texlive:latest-medium \
  sh -c '
    set -e
    tl_mgr() { tlmgr install "$@"; }
    # ieeetran pulls in the IEEEtran package; pgfplots pulls in pgf.
    # Best-effort: CTAN mirrors are occasionally stale, and the packages are
    # already present in the image if a previous run installed them.
    tlmgr install ieeetran pgfplots || echo "tlmgr skipped (packages already present or mirror unavailable)"
    kpsewhich IEEEtran.cls >/dev/null
    for pass in 1 2 3; do
      pdflatex -interaction=nonstopmode -halt-on-error main.tex >"/tmp/pass$pass.log" 2>&1 || {
        echo "=== pass $pass failed ==="
        grep -n -A4 "^!" "/tmp/pass$pass.log" | head -80
        exit 1
      }
    done
    echo "=== undefined refs / citations ==="
    grep -E "LaTeX Warning: (Citation|Reference) .* undefined" /tmp/pass3.log || echo "(none)"
    echo "=== overfull boxes ==="
    grep -E "^Overfull" /tmp/pass3.log || echo "(none)"
    echo "=== output ==="
    grep -o "Output written.*" /tmp/pass3.log || true
  '
