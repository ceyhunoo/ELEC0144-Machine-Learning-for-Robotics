#!/bin/bash
# Refreshes the submission folder from this repo and, once both PDFs are in it, builds the zip for Moodle.
#   bash build_submission.sh
# ELEC0144_Team10_submission/ holds only what the brief asks for and is not in git (.gitignore).
# Code, data and results are copied into it from this repo every time this script runs, so edit files in the repo,
# never in the submission folder. Export the two PDFs from Word straight into the submission folder:
#   ELEC0144_Report_Final_Draft.docx  ->  ELEC0144_Team10_submission/ELEC0144_Team10_Report.pdf
#   SelfAssessment_Team10.docx        ->  ELEC0144_Team10_submission/SelfAssessment_Team10.pdf
# Left out on purpose: videos, OpenExploration/cache, .venv, ThresholdHistograms.py (it contains hand-drawn
# outlines), OpenExploration/OE_Report_Draft.docx, the .docx files, the brief and template, "Claude outputs/".
set -euo pipefail
cd "$(dirname "$0")"
SUB=ELEC0144_Team10_submission

FILES=(
  pyproject.toml uv.lock
  Task1.py Task2.py WhatIf_Task1.py WhatIf_Task2.py WhatIf_Task1.png WhatIf_Task2.png
  Fruits.jpg RegressionData.csv InitialWeights.npz
  OpenExploration/oe_common.py OpenExploration/OE1_Grips.py OpenExploration/OE2_SessionCheck.py
  OpenExploration/OE3_Explore.py OpenExploration/FinalMethod.py OpenExploration/OE4_Extension.py
  OpenExploration/OE5_SkinCheck.py
)

mkdir -p "$SUB/OpenExploration/results"
cp SUBMISSION_README.md "$SUB/README.md"
for f in "${FILES[@]}"; do cp "$f" "$SUB/$f"; done
for f in OpenExploration/results/*; do                # every saved result: csv, png, md
  case "$(basename "$f")" in .DS_Store) ;; *) cp "$f" "$SUB/OpenExploration/results/" ;; esac
done
echo "Submission folder refreshed from the repo: $SUB/"

missing=0
for f in ELEC0144_Team10_Report.pdf SelfAssessment_Team10.pdf; do
  if [ ! -f "$SUB/$f" ]; then echo "  still missing: $SUB/$f (export it from Word)"; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
  echo "Zip not built yet. Add the PDFs, then run this script again."
  exit 0
fi

rm -f "$SUB.zip"
# leave out anything created by running the code inside the folder (uv's .venv, caches)
zip -qrX "$SUB.zip" "$SUB" -x '*.DS_Store' "$SUB/.venv/*" '*__pycache__*' "$SUB/OpenExploration/cache/*" '*.pyc'
unzip -l "$SUB.zip"
echo "Built $SUB.zip: this is the file to upload to Moodle."
