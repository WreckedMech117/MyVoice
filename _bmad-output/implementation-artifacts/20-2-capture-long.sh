set -e
cd /i/MyVoiceV2
OUT=_bmad-output/implementation-artifacts
for i in 1 2 3 4 5; do
  echo "### BEFORE $i"
  python310/python.exe tools/ttfa_spike_harness.py --runs 1 --warmup 0 --utterance long --compile auto \
    --out $OUT/20-2-before-long-r0$i.csv 2>&1 | grep -E "^  run 1|startup priming"
done
for i in 1 2 3 4 5; do
  echo "### AFTER $i"
  python310/python.exe tools/ttfa_spike_harness.py --runs 1 --warmup 0 --utterance long --compile auto --prime \
    --out $OUT/20-2-after-long-r0$i.csv 2>&1 | grep -E "^  run 1|startup priming"
done
echo "### DONE"
