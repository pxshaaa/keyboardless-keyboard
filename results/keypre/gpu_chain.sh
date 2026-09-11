cd /Users/pashaalidadi/Documents/misc/computer-vision-test
export PYTHONPATH=.
for i in 1 2 3 4; do
  .venv/bin/python -m phase0.analysis.keypre pix >> .cache/keypre/pix.log 2>&1; rc=$?
  echo "pix exit=$rc attempt=$i" >> .cache/keypre/pix.log
  [ $rc -eq 0 ] && break
done
for i in 1 2 3; do
  .venv/bin/python -m phase0.analysis.keypre deskpix --conds finetune,scratch >> .cache/keypre/deskpix.log 2>&1; rc=$?
  echo "deskpix exit=$rc attempt=$i" >> .cache/keypre/deskpix.log
  [ $rc -eq 0 ] && break
done
