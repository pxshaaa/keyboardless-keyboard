cd /Users/pashaalidadi/Documents/misc/computer-vision-test
base=$(grep -c "exit=" .cache/keypre/deskpix.log)
for cond in finetune scratch; do
  f=.cache/keypre/deskpix_${cond}_3x40.npy
  until grep -q -- "-> $f" .cache/keypre/deskpix.log; do
    tail -n +1 .cache/keypre/deskpix.log | grep "exit=" | tail -n +$((base+1)) | grep -q "exit=[1-9]" && { echo "deskpix failed, abort $cond"; exit 1; }
    sleep 20
  done
  rsync -az "$f" macmini:~/cvt/.cache/keypre/ || exit 1
  echo "shipped $f md5 $(md5 -q $f)"
  ssh macmini "cd ~/cvt && nice -n 10 env PYTHONPATH=. .venv/bin/python -m phase0.analysis.keypre desktable --cond $cond --procs 4 > .cache/keypre/desktable_$cond.log 2>&1; echo table_rc=\$?; nice -n 10 env PYTHONPATH=. .venv/bin/python -m phase0.analysis.keypre deskcv --cond $cond --procs 4 --examples 6 --json .cache/keypre/deskcv_$cond.json > .cache/keypre/deskcv_$cond.log 2>&1; echo cv_rc=\$?"
  rsync -az "macmini:~/cvt/.cache/keypre/table_base+rec_$cond.json" "macmini:~/cvt/.cache/keypre/desktable_$cond.log" "macmini:~/cvt/.cache/keypre/deskcv_$cond.json" "macmini:~/cvt/.cache/keypre/deskcv_$cond.log" .cache/keypre/
  echo "copied back $cond"
done
