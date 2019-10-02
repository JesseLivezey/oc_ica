for oc in 1 1.25 1.5 1.75 2 2.5 3 3.5; do
  for k in 1 3 5 10 16; do
    python compare_analysis_models.py -n 32 -o $oc -k $k
  done
done
