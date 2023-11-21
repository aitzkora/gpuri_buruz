(...)
fitxategia=$(mktemp --tmpdir --suffix=.sl)
echo "#!/bin/sh 
#SBATCH --time=00:05:00
#SBATCH -p cytech
#SBATCH -N 1
" >>$fitxategia
# append command parameters
echo $* >>$fitxategia

# retrieve the job_id
out=$(sbatch $fitxategia) 
job_id=$(echo $out | grep Sub | cut -d' ' -f 4)
out_filename=slurm-$job_id.out
me=$(who | cut -d' ' -f1)

# waiting for job is done
while true; do 
  st=$(squeue -u $me --format "%T" -j $job_id | grep -E -- 'RUNNING|PENDING')
  if [[ -z "$st" ]]; then
    break
  else  
    echo -ne "\rjob is  $st"
  fi
done
(...)
# display job output file content
less +F $out_filename
