#!/bin/bash
# Toda a bateria da tese, com a rede pequena (18,406 parametros). Uma rodada por vez
# (cada rodada usa ~9 GB do WSL). Pode ser rodado de novo: --resume pula o que ja esta completo.
#
#   bash scripts/run_all.sh calibracao   # so o Stackelberg silencioso (para conferir o Cm0)
#   bash scripts/run_all.sh              # tudo
#   bash scripts/run_all.sh shapley_loss # ablacao: Shapley com perda no lugar da acuracia (silent e noisy)
#   bash scripts/run_all.sh full_data    # so o Stackelberg com dados completos (comparacao principal), 8 cenarios
#   PY=/caminho/do/python bash scripts/run_all.sh   # outro interpretador (padrao: python3 do venv ativo)
#
# Cenarios: alpha (0.5, 0.1) x ataque (silent, low_effort, noisy, fr50). Em cada um, por seed:
# stackelberg (dados escolhidos pelo preco), stackelberg em full_data/ (dados completos: e o Stackelberg
# da comparacao principal), fedavg, shapley, shapley_v0 (sem exclusao), control (fedavg com o volume de
# dados do Stackelberg). Nos cenarios silent e fr50: tambem oracle (fedavg sem os free-riders, o detector
# perfeito). No silent: tambem fr0 (sem free-riders) e shapley com V=0.2 (shapley_v02).
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-python3}
SEEDS="42 43 44 45 46"
LOGS=results/logs
mkdir -p $LOGS

# As variaveis das funcoes sao `local`: no bash elas sao globais por padrao, e o `dir` do run() chegou a
# sobrescrever o `dir` do scenario(), gravando os resultados em pastas aninhadas.
run() {  # nome, pasta, args...
  local name=$1 dir=$2 status
  shift 2
  echo "$(date '+%F %T') inicio $name" >> $LOGS/progresso.txt
  $PY scripts/run_experiment.py --resume --model small --results-dir $dir "$@" > $LOGS/$name.log 2>&1
  status=$?   # guardado antes do $(date), que sobrescreveria o $?
  echo "$(date '+%F %T') fim $name (codigo $status)" >> $LOGS/progresso.txt
}

scenario_args() {  # ataque -> argumentos do cenario
  case $1 in
    silent)     echo "--num-free-riders 3" ;;
    low_effort) echo "--num-free-riders 3 --free-rider-type low-effort" ;;
    noisy)      echo "--num-free-riders 3 --free-rider-type noisy" ;;
    fr50)       echo "--num-free-riders 5" ;;
  esac
}

voting_rate() { if [ "$1" = fr50 ]; then echo 0.5; else echo 0.3; fi; }
excluded() { if [ "$1" = fr50 ]; then echo "0,1,2,3,4"; else echo "0,1,2"; fi; }

stackelberg_only() {  # alpha ataque
  local alpha attack dir args seed
  alpha=$1; attack=$2; dir=results/a${alpha/./}_$attack; args=$(scenario_args $attack)
  for seed in $SEEDS; do
    run a${alpha/./}_${attack}_stackelberg_s$seed $dir --strategy stackelberg --seed $seed --alpha $alpha $args
  done
}

scenario() {  # alpha ataque
  local alpha attack tag dir args v seed
  alpha=$1; attack=$2; tag=a${alpha/./}_$attack; dir=results/$tag
  args=$(scenario_args $attack); v=$(voting_rate $attack)
  for seed in $SEEDS; do
    run ${tag}_stackelberg_s$seed $dir --strategy stackelberg --seed $seed --alpha $alpha $args
    run ${tag}_fulldata_s$seed    $dir/full_data --strategy stackelberg --seed $seed --alpha $alpha $args --full-data
    run ${tag}_fedavg_s$seed      $dir --strategy fedavg      --seed $seed --alpha $alpha $args
    run ${tag}_shapley_s$seed     $dir --strategy shapley     --seed $seed --alpha $alpha $args --shapley-voting-rate $v
    run ${tag}_shapleyv0_s$seed   $dir/shapley_v0 --strategy shapley --seed $seed --alpha $alpha $args --shapley-voting-rate 0
    run ${tag}_control_s$seed     $dir/control --strategy fedavg --seed $seed --alpha $alpha $args --price-from $dir/stackelberg_seed$seed.json
  done
}

oracle() {  # alpha ataque (so silent e fr50: o oracle ignora os free-riders, qualquer que seja o ataque)
  local alpha attack tag args seed
  alpha=$1; attack=$2; tag=a${alpha/./}_$attack; args=$(scenario_args $attack)
  for seed in $SEEDS; do
    run ${tag}_oracle_s$seed results/$tag/oracle --strategy fedavg --seed $seed --alpha $alpha $args --exclude-clients $(excluded $attack)
  done
}

zero_free_riders() {  # alpha
  local alpha tag seed strategy
  alpha=$1; tag=a${alpha/./}_silent
  for seed in $SEEDS; do
    for strategy in stackelberg fedavg shapley; do
      run ${tag}_fr0_${strategy}_s$seed results/$tag/fr0 --strategy $strategy --seed $seed --alpha $alpha --num-free-riders 0
    done
  done
}

shapley_v02() {  # alpha
  local alpha tag seed
  alpha=$1; tag=a${alpha/./}_silent
  for seed in $SEEDS; do
    run ${tag}_shapleyv02_s$seed results/$tag/shapley_v02 --strategy shapley --seed $seed --alpha $alpha --num-free-riders 3 --shapley-voting-rate 0.2
  done
}

shapley_loss() {  # alpha ataque (ablacao: a utilidade v(S) do Shapley passa de acuracia para perda)
  # So o Shapley usa v(S): FedAvg e Stackelberg nao mudam, entao nao sao rodados de novo.
  # Grava em pasta propria (shapley_loss) e nao altera os resultados originais.
  local alpha attack tag args v seed
  alpha=$1; attack=$2; tag=a${alpha/./}_$attack
  args=$(scenario_args $attack); v=$(voting_rate $attack)
  for seed in $SEEDS; do
    run ${tag}_shapleyloss_s$seed results/$tag/shapley_loss --strategy shapley --seed $seed --alpha $alpha $args --shapley-voting-rate $v --shapley-utility loss
  done
}

if [ "$1" = shapley_loss ]; then
  # Ordem: o cenario do diagnostico primeiro (silent, alpha 0.5), depois os outros principais.
  for attack in silent noisy; do
    shapley_loss 0.5 $attack
    shapley_loss 0.1 $attack
  done
  echo "$(date '+%F %T') SHAPLEY_LOSS CONCLUIDO" >> $LOGS/progresso.txt
  exit 0
fi

if [ "$1" = full_data ]; then
  for alpha in 0.5 0.1; do
    for attack in silent low_effort noisy fr50; do
      tag=a${alpha/./}_$attack
      for seed in $SEEDS; do
        run ${tag}_fulldata_s$seed results/$tag/full_data --strategy stackelberg --seed $seed --alpha $alpha $(scenario_args $attack) --full-data
      done
    done
  done
  echo "$(date '+%F %T') FULL_DATA CONCLUIDO" >> $LOGS/progresso.txt
  exit 0
fi

if [ "$1" = calibracao ]; then
  stackelberg_only 0.5 silent
  stackelberg_only 0.1 silent
  echo "$(date '+%F %T') CALIBRACAO CONCLUIDA" >> $LOGS/progresso.txt
  exit 0
fi

for alpha in 0.5 0.1; do
  scenario $alpha silent
  oracle $alpha silent
  zero_free_riders $alpha
  shapley_v02 $alpha
done
for alpha in 0.5 0.1; do
  scenario $alpha low_effort
  scenario $alpha noisy
done
for alpha in 0.5 0.1; do
  scenario $alpha fr50
  oracle $alpha fr50
done
echo "$(date '+%F %T') TUDO CONCLUIDO" >> $LOGS/progresso.txt
