GPUS="4,5"
REPEATS=5

#SLEEP=5
#python run/sweep.py --config-file run/configs/f1-position.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/f1-dnf.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/f1-qualifying.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS

#SLEEP=240
#python run/sweep.py --config-file run/configs/amazon_churn.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/amazon_ltv.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/amazon_product_churn.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/amazon_product_ltv.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS

#SLEEP=10
#python run/sweep.py --config-file run/configs/stackex-badges.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/stackex-engage.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/stackex-votes.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS


#SLEEP=10
#python run/sweep.py --config-file run/configs/clinical-trial-site.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/clinical-trial-adverse.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/clinical-trial-outcome.yaml --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS

SLEEP=60
GRID_FILE="run/experiments/small_basic.yaml"
python run/sweep.py --config-file run/configs/math-stackex-votes.yaml --grid-file $GRID_FILE --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
#python run/sweep.py --config-file run/configs/math-stackex-badges.yaml --grid-file $GRID_FILE --gpu-ids $GPUS --sleep-time $SLEEP --repeats $REPEATS
