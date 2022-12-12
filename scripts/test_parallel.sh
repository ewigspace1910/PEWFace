source venv/bin/activate
#############################
#             CPU           #
#script 2 squence = baseline = 16 core, parallel 48 core
taskset -c 42-58  python examples/parallel_cpu.py --c "./configs/ensemble/soft.yaml"  --b 128 --l 5
taskset -c 42-58  python examples/parallel_cpu.py --c "./configs/ensemble/soft.yaml"  --e --b 128 --l 10 
taskset -c 42-58  python examples/parallel_cpu.py --c "./configs/ensemble/hard.yaml"  --e --b 128 --l 10

taskset -c 42-90  python examples/parallel_cpu.py --c "./configs/ensemble/soft.yaml"  --e --b 128 --l 5 --parallel
taskset -c 42-90  python examples/parallel_cpu.py --c "./configs/ensemble/hard.yaml"  --e --b 128 --l 10 --parallel

#############################
#             GPU          #
python examples/parallel_gpu.py --c "./configs/ensemble/soft.yaml"  --b 16 --l 10
python examples/parallel_gpu.py --c "./configs/ensemble/soft.yaml"  --e --b 16 --l 10 
python examples/parallel_gpu.py --c "./configs/ensemble/soft.yaml"  --e --b 16 --l 10 --parallel
deactivate
