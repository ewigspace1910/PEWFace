source venv/bin/activate
taskset -c 45-80 python tuning/ensemble_weights.py --c /home/k64t/face_recognition/configs/ensemble/tuning-weights/ensemble-1.yaml --t 80 --m 2 --d minimize --cpu --p > logs/tuning/weights-ensemble-1-legit.log
taskset -c 45-80 python tuning/ensemble_weights.py --c /home/k64t/face_recognition/configs/ensemble/tuning-weights/ensemble-2.yaml --t 80 --m 2 --d minimize --cpu --p > logs/tuning/weights-ensemble-2-legit.log
taskset -c 45-80 python tuning/ensemble_weights.py --c /home/k64t/face_recognition/configs/ensemble/tuning-weights/ensemble-3.yaml --t 80 --m 2 --d minimize --cpu --p > logs/tuning/weights-ensemble-3-legit.log
taskset -c 45-80 python tuning/ensemble_weights.py --c /home/k64t/face_recognition/configs/ensemble/tuning-weights/ensemble-4.yaml --t 80 --m 2 --d minimize --cpu --p > logs/tuning/weights-ensemble-4-legit.log
taskset -c 45-80 python tuning/ensemble_weights.py --c /home/k64t/face_recognition/configs/ensemble/tuning-weights/ensemble-5.yaml --t 80 --m 2 --d minimize --cpu --p > logs/tuning/weights-ensemble-5-legit.log

deactivate
