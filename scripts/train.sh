source venv/bin/activate
# python examples/train.py --c configs/cosface.yaml --n 4 > logs/cosloss.log 
# python examples/train.py --c configs/arcface.yaml --n 4 > logs/arcloss.log 
# python examples/train.py --c configs/softmax.yaml --n 4 > logs/softmax.log 
# python examples/train.py --c configs/adaface.yaml --n 4 > logs/adaface.log 
# python examples/train.py --c configs/elastic-cos.yaml --n 4 > logs/elastic-cos.log 
# python examples/train.py --c configs/elastic-arc.yaml --n 4 > logs/elastic-arc.log 
# python examples/train.py --c configs/magface.yaml --n 4 > logs/magloss.log 
# python examples/train.py --c configs/adacos.yaml --n 4 > logs/adacos.log 


# python examples/train.py --c configs/org/cosface.yaml --n 4 > logs/org-cosloss.log 
# python examples/train.py --c configs/org/arcface.yaml --n 4 > logs/org-arcloss.log 
python examples/train.py --c configs/org/magface.yaml --n 4 > logs/org-magloss.log
python examples/train.py --c configs/org/eaface.yaml --n 4 > logs/org-ealoss.log 
python examples/train.py --c configs/org/ecface.yaml --n 4 > logs/org-ecloss.log 
python examples/train.py --c configs/org/adaface.yaml --n 4 > logs/org-adaloss.log

deactivate
