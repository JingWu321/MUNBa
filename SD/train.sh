python MUNBa_nsfw.py --config_path './configs/stable-diffusion/v1-inference_nash.yaml' --munba --class_to_forget '0' --train_method 'noxattn' --lr 1e-5 --epochs 1 --device '3' --batch_size 4 --beta 100.0 # --with_l1 --alpha 1e-4
python MUNBa_nsfw.py --config_path './configs/stable-diffusion/v1-inference_nash.yaml' --munba --class_to_forget '0' --train_method 'noxattn' --lr 1e-6 --epochs 2 --device '3' --batch_size 4 --beta 100.0 # --with_l1 --alpha 1e-4
