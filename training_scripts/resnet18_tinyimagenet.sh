python3 main.py --p_type resnet18 --s_type resnet18 --kd --k_train 10 --exp_id 3 --temperature 1 --gamma1 1 --gamma2 100 --re_kd_temperature 1 --config_path ./configs/PeerAiD_resnet18_tinyimagenet.json --AA --dataset tinyimagenet --data_path {your_data_path} --fgsm_eval --pgd_eval --lamb1 0.035 --lamb2 35 --lamb3 20 --swa_s 