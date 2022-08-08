#Ens
CUDA_VISIBLE_DEVICES=0 python3 trainND.py  --batch_size 20 --epochs 5 --match_target -1  --eps 10  --lam 0  --save_dir result/Ens
CUDA_VISIBLE_DEVICES=0 python3 trainND.py  --batch_size 20 --epochs 5 --match_target 150  --eps 10  --lam 0  --save_dir result/Ens
#KL
CUDA_VISIBLE_DEVICES=0 python3 trainND.py  --batch_size 20 --epochs 5 --match_target -1  --eps 10  --lam 1  --KL  --save_dir result/Ens-KL
CUDA_VISIBLE_DEVICES=0 python3 trainND.py  --batch_size 20 --epochs 5 --match_target 150  --eps 10  --lam 1 --KL  --save_dir result/Ens-KL
#ND
CUDA_VISIBLE_DEVICES=0 python3 trainND.py  --batch_size 20 --epochs 5 --match_target -1  --eps 10  --lam 1   --save_dir result/Ens-ND
CUDA_VISIBLE_DEVICES=0 python3 trainND.py  --batch_size 20 --epochs 5 --match_target 150  --eps 10  --lam 1  --save_dir result/Ens-ND
#MM
CUDA_VISIBLE_DEVICES=0 python3 trainND-MM.py  --batch_size 20 --epochs 5 --match_target -1  --eps 10  --lam 0 --save_dir result/MM
CUDA_VISIBLE_DEVICES=0 python3 trainND-MM.py  --batch_size 20 --epochs 5 --match_target 150  --eps 10 --lam 0 --save_dir result/MM
#ND-MM
CUDA_VISIBLE_DEVICES=0 python3 trainND-MM.py  --batch_size 20 --epochs 5 --match_target -1  --eps 10  --lam 1 --save_dir result/MM-ND
CUDA_VISIBLE_DEVICES=0 python3 trainND-MM.py  --batch_size 20 --epochs 5 --match_target 150  --eps 10 --lam 1 --save_dir result/MM-ND
