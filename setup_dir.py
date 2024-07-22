import os

# make dataset dir
root_data_dir = 'dataset'
data_names = ['chgh', 'mmwhs', 'mmwhs2', 'segthor']
for d in data_names:
    data_dir = os.path.join(root_data_dir, d)
    os.makedirs(data_dir, exist_ok=True)
    print('mkdir data:', data_dir)


# make model dir
exp_dir = os.path.join('exps', 'exps')
model_names = ['unetcnx_a1', 'swinunetr', 'unetr', 'cotr', 'attention_unet', 'unet3d', 'DynUNet', 'unest', 'testnet']
for m in model_names:
    for d in data_names:
        # make model dir
        model_exp_dir = os.path.join(exp_dir, m, d, 'tune_results')
        os.makedirs(model_exp_dir, exist_ok=True)
        print('mkdir exp:', model_exp_dir)
        
        
        
