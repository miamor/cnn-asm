import os

for setname in ['train', 'test']:
    root = '/media/tunguyen/Devs/MtaAV_stuff/VAE/assembly/data/asm_final/{}'.format(setname)

    for lbl in os.listdir(root):
        all = []
        for filename in os.listdir(root+'/'+lbl):
            path = root+'/'+lbl+'/'+filename
            print('path', path)
            with open(path, 'r') as f:
                content = f.read()

                all.append(content)
        
        with open('data/{}_{}.txt'.format(lbl, setname), 'w') as f:
            f.write('\n'.join(all))