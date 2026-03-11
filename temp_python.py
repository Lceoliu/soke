import glob, os, pickle, numpy as np

pred_root = '/home/SOKE/results/mgpt/SOKE_LFQ4_ACC_LM/test_rank_0'
out_dir = '/home/SOKE/visualize/t2m_csl_0312/npy'
os.makedirs(out_dir, exist_ok=True)

# CSL-Daily 样例名通常就是 S000xxx_Pxxxx_Txx
pkls = sorted(glob.glob(os.path.join(pred_root, 'S*.pkl')))[:12]
if not pkls:
    raise SystemExit(f'No CSL-Daily pkl found in {pred_root}')

for p in pkls:
    name = os.path.splitext(os.path.basename(p))[0]
    with open(p, 'rb') as f:
        item = pickle.load(f)

    arr_pred = np.asarray(item['feats_rst'], dtype=np.float32)
    if arr_pred.ndim == 3 and arr_pred.shape[0] == 1:
        arr_pred = arr_pred[0]
    np.save(os.path.join(out_dir, f'{name}_pred.npy'), arr_pred)

    arr_gt = np.asarray(item['feats_ref'], dtype=np.float32)
    if arr_gt.ndim == 3 and arr_gt.shape[0] == 1:
        arr_gt = arr_gt[0]
    np.save(os.path.join(out_dir, f'{name}_gt.npy'), arr_gt)

    print('saved', name)
