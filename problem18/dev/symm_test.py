import numpy as np, sys; sys.path.insert(0,'dev'); import common, symm
d=common.load(); S=symm.site_symmetry(d['env_t'],d['mask_t']); np.savez('dev/cache/symm_test.npz',**S); print('done')
