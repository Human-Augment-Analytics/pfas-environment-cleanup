from ase.build import bcc110, add_adsorbate
from ase.io import read, write

slab = bcc110('Fe', size=(3,3,4), vacuum = 7.5)
tfa = read('TFAnoH.xyz')
add_adsorbate(slab,tfa,height = 2.0, position='ontop', offset=(7.5,7.5))
write('FE_with_TFA_new2.cif', slab)
