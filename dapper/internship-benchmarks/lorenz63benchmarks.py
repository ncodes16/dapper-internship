import time
import dapper as dpr
import dapper.da_methods as da



#Testing the Lorenz63 model (sakov2012)
from dapper.mods.Lorenz63.sakov2012 import HMM

HMM.tseq.T = 30 #for now

xx, yy = HMM.simulate()

### List of methods to test:
# 1. Climatology (baseline)
# 2. OptInterp
# 3. Var3d xB = 0.1, 1
# 4. ExtKF infl = 10, 100
# 5. EnKF sqrt, PertObs; N = 10, 100, 300; infl = 1
# 6. PartFilt N = 100, 800, 3000; reg = 1; NER = 0.1


xps = dpr.xpList()

xps += da.Climatology()

xps += da.OptInterp()

xps += da.Var3D(xB = 0.1)
xps += da.Var3D(xB = 1)

xps += da.ExtKF(infl = 10)
xps += da.ExtKF(infl = 100)

xps += da.EnKF('Sqrt', N=10)
xps += da.EnKF('Sqrt', N=100)
xps += da.EnKF('Sqrt', N=300)
xps += da.EnKF('PertObs', N=10)
xps += da.EnKF('PertObs', N=100)
xps += da.EnKF('PertObs', N=300)

xps += da.PartFilt(N=100, reg=1, NER = 0.1)
xps += da.PartFilt(N=800, reg=1, NER = 0.1)
xps += da.PartFilt(N=3000, reg=1, NER=0.1)

results = ""
for seed in range(3000, 3005):
    for xp in xps:
        xp.seed = seed
    save_as = xps.launch(HMM, liveplots = False)
    print(xps.tabulate_avrgs(statkeys = ['err.rms.a', 'rmv.a', 'duration']) + "\n")
    results += (xps.tabulate_avrgs(statkeys = ['rmse.a', 'rmv.a', 'duration'], colorize=False) + "\n")

with open ('results.txt', 'w') as results_file:
    results_file.write(results)