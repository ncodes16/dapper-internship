import time
import dapper as dpr
import dapper.da_methods as da



#Testing the Lorenz63 model (sakov2012)
from dapper.mods.KS.bocquet2019 import HMM

HMM.tseq.T = 30 #for now

xx, yy = HMM.simulate()




xps = dpr.xpList()

#Methods in paper:
xps += da.LETKF(N=4 , loc_rad=15/1.82, infl=1.11,rot=True,taper='GC') # 0.18
xps += da.LETKF(N=6,  loc_rad=25/1.82, infl=1.06,rot=True,taper='GC') # 0.14
xps += da.LETKF(N=16, loc_rad=51/1.82, infl=1.02,rot=True,taper='GC') # 0.11


#recommended in bocquet2019.py:
xps += da.Climatology()                                               # 1.3
xps += da.OptInterp()                                                 # 0.5
xps += da.EnKF('Sqrt', N=13,           infl=1.60,rot=True)            # 0.5
xps += da.EnKF('Sqrt', N=20,           infl=1.03,rot=True)            # 0.115
xps += da.EnKF('PertObs',N=23, infl = 0.98, rot=True)
xps += da.EnKF('PertObs',N=30, infl = 1, rot=True)
xps += da.Var3D(xB = 0.1)
xps += da.PartFilt(N=50, reg = 1, NER = 0.1)
#xps += da.Var4D(xB =0.1)

#others to see if we can beat those- add below

# xps += da.Var3D(xB = 0.1)
# xps += da.Var3D(xB = 1)

# xps += da.ExtKF(infl = 10)
# xps += da.ExtKF(infl = 100)

# xps += da.EnKF('Sqrt', N=10)
# xps += da.EnKF('Sqrt', N=100)
# xps += da.EnKF('Sqrt', N=300)
# xps += da.EnKF('PertObs', N=10)
# xps += da.EnKF('PertObs', N=100)
# xps += da.EnKF('PertObs', N=300)

# xps += da.PartFilt(N=100, reg=1, NER = 0.1)
# xps += da.PartFilt(N=800, reg=1, NER = 0.1)
# xps += da.PartFilt(N=3000, reg=1, NER=0.1)

results = ""
for seed in range(3000, 3005):
    for xp in xps:
        xp.seed = seed
    save = xps.launch(HMM, liveplots = False)
    time.sleep(1)
    print(xps.tabulate_avrgs(statkeys = ['err.rms.a', 'rmv.a', 'duration']) + "\n")
    results += (xps.tabulate_avrgs(statkeys = ['rmse.a', 'rmv.a', 'duration'], colorize=False) + "\n")

with open ('kuramoto_sivashinsky1.txt', 'w') as results_file:
    results_file.write(results)