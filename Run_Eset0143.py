
import numpy as np
from ClassDefs import Eset, TrpExp, ReadExpPars
from consensusPeak import AutoencoderCP

datapath = '/Volumes/ANPC_ext1/trp_qc'
expFile = '/Users/TKimhofer/Downloads/Torben19Aug.exp'

# Import experiment set (autoconverts vendor file format to xml-based open data format)
ss=Eset.importbinary(dpath=datapath, epath=expFile)

# Import and quantify peak integrals in a single experiment
# efile = TrpExp.waters(dpath=datapath + 'NW_TRP_023_manual_Spiking_05Aug2022_URN_healthy_92.raw', efun=ReadExpPars(expFile), convert=False)
# efile.q(height=0.1, distance=10, prominence=1, width=7, wlen=9, rel_height=0.7, plot=False)

# visualise individual functions
# fA='FUNCTION 2'
# efile.featplot(efile.qs[fA][0])

# check compound - internal standard pair (id: a1 to a10)
pair='a4'
s=0 # first sample
sic=0 # select iom transition

# map to assay functions
fIS=list(ss.exp[s].fmap[pair]['std'].keys())[sic] # fIS='FUNCTION 27' sic 0
fA=list(ss.exp[s].fmap[pair]['analyte'].keys())[sic]

# visualise SIC's for analyte and internal standard
ss.exp[s].featplot(ss.exp[s].qs[fIS][sic])
ss.exp[s].featplot(ss.exp[s].qs[fA][sic])

# xx.featplot(xx.qs[fIS][0])
# xx.featplot(xx.qs[fA][1])

# np.argsort(-dp)
# i=71
# exp1[i].featplot(exp1[i].qs[fIS][1])

# oddities
# a15: Fx.0 and Fx.1 matching
# a14: only Fx.0 matching pair, Fx.1 not exist for IS
# a13: F36.0 (IS), F32.1 (A) matching, F32.0 not paired
# a12: Xanthurenic acid: additoinal signal that is max but nor relevant
# a6: F6.0 with F14.1 (no 6.1 and no 14.0)
# a5: more than peak in analyte Fx0 and Fx1 (F.x2 the same)
# a7: 17.1 and 15.1 not matching in transition
# a8: 19.0 with 18.1 (forget the others)
# def getA(dat):
#     iid=np.argmax([x['A'] for x in dat[1]])
#     return dat[1][iid]['A']
#
# f='FUNCTION 11' # piclolinic acid
# # fa='FUNCTION 9' # piclolinic acid
# getA(x.qs[f][sir])
#
# # extract sics from eset and run consensus peak autoencoder model
# cp = AutoencoderCP()
# cp.fit()