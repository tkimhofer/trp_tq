import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import re

def matchC(rec, fis, fa, sirIS, sirA, locDev=60/30, sumC=False, maxi=False):
    # select fitted components of IS and match to components of analyte
    # locDev: allowed rt deviation (10%->6 sec)
    if fis not in rec.qs.keys() or fa not in rec.qs.keys():
        return [dict(info= 'function not found', sid=rec.fname, fID=fIS, fA=fA, tA=None, tIS = None, sirIS=sirIS, sirA=sirA, cidIS=None, cidA=None, aIS=None, aA=None, aR=None)]

    recIS = rec.qs[fis]
    recA = rec.qs[fa]
    # print(recIS[sirIS][1])
    # print(rec.efun.funcs[fIS].reactions)
    # print(rec.efun.funcs[fIS].reactions)
    tis = rec.efun.funcs[fIS].reactions[str(sirIS + 1)]
    ta = rec.efun.funcs[fA].reactions[str(sirA + 1)]

    if not isinstance(recIS[sirIS], list) or not isinstance(recA[sirA], list):
        return [dict(info= 'sir not found', sid=rec.fname, fID=fIS, fA=fA, tA=ta, tIS = tis, sirIS=sirIS, sirA=sirA, cidIS=None, cidA=None, aIS=None, aA=None, aR=None)]
    locIS = [x['mu_decon'] for x in recIS[sirIS][1]]
    locA = [x['mu_decon'] for x in recA[sirA][1]]


    # find matches
    mm=[]
    for i, li in enumerate(locIS):
        a_is = recIS[sirIS][1][i]['A']
        res = [(k, la) for k, la in enumerate(locA) if abs(li-la)<locDev]

        if len(res) == 0:
            d=dict(sid=rec.fname, fID=fIS, fA=fA, tA=ta, tIS = tis, sirIS=sirIS, sirA=sirA, cidIS=i, cidA=None, aIS=a_is, aA=None, aR=None)
        else:
            vals = [x[1] for x in res]
            idx_min=res[vals.index(min(vals))][0]
            a_a = recA[sirA][1][idx_min]['A']
            d=dict(info= 'quantified comp', sid=rec.fname, fID=fIS, fA=fA, tA=ta, tIS = tis, sirIS=sirIS, sirA=sirA, cidIS=i, cidA=idx_min, aIS=a_is, aA=a_a, aR=a_a/a_is)
        mm.append(d)

    if sumC:
        ais=0
        aa=0
        for i, dd in enumerate(d):
            ais += d['aIS']
            aa += d['aA']
        mm=[dict(info= 'quantified sum', sid=rec.fname, fID=fIS, fA=fA, tA=ta, tIS = tis, sirIS=sirIS, sirA=sirA, cidIS=i, cidA=idx_min, aIS=ais, aA=aa, aR=aa/ais)]

    if maxi and len(mm)>1:
        intis = [x['aIS'] for x in mm]
        idx=intis.index(max(intis))
        mm=[mm[idx]]

    return mm




    #
    #
    # locIS = [x['mu_decon'] for x in rec[sirIS][1]]
    # locA = [x['mu_decon'] for x in rec[sirA][1]]
    #
    # # return ratios A/IS

def cat(x):
    if bool(re.findall('22_DB_[0-9].*', x)):
        return 'DB'
    elif bool(re.findall('22_Cal[0-9]_[0-9].*', x)):
        return x.split('_')[-2]
    elif bool(re.findall('22_PLA_unhealthy_[0-9].*', x)):
        return 's1P'
    elif bool(re.findall('22_SER_unhealthy_[0-9].*', x)):
        return 's1S'
    elif bool(re.findall('22_URN_unhealthy_[0-9].*', x)):
        return 's1U'
    elif bool(re.findall('22_PLA_healthy_[0-9].*', x)):
        return 's0P'
    elif bool(re.findall('22_SER_healthy_[0-9].*', x)):
        return 's0S'
    elif bool(re.findall('22_URN_healthy_[0-9].*', x)):
        return 's0U'
    elif bool(re.findall('22_PLA_LTR_[0-9].*', x)):
        return 'rP'
    elif bool(re.findall('22_SER_LTR_[0-9].*', x)):
        return 'rS'
    elif bool(re.findall('22_URN_LTR_[0-9].*', x)):
        return 'rU'
    elif bool(re.findall('22_PLA_unhealthy_Cal[0-9]_[0-9].*', x)):
        return 's1P_'+x.split('_')[-2]
    elif bool(re.findall('22_SER_unhealthy_Cal[0-9]_[0-9].*', x)):
        return 's1S_'+x.split('_')[-2]
    elif bool(re.findall('22_URN_unhealthy_Cal[0-9]_[0-9].*', x)):
        return 's1U_'+x.split('_')[-2]
    elif bool(re.findall('22_PLA_healthy_Cal[0-9]_[0-9].*', x)):
        return 's0P_' + x.split('_')[-2]
    elif bool(re.findall('22_SER_healthy_Cal[0-9]_[0-9].*', x)):
        return 's0S_' + x.split('_')[-2]
    elif bool(re.findall('22_URN_healthy_Cal[0-9]_[0-9].*', x)):
        return 's0U_' + x.split('_')[-2]

    elif bool(re.findall('22_PLA_LTR_Cal[0-9]_[0-9].*', x)):
        return 'rP_'+x.split('_')[-2]
    elif bool(re.findall('22_SER_LTR_Cal[0-9]_[0-9].*', x)):
        return 'rS_'+x.split('_')[-2]
    elif bool(re.findall('22_URN_LTR_Cal[0-9]_[0-9].*', x)):
        return 'rU_'+x.split('_')[-2]
    else:
        return 'unassigned'

def createMetaDf(test):
    nam = [x.fname for x in test.exp]
    cats = [cat(x.fname) for x in test.exp]
    df = pd.DataFrame({'cats': cats, 'nam': nam})
    df['cats'].value_counts()
    rep = df[df['cats'] == '2']['nam'].str.split('_').str[-3]
    df.loc[df['cats'] == '2', 'cats'] = rep
    df['dt'] = [x.aDt for x in test.exp]
    add = ['Average System Pressure', 'Minimum System Pressure', 'Maximum System Pressure',
           'Total Injections on Column', 'Sample Description', 'Bottle Number']
    meta = pd.DataFrame([x.edf for x in test.exp])
    df1 = pd.concat([df, meta[add]], axis=1)
    df1['ind'] = np.arange(df1.shape[0])
    df1 = df1.sort_values('dt')
    return df1


class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(500, activation="relu"),
      layers.Dense(256, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(256, activation="sigmoid"),
      layers.Dense(500, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def fnorm(dat, xnew):
    import scipy.interpolate as sint
    iid=np.argmax([x['A'] for x in dat[1]])
    xn=dat[0]['x']- dat[1][iid]['mu_decon']
    yn = dat[0]['yo']/dat[1][iid]['A']
    fi = sint.interp1d(xn, yn, bounds_error=False, fill_value="extrapolate")
    yi = fi(xnew)
    return yi / max(yi)
    # return yi
  
ss=Eset.importbinary(dpath='/Volumes/ANPC_ext1/trp_qc', epath='/Users/TKimhofer/Downloads/Torben19Aug.exp', pat='', n=600)
# ss=Eset.imp(dpath='/Volumes/ANPC_ext1/trp_qc', epath='/Users/TKimhofer/Downloads/Torben19Aug.exp', pat='', n=600, alwaysConvert=False)

df=createMetaDf(ss)
ttt=df[['cats', 'nam', 'Sample Description',  'Total Injections on Column']]
df.index=df.nam

# select calibration samples
df['rid'] = df['nam'].str.split('_|\.').str[-2].astype(int)
calibMan = df[(df['rid'] > 131) & (df['rid'] < 140)]
df['Total Injections on Column'] = df['Total Injections on Column'].astype(int)
calibRob = df[(df['Total Injections on Column'] > 1058) & (df['Total Injections on Column'] < 1067)]
calibRob.index = calibRob.nam
calibMan.index = calibMan.nam



# plot calibration curve for calibMan
for i in range(calibMan.shape[0]):
    print(i)
    [x.fname for x in enumerate(ss.exp) if ]




epath='/Users/TKimhofer/Downloads/Torben19Aug.exp'
efile = TrpExp.waters(dpath='/Volumes/ANPC_ext1/trp_qc/NW_TRP_023_manual_Spiking_05Aug2022_URN_healthy_92.raw', efun=ReadExpPars(epath), convert=False)
efile.q()
fA='FUNCTION 2'
efile.featplot(efile.qs[fA][0])



efile.q(height=0.1, distance=10, prominence=1, width=7, wlen=9, rel_height=0.7, plot=False)
efile.qs['FUNCTION 27'][0][0].keys()
efile.qs['FUNCTION 31'][0][1][0].keys()

plt.plot(efile.qs['FUNCTION 1'][0][0]['yres'])
plt.title(efile.qs['FUNCTION 31'][0][0]['resNorm'])




# get sd normalised residuals
import statistics
{i: sum(abs(x.qs[fIS][sic][0]['yo'] - x.qs[fIS][sic][0]['yest'])) / statistics.stdev(x.qs[fIS][sic][0]['yo']) for i, x in enumerate(ss.exp) if isinstance(x.qs[fIS][0], list)}


fA='FUNCTION 2'
efile.featplot(ss.exp[1].qs[fA][0])
ss.exp[1].featplot(ss.exp[1].qs[fA][0])
ss.exp[6].featplot(ss.exp[6].qs[fA][0])
ss.exp[4].featplot(ss.exp[4].qs[fA][sic])

ss.exp[5].featplot(ss.exp[5].qs[fIS][sic])
ss.exp[8].featplot(ss.exp[8].qs[fIS][sic])





df=createMetaDf(ss)
df.index=df.nam
ttt=df[['cats', 'nam', 'Sample Description']]


# ss=Eset.imp(dpath='/Users/torbenkimhofer/tdata_trp/', epath='/Users/torbenkimhofer/Desktop/Torben19Aug.exp', pat='', n=1000)

# ss.exp[0]
# f='FUNCTION 5' # piclolinic acid
# x=ss.exp[5]
# x.fname
# x.featplot(x.qs[f][1])
#
# kwargs = dict(height=0.1, distance=10, prominence=1, width=7, wlen=9, rel_height=0.7)
# x.qFunction(f, 2, plot=True, **kwargs, )
pair='a4'
s=0
sic=0
# xx=[x for x in ss.exp if x.fname == 'RCY_TRP_023_Robtot_Spiking_04Aug2022_SER_unhealthy_Cal2_80.raw'][0]
fIS=list(ss.exp[s].fmap[pair]['std'].keys())[0]
# fIS='FUNCTION 27'
# fIS='FUNCTION 10'
# fIS='FUNCTION 28'
# fIS='FUNCTION 14'

fA=list(ss.exp[s].fmap[pair]['analyte'].keys())[0]

ss.exp[s].featplot(ss.exp[s].qs[fIS][sic])
ss.exp[s].featplot(ss.exp[s].qs[fA][sic])

# xx.featplot(xx.qs[fIS][0])
# xx.featplot(xx.qs[fA][1])

# np.argsort(-dp)
# i=71
# exp1[i].featplot(exp1[i].qs[fIS][1])

# a15: Fx.0 and Fx.1 matching
# a14: onlye Fx.0 matching pair, Fx.1 not exist for IS
# a13: F36.0 (IS), F32.1 (A) matching, F32.0 not paired
# a12: Xanthurenic acid: additoinal signal that is max but nor relevant
# a6: F6.0 with F14.1 (no 6.1 and no 14.0)
# a5: more than peak in analyte Fx0 and Fx1 (F.x2 the same)
# a7: 17.1 and 15.1 not matching in transition
# a8: 19.0 with 18.1 (forget the others)
def getA(dat):
    iid=np.argmax([x['A'] for x in dat[1]])
    return dat[1][iid]['A']

getA(x.qs[f][sir])
11,9
20,12
14,13
17, 15

# f='FUNCTION 11' # piclolinic acid
# fa='FUNCTION 9' # piclolinic acid
df['assay'] = df['nam'].apply(lambda x: 'r' if bool(re.findall('.*Robtot.*', x)) else 'm')
# df['assay'] = df['nam'].apply(lambda x:  print(bool(re.findall('.*.*', str(x)))))

df['assay'].value_counts()
df['Total Injections on Column'] = df['Total Injections on Column'].astype(int)
df['Sample Description'].value_counts()
df['qc'] = df['Sample Description'].str.contains("^ ?Cal.*", regex=True, flags=re.IGNORECASE)
df.qc.value_counts()


# data = []
fIS='FUNCTION 20'
fA='FUNCTION 1'
sirIS=0
sirA=1
ar = []
quants=[]
for i, x in enumerate(ss.exp):
    # x.featplot(x.qs[f][1])
    rec = x
    quants.append(matchC(rec, fIS, fA, sirIS, sirA, sumC=True, maxi=False))
len(quants)
test=pd.DataFrame([item for sublist in quants for item in sublist])
# plt.scatter(np.arange(test.shape[0]), test.aR)

test.index=test.sid
df1=df.join(test)
df1['Total Injections on Column']=df1['Total Injections on Column'].astype(int)


# dman = calibMan.join(test)
# dman['cpoint'] = dman['Sample Description'].str.replace('Cal', '').astype(int)
# #
# fig, axs = plt.subplots(1, 2, sharey=True)
# axs[0].scatter(drob['cpoint'], 1/(1+drob['aR']), label='Dopamine Robot')
# # axs[0].set_yscale('log')
# axs[1].scatter(dman['cpoint'], 1 / (1+dman['aR']), label='Dopamine Manual')
# # axs[1].set_yscale('log')
# fig.legend()
#
# fig, axs = plt.subplots(1, 2, sharey=True)
# axs[0].scatter(drob['cpoint'], drob['aR'], label='Dopamine Robot')
# axs[1].scatter(dman['cpoint'], dman['aR'], label='Dopamine Manual')
#
# fig.legend()

df1['cpoint']



fig, axs = plt.subplots(1, 2)

idx=df1.assay == 'r'
df1.assay.value_counts()
idx=df1.assay == 'r'

df1['cpoint']=None
# idx_cal = df1['Sample Description'].str.contains('^ ?Cal')
df1['cpoint'].loc[df1.qc] = df1['Sample Description'].loc[df1.qc].str.replace('^ ?Cal', '').astype(float)

fig, axs = plt.subplots(2, 2, sharey=True)

iidc = df1.qc & (df1.assay=='r')

axs[0, 0].scatter(df1['cpoint'].loc[df1.qc and (df1.assay=='r')], df1['aA'].loc[df1.qc & (df1.assay=='r')], label='integral A')
axs[0, 0].set_title('Calib Robot')
axs[0, 0].set_yscale('log', base=10)
axs[0, 0].legend()

df1['iid']=df['Total Injections on Column'].rank()

for i in df1['Sample Description'].loc[~df1.qc & (df1.assay=='r')].unique():
    idx = df1['Sample Description'] == i
    axs[0, 1].scatter(df1.iid.loc[idx], df1['aIS'].loc[idx], label=i)
axs[0, 1].legend()
axs[0, 1].set_yscale('log', base=10)


axs[0].scatter(df1['cpoint'].loc[df1.qc], df1['aIS'].loc[df1.qc], label='integral IS')
axs[0, 1].scatter(np.argsort(np.argsort(df1['aIS'].loc[~df1.qc].values)), df1['aIS'].loc[~df1.qc])
axs[1].scatter(np.argsort(np.argsort(df1['aA'].loc[~df1.qc].values)), df1['aA'].loc[~df1.qc], c=df1['Sample Description'].astype('category').cat.codes.loc[~df1.qc])
axs[1].set_yscale('log', base=10)


df1['Sample Description'].astype('category').cat.codes

axs[1].scatter(drob['cpoint'].loc[~idx], drob['aIS'].loc[~idx], label='integral IS')
axs[1].set_yscale('log', base=10)
axs[1].scatter(drob['cpoint'].loc[~idx], drob['aA'].loc[~idx], label='integral A')
axs[1].set_title('Calib Manual')
axs[1].set_yscale('log')
axs[1].legend()



fig, axs = plt.subplots(1, 2, sharey=True)

idx=df1.assay == 'r'
df1.assay.value_counts()

axs[0].scatter(df1['cpoint'].loc[idx], df1['aIS'].loc[idx], label='integral IS', c=df1['qc'].astype('category').cat.codes.loc[idx])
axs[0].scatter(drob['cpoint'].loc[idx], drob['aA'].loc[idx], label='integral A')
axs[0].set_title('Calib Robot')
axs[0].set_yscale('log', base=10)
axs[0].legend()

axs[1].scatter(drob['cpoint'].loc[~idx], drob['aIS'].loc[~idx], label='integral IS')
axs[1].set_yscale('log', base=10)
axs[1].scatter(drob['cpoint'].loc[~idx], drob['aA'].loc[~idx], label='integral A')
axs[1].set_title('Calib Manual')
axs[1].set_yscale('log')
axs[1].legend()

tttitle = f'{fIS}.{sirIS}, {fA}.{sirA}\n{ss.exp[0].efun.funcs[fA].reactions[str(sirA+1)]}'
fig.suptitle(tttitle)


caliMeta=pd.read_excel('/Users/TKimhofer/Downloads/Concentrations for the Try assay standards.xlsx')
caliMeta[caliMeta.Substance.str.contains('Dopam', na=False)].T.to_dict('index')
caliD = {'Dopamine': {1: 80, 2: 40, 3: 20, 4: 12, 5: 8, 6: 3.2, 7: 1.8, 8: 0.8}, '3-HAA': {1: 200, 2:100, 3:50, 4:30, 5:20, 6:8, 7:4, 8:2}, 'Serotonin': {1: 300, 2:150, 3:75, 4:45, 5:30, 6:12, 7:6, 8:3}, 'TMA': {1:400, 2:200, 3:100, 4:60, 5: 40, 6: 16, 7:8, 8:4}, 'TMAO': {1:400, 2:200, 3:100, 4:60, 5: 40, 6: 16, 7:8, 8:4}}

dman['conc']=[caliD['Dopamine'][x] for x in dman.cpoint]
drob['conc']=[caliD['Dopamine'][x] for x in drob.cpoint]

dman['conc']=[caliD['3-HAA'][x] for x in dman.cpoint]
drob['conc']=[caliD['3-HAA'][x] for x in drob.cpoint]

dman['conc']=[caliD['Serotonin'][x] for x in dman.cpoint]
drob['conc']=[caliD['Serotonin'][x] for x in drob.cpoint]

dman['conc']=[caliD['TMAO'][x] for x in dman.cpoint]
drob['conc']=[caliD['TMAO'][x] for x in drob.cpoint]

import scipy.stats as sss

idx=drob.assay == 'r'

fig, axs = plt.subplots(1, 2, sharey=True)

axs[0].scatter(drob['conc'].loc[idx], drob['aIS'].loc[idx], label='integral IS')
axs[0].scatter(drob['conc'].loc[idx], drob['aA'].loc[idx], label='integral A')
axs[0].set_yscale('log', base=10)
axs[0].set_xscale('log', base=10)
axs[0].legend()


axs[1].scatter(dman['conc'].loc[~idx], dman['aIS'].loc[~idx], label='integral IS')
axs[1].scatter(dman['conc'].loc[~idx], dman['aA'].loc[~idx], label='integral A')
axs[1].set_yscale('log', base=10)
axs[1].set_xscale('log', base=10)
axs[1].legend()
fig.suptitle(tttitle)



# slope, intercept, r, p, se = sss.linregress(dman['conc'].values, dman['aA'].values)
# slope, intercept, r, p, se = sss.linregress(drob['conc'].values, drob['aA'].values)
# axs[0].set_title(f'Calib Robot (r2={round(r, 2)}, m={round(slope, 2)}, b={round(intercept, 2)})')
# axs[1].set_title(f'Calib Manual (r2={round(r, 2)}, m={round(slope, 2)}, b={round(intercept, 2)})')


df2=df1[df1['cats'].str.contains('rP')]
esub = [x for x in ss.exp if x.fname in df2.index]
for e in esub:
    if not pd.isna(df1.loc[e.fname].aR):
        print(df1.loc[e.fname])
        e.featplot(e.qs[fIS][sic])
        e.featplot(e.qs[fA][sic])
# ct=test.sid.value_counts()
# rt =test.loc[ct[ct>1].index]



# fig, axs = plt.subplots(2,4)
vis(df1, title=f"{rec.efun.funcs[fA].reactions[str(sirA+1)]} with {rec.efun.funcs[fIS].reactions[str(sirIS+1)]}",\
    shy=True, ymax=None)


def vis(df1, title, fsize=(9, 6), shy=True, ymax=None):
    # plt.figure(figsize=fsize)
    import re
    def mapPanel(x):
        if 's1' in x:
            return 's1'
        elif 's0' in x:
            return 's0'
        elif bool(re.findall('^Cal', x)):
            return 'cal'
        elif 'r' in x:
            return 'ltr'
        else:
            return 'uknw'

    df1['type'] = [mapPanel(x) for x in df1.cats]
    pans = ['rP', 'rS', 'rU', 's1P', 's1S', 's1U']

    cols = plt.get_cmap('tab20c').colors
    colS = {'rP': cols[3], 'rP_Cal5': cols[2], 'rP_Cal2': cols[1],
            'rS': cols[7], 'rS_Cal5': cols[6], 'rS_Cal2': cols[5],
            'rU': cols[11], 'rU_Cal5': cols[10], 'rU_Cal2': cols[9],
            's1P': cols[3], 's1P_Cal5': cols[2], 's1P_Cal2': cols[1],
            's1S': cols[7], 's1S_Cal5': cols[6], 's1S_Cal2': cols[5],
            's1U': cols[11], 's1U_Cal5': cols[10], 's1U_Cal2': cols[9],
            'Cal': plt.get_cmap('viridis')(np.linspace(0, 1, 8))
            }
    fig = plt.figure(figsize=fsize)
    idp_x = 0
    idp_y = 0
    for s, i in enumerate(pans):
        if ((idp_y+1) % 4) == 0 :
            idp_x += 1
            idp_y +=1
        ds= df1[df1.cats.str.contains(pans[s])].copy()
        ds['rank']=ds['Total Injections on Column'].rank().astype(int)
        if s==0:
            sub1 = fig.add_subplot(2, 4, idp_y + 1)
            aa= sub1
        else:
            if shy:
                sub1 = fig.add_subplot(2, 4, idp_y + 1, sharey=aa)
            else:
                sub1 = fig.add_subplot(2, 4, idp_y + 1)
                plt.setp(sub1.get_yticklabels(), visible=False)
        if isinstance(ymax, list):
            sub1.set_ylim(ymax)
        sub1.set_yscale('log', base=10)
        sub1.set_xticks([])
        sub1.set_title(i)
        if (idp_y % 4) != 0:
            plt.setp(sub1.get_yticklabels(), visible=False)

        for r, t in enumerate(ds.cats.unique()):
            iid = ds.cats == t
            if bool(re.findall('^Cal', t)):
                print(int(t.replace('Cal', '')))
                ccs=colS['Cal'][int(t.replace('Cal', ''))-1]
            else:
                ccs=colS[t]
            # axs[idp_x, idp_y].scatter(ds['Total Injections on Column'][iid], ds.aR[iid], color=ccs, label=t)
            sub1.scatter(ds['rank'][iid], ds.aR[iid], color=ccs, label=t)
            sub1.scatter(ds['rank'][iid], ds.aR[iid], color='black', s=0.7)


        # axs[idp_x, idp_y].legend()
        idp_y += 1

    if shy:
        sub1 = fig.add_subplot(1, 4, 4, sharey=aa)
    else:
        sub1 = fig.add_subplot(1, 4, 4)
    ds = df1[df1.cats.str.contains('^Cal')].copy()
    ds['rank'] = ds['Total Injections on Column'].rank().astype(int)
    sub1.set_yscale('log')
    sub1.yaxis.tick_right()
    sub1.set_xticks([])
    # sub1.set_ylim([0, 5])
    for r, t in enumerate(ds.cats.unique()):
        iid = ds.cats == t
        ccs = colS['Cal'][int(t.replace('Cal', '')) - 1]
        sub1.scatter(ds['rank'][iid], ds.aR[iid], color=ccs, label=t)
        sub1.scatter(ds['rank'][iid], ds.aR[iid], color='grey', s=1.7)
    sub1.set_title('Cal')
    fig.suptitle(title)



# for i in range(len(ar)):
#     id=ar[i][3].replace('RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_', '').replace('.raw', '')
#     id = id.replace('NW_TRP_023_manual_Spiking_05Aug2022_URN_LTR_', '').replace('.raw', '')
#     if 'manual' in ar[i][3]:
#         col='green'
#     else:
#         col='blue'
#     plt.scatter(i, ar[i][2], c=col)
#     plt.annotate(id, (i, ar[i][2]))




# for i in data:
#     plt.plot(i)


autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
rr=tf.convert_to_tensor(data)
history = autoencoder.fit(rr, rr,
          epochs=100,
          batch_size=20,
          # validation_data=(test_data, test_data),
          shuffle=True)

plt.figure(figsize=(5,4))
plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title('F27.1 (Knynate-D5): Autoencoder training')
plt.xlabel('Epoch (batch size 20)')
plt.ylabel('Mean absolute error')
plt.legend()


encoded_data = autoencoder.encoder(rr).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(xnew, decoded_data.T, color='black', linewidth=0.1)
plt.plot(xnew, decoded_data.T[:,0], color='black', linewidth=0.1, label='AE reconstructed signal')

# rp = tf.convert_to_tensor(data)
# reconstructions = autoencoder.predict(rp)
# train_loss = tf.keras.losses.mae(reconstructions, rp)
plt.plot(xnew, np.mean(rr, 0), color='yellow', linewidth=2, label='Avg signal')
plt.plot(xnew, np.mean(rr, 0)+np.std(rr, 0), color='cyan', linewidth=2, label='Avg signal +/- 1 SD')
plt.plot(xnew, np.mean(rr, 0)-np.std(rr, 0), color='cyan', linewidth=2)
plt.plot(xnew, np.mean(decoded_data, 0), color='red', linewidth=2, label='AE consensus peak')
plt.xlabel('Normalised ST')
plt.ylabel('Normalised sum of counts')
plt.legend()

# plt.plot(xnew, np.mean(decoded_data, 0), color='red', linewidth=20)
plt.plot(xnew, rr.numpy().T, color='black', linewidth=0.1)
plt.plot(xnew, np.mean(rr, 0), color='yellow', linewidth=2, label='Avg signal')
plt.plot(xnew, np.mean(rr, 0)+np.std(rr, 0), color='cyan', linewidth=2, label='Avg signal +/- 1 SD')
plt.plot(xnew, np.mean(rr, 0)-np.std(rr, 0), color='cyan', linewidth=2)
plt.plot(xnew, np.mean(decoded_data, 0), color='red', linewidth=2, label='AE consensus peak')

plt.title(f'{fIS}.{sirIS}: {x.efun.funcs[fIS].reactions[str(sirIS+1)]}')
plt.xlabel('Normalised ST')
plt.ylabel('Normalised sum of counts')
plt.legend()

data = []
# sir=1
for i, x in enumerate(ss.exp):
    # x.featplot(x.qs[f][1])
    if i == 0:
        xra = [-0.3, 0.3, 500]
        xnew = np.linspace(*xra)
    if isinstance(x.qs[fA][sirA], list):
        data.append(fnorm(x.qs[fA][sirA], xnew))

data = []
exp1=[]
# sir=1
for i, x in enumerate(ss.exp):
    # x.featplot(x.qs[f][1])
    if i == 0:
        xra = [-0.3, 0.3, 500]
        xnew = np.linspace(*xra)
    if isinstance(x.qs[fIS][sirIS], list):
        exp1.append(x)
        data.append(fnorm(x.qs[fIS][sirIS], xnew))


fig, axs = plt.subplots(2,1)
ssc = (train_loss.numpy() - min(train_loss.numpy())) / (max(train_loss.numpy())- min(train_loss.numpy()))


cm=plt.cm.viridis(ssc)
import matplotlib as mp
for c, i in enumerate(datap):
    axs[0].plot(xnew, i, color=cm[c])
axs[0].title.set_text('Analyte (prediction)')
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
for c, i in enumerate(data):
    axs[1].plot(xnew, i, color=cm[c+len(datap)-1])
axs[1].title.set_text('IS (training)')
fig.subplots_adjust(hspace=0.5)
# cax = plt.axes([0.85, 0.1, 0.075, 0.8])
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
clb=fig.colorbar(sm, ax=axs[0:2])
clb.ax.set_title('N-MAE')

plt.figure(figsize=(6,4))
dp = train_loss.numpy()[0:len(datap)]
plt.scatter(np.arange(len(dp)), dp, c='red', label='prediction analyte')
dt = train_loss.numpy()[len(datap):]
# plt.scatter(np.arange(len(dt))+len(dp), dt, c='blue', label='training IS')
plt.legend()
plt.xlabel('Sample index')
plt.ylabel('Mean Absolute Error')

np.argmax(dp)














# epath = '/Users/torbenkimhofer/Desktop/Torben19Aug.exp'
epath='/Users/TKimhofer/Downloads/Torben19Aug.exp'
dpath='/Volumes/ANPC_ext1/tdata_trp/RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_Cal2_110.raw'
dpath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_PLA_LTR_Cal5_28.raw'
dpath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_SER_LTR_Cal5_65.raw'
dpath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_Cal5_106.mzML'
dpath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_PLA_unhealthy_18.raw'


epath='/Users/torbenkimhofer/Desktop/Torben19Aug.exp'
dpath='/Users/torbenkimhofer/tdata_trp/RCY_TRP_023_Robtot_Spiking_04Aug2022_SER_LTR_52.raw'
dpath='/Users/torbenkimhofer/tdata_trp/RCY_TRP_023_Robtot_Spiking_04Aug2022_PLA_LTR_15.raw'

dpath='/Users/torbenkimhofer/tdata_trp/NW_TRP_023_manual_Spiking_05Aug2022_PLA_LTR_13.raw'

import time
a=time.time()
dpath='/Users/torbenkimhofer/Downloads/test/NW_TRP_023_manual_Spiking_05Aug2022_PLA_LTR_16.raw'
pUH=TrpExp.waters(dpath, efun=ReadExpPars(epath))
b=time.time()
# pUH.qFunction(fid='FUNCTION 22', sir = None, plot=True, **kwargs)
# [x['A'] for x in pUH.qs['FUNCTION 22'][1][1]]
# # 31, 17, 15
c1=time.time()
kwargs= dict(height = 0.1, distance = 10, prominence = 1, width = 7, wlen=9, rel_height=0.7)
pUH.q(**kwargs)
# self=pUH

c=time.time()

import scipy.interpolate as sint
def fnorm(dat):
    iid=np.argmax([x['mu_pp'] for x in dat[1]])
    xn=dat[0]['x']- dat[0]['x'][dat[0]['peaks'][iid]]
    yn = dat[0]['yo']/dat[0]['yo'][dat[0]['peaks'][iid]]
    xnew=np.linspace(-0.38, 0.17, 300)
    fi = sint.interp1d(xn, yn, bounds_error=False, fill_value="extrapolate")
    yi = fi(xnew)
    return yi

def featplot1(ff, axs):
    # rec = {'x': x, 'yo': yo, 'ys': ys, 'bl': baseline, 'peaks': peaks, 'hh': hh, 'ithresh': ithresh} # ybl
    axs[2].text(1.03, 0, self.fname, rotation=90, fontsize=6, transform=axs[2].transAxes)
    pso = -(0.2 * max(ff[0]['yo']))
    pso_up = -(0.1 * max(ff[0]['yo']))
    pso_low = -(0.3 * max(ff[0]['yo']))

    # axs[0].fill_between(x=ff[0]['x'], y1=ff[0]['yo'], color='white', alpha=1, zorder=10)
    axs[0].plot(ff[0]['x'], ff[0]['yo'], c='black', label='ori', zorder=11)
    axs[0].plot(ff[0]['x'], ff[0]['ys'], label='sm', c='gray', linewidth=1, zorder=11)
    axs[0].plot(ff[0]['x'], ff[0]['ybl'], label='sm-bl', c='cyan', linewidth=1, zorder=11)
    axs[0].hlines(0.1, ff[0]['x'][0], ff[0]['x'][-1], color='gray', linewidth=1, linestyle='dashed', zorder=0)
    axs[0].vlines(ff[0]['x'][ff[0]['hh']['left_bases']], pso_low, pso_up, color='gray', zorder=11)
    axs[0].vlines(ff[0]['x'][ff[0]['hh']['right_bases']], pso_low, pso_up, color='gray', zorder=11)
    axs[0].vlines(ff[0]['x'][ff[0]['hh']['left_ips']], 0, ff[0]['hh']['width_heights'], color='gray', linewidth=1,
                  linestyle='dotted', zorder=11)
    axs[0].vlines(ff[0]['x'][ff[0]['hh']['right_ips']], 0, ff[0]['hh']['width_heights'], color='gray', linewidth=1,
                  linestyle='dotted', zorder=11)

    # axs[0].scatter(x[peaks], hh['peak_heights'], c='red')
    lyo = len(ff[0]['x'])
    cols = plt.get_cmap('Set1').colors
    ci = 0
    for pi, p in enumerate(ff[0]['peaks']):
        # axs[0].annotate(round(ff[0]['hh']['prominences'][pi], 1), (ff[0]['x'][p], ff[0]['hh']['peak_heights'][pi]),
        #                 textcoords='offset pixels', xytext=(-4, 10), rotation=90, zorder=12)
        peak_width = round(ff[0]['hh']['widths'][pi] / 2)
        idx_left = max([0, p - peak_width])
        idx_right = min([lyo - 1, p + peak_width])
        axs[0].hlines(pso, ff[0]['x'][idx_left], ff[0]['x'][idx_right], color=cols[ci])

        axs[1].plot(ff[0]['x'], ff[0]['ycomps'][pi], color=cols[ci], linewidth=1)
        axs[1].fill_between(x=ff[0]['x'], y1=ff[0]['ycomps'][pi], color=cols[ci], alpha=0.4)
        ci += 1
        if ci >= len(cols):
            ci = 0
    axs[2].plot(ff[0]['x'], ff[0]['yo'] - ff[0]['yest'], c='black')
    axs[0].scatter(ff[0]['x'][ff[0]['peaks']], np.repeat(pso, len(ff[0]['peaks'])), c='black', s=20)
    axs[0].scatter(ff[0]['x'][ff[0]['peaks']], np.repeat(pso, len(ff[0]['peaks'])), c='white', s=5, zorder=10)

    axs[1].plot(ff[0]['x'], ff[0]['yo'], c='black', label='ori')
    axs[1].plot(ff[0]['x'], np.sum(ff[0]['ycomps'], 0), label='psum', c='orange')

fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.4]})
r = 0
ar={}
f='FUNCTION 5'
c=2
yn = []


fig, ax =plt.subplots(1, 1)
ax.plot(df1.dt, df1['Average System Pressure'].astype(float))
ax.plot(df1.dt, df1['Average System Pressure'].astype(float))
ax.plot(df1.dt, df1['Minimum System Pressure'].astype(float))
ax.plot(df1.dt, df1['Maximum System Pressure'].astype(float))
df1['catcode']=df1.cats.astype('category').cat.codes
cols=plt.cm.tab20.colors + plt.cm.tab20b.colors
cc=[cols[i] for i in df1.catcode.values]
# ax.scatter(df1.dt, [5000]*df1.shape[0], color=cc, s=2)
for i in df1.catcode.unique():
    idx=df1.catcode==i
    ax.scatter(df1.dt[idx], [5000]*len(df1.dt[idx]), color=cc[i], label=df1.cats[idx].iloc[0])
fig.legend()

def cat(x):
    if bool(re.findall('22_DB_[0-9].*', x)):
        return 'DB'
    elif bool(re.findall('22_Cal[0-9]_[0-9].*', x)):
        return x.split('_')[-2]
    elif bool(re.findall('22_PLA_unhealthy_[0-9].*', x)):
        return 'unhealthyPLA'
    elif bool(re.findall('22_SER_unhealthy_[0-9].*', x)):
        return 'unhealthySER'
    elif bool(re.findall('22_URN_unhealthy_[0-9].*', x)):
        return 'healthyURN'
    elif bool(re.findall('22_PLA_healthy_[0-9].*', x)):
        return 'healthyPLA'
    elif bool(re.findall('22_SER_healthy_[0-9].*', x)):
        return 'healthySER'
    elif bool(re.findall('22_URN_healthy_[0-9].*', x)):
        return 'healthyURN'
    elif bool(re.findall('22_PLA_LTR_[0-9].*', x)):
        return 'LTR_PlA'
    elif bool(re.findall('22_SER_LTR_[0-9].*', x)):
        return 'LTR_SER'
    elif bool(re.findall('22_URN_LTR_[0-9].*', x)):
        return 'LTR_URN'


    elif bool(re.findall('22_PLA_unhealthy_Cal[0-9]_[0-9].*', x)):
        return 'unhealthyPLA_spikeIn_'+x.split('_')[-2]
    elif bool(re.findall('22_SER_unhealthy_Cal[0-9]_[0-9].*', x)):
        return 'unhealthyURN_spikeIn_'+x.split('_')[-2]
    elif bool(re.findall('22_URN_unhealthy_Cal[0-9]_[0-9].*', x)):
        return 'unhealthyURN_spikeIn_'+x.split('_')[-2]
    elif bool(re.findall('22_PLA_healthy_Cal[0-9]_[0-9].*', x)):
        return 'healthyPLA_spikeIn_' + x.split('_')[-2]
    elif bool(re.findall('22_SER_healthy_Cal[0-9]_[0-9].*', x)):
        return 'healthyURN_spikeIn_' + x.split('_')[-2]
    elif bool(re.findall('22_URN_healthy_Cal[0-9]_[0-9].*', x)):
        return 'healthyURN_spikeIn_' + x.split('_')[-2]

    elif bool(re.findall('22_PLA_LTR_Cal[0-9]_[0-9].*', x)):
        return 'LTR_PlA_spikeIn_'+x.split('_')[-2]
    elif bool(re.findall('22_SER_LTR_Cal[0-9]_[0-9].*', x)):
        return 'LTR_SER_spikeIn_'+x.split('_')[-2]
    elif bool(re.findall('22_URN_LTR_Cal[0-9]_[0-9].*', x)):
        return 'LTR_URN_spikeIn_'+x.split('_')[-2]

    # elif bool(re.findall('22_SER_URN_unhealthy_Cal[0-9]_[0-9].*', x)):
    #     return 'unhealthyURN_spikeIn_'+x.split('_')[-2]
    # elif bool(re.findall('22_URN_LTR_[0-9].*', x)):
    #     return 'LTR_URN'
    # elif bool(re.findall('22_URN_LTR_Cal[0-9]_[0-9].*', x)):
    #     return 'LTR_URN_spikeIn_'+x.split('_')[-2]
    else:
        return 'unassigned'

# extract metabolite data of calibration samples for one transition of IS
s='Cal2'
idxS=df1.ind.iloc[np.where(df1.cats == s)[0][0]]

f='FUNCTION 5' # piclolinic acid
x=ss.exp[0]
x.fname
x.featplot(x.qs[f][1])
pd.DataFrame(x.qs[f][1][1])
out=[]
for i in df1.ind.values:
    if isinstance(test.exp[i].qs[f][0][1], list):
        for k, s in enumerate(test.exp[i].qs[f][0][1]):
            test.exp[i].qs[f][0][1][k].update({'fname': test.exp[i].fname})
        out.append(test.exp[i].qs[f][0][1])


idxS = df1.ind.iloc[np.where(df1.cats == s)].values

out=[j for j in test.exp[i].qs[f][1][0] for i in df1.ind.values]
flat_list = [item for sublist in out for item in sublist]
df=pd.DataFrame(flat_list)

df1.index=df1.nam
df.index=df.fname
ds=df1.join(df)

dd=ds[ds.cats.str.contains('^Cal') & (ds['Total Injections on Column'].astype(int) < 1100)]
dd=dd.sort_values('dt')
plt.scatter(dd['Total Injections on Column'].astype(int), dd['A'], c=dd['Sample Description'].astype('category').cat.codes)
for i in range(dd.shape[0]):
    x=dd['Total Injections on Column'].astype(int).iloc[i]
    y=dd['A'].iloc[i]
    plt.annotate(dd['Sample Description'].iloc[i], (float(x), y))
plt.xlabel('Total Injections on Column')
plt.ylabel('Integral')
plt.title(f+' SIR 1')
plt.subplots()



i=53
test.exp[i].qs[f][0][1]
test.exp[i].p
i=150
test.exp[i].featplot(test.exp[i].qs[f][0])
test.exp[i].qs[f][0][1]

test=dd[['Sample Description', 'nam']]

# print(len(test.exp[i].qs[f]))
# find consensus signals using IS, then compare analyte peaks with consensus signal
f='FUNCTION 28'
stype='LTR_URN'
transition=1
sidx=df1[df1.cats == stype].ind.values



for i in range(len(test.exp)):
    pUH=test.exp[i]
    if isinstance(pUH.qs[f][r], list):
        yn.append(fnorm(pUH.qs[f][r]))
    try:
        out = fnorm(pUH.qs[f][r])
        if not any(np.isnan(out)):
            yn.append(fnorm(pUH.qs[f][r]))
            c += 1
        # featplot1(pUH.qs[f][r], axs)
        # ar[pUH.fname] = max([x['A'] for x in pUH.qs[f][r][1]])


    except:
        pass
print(c)
fig.suptitle(f'{f}.{r}\n{pUH.efun.funcs[f].reactions[str(r+1)].__repr__().rstrip()}, n={c}/{len(test.exp)}')

# consesnus:
# extract data, align data based on IS signal, minmax scaling - train autoencoder, calc similarity


for i in yn:
    plt.plot(i)

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(128, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(300, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
rr=tf.convert_to_tensor(yn)
history = autoencoder.fit(rr, rr,
          epochs=100,
          batch_size=20,
          # validation_data=(test_data, test_data),
          shuffle=True)

plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


encoded_data = autoencoder.encoder(rr).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(xnew, decoded_data.T)

reconstructions = autoencoder.predict(rr)
train_loss = tf.keras.losses.mae(reconstructions, rr)
# plt(train_loss)
plt.scatter(np.arange(len(rr)), train_loss.numpy())
plt.xlabel('Sample index')
plt.ylabel('Mean absolute error (AE)')

imin = np.argsort(train_loss.numpy())[2]
plt.plot(xnew, yn[imin], label='SIC')
plt.plot(xnew, reconstructions[imin], label=f'Autoencoder reconstruction\n(MAE={round(train_loss.numpy()[imin], 2)})')
plt.legend()

imax = np.argmax(train_loss.numpy())
plt.plot(xnew, yn[imax], label='SIC')
plt.plot(xnew, reconstructions[imax], label=f'Autoencoder reconstruction\n(MAE={round(train_loss.numpy()[imax], 2)})')
plt.legend()

dat=pUH.qs[f][r]
plt.plot(dat[0]['yo'])


plt.plot(xn, yn)
plt.plot(xnew, yi)





dd =pd.DataFrame(ar, index=[0]).T
dd.index.str.contains('manual')
c=['blue']*dd.shape[0]
cols=['blue' if 'manual' in x else 'red' for x in dd.index]

dd[dd.index.str.contains('manual|')].std()
plt.scatter(range(len(cols)), dd[0], c=cols)
plt.xlabel('index')
plt.ylabel('Area')
# axs[0].legend()
plt.suptitle(f"{f}\n{pUH.efun.funcs[f].reactions[str(r)]}")

{'a1': {'std': {'FUNCTION 5': 'Picolinic acid-D3'},
       'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
'a2': {'std': {'FUNCTION 6': 'Nicotinic acid-D4'},
       'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
'a3': {'std': {'FUNCTION 10': '3-HAA-D3'}, 'analyte': {'FUNCTION 8': '3-HAA'}},
'a4': {'std': {'FUNCTION 11': 'Dopamine-D4'}, 'analyte': {'FUNCTION 9': 'Dopamine'}},
'a5': {'std': {'FUNCTION 20': 'Serotonin-d4'}, 'analyte': {'FUNCTION 12': 'Serotonin'}},
'a6': {'std': {'FUNCTION 14': 'Tryptamine-d4'}, 'analyte': {'FUNCTION 13': 'Tryptamine'}},
'a7': {'std': {'FUNCTION 17': 'Quinolinic acid-D3'}, 'analyte': {'FUNCTION 15': 'Quinolinic acid'}},
'a8': {'std': {'FUNCTION 19': 'I-3-AA-D4'}, 'analyte': {'FUNCTION 18': 'I-3-AA'}},}

#prominence should be more than 5% of max prominence

# df=pUH.extractData(f)
# x=df['d'][0][1]
# y=df['d'][0][2]
# ff=pUH.featquant(x, y, f, 0, height = 0.1, distance = 1, prominence = 0.2, width = 3, wlen = 17)
# pUH.featplot(ff)
# plt.plot(x, y)

dpath='/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_PLA_unhealthy_Cal2_42.raw'
pUHcal2=TrpExp.waters(dpath, efun=ReadExpPars(epath))
pUHcal2.q(height = 0.1, distance = 1, prominence = 0.2, width = 3, wlen = 17)
pUHcal2.qFunction('FUNCTION 2', sir='2', height = 0.1, distance = 1, prominence = 0.2, width = 3, wlen = 17)
pUHcal2.qFunction('FUNCTION 2', sir='1', height = 0.1, distance = 1, prominence = 0.2, width = 3, wlen = 17)
pUHcal2.qFunction('FUNCTION 2', sir=None, height = 0.1, distance = 1, prominence = 0.2, width = 3, wlen = 17)



s=pUHcal2.qs['FUNCTION 3'][1][1]
pUH.qs['FUNCTION 1'][1][1]
pUHcal2.qs['FUNCTION 1'][1][0][0:3]
pUH.qs['FUNCTION 1'][1][0][0:3]

pUHcal2.qs['FUNCTION 1'][2][0][0:3]
pUH.qs['FUNCTION 1'][2][0][0:3]


pUHcal2.qs['FUNCTION 3'][0][0][0:3]
pUH.qs['FUNCTION 3'][0][0][0:3]


# build calibration curve
fh = ['/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal1_11.raw',
'/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal2_10.raw',
'/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal3_9.raw',
'/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal4_8.raw',
'/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal6_6.raw',
'/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal7_5.raw',
'/Volumes/ANPC_ext1/trp_qc/RCY_TRP_023_Robtot_Spiking_04Aug2022_Cal8_4.raw']


res= []
for f in fh:
    pobj = TrpExp.waters(f, efun=ReadExpPars(epath))
    pobj.quant(height=0.1, distance=1, prominence=0.2, width=3, wlen=17, plot=False)
    res.append(pobj)

# df=test.extractData('FUNCTION 3')

import scipy.signal as ss
ra=[]
for i,f in enumerate(test.efun.funcs.keys()):
    if '38' in f:
        continue
    f = list(test.efun.funcs.keys())[i]
    # f = 'FUNCTION 1'
    df = test.extractData(f)
    print(f)
    for l, r in enumerate(df['d']):
        # l=1
        # r = df['d'][l]
        x=r[1]
        y=r[2]
        self.featquant(x, y, height=0.1, distance=5, prominence=0.1, width=1)
        # plt.plot(x,y)
        res=test.ppick(x, y, plot=True, height=0.1, distance=1, prominence=0.0001, width=3, wlen=17)
        # res=ppick(x, y, plot=True, height=0.1, distance=1, prominence=0.0001, width=3, wlen=17)
        ra.append(res)
        if isinstance(res, list):
            plt.suptitle(f+'.'+str(l) +' : success: '+str(res[0][0])+ ', res: '+str(round(res[0][1])))

res={}
for i, r in enumerate(ra):
    if isinstance(r, list):
        res[i]={'success': r[0][0], 'res': r[0][1], 'area_comp': r[0][2], 'comp_loc': r[0][4]}

pd.DataFrame(res)


# ar = {'height': 0.1, 'distance': 1, 'prominence': 0.1, 'width': 5}
# i=14 # broad and ragged, strong right tail height=0.1, distance=1, prominence=0.2, width=5
# i=15  height=0.1, distance=1, prominence=0.01, width=3
# i 16 ragged and strong right tails
# i=17,18, 24 right shoulder
# i=19, sencond srm has smaller left signal that is not detected
# 27 smaller right peak not detected
# 22: high baseline troughout and no peak in third srm
# i = 32 # borad ragged

e=0
add = []
for i in range(1, len(g)):
    if np.sign(g[i - 1]) != np.sign(g[i]) and g[i] > 0:
        add.append((e - g[i]))
        e=g[i]


plt.plot(x, y)





# define experiment set for Trp data (import series of trp experiments)
# plot functions, QC and quantify signals
class ExpSet:
    def __init__(self, path='/Volumes/ANPC_ext1/trp_qc/', pat='22_URN_LTR'):
        import re, os
        import numpy as np
        # find all waters experiment files
        # try to annotate sample type (LTR vs Cal vs Sample)
        self.exp= []
        self.pat = pat
        # self.calib = []
        self.df = {}
        self.areas = {}
        c = 0
        for d in os.listdir(path):
            if bool(re.findall(".*raw$", d)) & bool(re.findall(self.pat, d)):
            #     print(d)
                try:
                    c += 1
                    efile = Trp.waters(os.path.join(path, d), efun=efun, convert=False)
                    if '2022_Cal' in efile.fname:
                        efile.stype = 'Calibration'
                        efile.cpoint = efile.fname.split('_')[-2]
                        self.exp.append(efile)
                        self.df.update({efile.fname: {'stype': efile.stype , 'cpoint': efile.cpoint, 'dt': efile.aDt}})
                    else:
                        self.exp.append(efile)
                        efile.stype = 'Sample'
                        self.df.update({efile.fname: {'stype': efile.stype,  'dt': efile.aDt}})
                    self.areas[efile.fname] = {}
                    for f in efile.efun.funcs:  # for every function
                        id = f.replace('FUNCTION ', '')
                        try:
                            efDat = efile.extractFunData(id)
                            for i, xd in enumerate(efDat['d']):
                                # xd = fd
                                try:
                                    self.areas[efile.fname].update({f + '_' + str(i): pbound(xd, plot=False)})
                                except:
                                    pass
                        except:
                            pass

                except:
                    print(f'failed for file {d}')

        print(f'Imported a total of {c} files')

        dt = [x['dt'] for k, x in self.df.items()]
        ro = np.argsort(dt)

        dd = pd.DataFrame(self.df).T
        dd['RunOrder'] = ro
        dd['Sample'] = dd.index
        dd = dd[['RunOrder', 'Sample', 'stype', 'cpoint', 'dt']]
        self.exp = [self.exp[i] for i in ro]
        dd = dd.sort_values('RunOrder')
        self.df = dd

    def vis(self):
        import plotly.graph_objects as go
        fig = go.Figure()
        areas = {}
        for rec in self.exp:
            areas[rec.fname] = {}
            for f in rec.efun.funcs:  # for every function
                id = f.replace('FUNCTION ', '')
                try:
                    efDat = rec.extractFunData(id)
                    for i, fd in enumerate(efDat['d']):
                        print(i)
                        xd = fd
                        try:
                            areas[rec.fname].update({f + '_' + str(i): pbound(xd, plot=False)})
                        except:
                            pass
                except:
                    pass
