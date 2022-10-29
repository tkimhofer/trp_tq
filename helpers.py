import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import re

def matchC(rec, fis, fa, sirIS, sirA, locDev=60/30, sumC=False, maxi=False):
    # for a given SIC, match peak integral(s) of internal standard with integral(s) of respective analyte
    # locDev: allowed deviation in scan time dimension (10%->6 sec)
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

def SampleList(x): # cat
    # generate sample type from file names
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
    # collect system metadata
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


def fnorm(dat, xnew):
    # axis normalisation (peak centering and intensity scaling)
    import scipy.interpolate as sint
    iid=np.argmax([x['A'] for x in dat[1]])
    xn=dat[0]['x']- dat[1][iid]['mu_decon']
    yn = dat[0]['yo']/dat[1][iid]['A']
    fi = sint.interp1d(xn, yn, bounds_error=False, fill_value="extrapolate")
    yi = fi(xnew)
    return yi / max(yi)

def vis(df1, title, fsize=(9, 6), shy=True, ymax=None):
    # vis(df1,
    #     title=f"{rec.efun.funcs[fA].reactions[str(sirA + 1)]} with {rec.efun.funcs[fIS].reactions[str(sirIS + 1)]}", \
    #     shy=True, ymax=None)
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

