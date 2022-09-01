import matplotlib.pyplot as plt

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
  
ss=Eset.importbinary(dpath='/Users/torbenkimhofer/tdata_trp/', epath='/Users/torbenkimhofer/Desktop/Torben19Aug.exp', pat='URN', n=100)
# ss=Eset.imp(dpath='/Users/torbenkimhofer/tdata_trp/', epath='/Users/torbenkimhofer/Desktop/Torben19Aug.exp', pat='', n=1000)

# ss.exp[0]
# f='FUNCTION 5' # piclolinic acid
# x=ss.exp[5]
# x.fname
# x.featplot(x.qs[f][1])
#
# kwargs = dict(height=0.1, distance=10, prominence=1, width=7, wlen=9, rel_height=0.7)
# x.qFunction(f, 2, plot=True, **kwargs, )
i=2
fIS=list(ss.exp[i].fmap['a9']['std'].keys())[0]
fA=list(ss.exp[i].fmap['a9']['analyte'].keys())[0]
ss.exp[i].featplot(ss.exp[i].qs[fIS][1])
ss.exp[i].featplot(ss.exp[i].qs[fA][1])

def getA(dat):
    iid=np.argmax([x['A'] for x in dat[1]])
    return dat[1][iid]['A']

getA(x.qs[f][sir])
11,9
20,12
14,13
17, 15

f='FUNCTION 11' # piclolinic acid
fa='FUNCTION 9' # piclolinic acid
f=fIS
fa=fA
data = []
sirIS=0
sirA=1
ar = []
for i, x in enumerate(ss.exp):
    # x.featplot(x.qs[f][1])
    if i==0:
        xra = [-0.3, 0.3, 500]
        xnew = np.linspace(*xra)
    if isinstance(x.qs[f][sirIS], list):
        data.append(fnorm(x.qs[f][sirIS], xnew))
    try:
        ais = getA(x.qs[f][sirIS])
        aa = getA(x.qs[fa][sirA])
        ar.append((aa, ais, aa/ais, x.fname))
    except:
        pass

for i in range(len(ar)):
    id=ar[i][3].replace('RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_', '').replace('.raw', '')
    id = id.replace('NW_TRP_023_manual_Spiking_05Aug2022_URN_LTR_', '').replace('.raw', '')
    if 'manual' in ar[i][3]:
        col='green'
    else:
        col='blue'
    plt.scatter(i, ar[i][2], c=col)
    plt.annotate(id, (i, ar[i][2]))

for i in data:
    plt.plot(i)


autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
rr=tf.convert_to_tensor(data)
history = autoencoder.fit(rr, rr,
          epochs=100,
          batch_size=20,
          # validation_data=(test_data, test_data),
          shuffle=True)

# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()


encoded_data = autoencoder.encoder(rr).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


rp = tf.convert_to_tensor(datap+data)
reconstructions = autoencoder.predict(rp)
train_loss = tf.keras.losses.mae(reconstructions, rp)

# plt.plot(xnew, np.mean(decoded_data, 0), color='red', linewidth=20)
plt.plot(xnew, rr.numpy().T, color='black', linewidth=1)
plt.plot(xnew, np.mean(decoded_data, 0), color='red', linewidth=2, label='Autoencoder')

plt.legend()

datap = []
# sir=1
for i, x in enumerate(ss.exp):
    # x.featplot(x.qs[f][1])
    if i == 0:
        xra = [-0.3, 0.3, 500]
        xnew = np.linspace(*xra)
    if isinstance(x.qs[fa][sirA], list):
        datap.append(fnorm(x.qs[fa][sirA], xnew))


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


dp = train_loss.numpy()[0:len(datap)]
plt.scatter(np.arange(len(dp)), dp, c='red', label='prediction analyte')
dt = train_loss.numpy()[len(datap):]
plt.scatter(np.arange(len(dt))+len(dp), dt, c='blue', label='training IS')
plt.legend()
plt.xlabel('Sample index')
plt.ylabel('Mean Absolute Error')
















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

nam = [x.fname for x in test.exp]
cats=[cat(x.fname) for x in test.exp]
df=pd.DataFrame({'cats': cats, 'nam': nam})
df['cats'].value_counts()
rep=df[df['cats']=='2']['nam'].str.split('_').str[-3]
df.loc[df['cats']=='2', 'cats']=rep
df['dt'] = [x.aDt for x in test.exp]

add=['Average System Pressure', 'Minimum System Pressure', 'Maximum System Pressure', 'Total Injections on Column', 'Sample Description', 'Bottle Number']

meta=pd.DataFrame([x.edf for x in test.exp])
df1=pd.concat([df, meta[add]], axis=1)
df1['ind']=np.arange(df1.shape[0])
df1=df1.sort_values('dt')

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
f='FUNCTION 5'
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