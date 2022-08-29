import os, pickle, re, time
import datetime as dt
import docker as dock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize
import scipy.stats as st
import scipy.signal as ss
import sys
import xml.etree.cElementTree as ET

# import scipy.signal as ss


sys.path.append('/Users/torbenkimhofer/PycharmProjects/pyms')
from msmate.helpers import _children, _get_obo, _node_attr_recurse, _collect_spectra_chrom

class SirFunction:
    '''Reads/defines single ion reaction defined in a function (targeted assay)
    Subclass of Mrm
    Input:
        d: part of a dictionary of exp parameters exptracted from .exp waters parameterisation file
    '''
    def __init__(self, d):
        self.massPrecursor = d['SIRMass']
        self.massProduct = d['SIRMass_2_']

        if (self.massPrecursor != d["Mass(amu)_"]) | (self.massProduct != d["Mass2(amu)_"]):
            raise Warning('diverging SIR and amu masses')
        self.autoDwell = d['SIRAutoDwell']
        self.autoDwellTime = d['SIRAutoDwell']
        self.delay = d['SIRDelay']
        self.useAsLockMass = d['UseAsLockMass']
        self.dwell_s = d['Dwell(s)_']
        self.conevoltageV = d['ConeVoltage(V)_']
        self.collisionEnergy = d['CollisionEnergy(V)_']
        self.interchanneldelay = d['InterChannelDelay(s)_']
        self.compoundName = d['CompoundName_']
        self.compoundFormula = d['CompoundFormula_']

    def __repr__(self):
        stf = f"{self.compoundName}: {self.massPrecursor} m/z ---> {self.massProduct} m/z\n"
        return stf

class Mrm:
    '''Reads/defines functions from .exp waters parameterisation file  (targeted assay)
        Subclass of ReadExpPars
       Input:
           d: dictionary of exp parameters exptracted from .exp waters parameterisation file
       '''
    # defines SIRs and parameters of multiple reaction monitoring for targeted assay
    def __init__(self, d):
        self.ftype = d['FunctionType']
        self.polarity = d['FunctionPolarity']
        self.stStart_min = d['FunctionStartTime(min)']
        self.stEnd_min = d['FunctionEndTime(min)']

        self.reactions = {k: SirFunction(s) for k, s in d['sim'].items()}
        self.name = '/'.join(list(set([dd.compoundName for k, dd in self.reactions.items()])))

        self.other = d

    def __repr__(self):
        ssr = "\n\t".join([f"{r.massPrecursor} m/z ---> {r.massProduct} m/z" for k, r in self.reactions.items()])
        return f"{self.name}\n\t" + ssr + '\n'

class ReadExpPars:
    '''reads Waters .exp file, containing function information, eg. compound name
           Input:
               epath: path to .exp file
    '''
    # reads Waters .exp file, containing function information, eg. compound name
    # function data for targeted assay, comprising of individuals MRM functions
    def __init__(self, epath):
        import re
        pars = {}
        with open(epath, 'rb') as f:
            for t in f.readlines():
                td = t.strip().decode('latin1')
                if bool(td.isupper()):
                    feat = td
                    pars[feat] = {}
                    pars[feat]['sim'] = {}
                    continue
                if ',' in td:
                    add = td.split(',')
                    if bool(re.findall('[0-9]$', add[0])) and 'FUN' in feat:
                        key, featid, _ = re.split('([0-9])$', add[0])
                        if featid not in pars[feat]['sim']:
                            pars[feat]['sim'][featid] = {}
                        pars[feat]['sim'][featid][key] = add[1]
                    else:
                        pars[feat][add[0]] = add[1]

        self.generalInfo = pars['GENERAL INFORMATION']
        self.funcs = {k: Mrm(x) for k, x in pars.items() if 'FUN' in k}

    def __repr__(self):
        return f"{len(self.funcs)} functions"

class ReadWatersRaw:
    '''Import a single Waters MRM experiment file (.raw). Import is done dynamically depending on which files are available.
       Import priority: 1. mm8 binary file within .raw experiment folder 2. mzml file in same parent folder as .raw 3. vendor file (.raw). Vendor files are converted using official
       msconvert docker image. Imported are also _INLET and _Header files (in .raw directory) as these contain experimental information
       that is not in mzml file. Therefore, it is important to keep both, .mzml and .raw files in the same parent folder.
               Input:
                   dpath: path to .raw experiment file
                   docker: dict with key 'repo' describing docker image:tag
                   convert: bool, if true ignores any mzml and mm8 binary and starts converting .raw to mzml to binary
        '''
    def __init__(self, dpath: str, docker: dict = {'repo': 'chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest'}, convert:bool = False):
        self.mslevel = 'MRM Waters'
        self.convert = convert
        self.docker = {'repo': docker['repo']}

        if bool(re.findall("mzML$", dpath)):
            self.mzmlfn = os.path.abspath(dpath)
            self.dpath = os.path.abspath(re.sub('mzML$', 'raw', dpath))
        elif bool(re.findall("raw$", dpath)):
            self.dpath = os.path.abspath(dpath)
            self.mzmlfn = os.path.abspath(re.sub('raw$', 'mzML', dpath))
        self.msmfile = os.path.join(re.sub('mzML$', 'raw', self.dpath), f'mm8v3_edata.p')
        self.fname = os.path.basename(self.dpath)

        self.edf = {} # experiment pd.dataframe
        try:
            f1 = os.path.join(self.dpath, '_INLET.INF')
            iadd = self._pInlet(f1)
            self.edf.update(iadd)
        except:
            print('_INLET file not found')
        try:
            f1 = os.path.join(self.dpath, '_HEADER.TXT')
            hadd = self._pInlet(f1)
            self.edf.update(hadd)
        except:
            print('_HEADER file not found')

        self.aDt = dt.datetime.strptime(self.edf['Acquired Date']+self.edf['Acquired Time'], ' %d-%b-%Y %H:%M:%S')
        self._importData()

    def _importData(self):
        if not self.convert:
            if os.path.exists(self.msmfile):
                self._readmsm8()
            elif os.path.exists(self.mzmlfn):
                self._read_mzml()
                self._createSpectMat()
                self._savemsm8()
            else:
                self._convoDocker()
                self._read_mzml()
                self._createSpectMat()
                self._savemsm8()
        else:
            self._convoDocker()
            self._read_mzml()
            self._createSpectMat()
            self._savemsm8()

    def _readmsm8(self):
        try:
            expmm8 = pickle.load(open(self.msmfile, 'rb'))
            self.edf = expmm8['edf']
            self.dfd = expmm8['dfd']
            self.xrawd = expmm8['xrawd']
        except:
            raise ValueError('Can not open mm8 binary')

    def _savemsm8(self):
        try:
            with open(self.msmfile, 'wb') as fh:
                pickle.dump({'dfd': self.dfd, 'xrawd': self.xrawd, 'edf': self.edf}, fh)
        except:
            print('Cant save mm8 experiment file - check disk write permission')

    def _convoDocker(self):
        """Convert Bruker 2D MS experiment data to msmate file/obj using a custom build Docker image"""
        t0 = time.time()
        client = dock.from_env()
        client.info()
        client.containers.list()
        # img = [x for x in client.images.list() if self.docker["repo"] in x.tags]
        img = [x for x in client.images.list() if self.docker["repo"] in x.tags]
        if len(img) < 1:
            raise ValueError('Image not found')
        ec = f'docker run -i --rm -e WINEDEBUG=-all -v "{os.path.dirname(self.dpath)}":/data {self.docker["repo"]} wine msconvert "/data/{self.fname}"'
        self.ex = ec
        self.rrr = os.system(ec)
        t1 = time.time()
        print(f'Conversion time: {round(t1 - t0)} sec')
        self.mzmlfn = re.sub('raw$', 'mzML', self.dpath)
        self.fpath = re.sub('raw$', 'mzML', self.dpath)

    def _read_mzml(self):
        """Extracts MS data from mzml file"""
        # this is for mzml version 1.1.0
        # schema specification: https://raw.githubusercontent.com/HUPO-PSI/mzML/master/schema/schema_1.1/mzML1.1.0.xsd
        # read in data, files index 0 (see below)
        tree = ET.parse(self.mzmlfn )
        root = tree.getroot()
        child = _children(root)
        imzml = child.index('mzML')
        mzml_children = _children(root[imzml])
        obos = root[imzml][mzml_children.index('cvList')]  # controlled vocab (CV
        self.obo_ids = _get_obo(obos, obo_ids={})
        seq = np.where(~np.isin(mzml_children, ['cvList', 'run']))[0]
        pp = {}
        for j in seq:
            filed = _node_attr_recurse(s=root[imzml][j], d=4, c=0, ii=[])
            dn = {}
            for i in range(len(filed)):
                dn.update(
                    dict(zip([filed[i]['path'] + '_' + re.sub('\{.*\}', '', x) for x in list(filed[i].keys())[1:]],
                             list(filed[i].values())[1:])))
            pp.update({mzml_children[j]: dn})
        run = root[imzml][mzml_children.index('run')]
        self.out31 = _collect_spectra_chrom(s=run, ii={}, d=20, c=0, flag=self.mslevel, tag='', obos=self.obo_ids)

    def prep_df(self, df):
        if 'MS:1000127' in df.columns:
            add = np.repeat('centroided', df.shape[0])
            add[~(df['MS:1000127'] == True)] = 'profile'
            df['MS:1000127'] = add
        if 'MS:1000128' in df.columns:
            add = np.repeat('profile', df.shape[0])
            add[~(df['MS:1000128'] == True)] = 'centroided'
            df['MS:1000128'] = add

        df = df.rename(
            columns={'UO:0000010': 'time_unit', 'UO:0000031': 'time_unit', 'defaultArrayLength': 'n',
                     'MS:1000129': 'polNeg', 'MS:1000130': 'polPos',
                     'MS:1000127': 'specRepresentation', 'MS:1000128': 'specRepresentation',
                     'MS:1000505': 'MaxIntensity', 'MS:1000285': 'SumIntensity',
                     'MS:1000016': 'Rt'
                     })
        df.columns = [self.obo_ids[x]['name'].replace(' ', '_') if x in self.obo_ids.keys() else x for x in
                      df.columns.values]
        df['fname'] = self.dpath
        return df

    @staticmethod
    def _pInlet(ipath):
        # extract inlet and header information
        pars = {}
        prefix = ''
        with open(ipath, 'rb') as f:
            for t in f.readlines():
                td = t.strip().decode('latin1')
                if bool(re.findall('^-- ', td)):
                    if not 'END' in td:
                        prefix = td.replace('-', '').lstrip().rstrip() + '_'
                    else:
                        prefix = ''

                if ':' in td:
                    add = td.replace('$$ ', '').split(':', 1)
                    pars[prefix + add[0]] = add[1]

        return pars

    def _createSpectMat(self):
        # tyr targeted data has two dimensions: 1. rt, 2. intensity
        """Organise raw MS data and scan metadata"""

        def srmSic(id):
            ss = re.split('=| ', id)
            return {'q1': ss[-7], 'q3': ss[-5], 'fid': ss[-3], 'offset': ss[-1]}


        sc_msl1 = [(i, len(x['data']['Int']['d'])) for i, x in self.out31.items() if
                   len(x['data']) == 2]  # sid of ms level 1 scans

        self.df = pd.DataFrame([x['meta'] for i, x in self.out31.items() if len(x['data']) == 2])
        self.df['defaultArrayLength'] = self.df['defaultArrayLength'].astype(int)

        from collections import defaultdict, Counter
        xrawd = {}
        if 'MS:1000129' in self.df.columns:
            nd = self.df['defaultArrayLength'][self.df['MS:1000129'] == True].sum()
            xrawd['1N'] = np.zeros((4, nd))

        if 'MS:1000130' in self.df.columns:
            nd = self.df['defaultArrayLength'][self.df['MS:1000130'] == True].sum()
            xrawd['1P'] = np.zeros((4, nd))

        row_counter = Counter({'1P': 0, '1N': 0})
        sid_counter = Counter({'1P': 0, '1N': 0})
        dfd = defaultdict(list)
        for i, s in enumerate(sc_msl1):
            d = self.out31[s[0]]
            if s[1] != self.df['defaultArrayLength'].iloc[i]:
                raise ValueError('Check data extraction')
            cbn = [[d['meta']['index']] * s[1]]
            for k in d['data']:
                cbn.append(d['data'][k]['d'])
            if 'MS:1000129' in d['meta']:
                fstr = '1N'
            elif 'MS:1000130' in d['meta']:
                fstr = '1P'
            cbn.append([sid_counter[fstr]] * s[1])
            add = np.array(cbn)
            xrawd[fstr][:, row_counter[fstr]:(row_counter[fstr] + s[1])] = add
            dfd[fstr].append(d['meta'])
            sid_counter[fstr] += 1
            row_counter[fstr] += (s[1])

        for k, d, in dfd.items(): # for each polarity
            for j in range(len(d)): # for each sample

                out = srmSic(d[j]['id'])
                dfd[k][j].update(out)

            dfd[k] = self.prep_df(pd.DataFrame(dfd[k]))

        self.dfd = dfd
        self.xrawd = xrawd
        # self.ms0string = row_counter.most_common(1)[0][0]
        # self.ms1string = None

class TrpExp:
    '''Import TQX Trp data (.raw) with ReadWatersRaw, define utility functions for plotting and peak detection/quantification
            '''
    assay = 'Quant of tryptophane pathway-related compounds'

    @classmethod
    def waters(cls,  dpath: str, docker: dict = {'repo': 'chambm/pwiz-skyline-i-agree-to-the-vendor-licenses:latest'}, convert:bool = False, efun=None):
        da = ReadWatersRaw(dpath, docker, convert)
        return cls(dpath=da.dpath, xrawd=da.xrawd, dfd=da.dfd, edf =da.edf, efun=efun, dt=da.aDt)

    def __init__(self, dpath, xrawd, dfd, edf, efun, dt):
        self.mslevel = 'MRM Waters'
        self.dpath = os.path.abspath(dpath)
        self.fname = os.path.basename(dpath)
        self.xrawd = xrawd
        self.dfd = dfd
        self.edf =edf
        self.efun = efun
        self.aDt = dt
        self.fmap = {
            'a1': {'std': {'FUNCTION 5': 'Picolinic acid-D3'}, 'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
            'a2': {'std': {'FUNCTION 6': 'Nicotinic acid-D4'}, 'analyte': {'FUNCTION 4': 'Picolinic/nicotinic acid/Nicolinic acid'}},
            'a3': {'std': {'FUNCTION 10': '3-HAA-D3'}, 'analyte': {'FUNCTION 8': '3-HAA'}},
            'a4': {'std': {'FUNCTION 11': 'Dopamine-D4'}, 'analyte': {'FUNCTION 9': 'Dopamine'}},
            'a5': {'std': {'FUNCTION 20': 'Serotonin-d4'}, 'analyte': {'FUNCTION 12': 'Serotonin'}},
            'a6': {'std': {'FUNCTION 14': 'Tryptamine-d4'}, 'analyte': {'FUNCTION 13': 'Tryptamine'}},
            'a7': {'std': {'FUNCTION 17': 'Quinolinic acid-D3'}, 'analyte': {'FUNCTION 15': 'Quinolinic acid'}},
            'a8': {'std': {'FUNCTION 19': 'I-3-AA-D4'}, 'analyte': {'FUNCTION 18': 'I-3-AA'}},
            'a9': {'std': {'FUNCTION 27': 'Kynurenic acid-D5'}, 'analyte': {'FUNCTION 22': 'Kynurenic acid'}},
            'a10': {'std': {'FUNCTION 28': '5-HIAA-D5'}, 'analyte': {'FUNCTION 26': '5-HIAA'}},
            'a11': {'std': {'FUNCTION 35': 'Tryptophan-d5'}, 'analyte': {'FUNCTION 30': 'Tryptophan'}},
            'a12': {'std': {'FUNCTION 33': 'Xanthurenic acid-D4'}, 'analyte': {'FUNCTION 31': 'Xanthurenic acid'}},
            'a13': {'std': {'FUNCTION 36': 'Kynurenine-D4'}, 'analyte': {'FUNCTION 32': 'Kynurenine'}},
            'a14': {'std': {'FUNCTION 39': '3-HK 13C15N'}, 'analyte': {'FUNCTION 38': '3-HK'}},
            'a15': {'std': {'FUNCTION 41': 'Neopterin-13C5'}, 'analyte': {'FUNCTION 40': 'Neopterin'}},
            'a16': {'analyte': {'FUNCTION 3': '2-aminophenol'}},
            'a17': {'analyte': {'FUNCTION 42': '3'}},
            'a18': {'analyte': {'FUNCTION 16': '3-methoxy-p-tyramine'}},
            'a19': {'analyte': {'FUNCTION 34': '4-hydroxyphenylacetylglycine'}},
            'a20': {'analyte': {'FUNCTION 23': '5-ME_TRYT'}},
            'a21': {'analyte': {'FUNCTION 37': '5-OH-tryptophan'}},
            'a22': {'analyte': {'FUNCTION 21': '(-)-Epinephrine'}},
            'a23': {'analyte': {'FUNCTION 29': 'L-Dopa'}},
            'a24': {'analyte': {'FUNCTION 25': 'L-tryptophanol'}},
            'a25': {'analyte': {'FUNCTION 43': 'N-acetyl-L-tyrosine'}},
            'a26': {'analyte': {'FUNCTION 24': 'N-methylseotonin'}},
            'a27': {'analyte': {'FUNCTION 1': 'Trimethylamine'}},
            'a28': {'analyte': {'FUNCTION 2': 'Trimethylamine-N-oxide'}},
            'a29': {'analyte': {'FUNCTION 7': 'Tyramine'}},
        }

    @staticmethod
    def blcor(x, y, wlenF=0.4, rety=True):
        import scipy.interpolate as si
        wlen = round(len(x) * wlenF)
        n = len(y) // wlen
        cb = [np.argmin(y[(wlen * i - wlen):(wlen * i)]) + (wlen * i - wlen) for i in range(1, n + 1)]
        if (len(y) % wlen) > 0:
            cb = cb + [np.argmin(y[((n) * wlen):]) + ((n) * wlen)]
        # yb = si.pchip_interpolate(x[cb], y[cb], x)
        f = si.interp1d(x[cb], y[cb], fill_value="extrapolate")

        if rety:
            ybl = y - f(x)
            return np.array([y[i] if ybl[i] > y[i] else ybl[i] for i in range(len(ybl))])
        else:
            return f(x)

    @staticmethod
    def smooth(y, wlen=21):
        ys = ss.savgol_filter(y, wlen, 3)
        return ys
        # return ys

    @staticmethod
    def decon1(x, y, peaks, hh, plot=True, ax=[None, None]):
        import math
        from scipy.signal import find_peaks
        # lower limit of detection is 20k

        def cost(params):
            n = round(len(params) / 4)
            est = np.zeros_like(x)
            for i in range(1, n + 1):
                pi = params[(i * 4 - 4):(i * 4)]
                # print(pi)
                est += g(x, *pi)
            return np.sum(np.power(est - y, 2)) / len(x)

        def g(x, A, a, µ, σ):
            yy = np.squeeze(st.skewnorm.pdf(x, a, µ, σ))
            return (yy / max(yy)) * A

            # gaussian peak shape (x, magnituded (max/40), mean and sd
            # return A / (σ * math.sqrt(2 * math.pi)) * np.exp(-(x - µ) ** 2 / (2 * σ ** 2))

        # def parameters: a, A, loc, scale (sqrt of fwhm)
        param = []
        bounds = []
        for i in range(len(peaks)):
            A = hh['peak_heights'][i]
            if A < 0:
                continue
            a = 2
            mu = x[peaks[i]]
            w = round(hh['widths'][i] / 2)
            sd = (x[peaks[i] + w] - x[max([peaks[i] - w, 0])]) / 2
            asc = (((sd * x[peaks[i]]) / a) if a != 0 else 0)
            lloc = mu
            p = [A, a, lloc, sd]
            param += p
            b = [(hh['peak_heights'][i] / 10, hh['peak_heights'][i]), (-10, 30),
                 (max([lloc - (asc / 2), 0]), min([lloc, lloc + (asc / 2)])), (sd / 4, sd * 2)]
            bounds += b

        result = optimize.minimize(cost, param, bounds=bounds)

        psum = np.zeros_like(x)
        acomp = []
        yest = []
        for i in range(1, (len(param) // 4) + 1):
            p = result.x[(i * 4 - 4):(i * 4)]
            est = g(x, *p)
            yest.append(est)
            acomp.append(np.sum(est))
            psum += est
        if plot:
            for i in range(1, (len(param) // 4) + 1):
                ax[0].plot(x, est, label='c' + str(i))
                # pstart = param[(i * 4 - 4):(i * 4)]
            ax[1].plot(x, psum - y)
            # plt.plot(x, s, label='sum')
            ax[0].plot(x, psum, label='psum')
            ax[0].plot(x, y, label='ori')
            ax[0].legend()

        return [result.success, np.sum((psum - y) ** 2), acomp, yest, result.x]

    def ppick(self, x, y, plot=True, wlen=21, **kwargs):
        # bline cor
        yo = y
        ys = self.smooth(y, wlen=wlen)
        ybl = self.blcor(x, ys)
        baseline = self.blcor(x, ys, rety=False)
        yo = yo - baseline
        # scale down or up depending on ymax
        if max(ybl) > 2e4:
            s = 0.1 / 2e4
            ybl = ybl * s
            ys = ys * s
            yo = yo * s
        else:
            if plot:
                plt.figure()
                plt.plot(x, yo, c='green', label='ori')
                plt.plot(x, ys, label='sm')
                plt.plot(x, ybl, label='sm-bl')
                plt.legend()
            return 'Signals below noise threshold'
        # all noise, no peak detected

        # height=0.1, distance=5, prominence=0.1, width=1
        # peaks, hh = ss.find_peaks(ys[10:-10], height=0.1, distance=1, prominence=0.1, width=5)
        peaks, hh = ss.find_peaks(ybl[10:-10], **kwargs)
        peaks = [i + 10 for i in peaks]

        if len(peaks) == 0:
            return 'No peaks found'
        hh['peak_heights'] = yo[peaks]

        if plot:
            fig, axs = plt.subplots(3, 1, sharey=True)
            axs[0].plot(x, yo, c='green', label='ori')
            axs[0].plot(x, ys, label='sm')
            axs[0].plot(x, ybl, label='sm-bl')
            axs[0].hlines(0.1, x[0], x[-1], color='gray')
            axs[0].vlines(x[hh['left_bases']], 0, max(ybl), color='gray')
            axs[0].vlines(x[hh['left_ips'].astype(int)], 0, max(ybl), color='cyan')
            axs[0].vlines(x[hh['right_ips'].astype(int)], 0, max(ybl), color='cyan')

            axs[0].vlines(x[hh['right_bases']], 0, max(ybl), color='gray')
            axs[0].scatter(x[peaks], hh['peak_heights'], c='red')
            for pi, p in enumerate(hh['prominences']):
                axs[0].annotate(round(hh['prominences'][pi], 1), (x[peaks][pi], hh['peak_heights'][pi]),
                                textcoords='offset pixels', xytext=(4, 4))
                # plt.hlines(hh['peak_heights'][pi]/2, x[peaks[pi] - (round(hh['widths'][pi]/2))], x[peaks[pi] + (round(hh['widths'][pi]/2))])
                axs[0].hlines(hh['width_heights'][pi], x[peaks[pi] - (round(hh['widths'][pi] / 2))],
                              x[peaks[pi] + (round(hh['widths'][pi] / 2))])
                axs[0].annotate(round(hh['widths'][pi], 1), (x[peaks][pi], max(yo)))
            dec = self.decon1(x, yo, peaks, hh, ax=[axs[1], axs[2]], plot=True)
            axs[0].legend()
        else:
            dec = self.decon1(x, yo, peaks, hh, ax=[None, None], plot=False)
        return [dec, s, peaks, hh]


    # run ppick over all functions, record residuals and


    def quant(self, xd, **xargs):
        # peak fitting
        from scipy.signal import find_peaks
        peaks, heights = find_peaks(xd[2], **xargs)
        return peaks, heights
        # print(len(peaks))

    def extractData(self, fid):
        ff = self.efun.funcs[fid]
        iid = fid.replace('FUNCTION ', '')
        p = '1P' if ff.polarity == 'Positive' else '1N'
        sub = self.dfd[p][self.dfd[p]['fid'] == str(iid)]
        React = list(self.efun.funcs['FUNCTION ' + str(iid)].reactions)
        nReact = len(React)

        # xd = self.xrawd[p][:, self.xrawd[p][0] == float(sub.fid.values[0])]
        ret = {'d': [], 'm': []}
        for i in range(nReact):
            sidx = sub['index'].iloc[i]
            ret['d'].append(self.xrawd[p][:, self.xrawd[p][0] == float(sidx)])
            ret['m'].append(sub.iloc[i])
        return ret

    def plotEfun(self, id, **xargs):
        ff = self.efun.funcs['FUNCTION ' + str(id)]
        p='1P' if ff.polarity == 'Positive' else '1N'
        sub = self.dfd[p][ self.dfd[p]['fid'] == str(id) ]
        React = list(self.efun.funcs['FUNCTION ' + str(id)].reactions)
        nReact = len(React)
        if nReact != len(sub.index.values):
            print('Product ion missing')
        if nReact == 1:
            i=0
            xd = self.xrawd[p][:,  self.xrawd[p][0] == float(sub.fid.values[0]) ]
            fig, axs = plt.subplots(nReact, sharex=True, sharey=True)
            axs.plot(xd[1], xd[2])
            pidx, pint = self.quant(xd, **xargs)
            axs.scatter(xd[1][pidx], pint['peak_heights'], c='red')
            axs.set_ylabel(r"$\bfCount$")
            axs.text(0.77, 0.9, f'Pr@ {ff.reactions[str(i + 1)].massProduct} m/z', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.8, f'st:  {ff.stEnd_min}-{ff.stStart_min} min', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.7, f'{ff.reactions[str(i + 1)].compoundName}', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.5, f'Co-E: {ff.reactions[str(i + 1)].collisionEnergy} eV', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.4, f'Conevolt: {ff.reactions[str(i + 1)].conevoltageV}', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.3, f'Polarity: {ff.polarity}', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            axs.text(0.77, 0.2, f'P@ {ff.reactions[str(i + 1)].massPrecursor} m/z', rotation=0, fontsize=8,
                        transform=axs.transAxes,
                        ha='left')
            fig.suptitle(f"F{str(id)}:  {ff.name} @ {ff.reactions[str(i + 1)].massPrecursor} m/z")
            axs.text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=axs.transAxes)
            axs.set_xlabel(r"$\bfScantime (min)$")
        else:
            fig, axs = plt.subplots(nReact, sharex=True, sharey=True)
            for i in range(nReact):
                sidx = sub['index'].iloc[i]
                xd =  self.xrawd[p][:, self.xrawd[p][0] == float(sidx)]
                axs[i].plot(xd[1], xd[2])
                pidx, pint = self.quant(xd, **xargs)
                axs[i].scatter(xd[1][pidx], pint['peak_heights'], c='red')
                axs[i].set_ylabel(r"$\bfCount$")
                print(self.efun.funcs['FUNCTION ' + str(id)].reactions[str(i + 1)])
                axs[i].text(0.77, 0.9, f'Pr@ {ff.reactions[str(i + 1)].massProduct} m/z', rotation=0, fontsize=8, transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.8, f'st:  {ff.stEnd_min}-{ff.stStart_min} min', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.7, f'{ff.reactions[str(i + 1)].compoundName}', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.5, f'Co-E: {ff.reactions[str(i + 1)].collisionEnergy} eV', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.4, f'Conevolt: {ff.reactions[str(i + 1)].conevoltageV}', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.3, f'Polarity: {ff.polarity}', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                axs[i].text(0.77, 0.2, f'P@ {ff.reactions[str(i + 1)].massPrecursor} m/z', rotation=0, fontsize=8,
                            transform=axs[i].transAxes,
                            ha='left')
                fig.suptitle(f"F{str(id)}:  {ff.name} @ {ff.reactions[str(i + 1)].massPrecursor} m/z")

            axs[i].text(1.04, 0, self.fname, rotation=90, fontsize=6, transform=axs[i].transAxes)
            axs[i].set_xlabel(r"$\bfScantime (min)$")

epath = '/Users/torbenkimhofer/Desktop/Torben19Aug.exp'
dpath='/Users/torbenkimhofer/tdata_trp/RCY_TRP_023_Robtot_Spiking_04Aug2022_URN_LTR_Cal2_110.raw'
dpath='/Users/torbenkimhofer/tdata_trp/RCY_TRP_023_Robtot_Spiking_04Aug2022_SER_unhealthy_Cal5_69.raw'
test=TrpExp.waters(dpath, efun=ReadExpPars(epath))

df=test.extractData('FUNCTION 3')

import scipy.signal as ss
ra=[]
for i,f in enumerate(test.efun.funcs.keys()):
    if '38' in f:
        continue
    f = list(test.efun.funcs.keys())[i]
    # f = 'FUNCTION 12'
    df = test.extractData(f)
    print(f)
    for l, r in enumerate(df['d']):
        # l=1
        # r = df['d'][l]
        x=r[1]
        y=r[2]
        # plt.plot(x,y)
        test.ppick(x, y, plot=True, height=0.1, distance=1, prominence=0.0001, width=3, wlen=17)
        # res=ppick(x, y, plot=True, height=0.1, distance=1, prominence=0.0001, width=3, wlen=17)
        ra.append(res)
        if isinstance(res, list):
            plt.suptitle(f+': success: '+str(res[0][0])+', res: '+str(round(res[0][1])))

res={}
for i, r in enumerate(ra):
    if isinstance(r, list):
        res[i]={'success': r[0][0], 'res': r[0][1], 'area_comp': r[0][2], 'comp_loc': r[0][4]}




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





#  how often changes
rr = [x[2] for x in ra][0]
plt.plot(np.sort(rr))
id=np.argsort([x for x in rr])
s=0
for s in range(20):
    f=ra[id[s]][0]
    r=test.extractData(f)['d'][ra[id[s]][1]]
    plt.figure()
    plt.plot(r[1], r[2])


ra[id[s]][0]
ra[25][0]
ra[25][1]
g=np.gradient(y)
gq=[1 for i in range(1,len(g)) if np.sign(g[i-1]) != np.sign(g[i])]
plt.plot(x[1:], gq)
np.sum(gq)/len(x)
plt.plot(gq)
plt.plot(x, y)
plt.plot(x, ss.detrend(y, axis=0))
out=ss.periodogram(y)
plt.plot(out[0], out[1])

out=si.PchipInterpolator(x, y)
out.x
out.

win=20
n = len(y) // win
cb = [np.argmin(y[(win*i - win):(win*i)])+(win*i - win) for i in range(1, n+1)] + [np.argmin(y[((n) * win):])+((n) * win)]
yb = si.pchip_interpolate(x[cb], y[cb], x)
plt.scatter(x[cb], y[cb], c='red')
plt.plot(x,y)
plt.plot(x,y-yb)

test.plotEfun('3')
df=test.extractData('FUNCTION 3')
sub=df['d'][0]
sub=df['d'][1]
plt.plot(sub[1], sub[2])
efun = ReadExpPars(epath)





# define experiment set for Trp data (import series of trp experiments)
# plot functions, QC and quantify signals
class ExpSet:
    def __init__(self, path='/Volumes/ANPC_ext1/trp_qc/', convert=False):
        import re, os
        import numpy as np
        # find all waters experiment files
        # try to annotate sample type (LTR vs Cal vs Sample)
        self.exp= []
        # self.calib = []
        self.df = {}
        self.areas = {}
        c = 0
        for d in os.listdir(path):
            if bool(re.findall(".*raw$", d)):
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