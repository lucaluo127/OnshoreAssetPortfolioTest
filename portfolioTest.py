import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class plusDataExcel(object):
    # 类似的数据结构即可以
    def __init__(self, path):
        fields = ['Amt', 'Close', 'MktCap']
        self.DB = {}

        for field in fields:
            self.DB[field] = pd.read_excel(f'{path}t{field}.xlsx', index_col=0, engine='openpyxl')
            self.DB[field].index = pd.to_datetime(self.DB[field].index)

        self.char = pd.read_excel(f'{path}tChar.xlsx', engine='openpyxl', index_col=0)['Type']
        self.unit = {'Amt': 10000, 'Close': 1, 'MktCap': 10000, 'Outstanding': 10000}


def getStartLoc(dataframe, trade_dt):
    return dataframe.index.get_loc(trade_dt)


def roundAdjust(dataframe, start, n):
    i = dataframe.index.get_loc(start)

    return dataframe.index[i:][::n]


def checkBook(database, dfRet, dfAssetBook, cash, trade_dt):
    if trade_dt == dfRet.index[0]:

        dfRet.loc[trade_dt]['NAV'] = cash

    else:

        i = dfRet.index.get_loc(trade_dt)

        if len(dfAssetBook.index) == 1 and dfAssetBook.index[0] == 'Nothing':
            dfRet.iloc[i]['NAV'] = dfRet.iloc[i - 1]['NAV']

        else:
            codes = list(dfAssetBook.index)
            dfAssetBook['marketPrice'] = database.DB['Close'].loc[trade_dt, codes]

            t1 = (dfAssetBook['marketPrice'] * dfAssetBook['q']).sum() + cash
            t0 = (dfAssetBook['costPrice'] * dfAssetBook['q']).sum() + cash

            dfRet.iloc[i]['NAV'] = dfRet.iloc[i - 1]['NAV'] * (t1 / t0)

            dfAssetBook['w'] = (dfAssetBook['marketPrice'] * dfAssetBook['q']) / \
                               ((dfAssetBook['marketPrice'] * dfAssetBook['q']).sum() + cash)

            dfAssetBook['costPrice'] = dfAssetBook['marketPrice']

    return dfRet, dfAssetBook, cash


def selectCodes(database, dfAssetBook, trade_dt):
    i = getStartLoc(database.DB['Amt'], trade_dt)
    n = min([i, 10])

    tempCodes = database.DB['Amt'].iloc[i - n:i + 1].dropna(axis=1).columns

    if database.selStkMethod:
        qualStks = database.char[database.char == 'Stock'].index.intersection(set(tempCodes))
        pickStks = database.selStkMethod(database, qualStks, trade_dt, dfAssetBook)
        tempCodes = set(tempCodes).difference(set(qualStks).difference(set(pickStks)))

    if database.selBondMethod:
        qualBonds = database.char[database.char == 'Bond'].index.intersection(set(tempCodes))
        pickBonds = database.selBondMethod(database, qualBonds, trade_dt, dfAssetBook)
        tempCodes = set(tempCodes).difference(set(qualBonds).difference(set(pickBonds)))

    if database.selCBMethod:
        qualCBs = database.char[database.char == 'CB'].index.intersection(set(tempCodes))
        pickCBs = database.selCBMethod(database, qualCBs, trade_dt, dfAssetBook)
        tempCodes = set(tempCodes).difference(set(qualCBs).difference(set(pickCBs)))

    return tempCodes


def sizeAdjust(database, weights, nav, trade_dt, adjustRate):
    tempCodes = weights.index
    qualBonds = database.char[database.char == 'Bond'].index.intersection(set(tempCodes))

    tradingLimit = database.DB['Amt'].loc[trade_dt, tempCodes] * adjustRate / (nav / database.unit['Amt'])
    srsWgts = pd.Series(np.minimum(weights, tradingLimit))
    srsWgts[qualBonds] = weights[qualBonds]

    return srsWgts


def returnAvg(database, codes, trade_dt):
    return pd.Series(np.ones(len(codes)) / float(len(codes)), index=codes)


def returnWeighted(database, codes, trade_dt):
    srsMktCap = database.DB['MktCap'].loc[trade_dt, codes]

    return srsMktCap / srsMktCap.sum()


def _offset(database, trade_dt, n=1):
    idx = database.DB["Amt"].index
    return list(idx)[idx.get_loc(trade_dt) - n]


def getSubWeight(database, codes, trade_dt, asset='Bond'):
    wgtMethodDict = {'Bond': database.weightBondMethod,
                     'Stock': database.weightStkMethod,
                     'CB': database.weightCBMethod}

    weightMethod = wgtMethodDict[asset]

    if not weightMethod:

        return returnAvg(database, codes, trade_dt)

    else:

        if type(weightMethod).__name__ == 'str':

            if weightMethod == 'Ev':

                return returnWeighted(database, codes, trade_dt)

            else:
                return returnAvg(database, codes, trade_dt)

        elif type(weightMethod).__name__ == 'function':

            return weightMethod(database, codes, trade_dt)


def checkSubWeightDict(targetDict):
    required_keys = {'Bond', 'Stock', 'CB'}

    if required_keys.issubset(targetDict):
        total_sum = sum(targetDict[k] for k in required_keys)

        if round(total_sum, 3) == 1:
            return True
        else:
            return False


def getSectWeight(database, codes, trade_dt):
    if not database.weightPortMethod:
        return {'Bond': 1, 'Stock': 0, 'CB': 0}

    if type(database.weightPortMethod).__name__ == 'dict':
        if checkSubWeightDict(database.weightPortMethod):
            return database.weightPortMethod
        else:
            raise ValueError('组合配置的权重字典存在问题')

    elif type(database.weightPortMethod).__name__ == 'str':

        if database.weightPortMethod == 'average':
            return {'Bond': 0.34, 'Stock': 0.33, 'CB': 0.33}
        elif database.weightPortMethod == '股一债九':
            return {'Bond': 0.9, 'Stock': 0.1, 'CB': 0}
        elif database.weightPortMethod == '股二债八':
            return {'Bond': 0.8, 'Stock': 0.2, 'CB': 0}
        elif database.weightPortMethod == '股三债七':
            return {'Bond': 0.8, 'Stock': 0.3, 'CB': 0}

    elif type(database.weightPortMethod).__name__ == 'function':

        return database.weightPortMethod(database, codes, trade_dt)

    return None


def getWeight(database, codes, trade_dt):
    dfWgt = pd.DataFrame(index=codes, columns=['Type', 'TypeWgt', 'SubWgt'])
    dfWgt['Type'] = database.char
    targetWgt = getSectWeight(database, codes, trade_dt)
    dfWgt['TypeWgt'] = dfWgt['Type'].map(targetWgt)

    for asset_type in set(dfWgt['Type']):
        tempCodes = database.char[database.char == asset_type].index.intersection(set(codes))
        dfWgt.loc[tempCodes, 'SubWgt'] = getSubWeight(database, tempCodes, trade_dt, asset=asset_type)

    dfWgt.fillna(0, inplace=True)

    return dfWgt['TypeWgt'] * dfWgt['SubWgt']


class portfolioBT(object):

    def __init__(self, database, startDate='2021/12/31', endDate=None,
                 selStkMethod=None, selBondMethod=None, selCBMethod=None,
                 weightStkMethod=None, weightBondMethod=None, weightCBMethod=None,
                 weightPortMethod=None, roundMethod=21, **kwargs):

        defaultAttrs = {'selStkMethod': selStkMethod, 'selBondMethod': selBondMethod, 'selCBMethod': selCBMethod,
                        'weightStkMethod': weightStkMethod, 'weightBondMethod': weightBondMethod,
                        'weightCBMethod': weightCBMethod, 'weightPortMethod': weightPortMethod,
                        'roundMethod': roundMethod}

        for attr_name, attr_value in database.__dict__.items():
            setattr(self, attr_name, attr_value)

        for attr_name, attr_value in defaultAttrs.items():
            setattr(self, attr_name, attr_value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.startDate = startDate
        self.endDate = endDate if endDate else self.DB['Close'].index[-1]

    def runStrategy(self, cash=1e6, cost=0., sizeLimit=False, sizeAdjustRate=0.1):

        dfRet = pd.DataFrame(columns=['NAV', 'LOG:SEL', 'LOG:WEIGHT'],
                             index=self.DB['Close'].loc[self.startDate:self.endDate].index)

        cash = cash
        dfAssetBook = pd.DataFrame(index=['Nothing'], columns=['costPrice', 'marketPrice', 'w', 'q'])

        isAdjustDates = roundAdjust(self.DB['Amt'], self.startDate, self.roundMethod)

        for trade_dt in dfRet.index:

            dfRet, dfAssetBook, cash = checkBook(self, dfRet, dfAssetBook, cash, trade_dt)

            if trade_dt in isAdjustDates:

                nav = (dfAssetBook['marketPrice'] * dfAssetBook['q']).sum() + cash

                sel = selectCodes(self, dfAssetBook, trade_dt)

                if len(sel) >= 1:

                    weights = getWeight(self, sel, trade_dt)
                    if sizeLimit:
                        weights = sizeAdjust(self, weights, nav, trade_dt, sizeAdjustRate)

                    srsWeightDiff = weights - dfAssetBook['w'].reindex(weights.index).fillna(0)
                    dfAssetBook = pd.DataFrame(index=sel, columns=['costPrice', 'marketPrice', 'w', 'q'])

                    dfAssetBook["w"] = weights

                    dfAssetBook["costPrice"] = self.DB['Close'].loc[trade_dt, sel] * (
                            1 + cost * srsWeightDiff / weights)
                    dfAssetBook["marketPrice"] = self.DB['Close'].loc[trade_dt, sel]

                    q = nav * weights / dfAssetBook["marketPrice"]
                    q *= weights.sum() / (weights * (1 + cost * srsWeightDiff / weights)).sum()
                    cash = nav - (dfAssetBook["costPrice"] * q).sum()

                    dfAssetBook['q'] = q

                else:
                    sel = ['Nothing']
                    weights = 0.0

                    dfAssetBook = pd.DataFrame(index=sel, columns=['costPrice', 'marketPrice', 'w', 'q'])
                    dfAssetBook['w'] = weights

            dfAssetBook = dfAssetBook[dfAssetBook['w'] != 0]

            dfRet['LOG:SEL'][trade_dt] = ','.join(list(dfAssetBook.index))
            dfRet['LOG:WEIGHT'][trade_dt] = ','.join([str(t) for t in list(dfAssetBook['w'])])

        return dfRet


def avgBond7CB2Stk1(database, codes, trade_dt):
    targetWeights = {'Stock': 0.1, 'Bond': 0.7, 'CB': 0.2}
    srsWeights = pd.Series(index=codes, data=0)

    for k, v in targetWeights.items():
        selAssets = database.char[database.char == k].index.intersection(set(codes))
        srsWeights[selAssets] = returnAvg(database, selAssets, trade_dt) * v

    return srsWeights


def avgBond8CB2(database, codes, trade_dt):
    targetWeights = {'Stock': 0, 'Bond': 0.8, 'CB': 0.2}
    srsWeights = pd.Series(index=codes, data=0)

    for k, v in targetWeights.items():
        selAssets = database.char[database.char == k].index.intersection(set(codes))
        srsWeights[selAssets] = returnAvg(database, selAssets, trade_dt) * v

    return srsWeights


if __name__ == '__main__':

    obj0 = plusDataExcel(path='data/')
    
    portfolio0 = portfolioBT(obj0, startDate='2020/12/31',
                            selCBMethod=None, selStkMethod=None, selBondMethod=None,
                            weightCBMethod=None, weightStkMethod=None, weightBondMethod=None,
                            weightPortMethod={'Bond': 1, 'Stock': 0, 'CB': 0})
    
    line0 = portfolio0.runStrategy(cash=1e8, cost=0.002)
