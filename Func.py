# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import scipy.stats as st 
import scipy
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# import cufflinks as cf
import chart_studio as cf

# from _plotly_future_ import v4_subplots
from business_duration import businessDuration

from plotly.subplots import make_subplots
# import plotly.graph_objects as go
import plotly.graph_objs as go

import plotly.figure_factory as ff
from plotly.offline import plot
import eikon as ek

from IPython.display import display, HTML
from IPython.display import display_html 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import seaborn as sns


def Screen(optionsInfo,Selec_C,Selec_P):
    CSS = """
    .output {
        flex-direction: row;
    }
    """

    HTML('<style>{}</style>'.format(CSS))

    cmc = sns.light_palette("green", as_cmap=True,reverse=False)
    cmp = sns.light_palette("green", as_cmap=True,reverse=True)

    call=optionsInfo[optionsInfo.Instrument.isin(Selec_C[0])][['Instrument','TRDPRC_1','STRIKE_PRC','VOL_X_PRC1']]
    put=optionsInfo[optionsInfo.Instrument.isin(Selec_P[0])][['Instrument','TRDPRC_1','STRIKE_PRC','VOL_X_PRC1']]

    td_props = [
      ('font-size', '11px')
      ]

    th_props = [
      ('font-size', '11px'),
      ('text-align', 'center'),
      ('font-weight', 'bold'),
      ('color', '#6d6d6d'),
      ('background-color', '#f7f7f9')
      ]

    styles = [
      dict(selector="th", props=th_props),
      dict(selector="td", props=td_props)
      ]

    t1=(call.style
      .background_gradient(cmap=cmc, subset=['STRIKE_PRC'])
      .set_caption('Call options')
      .format({'total_amt_usd_pct_diff': "{:.10%}"})
      .set_table_styles(styles))

    t2=(put.style
      .background_gradient(cmap=cmp, subset=['STRIKE_PRC'])
      .set_caption('Put options')
      .format({'total_amt_usd_pct_diff': "{:.10%}"})
      .set_table_styles(styles))
    return(t1,t2,call,put)
    
def ImpliedVolaObjective(S, K, R, D, OPprice,T1,sty):
    B=0
    dcalend=252
    days=T1/dcalend
    D1=(np.log(S/K)+((R-B+((D**2)/2))*days))/(D*(days**0.5))
    D2=((np.log(S/K)+((R-B+((D**2)/2))*days))/(D*(days**0.5)))-D*(days**0.5)
    if "C" in sty:
        return(OPprice - (S*st.norm.cdf(D1)-K*np.exp(-R*days)*st.norm.cdf(D2)))
    else:
        return(OPprice - (K*np.exp(-R*days)*st.norm.cdf(-D2)-S*st.norm.cdf(-D1)))

    
def ImpliedVola(S, K, R, D, OPprice,T1,sty, a=-8.0, b=8.0, xtol=1e-6):
    _S, _K, _R, _T1, _OPprice = S, K, R, T1, OPprice
    def fcn(D):
	    # returns the difference between market and model price at given volatility
        return ImpliedVolaObjective(_S, _K, _R, D, _OPprice, _T1,sty)
   # first we try to return the results from the brentq algorithm
    try:
        result = brentq(fcn, a=a, b=b, xtol=xtol)
        # if the results are *too* small, sent to np.nan so we can later interpolate
        return result
        # return np.nan if result <= xtol else result
    # if it fails then we return np.nan so we can later interpolate the results
    except ValueError:
        return np.nan


def OpitionsPrice(S, K, R, D,T):
    dcalend=252
    days=T/dcalend
    B=0
    D1=(np.log(S/K)+((R-B+((D**2)/2))*days))/(D*(days**0.5))
    D2=((np.log(S/K)+((R-B+((D**2)/2))*days))/(D*(days**0.5)))-D*(days**0.5)

    ValorC=S*st.norm.cdf(D1)-K*np.exp(-R*days)*st.norm.cdf(D2)
    DeltaC=st.norm.cdf(D1)
    GammaC=st.norm.pdf(D1)/(S*D*np.sqrt(days))
    VegaC=S*st.norm.pdf(D1)*(np.sqrt(days))
    ThetaC=(-((S*st.norm.pdf(D1)*D)/(2*np.sqrt(days)))-(R*K*np.exp(-R*days))*st.norm.cdf(D2))/dcalend #252
    RhoC=K*days*np.exp(-R*days)*st.norm.cdf(D2)
    EtasC=DeltaC*S/ValorC

    ValorP=K*np.exp(-R*days)*st.norm.cdf(-D2)-S*st.norm.cdf(-D1)
    DeltaP=-st.norm.cdf(-D1)
    GammaP=st.norm.pdf(D1)/(S*D*np.sqrt(days))
    VegaP=S*st.norm.pdf(D1)*(np.sqrt(days))
    ThetaP=(-((S*st.norm.pdf(D1)*D)/(2*np.sqrt(days)))+(R*K*np.exp(-R*days))*st.norm.cdf(-D2))/dcalend #252
    RhoP=-K*days*np.exp(-R*days)*st.norm.cdf(-D2)
    EtasP=DeltaP*S/ValorP
    return pd.DataFrame({"ValorC":ValorC,
            "DeltaC":DeltaC,
            "GammaC": GammaC,
            "VegaC":VegaC,
            "ThetaC":ThetaC,
            "RhoC": RhoC,
            "EtasC": EtasC,
            "ValorP":ValorP,
            "DeltaP":DeltaP,
            "GammaP": GammaP,
            "VegaP":VegaP,
            "ThetaP":ThetaP,
            "RhoP": RhoP,
            "EtasP": EtasP})



def OpitionsPriceImplied(base,x):
    S=base["CLOSE"+str("" if (x-1)==0 else (x-1))+"_ATIVO"].values
    K=base["STRIKE_PRC"].values
    R=base["FreeRiskCLOSE"+str("" if (x-1)==0 else ("_"+(str(x-1))))].values/100
    D=base["BLAKvolat"+str(x)].values
    dcalend=252
    days=base["dayTOexp"].values/dcalend
    B=0
    D1=(np.log(S/K)+((R-B+((D**2)/2))*days))/(D*(days**0.5))
    D2=((np.log(S/K)+((R-B+((D**2)/2))*days))/(D*(days**0.5)))-D*(days**0.5)
    ValorC=S*st.norm.cdf(D1)-K*np.exp(-R*days)*st.norm.cdf(D2)
    DeltaC=st.norm.cdf(D1)
    GammaC=st.norm.pdf(D1)/(S*D*np.sqrt(days))
    VegaC=S*st.norm.pdf(D1)*(np.sqrt(days))
    ThetaC=(-((S*st.norm.pdf(D1)*D)/(2*np.sqrt(days)))-(R*K*np.exp(-R*days))*st.norm.cdf(D2))/dcalend #252
    RhoC=K*days*np.exp(-R*days)*st.norm.cdf(D2)
    EtasC=DeltaC*S/ValorC

    ValorP=K*np.exp(-R*days)*st.norm.cdf(-D2)-S*st.norm.cdf(-D1)
    DeltaP=-st.norm.cdf(-D1)
    GammaP=st.norm.pdf(D1)/(S*D*np.sqrt(days))
    VegaP=S*st.norm.pdf(D1)*(np.sqrt(days))
    ThetaP=(-((S*st.norm.pdf(D1)*D)/(2*np.sqrt(days)))+(R*K*np.exp(-R*days))*st.norm.cdf(-D2))/dcalend #252
    RhoP=-K*days*np.exp(-R*days)*st.norm.cdf(-D2)
    EtasP=DeltaP*S/ValorP
    return pd.DataFrame({"ValorC":ValorC,
            "DeltaC":DeltaC,
            "GammaC": GammaC,
            "VegaC":VegaC,
            "ThetaC":ThetaC,
            "RhoC": RhoC,
            "EtasC": EtasC,
            "ValorP":ValorP,
            "DeltaP":DeltaP,
            "GammaP": GammaP,
            "VegaP":VegaP,
            "ThetaP":ThetaP,
            "RhoP": RhoP,
            "EtasP": EtasP})

    
def DeltaStreikFilter(optionsInfo_C,ll,numOpc=None,perDelta=None):
    if ((numOpc==None) & (perDelta==None)) or ((numOpc!=None) & (perDelta!=None)):
        return("You need to define only one filter")
    else:
        if numOpc!=None:
            tempRes=[]
            for exp in ll:
                temps=optionsInfo_C[(optionsInfo_C.EXPIR_DATE.astype('datetime64')==np.array(exp).astype('datetime64'))]
                temps=temps.reindex(temps.Delta.abs().sort_values().index)[:int(numOpc)]
                temps=temps.reindex(temps.STRIKE_PRC.abs().sort_values().index)
                temps=temps.sort_values("STRIKE_PRC")
                temps.index=range(len(temps))
                temps=temps['Instrument'].values
                tempRes=tempRes+[(list(temps))]
            return(list((tempRes)))
        else:
            tempRes=[]
            for exp in ll:
                temps=optionsInfo_C[(optionsInfo_C.EXPIR_DATE.astype('datetime64')==np.array(exp).astype('datetime64'))]
                temps=temps[(temps.Delta.abs()<=perDelta).tolist()]
                temps=temps.reindex(temps['Delta'].sort_values().index)
                temps.index=range(len(temps))
                temps=temps['Instrument'].values
                tempRes=tempRes+[(list(temps))]
            return(list((tempRes)))


def StrPL(Distrib,pay,pays,tempTT,days,dists,data):
    
    fig = go.FigureWidget(make_subplots(shared_xaxes=True,
                                        specs=[[{"secondary_y": True}]],print_grid=False))

#     fig = go.Figure(make_subplots(shared_xaxes=True,
#                                         specs=[[{"secondary_y": True}]],print_grid=False))

    trace3 = go.Scatter(name="0 line",x=pay.index,
                           y=np.array([0 for i in Distrib]),
                           xaxis = 'x1',yaxis = 'y2',
                           line = dict(color='black', width=2, dash='dash'))
    fig.add_trace(trace3, secondary_y=False)

    trace1 = go.Scatter(name="Payoff",x=pay.index,
                           y=pay.values,xaxis = 'x1',yaxis = 'y2',
                           mode='lines',fill='tozeroy')

    fig.add_trace(trace1, secondary_y=False)

    for i in range(len(tempTT)):
        trace5 = go.Scatter(name="Price - "+str(days[i])+' Days',x=pay.index,
                               y=pd.DataFrame(tempTT[i]).sum().values,
                               xaxis = 'x1',yaxis = 'y2',
                               mode='lines')
        fig.add_trace(trace5, secondary_y=False)


    for lin,i in zip(pays.values,pays.index):
        trace4 = go.Scatter(name=data.Instrument.values[i],x=pay.index,
                               y=lin,xaxis = 'x1',yaxis = 'y2',
                               line = dict(width=2, dash='dash'))
        fig.add_trace(trace4, secondary_y=False)

    for i,j in zip(days,dists):
        trace2 = ff.create_distplot([j],[str(i)+" Days - Probabilidade"],bin_size=.5,curve_type='normal',show_hist=False,show_rug=False)
        fig.add_trace(trace2['data'][0], secondary_y=True)
    
    fig.layout.update({'height': 800,
                       'width': 1000})
    return(fig)



def all(spot,data):
    ssp=spot.TRDPRC_1.values[0]

    data['Amount']=data['Amount'].astype(float)
    data['TRDPRC_1']=data['TRDPRC_1'].astype(float)
    data['STRIKE_PRC']=data['STRIKE_PRC'].astype(float)
    data["TRADE_DATE"]=pd.to_datetime(data["TRADE_DATE"])
    data["EXPIR_DATE"]=pd.to_datetime(data["EXPIR_DATE"])

    days=(data.apply(lambda row: businessDuration(startdate=row['TRADE_DATE'],enddate=row['EXPIR_DATE'],unit='day'), axis=1)).values

    ImpyVola=[round(ImpliedVola(spot.TRDPRC_1.values[0].astype(float),
                data["STRIKE_PRC"].values[j].astype(float),
                spot.TRDPRC_1.values[1]/100,
                .70,
                data["TRDPRC_1"].values[j].astype(float),
                days[j],
                # (((data["EXPIR_DATE"].values[j])-(data["TRADE_DATE"].values[j])).astype('timedelta64[D]').astype(int)),
                data["PUTCALLIND"].values[j],a=-12.0, b=12.0, xtol=1e-8)*100,2) for j in range(len(data))]

    GREEKS=OpitionsPrice(spot.TRDPRC_1.values[0].astype(float),
                                data["STRIKE_PRC"].values.astype(float),
                                spot.TRDPRC_1.values[1]/100,
                                np.array(ImpyVola)/100,
                                days)

    vola, err = ek.get_data([spot[spot["Instrument"]!="BRSELICD=CBBR"].Instrument.values[0]],
                                ['TR.Volatility5D','TR.Volatility10D',"TR.Volatility20D",
                                "TR.Volatility30D","TR.Volatility60D","TR.Volatility90D"])
    vola.columns=['Instrument',5,10,20,30,60,90]
    vola[[5,10,20,30,60,90]]=vola[[5,10,20,30,60,90]]/100
    days=list(np.unique(days))

    dists=[]
    vollist=[]
    for d in days:
        y_interp = scipy.interpolate.interp1d(vola.columns[1:].astype(float),vola.T[0].values[1:].astype(float))
        volaV=y_interp(d)
        Distrib=np.random.lognormal(0,(volaV/(252**0.5)*(d**0.5)), size=5000)
        Distrib=Distrib*ssp

        vollist.append(volaV)
        dists.append(Distrib)

    for i in range(len(dists)):
        dists[i]=dists[i][(dists[i]>ssp*0.8)&(dists[i]<ssp*1.2)]

    Price=np.array(dists).reshape(np.array(dists).shape[0]*np.array(dists).shape[1])
    # dists=[Price]
    Price.sort()

    net=[]
    for k in range(len(data)):
        if data['Amount'].iloc[k]>=0:
            if "C" in data['PUTCALLIND'].iloc[k]:
                Rt= -data['STRIKE_PRC'].iloc[k]-data['TRDPRC_1'].iloc[k] +Price
                Rt[Rt<=-data['TRDPRC_1'].iloc[k]]=-data['TRDPRC_1'].iloc[k]
                Rt=Rt*abs(data['Amount'].iloc[k])
                net.append(list(Rt))
            else:
                Rt= data['STRIKE_PRC'].iloc[k] -data['TRDPRC_1'].iloc[k] -Price
                Rt[Rt<=-data['TRDPRC_1'].iloc[k]]=-data['TRDPRC_1'].iloc[k]
                Rt=Rt*abs(data['Amount'].iloc[k])
                net.append(list(Rt))

        else:
            if "C" in data['PUTCALLIND'].iloc[k]:
                Rt= data['STRIKE_PRC'].iloc[k] +data['TRDPRC_1'].iloc[k] -Price
                Rt[Rt>=data['TRDPRC_1'].iloc[k]]=data['TRDPRC_1'].iloc[k]
                Rt=Rt*abs(data['Amount'].iloc[k])
                net.append(list(Rt))
            else:
                Rt= -data['STRIKE_PRC'].iloc[k] +data['TRDPRC_1'].iloc[k] +Price
                Rt[Rt>=data['TRDPRC_1'].iloc[k]]=data['TRDPRC_1'].iloc[k]
                Rt=Rt*abs(data['Amount'].iloc[k])
                net.append(list(Rt))

    tempTT=[]
    for j in range(len(set(days))):
        tempT=[]
        for s in range(len(data["STRIKE_PRC"])):
            temp=OpitionsPrice(Price,
                                data["STRIKE_PRC"].values.astype(float)[s],
                                spot.TRDPRC_1.values[1]/100,
                                np.array(ImpyVola[j])/100,
                                days[j])
            
            if "C" in data["PUTCALLIND"].values[s]:
                temp=list((temp.ValorC.values-data['TRDPRC_1'].values[s])*data['Amount'].values[s])
            else:
                temp=list((temp.ValorP.values-data['TRDPRC_1'].values[s])*data['Amount'].values[s])

            tempT.append(temp)
        tempTT.append(tempT)

    pay=pd.DataFrame(net)
    pay.columns=Price
    pays=pay
    pay=pay.sum()

    fig=StrPL(Distrib,pay,pays,tempTT,days,dists,data)

    prob=round(sum(pay.values>0)/len(Distrib)*100,2)
    return(fig,prob)


