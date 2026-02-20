#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
#%% LEctor de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype={'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% Extraigo valores de las tablas de resultados todo a 300 kHz
resultados_300 = glob("**/**/*resultados.txt")
resultados_300.sort()
for p in resultados_300:
    print(p)
#%%
t_300_100,t_300_125,t_300_150 = [],[],[]
T_300_100,T_300_125,T_300_150 = [],[],[]

tau_300_100,tau_300_125,tau_300_150 = [],[],[]
SAR_300_100,SAR_300_125,SAR_300_150 = [],[],[]

Hc_300_100,Hc_300_125,Hc_300_150 = [],[],[]
Mr_300_100,Mr_300_125,Mr_300_150 = [],[],[]

for f in resultados_300:
    if '100dA' in f:
        meta,_,t,T,Mr,Hc,_,_,_,_,_,_,SAR,tau,_ = lector_resultados(f)
        t_300_100.append(t-t[0])
        T_300_100.append(T)
        tau_300_100.append(tau)
        SAR_300_100.append(SAR)
        Mr_300_100.append(Mr)
        Hc_300_100.append(Hc)
    elif '125dA' in f:
        meta,_,t,T,Mr,Hc,_,_,_,_,_,_,SAR,tau,_ = lector_resultados(f)
        t_300_125.append(t-t[0])
        T_300_125.append(T)
        tau_300_125.append(tau)
        SAR_300_125.append(SAR)
        Mr_300_125.append(Mr)
        Hc_300_125.append(Hc)
    elif '150dA' in f:
        meta,_,t,T,Mr,Hc,_,_,_,_,_,_,SAR,tau,_ = lector_resultados(f)
        t_300_150.append(t-t[0])
        T_300_150.append(T)
        tau_300_150.append(tau)
        SAR_300_150.append(SAR)
        Mr_300_150.append(Mr)
        Hc_300_150.append(Hc)

#%% Tau vs t / T
fig1, ((ax,axa),(ax2,axb),(ax3,axc)) = plt.subplots(nrows=3,ncols=2,figsize=(12,8),constrained_layout=True,sharex='col',sharey='row')
ax.plot(t_300_100[0],tau_300_100[0],'.-')
ax.plot(t_300_100[1],tau_300_100[1],'.-')
ax.plot(t_300_100[2],tau_300_100[2],'.-')

ax2.plot(t_300_125[0],tau_300_125[0],'.-')
ax2.plot(t_300_125[1],tau_300_125[1],'.-')
ax2.plot(t_300_125[2],tau_300_125[2],'.-')

ax3.plot(t_300_150[0],tau_300_150[0],'.-')
ax3.plot(t_300_150[1],tau_300_150[1],'.-')
ax3.plot(t_300_150[2],tau_300_150[2],'.-')

axa.plot(T_300_100[0],tau_300_100[0],'.-')
axa.plot(T_300_100[1],tau_300_100[1],'.-')
axa.plot(T_300_100[2],tau_300_100[2],'.-')

axb.plot(T_300_125[0],tau_300_125[0],'.-')
axb.plot(T_300_125[1],tau_300_125[1],'.-')
axb.plot(T_300_125[2],tau_300_125[2],'.-')

axc.plot(T_300_150[0],tau_300_150[0],'.-')
axc.plot(T_300_150[1],tau_300_150[1],'.-')
axc.plot(T_300_150[2],tau_300_150[2],'.-')


ax.set_title('tau - 38 kA/m',loc='left')
ax2.set_title('tau - 47 kA/m',loc='left')
ax3.set_title('tau - 57 kA/m',loc='left')
ax3.set_xlabel("t (s)")
axc.set_xlabel("T (°C)")


for a in [ax,ax2,ax3,axa,axb,axc]:
    a.grid()
    #a.legend(title="Frecuencia (kHz)",ncol=2)
for a in [ax,ax2,ax3]:
    a.set_ylabel("τ (ns)")
plt.suptitle('tau vs t/T \nNF@citrato_reconc 260203 - 15.0 g/L Fe$_3$O$_4$')    
plt.show()
#%% SAR vs t / T
fig2, ((ax,axa),(ax2,axb),(ax3,axc)) = plt.subplots(nrows=3,ncols=2,figsize=(12,8),constrained_layout=True,sharex='col',sharey='row')

ax.plot(t_300_100[0],SAR_300_100[0],'.-')
ax.plot(t_300_100[1],SAR_300_100[1],'.-')
ax.plot(t_300_100[2],SAR_300_100[2],'.-')

ax2.plot(t_300_125[0],SAR_300_125[0],'.-')
ax2.plot(t_300_125[1],SAR_300_125[1],'.-')
ax2.plot(t_300_125[2],SAR_300_125[2],'.-')

ax3.plot(t_300_150[0],SAR_300_150[0],'.-')
ax3.plot(t_300_150[1],SAR_300_150[1],'.-')
ax3.plot(t_300_150[2],SAR_300_150[2],'.-')


axa.plot(T_300_100[0],SAR_300_100[0],'.-')
axa.plot(T_300_100[1],SAR_300_100[1],'.-')
axa.plot(T_300_100[2],SAR_300_100[2],'.-')

axb.plot(T_300_125[0],SAR_300_125[0],'.-')
axb.plot(T_300_125[1],SAR_300_125[1],'.-')
axb.plot(T_300_125[2],SAR_300_125[2],'.-')

axc.plot(T_300_150[0],SAR_300_150[0],'.-')
axc.plot(T_300_150[1],SAR_300_150[1],'.-')
axc.plot(T_300_150[2],SAR_300_150[2],'.-')


ax.set_title('SAR - 38 kA/m',loc='left')
ax2.set_title('SAR - 47 kA/m',loc='left')
ax3.set_title('SAR - 57 kA/m',loc='left')
ax3.set_xlabel("t (s)")
axc.set_xlabel("T (°C)")

for a in [ax,ax2,ax3,axa,axb,axc]:
    a.grid()
    #a.legend(title="Frecuencia (kHz)",ncol=2)
for a in [ax,ax2,ax3]:
    a.set_ylabel("SAR (W/g)")
plt.suptitle('SAR vs t/T \nNF@citrato_reconc 260203 - 15.0 g/L Fe$_3$O$_4$')    
plt.show()

#%% Temp vs tiempo

fig3,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(9,7),constrained_layout=True)

ax.plot(t_300_100[0],T_300_100[0],'.-')
ax.plot(t_300_100[1],T_300_100[1],'.-')
ax.plot(t_300_100[2],T_300_100[2],'.-')

ax2.plot(t_300_125[0],T_300_125[0],'.-')
ax2.plot(t_300_125[1],T_300_125[1],'.-')
ax2.plot(t_300_125[2],T_300_125[2],'.-')

ax3.plot(t_300_150[0],T_300_150[0],'.-')
ax3.plot(t_300_150[1],T_300_150[1],'.-')
ax3.plot(t_300_150[2],T_300_150[2],'.-')

ax.set_title('Temperatura - 38 kA/m',loc='left')    
ax2.set_title('Temperatura - 47 kA/m',loc='left')
ax3.set_title('Temperatura - 57 kA/m',loc='left')
ax3.set_xlabel("t (t)")

for a in [ax,ax2,ax3]:
    a.grid()
    a.set_ylabel("T (°C)")
    a.set_xlim(0,)
    a.set_ylim(20,90)
plt.suptitle('Temperatura  vs t\nNF@citrato_reconc 260203 - 15.0 g/L Fe$_3$O$_4$')    
plt.show()

#%% Hc vs t/ T

fig4,((ax,axa),(ax2,axb),(ax3,axc)) = plt.subplots(nrows=3,ncols=2,figsize=(12,8),constrained_layout=True,sharex='col',sharey='row')

ax.plot(t_300_100[0],Hc_300_100[0],'.-')
ax.plot(t_300_100[1],Hc_300_100[1],'.-')
ax.plot(t_300_100[2],Hc_300_100[2],'.-')

ax2.plot(t_300_125[0],Hc_300_125[0],'.-')
ax2.plot(t_300_125[1],Hc_300_125[1],'.-')
ax2.plot(t_300_125[2],Hc_300_125[2],'.-')

ax3.plot(t_300_150[0],Hc_300_150[0],'.-')
ax3.plot(t_300_150[1],Hc_300_150[1],'.-')
ax3.plot(t_300_150[2],Hc_300_150[2],'.-')


axa.plot(T_300_100[0],Hc_300_100[0],'.-')
axa.plot(T_300_100[1],Hc_300_100[1],'.-')
axa.plot(T_300_100[2],Hc_300_100[2],'.-')

axb.plot(T_300_125[0],Hc_300_125[0],'.-')
axb.plot(T_300_125[1],Hc_300_125[1],'.-')
axb.plot(T_300_125[2],Hc_300_125[2],'.-')

axc.plot(T_300_150[0],Hc_300_150[0],'.-')
axc.plot(T_300_150[1],Hc_300_150[1],'.-')
axc.plot(T_300_150[2],Hc_300_150[2],'.-')


ax.set_title('Hc - 38 kA/m',loc='left')
ax2.set_title('Hc - 47 kA/m',loc='left')
ax3.set_title('Hc - 57 kA/m',loc='left')
ax3.set_xlabel("t (s)")
axc.set_xlabel("T (°C)")

for a in [ax,ax2,ax3,axa,axb,axc]:
    a.grid()
    #a.legend(title="Frecuencia (kHz)",ncol=2)
for a in [ax,ax2,ax3]:
    a.set_ylabel("Hc (kA/m)")
plt.suptitle('Hc vs t/T \nNF@citrato_reconc 260203 - 15.0 g/L Fe$_3$O$_4$')    
plt.show()

#%% Mr vs t/T
fig5,((ax,axa),(ax2,axb),(ax3,axc)) = plt.subplots(nrows=3,ncols=2,figsize=(12,8),constrained_layout=True,sharex='col',sharey='row')

ax.plot(t_300_100[0],Mr_300_100[0],'.-')
ax.plot(t_300_100[1],Mr_300_100[1],'.-')
ax.plot(t_300_100[2],Mr_300_100[2],'.-')

ax2.plot(t_300_125[0],Mr_300_125[0],'.-')
ax2.plot(t_300_125[1],Mr_300_125[1],'.-')
ax2.plot(t_300_125[2],Mr_300_125[2],'.-')

ax3.plot(t_300_150[0],Mr_300_150[0],'.-')
ax3.plot(t_300_150[1],Mr_300_150[1],'.-')
ax3.plot(t_300_150[2],Mr_300_150[2],'.-')

axa.plot(T_300_100[0],Mr_300_100[0],'.-')
axa.plot(T_300_100[1],Mr_300_100[1],'.-')
axa.plot(T_300_100[2],Mr_300_100[2],'.-')

axb.plot(T_300_125[0],Mr_300_125[0],'.-')
axb.plot(T_300_125[1],Mr_300_125[1],'.-')
axb.plot(T_300_125[2],Mr_300_125[2],'.-')

axc.plot(T_300_150[0],Mr_300_150[0],'.-')
axc.plot(T_300_150[1],Mr_300_150[1],'.-')    
axc.plot(T_300_150[2],Mr_300_150[2],'.-')    

ax.set_title('Mr - 38 kA/m',loc='left')
ax2.set_title('Mr - 47 kA/m',loc='left')
ax3.set_title('Mr - 57 kA/m',loc='left')
ax3.set_xlabel("t (s)")
axc.set_xlabel("T (°C)")

for a in [ax,ax2,ax3,axa,axb,axc]:    
    a.grid()
    #a.legend(title="Frecuencia (kHz)",ncol=2)
for a in [ax,ax2,ax3]:
    a.set_ylabel("Mr (A/m)")
plt.suptitle('Mr vs t/T \nNF@citrato_reconc 260203 - 15.0 g/L Fe$_3$O$_4$')    
plt.show()

#%% Salvo las figuras
for f in zip([fig1,fig2,fig3,fig4,fig5],['tau_vs_t&T','SAR_vs_t&T','templogs','Hc_vs_t&T','Mr_vs_t&T']):
    f[0].savefig('NF_citrato_reconc_300_'+f[1]+'.png')


# %%
