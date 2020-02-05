import pickle
import numpy as np
from tqdm import tqdm
modes = ['equal']
plates = ['3496']
#nuggets are the samples from the top percentile and onenugget is the binary sequence if one sample was found from the top percentile
results_al = {k:{'nuggets':{},'rmses':{},'onenugget':{}} for k in modes}
inits = 50
nsteps = 300


#the raw data needs a preprocessing step to be easily plottable

#get nugget dist
for mi,pair in enumerate([[1,1]]):
    mode = modes[mi]
    k = mode
    print(mode)
    pm = pair[0]
    sm = pair[1]
    for plate in plates:
        print(plate)
        results_al[k]['nuggets'][plate] = {i:{} for i in range(inits)}
        results_al[k]['rmses'][plate] = {i:{} for i in range(inits)}
        results_al[k]['onenugget'][plate] = {i:{} for i in range(inits)}
        d = pickle.load(open(r'active_learning_benchmark_{}_{}_sm{}_pm{}.pck'.format(mode,plate,sm,pm),'rb'))

        for init in tqdm(range(inits)):
            results_al[k]['nuggets'][plate][init] = {s:[] for s in range(1,nsteps)}
            results_al[k]['rmses'][plate][init] = {s:[] for s in range(1,nsteps)}
            results_al[k]['onenugget'][plate][init] = {s:[] for s in range(1,nsteps)}
            if not d == []:
                for step in range(1,nsteps):
                    #get the nuggets
                    train_list = d[init][step]['train_list']
                    test_list = d[init][step]['test_list']
                    top_perc = d[init][step]['top_perc']
                    nuggl = [i for i in train_list if i in top_perc]
                    results_al[k]['nuggets'][plate][init][step] = len(nuggl)/len(top_perc)
                    #get first y/n
                    if len(nuggl)>0:
                        results_al[k]['onenugget'][plate][init][step] = 1
                    else:
                        results_al[k]['onenugget'][plate][init][step] = 0
                    #calc the RMSE
                    val = d[init][step-1]['val_test']
                    results_al[k]['rmses'][plate][init][step] = np.mean(np.abs(d[init][step]['pred_test'][test_list]-val[test_list]))

pickle.dump(results_al,open(r'results_al.pck','wb'))

random_results = {p:{'nuggets':{},'rmses':{},'onenugget':{}} for p in plates}

for plate in plates:
    print(plate)
    random_results[plate]['nuggets'] = {i:{} for i in range(inits)}
    random_results[plate]['rmses'] = {i:{} for i in range(inits)}
    random_results[plate]['onenugget'] = {i:{} for i in range(inits)}
    d = pickle.load(open(r'active_learning_benchmark_random_baseline_{}.pck'.format(plate),'rb'))

    for init in tqdm(range(inits)):
        random_results[plate]['nuggets'][init] = {s:[] for s in range(1,nsteps)}
        random_results[plate]['rmses'][init] = {s:[] for s in range(1,nsteps)}
        random_results[plate]['onenugget'][init] = {s:[] for s in range(1,nsteps)}
        for step in range(1,nsteps):
            #get the nuggets
            train_list = d[init][step]['train_list']
            test_list = d[init][step]['test_list']
            top_perc = d[init][step]['top_perc']
            nuggl = [i for i in train_list if i in top_perc]
            random_results[plate]['nuggets'][init][step] = len(nuggl)/len(top_perc)
            #get first y/n
            if len(nuggl)>0:
                random_results[plate]['onenugget'][init][step] = 1
            else:
                random_results[plate]['onenugget'][init][step] = 0
            #calc the RMSE
            val = d[init][step-1]['val_test']
            random_results[plate]['rmses'][init][step] = np.mean(np.abs(d[init][step]['pred_test'][test_list]-val[test_list]))
pickle.dump(random_results,open(r'results_random.pck','wb'))

#if this is run for the entire benchmarking dataset the above code takes a while such that here we
#load the saved data

results_al = pickle.load(open(r'results_al.pck','rb'))
random_results = pickle.load(open(r'results_random.pck','rb'))

#put together all the modes for averaging in a covenient dict and then numpy arrays for faster math
#end = int(0.7*2121)#this is how we used it in the paper as we used a 70/30 split
end = nsteps
all_nuggets = {k:{plate:np.empty((50,end)) for plate in plates} for k in modes}
all_rmses = {k:{plate:np.empty((50,end)) for plate in plates} for k in modes}
all_onenugget ={k:{plate:np.empty((50,end)) for plate in plates} for k in modes}
all_random_nuggets = {k:{plate:np.empty((50,end)) for plate in plates} for k in modes}
all_random_rmses = {k:{plate:np.empty((50,end)) for plate in plates} for k in modes}
all_random_onenugget = {k:{plate:np.empty((50,end)) for plate in plates} for k in modes}

for k in modes:
    print(k)
    for plate in plates:
        print(plate)
        for init in tqdm(range(inits)):
            for step in range(1,end):
                    all_nuggets[k][plate][init,step] = results_al[k]['nuggets'][plate][init][step]
                    all_rmses[k][plate][init,step] = results_al[k]['rmses'][plate][init][step]
                    all_onenugget[k][plate][init,step] = results_al[k]['onenugget'][plate][init][step]
                    all_random_nuggets[k][plate][init,step] = random_results[plate]['nuggets'][init][step]
                    all_random_rmses[k][plate][init,step] = random_results[plate]['rmses'][init][step]
                    all_random_onenugget[k][plate][init,step] = random_results[plate]['onenugget'][init][step]

#we define a baseline for the median error
rmconst = {m:{plate:0 for plate in plates} for m in modes}
for m in modes:
    for plate in plates:
        rmconst[m][plate] = np.median(all_random_rmses[m][plate][init,-1])


from scipy import interpolate
from scipy import signal
import scipy

reps = 50
splits = 5
acc_onenugget = {k:{plate:[[] for i in range(splits*reps)] for plate in plates} for k in modes}
x_onenugget = {k:{plate:[] for plate in plates} for k in modes}
acc_nugget = {k:{plate:[[] for i in range(reps*reps)] for plate in plates} for k in modes}
x_nuggets = {k:{plate:[] for plate in plates} for k in modes}
acc_rmses = {k:{plate:[[] for i in range(reps*reps)] for plate in plates} for k in modes}
x_rmses = {k:{plate:[] for plate in plates} for k in modes}

def closed_from_random(N=2121):
    M = N/100.
    P,E,A,cA=np.zeros((4,N),dtype='float64')
    P[0]=1.*M/N
    E[0]=1.*M/N
    A[0]=1.*M/N

    for i in range(1,N):
        P[i]=1.*(M-E[i-1])/(N-i)
        E[i]=P[:i+1].sum()
        A[i]=P[i]*(1.-P[:i]).prod()
        cA[i]=A[:i+1].sum()
    Ef=E/M
    return Ef,cA

def split(arr,splits):
    l = int(np.floor(len(arr)/splits))
    split_inds = []
    indexes = [i for i in range(l*splits)]
    for i in range(splits):
        temp = np.random.choice(indexes,l,replace=False)
        split_inds.append([q for q in np.copy(temp)])
        indexes = [i for i in indexes if not i in temp]
    return np.array([arr[split_inds[i],:] for i in range(splits)])

def rsplit(arr,splits,reps):
    return np.array([split(arr,splits) for x in range(reps)])


def inter(yr,ya,st):
    sfr = scipy.signal.savgol_filter(yr,21,3,mode='nearest')
    sfa = scipy.signal.savgol_filter(ya,21,3,mode='nearest')
    fr = interpolate.interp1d(sfr,st,kind='linear',bounds_error=False,fill_value=np.nan)
    fa = interpolate.interp1d(sfa,st,kind='linear',bounds_error=False,fill_value=np.nan)


    interpr = max([min(sfr),min(sfa)]),min(max([max(sfr)/max(sfa)]),1)
    ist = np.linspace(0,1,1000)
    yri,yai = [],[]
    for i in ist:
        if i>interpr[0]:
            if i<interpr[1]:
                yri.append(fr(i))
                yai.append(fa(i))
            else:
                yri.append(np.nan)
                yai.append(np.nan)
        else:
            yri.append(np.nan)
            yai.append(np.nan)
    return yri,yai

#alternative code
# def inter2(yr,ya,st):
#     sfr = scipy.signal.savgol_filter(yr,51,3,mode='nearest')
#     sfa = scipy.signal.savgol_filter(ya,51,3,mode='nearest')
#
#     fr = interpolate.interp1d(sfr,st,kind='linear',bounds_error=False,fill_value=np.nan)
#     fa = interpolate.interp1d(sfa,st,kind='linear',bounds_error=False,fill_value=np.nan)
#
#
#     interpr = max([min(sfr),min(sfa)]),min(max([max(sfr)/max(sfa)]),1)
#     ist = np.linspace(0,1,1000)
#     yri,yai = [],[]
#     for i in ist:
#         yri.append(fr(i))
#         yai.append(fa(i))
#     return yri,yai

#this is the code that calculates the acceleration factors etc
for k in modes:
    print(k)
    for plate in plates:
        print(plate)
        ist = np.linspace(0,1,1000)
        #we use the closed form
        eF,cA = closed_from_random()
        yr = np.array(cA[0:end])
        #alternatively one could calculate the results from the random run
        #this is instructive for checking the averaging procedure
        #yr = np.mean(all_random_onenugget[k][plate],axis=0) #get one for the random averaging to compare to

        yas = rsplit(all_onenugget[k][plate],splits,reps)
        count = 0
        for s in range(splits):
            for rep in range(reps):
                #onenugg
                st = np.array([i for i in range(len(yr))])
                yri,yai = inter(yr,np.mean(yas[rep][s],axis=0),st)
                acc_onenugget[k][plate][count] = (np.array(yri)/np.array(yai))
                count += 1
        x_onenugget[k][plate] = ist #this is the x-value of finding one top percentile sample

        #yr = np.mean(all_random_nuggets[k][plate],axis=0)
        yr = eF[0:end]
        yas = rsplit(all_nuggets[k][plate],splits,reps)
        count = 0
        for s in range(splits):
            for rep in range(reps):
                #ALLNUG
                st = np.array([i for i in range(len(yr))])
                yri,yai = inter(yr,np.mean(yas[rep][s],axis=0),st)
                acc_nugget[k][plate][count] = (np.array(yri)/np.array(yai))
                count += 1
        x_nuggets[k][plate] = ist #this is the x-value of finding all top percentile samples

        yr = np.median(rmconst[k][plate]/all_random_rmses[k][plate],axis=0)
        yas = rsplit(rmconst[k][plate]/all_rmses[k][plate],splits,reps)


        count = 0
        for s in range(splits):
            for rep in range(reps):
                #   MAE
                st = np.array([i for i in range(len(yr))])

                sfr = scipy.signal.savgol_filter(yr,51,3,mode='nearest')
                sfa = scipy.signal.savgol_filter(np.median(yas[rep][s],axis=0),51,3,mode='nearest')

                fr = interpolate.interp1d(sfr,st,kind='linear',bounds_error=False,fill_value=np.nan)
                fa = interpolate.interp1d(sfa,st,kind='linear',bounds_error=False,fill_value=np.nan)


                interpr = max([min(sfr),min(sfa)]),min(max([max(sfr)/max(sfa)]),1)
                ist = np.linspace(0,1,1000)
                yri,yai = [],[]
                for i in ist:
                    yri.append(fr(i))
                    yai.append(fa(i))


                acc_rmses[k][plate][count] = np.array(yri)/np.array(yai)
                x_rmses[k][plate] = ist
                count += 1



import matplotlib.pyplot as plt
from matplotlib import cm
colors = cm.tab10([i/5. for i in range(len(plates))])
cplates = {p:colors[i] for i,p in enumerate(plates)}

#from averaging and bagging we recieve a spread that we can then plot as bands
def plot_data_gen(d,xe,lo=5,hi=95):
    d = d[0:xe]
    x = np.where(np.array([len(np.where(np.isnan(np.array(d)[:,i]))[0]) for i in range(1000)])<xe/5)[0]
    med = np.nanmedian(d,axis=0)
    lower = np.nanpercentile(d,lo,axis=0)
    upper = np.nanpercentile(d,hi,axis=0)
    return med[x],lower[x],upper[x],x

fig,ax = plt.subplots(1,3,figsize=(17/2*1.2,11.5/2*1.2),sharey=True)
ax = ax.flatten()
for plate in plates:
    xe = splits*reps-1
    m,s,S,xi = plot_data_gen(acc_onenugget[mode][plate],xe)
    ax[0].plot(x_onenugget[mode][plate][xi],m,'-',label=plate,color=cplates[plate],linewidth=0.5)
    ax[0].fill_between(x_onenugget[mode][plate][xi],s,S,label=None,color=cplates[plate],alpha=0.2)
    ax[0].plot([0,1],[1,1],'--',color='black',label=None)
    ax[0].set_yscale('log')
    #ax[0].set_xlim(0,1)
    ax[0].set_ylim(0.01,100)


    m,s,S,xi = plot_data_gen(acc_nugget[mode][plate],xe)
    ax[1].plot(x_nuggets[mode][plate][xi],m,'-',label=plate,color=cplates[plate],linewidth=0.5)
    ax[1].fill_between(x_nuggets[mode][plate][xi],s,S,label=None,color=cplates[plate],alpha=0.2)
    ax[1].plot([0,1],[1,1],'--',color='black',label=None)
    ax[1].set_yscale('log')
    #ax[1].set_xlim(0,1)
    ax[1].set_ylim(0.01,100)

    m,s,S,xi = plot_data_gen(acc_rmses[mode][plate],xe)
    ax[2].plot(x_rmses[mode][plate][xi],m,'-',label=plate,color=cplates[plate],linewidth=0.5)
    ax[2].fill_between(x_rmses[mode][plate][xi],s,S,label=None,color=cplates[plate],alpha=0.2)
    ax[2].plot([0,1],[1,1],'--',color='black',label=None)
    ax[2].set_yscale('log')
    ax[2].set_ylim(0.01,100)
    ax[0].set_xlabel('{any}^ALM')
    ax[1].set_xlabel('{all}^ALM}')
    ax[2].set_xlabel('{model}^ALM')
    ax[0].set_ylabel('AF')

ax[2].legend(fontsize='x-small',frameon=False)
for y in range(3):
    ax[y].tick_params(direction='in',which='both')
    #ax[x,y].set_xticks(np.arange(0.25,1,step=0.25))
    if y == 0:
        ax[y].set_xlim(0,1)
        ax[y].set_xticks([0.,0.25,0.5,0.75])
    if y == 1:
        ax[y].set_xlim(0,0.75)
    if y == 2:
        ax[y].set_xlim(0.4,1)
    ax[y].set_yticks([0.01,0.1,1,10,100])
plt.subplots_adjust(wspace=0.05, hspace=0.05)
#plt.tight_layout()
plt.savefig(r'acceleration_plot.svg',dpi=400)
