#!/Users/christopherspringob/anaconda/bin/python

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import math
import numpy as np
import pandas as pd
import scipy as sc
import scipy.stats
import seaborn as sns
from scipy.optimize import curve_fit



f1 = open("correlations.txt","w")
f2 = open("npdemos.txt","w")
f3 = open("correlations2.txt","w")


#This program reads in US county demographic data, and then fits a model to predict the population growth rate by county, using the demographic variables as inputs.

#Read in the data:
demos = pd.read_csv('county_facts.csv')

#Only consider counties with populations greater than 10,000, to reduce shot noise:
bigdemo = demos.loc[demos['PST045214'] > 10000]

#Now we want to eliminate big outliers, so we calculate the mean and rms scatter of population growth, and then toss out counties that are more than 3 sigma away from the mean:
stdgrow = bigdemo.std(axis=0)['PST120214']
print(stdgrow)

meangrow = bigdemo.mean(axis=0)['PST120214']
print(meangrow)

bigdemos = bigdemo.loc[(bigdemo['PST120214'] > (meangrow - (3.0*stdgrow))) & (bigdemo['PST120214'] < (meangrow + (3.0*stdgrow)))]

#Now re-calculate the rms scatter with just the remaining counties, as this'll be important later:
newstdgrow = bigdemos.std(axis=0)['PST120214']
print(newstdgrow)


#Make correlation plot:
"""corr = bigdemos.corr()
corr.to_csv('corr.csv')
fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
ax1.grid(True)
plt.title('Demographic Correlations')

plt.savefig('correlations.eps',bbox_inches='tight',pad_inches=0.03)"""


#Now calculate the correlations between population growth and every other parameter, then rank the correlations and output to a file:
grocorr = bigdemos.corr()['PST120214']
grocorder = grocorr.order(kind="quicksort")
print(grocorder,file=f1)

#The top three correlations are with EDU685213 (percentage of the population over age 25 who have a bachelor's degree), INC110213 (median household income), and AGE775214 (percentage of the population over the age of 65).  The first two are positive correlations, while the last one is a negative correlation.

#We now make scatter plots of population growth vs. the three most closely correlated demographic parameters:


bigdemos['logpop']=np.log10(bigdemos['POP010210'])
fs2=14

fig2 = plt.figure(figsize=(11,9))
fig2.subplots_adjust(wspace=0.0,hspace=0.0)
ax2 = fig2.add_subplot(311,xlim=[-17,17],ylim=[20001,130000],autoscale_on=False)
plot2 = ax2.scatter(bigdemos['PST120214'],bigdemos['INC110213'],cmap='jet',s=5,c=bigdemos['logpop'],marker='s',vmin=4.0,vmax=6.0,edgecolors='none')

ax2.set_ylabel("median household income")

ax_cb = fig2.add_axes([0.91,0.62,0.01,0.26])
cb2 = plt.colorbar(plot2,cax=ax_cb,extend='both')
cb2.set_label("log(population)",fontsize=fs2)

#ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))
#ax2.yaxis.set_minor_locator(ticker.MultipleLocator(10))

ax3 = fig2.add_subplot(312,xlim=[-17,17],ylim=[51,99],autoscale_on=False)
plot3 = ax3.scatter(bigdemos['PST120214'],bigdemos['EDU635213'],cmap='jet',s=5,c=bigdemos['logpop'],marker='s',vmin=4.0,vmax=6.0,edgecolors='none')

ax3.set_ylabel("% adults w/ bachelor's degree")

ax_cb3 = fig2.add_axes([0.91,0.36,0.01,0.26])
cb3 = plt.colorbar(plot3,cax=ax_cb3,extend='both')
cb3.set_label("log(population)",fontsize=fs2)




ax4 = fig2.add_subplot(313,xlim=[-17,17],ylim=[0,39],autoscale_on=False)
plot4 = ax4.scatter(bigdemos['PST120214'],bigdemos['AGE775214'],cmap='jet',s=5,c=bigdemos['logpop'],marker='s',vmin=4.0,vmax=6.0,edgecolors='none')

ax4.set_xlabel("% population growth (2010-2014)")
ax4.set_ylabel("% population over 65")

ax_cb4 = fig2.add_axes([0.91,0.10,0.01,0.26])
cb4 = plt.colorbar(plot4,cax=ax_cb4,extend='both')
cb4.set_label("log(population)",fontsize=fs2)

ax2.axes.get_xaxis().set_ticks([])
ax3.axes.get_xaxis().set_ticks([])

#ax5 = fig2.add_subplot(224)
#plot5 = ax5.hist(bigdemos['PST120214'],bins=18)

plt.savefig('scatterplots.eps',bbox_inches='tight',pad_inches=0.03)



#OK, now we want to create an array that we can use to make the model:

shortdemos=bigdemos[['fips', 'PST120214', 'INC110213', 'EDU685213', 'AGE775214']].copy()
npdemos=shortdemos.values
print(npdemos[:,1],file=f2)



print(shortdemos.std(axis=0)['PST120214'])
#print(shortdemos.std(axis=1))


#Now we define the form of the model.  Based on what we observed in the scatter plots, we're going to assume that the relationship with population growth is linear for INC110213 and EDU685213, but quadratic for AGE775214.  So we have this:

def fitFunc(x,a,b,c,d,f):
    return a + b*x[:,2] + c*x[:,3] + d*x[:,4] + f*(x[:,4]**2)

#Here are some initial guesses for the values of the parameters:
p0 = [-7, 0.0002, 0.2, -0.4, 0.0]

#Now do the fit, then calculate the model predictions and residuals:
fitParams, fitCovariances = curve_fit(fitFunc, npdemos, npdemos[:,1], p0)
print(fitParams)
shortdemos['model'] = fitParams[0] + fitParams[1]*shortdemos['INC110213'] + fitParams[2]*shortdemos['EDU685213'] + fitParams[3]*shortdemos['AGE775214'] + fitParams[4]*(shortdemos['AGE775214']**2)
shortdemos['residual'] = shortdemos['PST120214'] - shortdemos['model']

shortdemos.to_csv('shortdemos.csv')

#Merge the model predictions and residuals back into the larger table:
longtable = pd.merge(bigdemos,shortdemos,left_index=True,right_index=True)




#Print the rms scatter of population growth itself, and then the rms scatter of the residual, and then see how much the scatter has been reduced:

print(newstdgrow)
stdresid = longtable.std(axis=0)['residual']
print(stdresid)
print(1.0-(stdresid/newstdgrow))

#Now calculate the R squared value:
meangrow2 = longtable.mean(axis=0)['PST120214_x']
longtable['valsq'] = (longtable['PST120214_x']-meangrow2)**2
meanresid = longtable.mean(axis=0)['residual']
longtable['ressq'] = (longtable['residual']-meanresid)**2
valsqval = longtable.sum(axis=0)['valsq']
ressqval = longtable.sum(axis=0)['ressq']
rsquared = 1.0 - (ressqval/valsqval)
print(rsquared)
longtable.to_csv('longtable.csv')



#Now calculate the correlations between residuals and other parameters, and then print those out:
longcorr = longtable.corr()['residual']
longcorder = longcorr.order(kind="quicksort")
print(longcorder,file=f3)

#Now make a plot of residuals vs. fit:
fig3 = plt.figure(figsize=(11,9))
ax5 = fig3.add_subplot(111,xlim=[-3,12],ylim=[-13,13],autoscale_on=False)
plot5 = ax5.scatter(longtable['model'],longtable['residual'],cmap='jet',s=5,c=longtable['logpop'],marker='s',vmin=4.0,vmax=6.0,edgecolors='none')


ax5.set_xlabel("model % population growth")
ax5.set_ylabel("residual % population growth")

ax_cb5 = fig3.add_axes([0.91,0.10,0.01,0.80])
cb5 = plt.colorbar(plot5,cax=ax_cb5,extend='both')
cb5.set_label("log(population)",fontsize=fs2)

plt.savefig('residfit.eps',bbox_inches='tight',pad_inches=0.03)



f1.close()
f2.close()
f3.close()
#pp.show()
