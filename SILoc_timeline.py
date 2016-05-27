#temporal analisis of the slum index
from numpy import *
import pandas as pd
from sklearn.decomposition import PCA

def str2(num):
	return u'%02d' % (num,)
def str3(num):
	return u'%03d' % (num,)
def str4(num):
	return u'%04d' % (num,) 
def floatClean(string):
	if string == u'*' or string == u'N/D':
		return 0.0
	else :
		return float(string)
def freq(data,index,reference):
	return array(map(floatClean,data[index]))/array(map(floatClean,data[reference]))

def SlumIndex(year, keysList):
	#year as string '90'
	#keysList must have [totpop, water, dirtfloor, toilet, AvrPers]
	#read
	df = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS" + year + ".xls")
	mex = pd.read_excel"/Users/D/Dropbox/slums/ITER_15XLS" + year + ".xls")
	df_mex = df.append(mex, ignore_index = True)

	#identify
	df_mex['CLAVE_LOC'] = map(df_mex.ENTIDAD*1e7+df_mex.MUN*1e4+df_mex.LOC)

	#prepareData
	frequencies = df_mex[['CLAVE_LOC']]

	#then we calculate for the quantities we are interested in the localities
	#as well as in the data at all levels of aggregation

	# The frequencies we are going to use for the slum index
	frequencies['NoWater']=freq(df_mex,keysList[1],keysList[0])
	frequencies['DirtFloor']=freq(df_mex,keysList[2],keysList[0])
	frequencies['Toilet']=freq(df_mex,keysList[3],keysList[0])
	frequencies['AvrPersPerRoom']=map(floatClean,df_mex[ keysList[4]])
	# To set the scale between 0 and 1. 1 being 0 habitants
	frequencies['AvrPersPerRoom']= 1.-1./(frequencies.AvrPersPerRoom + 1.)

	#calculate Slum index
	pca = PCA(n_components=4)
	pca.fit(frequencies[['NoWater','DirtFloor','AvrPersPerRoom','Toilet']])

	#Here we STORE PARAMETER FOR SLUM IMPACT using the weights and vectors from pca
	df_mex['SlumIndex'] = zeros(len(df_mex))
	weights=pca.explained_variance_ratio_
	new_vectors = dot(pca.components_, transpose(frequencies[['NoWater','DirtFloor','AvrPersPerRoom','Toilet']].values))
	#In the localities we only need to calculate for the urban ones, the rural 
	#doesn't have data at the block level
	df_mex['SlumIndex'] = transpose(dot(weights,new_vectors))

	return df_mex.CLAVE_LOC.values, df_mex.SlumIndex.values






# The localities file is easier to read since its in 1 page
df_90 = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS90.xls")
df_95 = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS95.xls")
df_00 = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS00.xls")
df_05 = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS05.xls")
df_10 = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS10.xls")

mex_90 = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS90.xls")
mex_95 = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS95.xls")
mex_00 = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS00.xls")
mex_05 = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS05.xls")
mex_10 = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS10.xls")


localidadesMex = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS10.xls")
df_mex = df_mex.append(localidadesMex, ignore_index=True)

#we establish a key for each locality ( later we use this for the gini coef)

df_mex['CLAVE_LOC'] = map(int,df_mex.ENTIDAD*1e7+df_mex.MUN*1e4+df_mex.LOC)
# In the other data we also establish the same number so we know to wich locality 
