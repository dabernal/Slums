#temporal analisis of the slum index
from numpy import *
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis

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

def getRates(year,keysList):
	#year as string '90'
	#keysList must have [totViv, water, dirtfloor, toilet, AvrPers]
	#read
	df = pd.read_excel("/Users/D/Dropbox/slums/ITER_09XLS" + year + ".xls")
	mex = pd.read_excel("/Users/D/Dropbox/slums/ITER_15XLS" + year + ".xls")
	df_mex = df.append(mex, ignore_index = True)
	#to take out \t in the keys
	df_mex.rename(columns=lambda x: x.strip('\t'),inplace=True)

	#identify
	df_mex['ID'] = map(int,df_mex.ENTIDAD*1e7+df_mex.MUN*1e4+df_mex.LOC)

	#prepareData
	rates = df_mex[['ID']]

	#then we calculate the rates for the quantities we are interested
	

	# The rates we are going to use for the slum index
	rates['NoWater']=1.0-freq(df_mex,keysList[1],keysList[0])
	rates['DirtFloor']=1.0-freq(df_mex,keysList[2],keysList[0])
	rates['NoSewage']=1.0-freq(df_mex,keysList[3],keysList[0])
	rates['AvrPersPerRoom']=map(floatClean,df_mex[ keysList[4]])
	# To set the scale between 0 and 1. 1 being 0 habitants
	rates['AvrPersPerRoom']= 1.-1./(rates.AvrPersPerRoom + 1.)
	return rates

def SlumIndex(rates):
	#calculate Slum index

	#PCA with 4 components
	# pca = PCA(n_components=4)
	# pca.fit(rates[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']])

	# #Here we STORE PARAMETER FOR SLUM IMPACT using the weights and vectors from pca
	# rates['SlumIndex'] = zeros(len(rates))
	# weights=pca.explained_variance_ratio_
	# #pca.components_ are the transformation vectors, rates are the original ones
	# new_vectors = dot(transpose(pca.components_), transpose(rates[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']].values))
	
	# #Finally we get the index with the eigenvalues
	# rates['SlumIndex'] = transpose(dot(weights,new_vectors))

	# pca = PCA(n_components=4)
	# new_vectors = pca.fit_transform(rates[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']])
	# rates['SlumIndex'] = dot(pca.explained_variance_ratio_,transpose(new_vectors))
	facAn = FactorAnalysis(n_components = 1)
	facAn.fit(rates[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']])
	rates['SlumIndex'] = dot(facAn.components_**2,transpose(rates[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']].values))[0]
	
	# rates['SlumIndex'] = rates[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']].values.sum(axis=1)
	return rates[['ID','SlumIndex']]

# this function inserts data to the existing database (destination) of another year (origin)
def assignColumn(destination,origin,year,columnName='SlumIndex',missingValue=nan):
	destination[year] = zeros(len(destination))*missingValue
	for i in range(len(origin)):
		destination.loc[destination['ID']==origin.ID[i],year] = origin[columnName][i]


#We need to track all the different localities with their ID's
#since there are new localities or some change
localitiesID = array([])

#First we get the rates (frequencies)

# 90 also has roof TECHO_LA, wall PARED_LA, electricity C_E_ELEC owner of the house VIV_PPROP
# floor value refers to houses without dirtfloor, also with water and with sewer
rates90 = getRates('90',['T_VIVHAB','C_AGUA_ENT','PISO_TIE','C_DRENAJE','PROM_CUA'])
localitiesID = sort(unique(append(localitiesID,rates90.ID.values)))
# VP_ELECTR, VP_TECDES VP_PARDES VP_SERSAN, VP_PROPIA
rates00 = getRates('00',['TOTVIVHAB','VP_AGUENT','VP_PISDES','VP_DRENAJ','PRO_OCVP'])
localitiesID = sort(unique(append(localitiesID,rates00.ID.values)))
rates05 = getRates('05',['T_VIVHAB','VPH_AGDV','VPH_PIDT','VPH_DREN','PRO_C_VP'])
localitiesID = sort(unique(append(localitiesID,rates05.ID.values)))
rates10 = getRates('10',['TVIVHAB','VPH_AGUADV','VPH_PISODT','VPH_DRENAJ','PRO_OCUP_C'])
localitiesID = sort(unique(append(localitiesID,rates10.ID.values)))

# Then we calculate the slum index and store all in one database
slumIndexHist = pd.DataFrame(localitiesID,columns=['ID'])

newYear = SlumIndex(rates90)
assignColumn(slumIndexHist,newYear,'90')

newYear = SlumIndex(rates00)
assignColumn(slumIndexHist,newYear,'00')

newYear = SlumIndex(rates05)
assignColumn(slumIndexHist,newYear,'05')

newYear = SlumIndex(rates10)
assignColumn(slumIndexHist,newYear,'10')

slumIndexHist[['90','00','05','10']].boxplot() 
