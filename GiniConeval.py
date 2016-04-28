from numpy import *
import pandas as pd
from sklearn.decomposition import PCA

def str3(num):
	return '%03d' % (num,)
def str4(num):
	return '%04d' % (num,) 
def floatClean(string):
	if string == u'*' or string == u'N/D':
		return 0.0
	else :
		return float(string)
def freq(data,index,reference):
	return array(map(floatClean,data[index]))/array(map(floatClean,data[reference]))

agebExcel = pd.ExcelFile("/Users/D/Dropbox/slums/RESAGEBURB_09XLS10.xls")
d1Ageb = agebExcel.parse(u'ITER AGEB URBANAS MUN 002 - 014')
d2Ageb =agebExcel.parse(u'ITER AGEB URBANAS MUN 014 - 017')
dataLoc2010 = pd.read_excel("/Users/D/Dropbox/slums/DatosDF2010.xls")
dataAgeb2010 = d1Ageb.append(d2Ageb,ignore_index=True)

conevalIndex = pd.read_excel("/Users/D/Dropbox/slums/conevalIndex.xlsx")

#cleaning


#ID

dataLoc2010['CLAVE_LOC'] = map(int,9e7+dataLoc2010.MUN*1e4+dataLoc2010.LOC)

#level of aggregation


# To compare the coneval index
freqLoc = pd.DataFrame(columns=['Index'])
freqLoc['Index'] = arange(len(dataLoc2010))

freqLoc['NoSewage']=freq(dataLoc2010,'VPH_NODREN','VIVTOT')
freqLoc['NoHealthIns']=freq(dataLoc2010,'PSINDER','POBTOT')
freqLoc['NoElec']=freq(dataLoc2010,'VPH_S_ELEC','VIVTOT')
freqLoc['NoWater']=freq(dataLoc2010,'VPH_AGUAFV','VIVTOT')
freqLoc['DirtFloor']=freq(dataLoc2010,'VPH_PISOTI','VIVTOT')
freqLoc['Toilet']=freq(dataLoc2010,'VPH_EXCSA','VIVTOT')
freqLoc['AvrPersPerRoom']=map(floatClean,dataLoc2010['PRO_OCUP_C'])

#freqLoc.ID = lev4_Clean.ID.values

# Some frequencies

dataAgeb2010['CLAVE_LOC'] = map(int,9e7+dataAgeb2010.MUN*1e4+dataAgeb2010.LOC)
dataAgeb2010['MUN']= map(str3,dataAgeb2010['MUN'])
dataAgeb2010['LOC']= map(str4,dataAgeb2010['LOC'])
dataAgeb2010['MZA']= map(str3,dataAgeb2010['MZA'])
dataAgeb2010['ID'] = dataAgeb2010['MUN']+dataAgeb2010['LOC']+dataAgeb2010['AGEB']+dataAgeb2010['MZA']

level1_2010 = dataAgeb2010[dataAgeb2010['NOM_LOC']==u'Total del municipio']
level2_2010 = dataAgeb2010[dataAgeb2010['NOM_LOC']==u'Total de la localidad urbana']
level3_2010 = dataAgeb2010[dataAgeb2010['NOM_LOC']==u'Total AGEB urbana']
level4_2010 = dataAgeb2010[(dataAgeb2010['NOM_LOC']!=u'Total del municipio') & (dataAgeb2010['NOM_LOC']!=u'Total de la localidad urbana') & (dataAgeb2010['NOM_LOC']!=u'Total AGEB urbana') & (dataAgeb2010['NOM_LOC']!=u'Total de la entidad') ]

lev4_Clean = level4_2010[(level4_2010.VIVTOT > 10) & (level4_2010.POBTOT > 30)]


freqLev4 = pd.DataFrame(columns=['Index','ID'])
freqLev4.Index = arange(len(lev4_Clean.ID))
freqLev4.ID = lev4_Clean.ID.values
freqLev4['CLAVE_LOC'] = lev4_Clean.CLAVE_LOC.values

freqLev4['NoWater']=freq(lev4_Clean,'VPH_AGUAFV','VIVTOT')
freqLev4['DirtFloor']=freq(lev4_Clean,'VPH_PISOTI','VIVTOT')
freqLev4['Toilet']=freq(lev4_Clean,'VPH_EXCSA','VIVTOT')
freqLev4['AvrPersPerRoom']=map(floatClean,lev4_Clean['PRO_OCUP_C'])
freqLev4['NoSewage']=freq(lev4_Clean,'VPH_NODREN','VIVTOT')



#NEW PARAMETER FOR SLUM IMPACT
# weights = rand(4)
# freqLev4['SlumIndex']=weights[0]*freqLev4.NoWater + weights[1]*freqLev4.DirtFloor + weights[2]*freqLev4.AvrPersPerRoom - weights[3]*freqLev4.Toilet

#Comparison of slum Index and Coneval
dataLoc2010['conevalIndex'] = zeros(len(dataLoc2010))

for i in range(len(conevalIndex)):
	dataLoc2010.loc[dataLoc2010[dataLoc2010.CLAVE_LOC == conevalIndex.Clave_Loc[i]].index.values[0],'conevalIndex'] = conevalIndex.conevalIndex[i] 


dataLoc2010['SlumIndex'] = zeros(len(dataLoc2010))
#PCA using sklearn

pca = PCA(n_components=2)
pca.fit(freqLoc[['NoWater','DirtFloor','AvrPersPerRoom','Toilet']][dataLoc2010.conevalIndex != 0.0])
weights=pca.explained_variance_ratio_
new_vectors = dot(pca.components_, transpose(freqLoc[['NoWater','DirtFloor','AvrPersPerRoom','Toilet']][dataLoc2010.conevalIndex != 0.0].values))
dataLoc2010['SlumIndex'][dataLoc2010.conevalIndex != 0.0]= transpose(dot(weights,new_vectors))

#Prueba con indice 2 pues Toilet casi todos tienen
dataLoc2010['SlumIndex2'] = zeros(len(dataLoc2010))
#PCA using sklearn

pca = PCA(n_components=2)
pca.fit(freqLoc[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']][dataLoc2010.conevalIndex != 0.0])
weights=pca.explained_variance_ratio_
new_vectors = dot(pca.components_, transpose(freqLoc[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']][dataLoc2010.conevalIndex != 0.0].values))
dataLoc2010['SlumIndex2'][dataLoc2010.conevalIndex != 0.0]= transpose(dot(weights,new_vectors))

#correlation of .7
# weights = r_[.1,.1,.1,.1]
# dataLoc2010['SlumIndex_w1']=weights[0]*freqLoc.NoWater + weights[1]*freqLoc.DirtFloor + weights[2]*freqLoc.AvrPersPerRoom - weights[3]*freqLoc.Toilet


#GINI coeficient

def gini(x):
	return sum([abs(xi-xj) for xi in x for xj in x])/(2*sum(x)*len(x))

dataLoc2010['giniWater'] = zeros(len(dataLoc2010))
dataLoc2010['giniFloor'] = zeros(len(dataLoc2010))
dataLoc2010['giniToilet'] = zeros(len(dataLoc2010))
dataLoc2010['giniAPR'] = zeros(len(dataLoc2010))

urbanLocalitiesNumber = dataLoc2010.CLAVE_LOC[(dataLoc2010.POBTOT > 2500) & (dataLoc2010.conevalIndex != 0.0)]

for i in urbanLocalitiesNumber.index:
	print i
	locId = urbanLocalitiesNumber[i]
	blocksInLocality = freqLev4.CLAVE_LOC == locId
	dataLoc2010['giniWater'][i] = gini(freqLev4.NoWater[blocksInLocality])
	dataLoc2010['giniFloor'][i] = gini(freqLev4.DirtFloor[blocksInLocality])
	dataLoc2010['giniToilet'][i] = gini(freqLev4.Toilet[blocksInLocality])
	dataLoc2010['giniAPR'][i] = gini(freqLev4.AvrPersPerRoom[blocksInLocality])

# SLUM INDEX

freqLev4['SlumIndex'] = zeros(len(freqLev4))
#PCA using sklearn

pca = PCA(n_components=4)
pca.fit(freqLev4[['NoWater','DirtFloor','AvrPersPerRoom','Toilet']])
weights=pca.explained_variance_ratio_
new_vectors = dot(pca.components_, transpose(freqLev4[['NoWater','DirtFloor','AvrPersPerRoom','Toilet']].values))
freqLev4['SlumIndex']= transpose(dot(weights,new_vectors))

#prueba 2
freqLev4['SlumIndex2'] = zeros(len(freqLev4))
#PCA using sklearn

# pca = PCA(n_components=2)
# pca.fit(freqLev4[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']])
# weights=pca.explained_variance_ratio_
# new_vectors = dot(pca.components_, transpose(freqLev4[['NoWater','DirtFloor','AvrPersPerRoom','NoSewage']].values))
# freqLev4['SlumIndex2']= transpose(dot(weights,new_vectors))

dataLoc2010['giniSlumIndex'] = zeros(len(dataLoc2010))
for i in urbanLocalitiesNumber.index:
	print i
	locId = urbanLocalitiesNumber[i]
	blocksInLocality = freqLev4.CLAVE_LOC == locId
	dataLoc2010['giniSlumIndex'][i] = gini(freqLev4.SlumIndex[blocksInLocality])


# Analizing similarities between the two indexes

urbanLocalities = dataLoc2010[(dataLoc2010.POBTOT > 2500) & (dataLoc2010.conevalIndex != 0.0)]

plot(urbanLocalities.conevalIndex,urbanLocalities.SlumIndex,'o')
corrcoef(urbanLocalities.conevalIndex,urbanLocalities.SlumIndex)

# figure(2)
# freqSort2 = freqLev4.sort(columns=['CLAVE_LOC','SlumIndex'])
# plot(freqSort2.AvrGrade,'o')
# xlabel('Urban units ordered by vale of Slum Index divided by locality')
# ylabel('Average Level of Scholarity')
# title('Average grade of the population ordered by increasing Slum Index')

#To check out distribution of Slum Index in each Locality
#plot(freqSort2.SlumIndex,'o')
#hist(freqSort.SlumIndex[(freqSort.SlumIndex<2) & (freqSort.CLAVE_LOC==90090033)].values,bins=30)


# Some plots to show correlation between SLum Index and other ambitos
freqLev4['Employed']=freq(lev4_Clean,'POCUPADA','PEA')
freqLev4['AvrGrade'] = map(floatClean, lev4_Clean['GRAPROES'])
freqLev4['NoHealthIns']=freq(lev4_Clean,'PSINDER','POBTOT')
freqLev4['Fecundity']=map(floatClean, lev4_Clean['PROM_HNV'])
freqLev4['NonImigrants']=freq(lev4_Clean,'PNACENT','POBTOT')
freqLev4['Indigenous']=freq(lev4_Clean,'P3YM_HLI','P_3YMAS')

freqSort = freqLev4.sort('SlumIndex')

# School
#freqLev4['NoSchool']=freq(lev4_Clean,'P15YM_SE','P_15YMAS')
#freqLev4['NoPrimary']=freq(lev4_Clean,'P15PRI_IN','P_15YMAS')
#freqLev4['Primary']=freq(lev4_Clean,'P15PRI_CO','P_15YMAS')
#freqLev4['Secondary']=freq(lev4_Clean,'P15SEC_COM','P_15YMAS')
#freqLev4['HigherSchool']=freq(lev4_Clean,'P18YM_PB','P_18YMAS')

plot(freqSort.AvrGrade,'o')
xlabel('Urban units ordered by value of Slum Index')
ylabel('Average Level of Scholarity')
title('Correlation between SI and Average scholarity level achieved by urban unit')
#Correlation between SI and Average scholarity level achieved by urban unit



# Work (Strange distribution with skips, almost no correlation with
# values evrywhere)

#freqLev4['UnEmployed']=freq(lev4_Clean,'PDESOCUP','PEA')
plot(freqSort.Employed,'o')
xlabel('Urban units ordered by value of Slum Index')
ylabel('Average Level of Scholarity')
title('Correlation with employment')

# With Amenities
# freqLev4['Toilet']=freq(lev4_Clean,'VPH_EXCSA','VIVTOT')
# freqLev4['OtherFloor']=freq(lev4_Clean,'VPH_PISODT','VIVTOT')
# freqLev4['AllServices']=freq(lev4_Clean,'VPH_C_SERV','VIVTOT')

# # Lacking Amenities
# freqLev4['NoSewage']=freq(lev4_Clean,'VPH_NODREN','VIVTOT')
# freqLev4['NoElec']=freq(lev4_Clean,'VPH_S_ELEC','VIVTOT')
# freqLev4['NoWater']=freq(lev4_Clean,'VPH_AGUAFV','VIVTOT')
# freqLev4['NoRadioCarCompEtc']=freq(lev4_Clean,'VPH_SNBIEN','VIVTOT')


# Health
plot(freqSort.NoHealthIns,'o')
xlabel('Urban units ordered by value of Slum Index')
ylabel('No Insurance')
title('Correlation with health')

#Fecundity
plot(freqSort.Fecundity,'o')
xlabel('Urban units ordered by value of Slum Index')
ylabel('Average Children Born')
title('Correlation with Fecundity')

#Migration
plot(freqSort.NonImigrants,'o')
xlabel('Urban units ordered by value of Slum Index')
ylabel('Native')
title('Correlation with Migration')

#Indigenous
plot(freqSort.Indigenous,'o')
xlabel('Urban units ordered by value of Slum Index')
ylabel('Indigenous People')
title('Correlation with Indegenous population')


