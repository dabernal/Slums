'''
READ THE DATA, CLEAN, PROCESS AND STORE IT IN A DATABASE

DATA: CENSUS 2010 at a BLOCK level in Mexico city (df) and conurbation (mex), for the CALCULATION of Slum Severity Index (SSI) 
 

TO RUN this code first be sure to have the following files IN ONE FOLDER:

Census data at the level of AGEB and blocks for 2010 in:

Mexico City (Federal District) df: "RESAGEBURB_09XLS.XLS"
State of Mexico mex: "RESAGEBURB_15XLS.XLS"

BE SURE to fill in the variable folderPath with the path to the folder with those files.

OUTCOME in the folderPath:
file "dataTrans.csv" with dataframe ready to be plotted, with slum characteristics: "Water", "Sanitation", "Structure" and "Density"
also it has the slum indexes as attributes with names "Factor", "K_Means", "LC"

files with desicion tree for different attributes: "de_tree.dot","wa_tree.dot",etc 

'''
### FILL IN THIS VARIBLE THE PATH TO THE FOLDER WHERE ALL THE FILES MENTIONED ABOVE CAN BE FOUND:
folderPath = "/Users/db/Dropbox/slums/"

from numpy import *
import json
import pandas as pd
from decimal import Decimal
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.ensemble import RandomForestClassifier


# this list contains the localities that form part of Mexico city and conurbation
cityLocalities = [ '090020001','090030001','090040001','090040020','090050001','090060001','090070001','090080001','090090001','090090011','090090015','090090017','090090024','090090029','090090033','090090036','090090152','090090300','090100001','090110001','090110011','090110021','090110024','090110026','090120001','090120019','090120026','090120027','090130001','090140001','090150001','090160001','090170001','130690001','130690002','130690008','130690010','130690019','130690042','130690044','150020001','150020005','150020008','150020011','150020012','150020015','150020016','150090001','150090005','150090006','150100001','150100007','150110001','150110002','150110004','150110013','150110029','150130001','150150002','150150004','150150006','150150007','150160001','150160006','150160016','150160017','150170001','150200001','150220001','150230001','150240001','150240088','150240111','150240124','150250001','150250002','150250002','150250010','150250012','150250011','150250014','150250016','150250017','150250019','150250020','150250021','150280001','150280004','150280006','150290001','150300001','150310001','150330001','150340001','150350001','150350006','150350008','150350009','150350012','150350013','150350019','150350027','150350028','150350030','150360001','150360006','150360007','150360008','150360009','150370001','150370005','150370009','150370013','150370018','150370019','150370011','150370023','150370024','150370025','150370026','150370071','150370089','150380001','150390001','150390003','150390004','150390007','150390011','150390012','150390064','150390151','150440001','150440020','150460001','150460003','150460005','150500001','150500002','150530001','150530005','150570001','150570088','150570098','150570267','150580001','150590001','150590017','150590058','150600001','150600003','150600004','150600016','150600018','150600020','150600025','150600082','150610001','150610004','150650001','150650007','150650011','150650019','150680001','150680007','150690001','150700001','150700005','150700008','150700009','150700013','150700017','150700019','150700036','150700037','150700038','150700039','150700040','150700041','150700001','150700042','150700043','150750001','150810001','150810004','150810012','150810019','150810025','150810050','150810098','150830001','150840001','150840004','150840011','150840013','150840014','150840017','150890001','150910001','150910010','150920001','150920002','150920013','150920019','150920020','150930001','150930003','150930014','150940001','150940002','150950001','150950005','150950021','150950026','150950073','150950074','150960001','150960008','150990001','150990012','150990016','150990020','150990022','150990024','150990025','150990029','150990030','150990035','150990041','150990042','150990043','150990045','150990048','151000001','151000002','151000007','151030001','151030005','151040001','151040105','151080001','151080014','151080063','151090001','151090003','151090025','151090068','151090069','151090072','151120001','151120011','151200001','151200013','151200014','151200023','151200041','151200045','151200046','151200054','151200056','151200057','151200175','151200198','151200204','151210001','151210020','151220001','151250001']
cityLocalities = map( int, cityLocalities )

# properties OF INTEREST for SSI
slumProperties = [ 'water', 'structure', 'sanitation', 'density', 'houses']

def floatClean(string):
	if string == u'*':
		return nan
	else :
		return float(string)

def addZeros( x, N):
	if( N > 0):
		x = addZeros( '0'+ x, N - 1)
	return x


def invAGEB( x ):
	x = hex(x)[ 2 : ]
	return addZeros( x, 4 - len(x) ).upper()

def fill_ref( x, ref, tot, data ):
	NaNs = data[x].isnull()
	data.loc[NaNs,x] = data.loc[NaNs,tot] - data.loc[NaNs,ref]
	return data

def fill_val( x, val, data ):
	NaNs = data[x].isnull()
	data.loc[NaNs,x] = val
	return data
def fill_3( x , ref1, ref2, tot, data):
	NaNs = data[x].isnull()
	data.loc[ NaNs, x ] = data.loc[ NaNs,tot ]  - data.loc[NaNs, ref1] - data.loc[NaNs, ref2]
	NaNs = data[x].isnull()
	data.loc[ NaNs, x ] = data.loc[ NaNs,tot ]  - data.loc[NaNs, ref1] 
	NaNs = data[x].isnull()
	data.loc[ NaNs, x ] = data.loc[ NaNs,tot ]  - data.loc[NaNs, ref2]

	return data

def fill_mean(attribute, reference, x):
	for i in x.AGEB.unique():
		AGEB = x.AGEB == i
		AGEB_i = x[AGEB].index
		NaNs = x.loc[ AGEB_i, attribute ].isnull()
		NaNs_i = x.loc[AGEB_i,attribute][NaNs].index
		if len(NaNs_i)>0 and len(AGEB_i)-len(NaNs_i)>0:
			x.loc[ NaNs_i , attribute] = (x.loc[ AGEB_i, attribute ]/x.loc[AGEB_i,reference]).mean() * x.loc[NaNs_i, reference]
	return x

def fill_mean_M(attribute, x):
	for i in x.AGEB.unique():
		AGEB = x.AGEB == i
		AGEB_i = x[AGEB].index
		NaNs = x.loc[ AGEB_i, attribute ].isnull()
		NaNs_i = x.loc[AGEB_i,attribute][NaNs].index
		if len(NaNs_i)>0 and len(AGEB_i)-len(NaNs_i)>0:
			x.loc[ NaNs_i , attribute] = x.loc[ AGEB_i, attribute ].mean() 
	return x


# This function reads the data for df and mex localities, index them with the locality key,
# chooses those that belongs to the city and renames the columns of interest for SSI
def cityLevelsData(year):
	df_mex = year
	# READING
	dfFile = pd.ExcelFile( folderPath + "RESAGEBURB_09XLS" + year + ".xls" )
	mexFile = pd.ExcelFile( folderPath + "RESAGEBURB_15XLS" + year + ".xls" )


	df_1 = dfFile.parse(u'ITER AGEB URBANAS MUN 002 - 014')
	df_2 = dfFile.parse(u'ITER AGEB URBANAS MUN 014 - 017')

	mex_1 = mexFile.parse(u'ITER AGEB URBANAS')
	mex_2 = mexFile.parse(u'ITER AGEB URBANAS_01')
	mex_3 = mexFile.parse(u'ITER AGEB URBANAS_02')

	df_mex = df_1.append(df_2, ignore_index = True)
	df_mex = df_mex.append(mex_1, ignore_index = True)
	df_mex = df_mex.append(mex_2, ignore_index = True)
	df_mex = df_mex.append(mex_3, ignore_index = True)

	#to take out \t in the keys
	df_mex.rename(columns=lambda x: x.strip('\t'),inplace=True)

	# identify and INDEX with Locality key (since there are letters in the attribute AGEB we had to change it to hex, to return it use invAGEB )
	df_mex['MUN'] = map( int, df_mex.ENTIDAD*Decimal(1e3) + df_mex.MUN*Decimal(1e0) )

	df_mex['LOC'] = map( int, df_mex.MUN*Decimal(1e4) + df_mex.LOC*Decimal(1e0) )

	df_mex['AGEB'] = map( int, df_mex.LOC*Decimal(1e5)  + array(map( lambda x: int(x,16), df_mex.AGEB ))*Decimal(1e0) )

	df_mex['ID'] = map( int, df_mex.AGEB*Decimal(1e3) + df_mex.MZA*Decimal(1e0))

	df_mex = df_mex.set_index('ID')

	#Stay only with data of the CITY
	# df_mex['CLAVE_LOC'] = map(int,df_mex.ENTIDAD*1e7+df_mex.MUN*1e4+df_mex.LOC)
	df_mex = pd.concat([df_mex[ df_mex.LOC==int(i) ] for i in cityLocalities])

	# SEPARATE the data by LEVEL of AGREGGATION
	df_mex['NOM_LOC'] = map(lambda x: x.lower(), df_mex.NOM_LOC)


	#first level (biggest) is district
	l1 = df_mex[df_mex['NOM_LOC']==u'total del municipio']
	# then  locality
	l2 = df_mex[df_mex['NOM_LOC']==u'total de la localidad urbana']
	# then group of blocks
	l3 = df_mex[df_mex['NOM_LOC']==u'total ageb urbana']
	# last (smallest) per block. We will work with this data
	l4 = df_mex[(df_mex['NOM_LOC']!=u'total del municipio') & (df_mex['NOM_LOC']!=u'total de la localidad urbana') & (df_mex['NOM_LOC']!=u'total ageb urbana') & (df_mex['NOM_LOC']!=u'total de la entidad') ]

	return l1, l2, l3, l4


def dataCleaning(x, columns=['NOM_ENT','NOM_MUN', 'NOM_LOC'],N=184):

	#Drop duplicates, columns with id names and N/D
	x = x.drop_duplicates( )
	x = x.drop( columns, axis=1 )
	x = x.drop( x[ x.applymap(lambda y: y == 'N/D').any(axis=1) ].index )
	# x = x.drop( x[ x.OCUPVIVPAR == u'N/D'] )

	# Convert to float
	for i in x.keys()[6:]:
		x.loc[:,i] = map(floatClean,x[i])

	# Remove samples with practicaly no data (correspond to estimates/ no info from occupants) and columns with no data
	# x = x.drop( x[ x.VIVTOT < 3] )
	# x = x.drop( x[ x.POBTOT < 3] )
	x = x.drop( x[ x.isnull().sum(axis=1) > N ].index )
	x = x.drop( x.keys()[x.isnull().sum(axis=0) > 100000], axis=1 )
	x = x.drop( x[ x.OCUPVIVPAR < 3].index )
	x = x.drop( x[ x.OCUPVIVPAR.isnull()].index )
	x.loc[:,'Houses'] = map(round,x['OCUPVIVPAR'].divide(x['PROM_OCUP']))

	# Remove Inconsistencies
	x = x.drop( x[ x.POBTOT > 5000].index )
	x = x.drop( x[ x.PRO_OCUP_C > x.PROM_OCUP].index)

	#Fill with opposite
	necessary = [(u'VPH_PISODT', u'VPH_PISOTI'), (u'VPH_1DOR', u'VPH_2YMASD'), (u'VPH_C_ELEC', u'VPH_S_ELEC'), (u'VPH_AGUADV', u'VPH_AGUAFV'),  (u'VPH_DRENAJ', u'VPH_NODREN') ]
	for i in necessary:
		x = fill_ref( i[0], i[1], 'Houses', x )
		x = fill_ref( i[1], i[0], 'Houses', x )

	x = fill_3( u'VPH_1CUART', u'VPH_2CUART', u'VPH_3YMASC', 'Houses', x )
	x = fill_3( u'VPH_2CUART', u'VPH_1CUART', u'VPH_3YMASC', 'Houses', x )
	x = fill_3( u'VPH_3YMASC', u'VPH_1CUART', u'VPH_2CUART', 'Houses', x )

	#Population
	x = fill_ref(u'POBMAS',u'POBFEM','POBTOT', x)
	x = fill_ref(u'POBFEM',u'POBMAS','POBTOT', x)

	for i in x.loc[:,'P_0A2':'P_60YMAS'].keys():
		if i[-1]!='F' and i[-1] !='M':
			x = fill_ref( i +'_F', i+'_M', i, x )
			x = fill_ref( i +'_M', i+'_F', i, x )


	x = fill_3( u'POB0_14', u'POB15_64', u'POB65_MAS', 'POBTOT', x )
	x = fill_3( u'POB15_64', u'POB0_14',  u'POB65_MAS', 'POBTOT', x )
	x = fill_3( u'POB65_MAS', u'POB0_14', u'POB15_64', 'POBTOT', x )

	for i in x.loc[:,'PNACENT': 'P3HLI_HE'].keys():
		if i[-1]!='F' and i[-1] !='M':
			x = fill_ref( i +'_F', i+'_M', i, x )
			x = fill_ref( i +'_M', i+'_F', i, x )

	x = fill_ref( u'P5_HLI_NHE', u'P5_HLI_HE', u'P5_HLI', x )
	x = fill_ref( u'P5_HLI_NHE', u'P5_HLI_HE', u'P5_HLI', x )


	x = fill_ref( 'P3A5_NOA_F', 'P3A5_NOA_M', 'P3A5_NOA', x )
	x = fill_ref( 'P3A5_NOA_M', 'P3A5_NOA_F', 'P3A5_NOA', x )


	for i in x.loc[:,'P6A11_NOA': 'P12A14NOA'].keys():
		if i[-1]!='F' and i[-1] !='M':
			x = fill_ref( i +'F', i+'M', i, x )
			x = fill_ref( i +'M', i+'F', i, x )


	for i in x.loc[:,'P15A17A': 'P15YM_SE'].keys():
		if i[-1]!='F' and i[-1] !='M':
			x = fill_ref( i +'_F', i+'_M', i, x )
			x = fill_ref( i +'_M', i+'_F', i, x )

	for i in x.loc[:,'P15PRI_IN': 'P15SEC_CO'].keys():
		if i[-1]!='F' and i[-1] !='M':
			x = fill_ref( i +'F', i+'M', i, x )
			x = fill_ref( i +'M', i+'F', i, x )

	for i in x.loc[:,'P18YM_PB': 'PDESOCUP'].keys():
		if i[-1]!='F' and i[-1] !='M':
			x = fill_ref( i +'_F', i+'_M', i, x )
			x = fill_ref( i +'_M', i+'_F', i, x )
	#Homes

	x = fill_ref( u'HOGJEF_M', u'HOGJEF_F', u'TOTHOG', x )
	x = fill_ref( u'HOGJEF_F', u'HOGJEF_M', u'TOTHOG', x )


	x = fill_ref( 'PHOGJEF_M', 'PHOGJEF_F', 'POBHOG', x )
	x = fill_ref( 'PHOGJEF_F', 'PHOGJEF_M', 'POBHOG', x )

	# Fill with mean

	for i in ['PROM_HNV', 'GRAPROES', 'GRAPROES_M', 'GRAPROES_F', 'PROM_OCUP', 'PRO_OCUP_C','REL_H_M']:
		x = fill_mean_M(i,x)

	for i in x.loc[:,'P_0A2':'P_60YMAS_F'].keys(): 
		print i
		x = fill_mean(i,'POBTOT', x)

	for i in x.loc[:,'POB0_14':'POB65_MAS'].keys():
		print i
		x = fill_mean(i,'POBTOT', x)

	for i in x.loc[:, 'PNACENT':'P18YM_PB_F'].keys():
		print i
		x = fill_mean(i,'POBTOT', x)

	for i in x.loc[:, 'PEA':'PSIN_RELIG'].keys():
		print i
		x = fill_mean(i,'POBTOT', x)

	for i in x.loc[:,'TOTHOG':'PHOGJEF_F'].keys():
		print i
		x = fill_mean(i,'Houses', x)

	for i in x.loc[:,'VPH_PISODT': ].keys():
			print i
			x = fill_mean(i,'Houses', x)
	# Fill with one the rest 
	x.fillna(1)
	return x
	
def dataTransformations(x):

	x.rename(columns={'OCUPVIVPAR': 'Dwellers'}, inplace=True)
	#water
	x['Water'] = x['VPH_AGUAFV']/x['Houses']

	#Sanitation use VPH_EXCSA and VPH_NODREN
	x['Sanitation'] = (x['Houses'] - x['VPH_EXCSA'] + x['VPH_NODREN']) / (2.*x['Houses'])

	#Overcrowding use VPH_1CUART and PRO_OCUP_C
	# x['Density'] = 1. - 1./(1. +x['PRO_OCUP_C'])
	x['Density'] = x['PRO_OCUP_C']-2.
	x.loc[x.Density<0,'Density'] = 0.
	x['Density'] = 1. - 1./(1. + x.Density)
	x['Density'] = x['Density']/x['Density'].max()
	
	#Structure VPH_1CUART and VPH_PISOTI
	x['Structure'] = (x['VPH_PISOTI'] + x['VPH_1CUART']) / (2*x['Houses'])

	ssiData = pd.DataFrame(normalize(x[['Water','Structure','Density','Sanitation']],axis=0), columns=['Water','Structure','Density','Sanitation'])

	# x.loc[:,'Factor'] = zeros(len(x)	
	facAn = FactorAnalysis(n_components = 1)
	facAn.fit(ssiData)
	x.loc[:,'Factor'] = dot(facAn.components_**2,transpose(ssiData.values))[0]

	#K-Means
	k_meansX = ssiData

	# do the clustering
	k_means = KMeans(n_clusters=4)
	k_means.fit(k_meansX) 
	x.loc[:,'K_Means'] = k_means.labels_

	#linear combination

	x.loc[:,'LC'] = x[['Water','Structure','Sanitation']].sum(axis=1) + (x['PRO_OCUP_C']/ x['PRO_OCUP_C'].max())

	


	#save x to csv
	# x.to_csv(folderPath+'dataTrans.csv')
	return x

def prepareMining(x):
	X = x.copy()
	#get percentages
	for i in x.loc[:,'P_0A2':'P_60YMAS_F'].keys():
		X[i] = x[i] / x['POBTOT'].values
	for i in x.loc[:,'POB0_14':'POB65_MAS'].keys():
		X[i] = x[i] / x['POBTOT'].values
	for i in x.loc[:,'PNACENT':'P18YM_PB_F'].keys():
		X[i] = x[i] / x['POBTOT'].values
	for i in x.loc[:,'PEA':'PSIN_RELIG'].keys():
		X[i] = x[i] / x['POBTOT'].values
	for i in x.loc[:,'TOTHOG':'PHOGJEF_F'].keys():
		X[i] = x[i] / x['POBTOT'].values
	for i in x.loc[:,'VPH_PISODT': ].keys():
		X[i] = x[i] / x['POBTOT'].values
	X = x.drop(x.columns[161:179],axis=1)
	X = X.drop(x.columns[190:],axis=1)
	return X


def binAttribute0(y, classes=5):
	binned_y = pd.DataFrame({'Class':zeros(len(y))})
	dClass = 100/classes
	clas = 0
	binned_y.loc[y==0.] = clas
	for i in range(0,100, dClass ):
		clas +=1
		binned_y.loc[ y.between( percentile(y[y>0.0],i), percentile(y[y>0],i+dClass)),'Class'] = clas
		
	return binned_y

def binAttribute(y, classes=5):
	binned_y = pd.DataFrame({'Class':zeros(len(y))})
	dClass = 100/classes
	clas = 0
	for i in range(0,100, dClass ):
		binned_y.loc[ y.between( percentile(y,i), percentile(y,i+dClass)),'Class'] = clas
		clas +=1
	return binned_y

# Data selection for mining

# X = prepareMining(x)

def dataMining(X,y, classes=5, name = "tree.dot", max_depth=5, binData=1):
	if binData == 1:
		Y_train = binAttribute0(y,classes)
	else:
		Y_train = y
	Y_train = np.asarray(Y_train, dtype="|S6") 
	
	dTree = tree.DecisionTreeClassifier(random_state=0,max_depth = max_depth)
	dTree = dTree.fit(X, Y_train)

	with open(folderPath + name, 'w') as f:
		f = tree.export_graphviz(dTree, out_file=f)
	return dTree




l1, l2, l3, l4 = cityLevelsData( '10' )
blocksClean = dataCleaning( l4 )
blocksAll = dataTransformations( blocksClean )
blocksAll.to_csv( folderPath + "dataTrans.csv")

X_train = prepareMining( blocksAll )


y_labels = blocksAll['Density']
de_tree = dataMining(X_train, y_labels,name='de_tree.dot')

y_labels = blocksAll['Structure']
st_tree = dataMining(X_train, y_labels, name='st_tree.dot')

y_labels = blocksAll['Sanitation']
sa_tree = dataMining(X_train, y_labels, name='sa_tree.dot')

y_labels = blocksAll['Water']
wa_tree = dataMining(X_train, y_labels, name='wa_tree.dot')

y_labels = blocksAll['Factor']
fa_tree = dataMining(X_train, y_labels, name='fa_tree.dot')

y_labels = blocksAll['K_Means']
km_tree = dataMining(X_train, y_labels, name='km_tree.dot', binData = 0)

y_labels = blocksAll['LC']
lc_tree = dataMining(X_train, y_labels,  name='lc_tree.dot')








