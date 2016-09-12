'''
CREATE THE SHAPEFILES OF THE CITY

Files needed:
dataTrans.csv  file with data of the blocks of the city, just to get the blocks that actually belong to the city
not all of the state

shapefiles/df_manzanas  all shapefiles for blocks in federal district
shapefiles/mex_manzanas  all shapefiles for blocks in state of mexico

output is a file with the blocks for Greater Mexico City in
cityBlocks/cityBlocks

'''
folderPath = "/Users/db/Dropbox/slums/"

import json
import pandas as pd
import shapefile

def addZeros( x, N):
	if( N > 0):
		x = addZeros( '0'+ x, N - 1)
	return x

def idAGEB_CLAVE(x):
	hexa = format(x,'x')
	return addZeros( hexa, 4 - len(hexa) )

def CLAVEtoID(clave):
	return (int(clave[:9])*100000 + int(clave[9:13],16))*1000 + int(clave[14:])

# file with localities in clean Data
cityData = pd.read_csv(folderPath + 'dataTrans.csv')
cityIds = cityData['ID'].values


df = shapefile.Reader( folderPath + 'shapefiles/' + "df_manzanas" )
mex = shapefile.Reader( folderPath + 'shapefiles/' + "mex_manzanas" )


shapes = df.shapes( )
shapes.extend( mex.shapes( ) )

records = df.records( )
records.extend( mex.records( ) )

#new shapefiles
cityShp= shapefile.Writer(shapefile.POLYGON)
cityShp.field('ID', 'N', 17, 0)
for record, shape in zip( records, shapes ):
	agebId = CLAVEtoID(record[0])
	if agebId in cityIds:
		cityShp.poly([shape.points])
		cityShp.record(agebId)
cityShp.save(folderPath + 'cityBlocks/cityBlocks')