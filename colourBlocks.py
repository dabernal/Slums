"""
CREATES A MAP OF THE BLOCKS IN MEXICO CITY USING DATA
FILES NEEDED:
"dataTrans.csv" file that contan the data to be plotted
FOLDER NEEDED:
cityBlocks with the shapefiles for the city

Output:
Map with blocks coloured according to values of an attribute in the dataframe 

"""

from mpl_toolkits.basemap import Basemap
from matplotlib.colors import rgb2hex

folderPath = "/Users/db/Dropbox/slums/"

import json
import pandas as pd
from sklearn.preprocessing import normalize

def getColor(value,minV=0.,maxV=.008,cmap=cm.jet):
	return rgb2hex(cmap((value - minV)/(maxV-minV)))


# with open('/Users/db/Dropbox/slums/agebSI.json') as f:
#     slumIndexDict = json.load(f)
# maxSI = max(slumIndexDict.values())

# read the data
cityData = pd.read_csv(folderPath + 'dataTrans.csv')
cityData = cityData.set_index('ID')

#Parameters for font size and family in map
rcParams['font.size'] = 18.
rcParams['font.family'] = 'Sans Serif'
rcParams['axes.labelsize'] = 18.
rcParams['xtick.labelsize'] = 14.
rcParams['ytick.labelsize'] = 14.

fig = figure(figsize=(10, 8))

# Coordinates of Mexico City (DF)
DF = [-99.0, 17.42]
shift = [0, 2]
#Characteristics of the map 
width = 110000
height = 110000
res = 'l'
proj = 'tmerc'

#Map for drawing the districts (l1), so we dont need to read sevral times a big file
m_mxLoc = Basemap(width=width, height=height, resolution=res, projection=proj,
            lon_0=DF[0]+shift[0], lat_0=DF[1]+shift[1])

#Map for drawing the blocks (l3)
m_mx = Basemap(width=width, height=height, resolution=res, projection=proj,
            lon_0=DF[0]+shift[0], lat_0=DF[1]+shift[1])
m_mx.readshapefile( folderPath + 'cityBlocks/cityBlocks','cityBlocks', drawbounds=False, linewidth=.01)

# To plot different maps, after running one time the code up to this point, its just necessary to run the code from this point forward
# 
#open a plotting environment
fig = figure(figsize=(10, 8))
colormap = cm.jet
# plot the reference
m_mx.drawcoastlines()
#plot a mapscale (not working)
#m_mx.drawmapscale(DF[0], DF[1], 0, 0, 10, barstyle='simple', units='km', fontsize=9, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)

#This are different attributes to be plotted
# plotData = cityData['VPH_NODREN'] #Houses without drainege
# plotData = cityData['VPH_1CUART'] #Houses with only 1 room
# plotData = cityData['VPH_AGUAFV'] #Houses without tubed water
# plotData = cityData['VPH_PISOTI'] #Houses with dirt floor
# plotData = cityData['PRO_OCUP_C'] # Average number of dwelleres per room
# plotData = cityData['Dwellers'] #Dwellers
# plotData = cityData['LC'] # Linear Combination SSI
plotData = cityData['Factor']/cityData['Factor'].max() #Factor Analysis SSI
# orderK = {0:2,1:1,2:4,3:3}ç

# Because there are many blocks sometimes we decided just to plot a percentage, based on percentile of the data to be plotted
lim = percentile(plotData,90)
maxSI=plotData.max()

# this for plots the data
for Blockdict,Block in zip(m_mx.cityBlocks_info,m_mx.cityBlocks):
    clave = Blockdict['ID']
    val = plotData[clave]
    # val =orderK[plotData[clave]]

    
    if (val > lim):# and (val<10):
    # if cityData.LOC[clave] == 90070001:
        xx,yy = zip(*Block)
        color = getColor(val ,maxV=maxSI,cmap=colormap)
        fill(xx,yy,color,linewidth=.01,alpha=2.)


#this plots the contour of the districts
m_mxLoc.readshapefile( folderPath + 'shapefiles/df_loc_urb', 'df_loc_urb',linewidth=.1)
m_mxLoc.readshapefile( folderPath + 'shapefiles/mex_loc_urb', 'mex_loc_urb',linewidth=.1)

#map details
title('Factor Analysis')

ax1 = fig.add_axes([0.2, 0.15, 0.01, 0.15])
cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=colormap,
                                norm=mpl.colors.Normalize(vmin=lim, vmax=maxSI),
                                orientation='vertical')
cb1.set_label('FA SSI')
tick_locator = mpl.ticker.MaxNLocator(nbins=4)
cb1.locator = tick_locator
cb1.update_ticks()

# move the Colorbar position
ax1.set_position([0.70, 0.7, 0.01, 0.15])

# drawmapscale(lon, lat, lon0, lat0, length, barstyle='simple', units='km', fontsize=9, yoffset=None, labelstyle='simple', fontcolor='k', fillcolor1='w', fillcolor2='k', ax=None, format='%d', zorder=None)





# title('Fraction of Block without water')

# cityData = cityData.drop(cityData.loc[cityData['PRO_OCUP_C']>cityData['PROM_OCUP'],'PROM_OCUP'].index)
# cityData = cityData.drop(cityData.loc[cityData['PRO_OCUP_C']==0.],'PROM_OCUP'].index)

# cityData.drop(cityData.loc[cityData['PRO_OCUP_C']==0.],'PROM_OCUP'].index)



# map.readshapefile('/Users/db/Desktop/David/scince/shps/df/df_manzanas','df_manzanas')
# map.readshapefile('/Users/db/Desktop/David/scince/shps/mex/mex_manzanas','mex_manzanas')
# cityLocalities = ['090020001','090030001','090040001','090040020','090050001','090060001','090070001','090080001','090090001','090090011','090090015','090090017','090090024','090090029','090090033','090090036','090090152','090090300','090100001','090110001','090110011','090110021','090110024','090110026','090120001','090120019','090120026','090120027','090130001','090140001','090150001','090160001','090170001','130690001','130690002','130690008','130690010','130690019','130690042','130690044','150020001','150020005','150020008','150020011','150020012','150020015','150020016','150090001','150090005','150090006','150100001','150100007','150110001','150110002','150110004','150110013','150110029','150130001','150150002','150150004','150150006','150150007','150160001','150160006','150160016','150160017','150170001','150200001','150220001','150230001','150240001','150240088','150240111','150240124','150250001','150250002','150250002','150250010','150250012','150250011','150250014','150250016','150250017','150250019','150250020','150250021','150280001','150280004','150280006','150290001','150300001','150310001','150330001','150340001','150350001','150350006','150350008','150350009','150350012','150350013','150350019','150350027','150350028','150350030','150360001','150360006','150360007','150360008','150360009','150370001','150370005','150370009','150370013','150370018','150370019','150370011','150370023','150370024','150370025','150370026','150370071','150370089','150380001','150390001','150390003','150390004','150390007','150390011','150390012','150390064','150390151','150440001','150440020','150460001','150460003','150460005','150500001','150500002','150530001','150530005','150570001','150570088','150570098','150570267','150580001','150590001','150590017','150590058','150600001','150600003','150600004','150600016','150600018','150600020','150600025','150600082','150610001','150610004','150650001','150650007','150650011','150650019','150680001','150680007','150690001','150700001','150700005','150700008','150700009','150700013','150700017','150700019','150700036','150700037','150700038','150700039','150700040','150700041','150700001','150700042','150700043','150750001','150810001','150810004','150810012','150810019','150810025','150810050','150810098','150830001','150840001','150840004','150840011','150840013','150840014','150840017','150890001','150910001','150910010','150920001','150920002','150920013','150920019','150920020','150930001','150930003','150930014','150940001','150940002','150950001','150950005','150950021','150950026','150950073','150950074','150960001','150960008','150990001','150990012','150990016','150990020','150990022','150990024','150990025','150990029','150990030','150990035','150990041','150990042','150990043','150990045','150990048','151000001','151000002','151000007','151030001','151030005','151040001','151040105','151080001','151080014','151080063','151090001','151090003','151090025','151090068','151090069','151090072','151120001','151120011','151200001','151200013','151200014','151200023','151200041','151200045','151200046','151200054','151200056','151200057','151200175','151200198','151200204','151210001','151210020','151220001','151250001']


# allcoordinates = {shapedict['CVEGEO']:array(shape).mean(axis=0) for shapedict,shape in zip(mapa.df_manzanas_info,mapa.df_manzanas) }
# allcoordinates.update({shapedict['CVEGEO']:array(shape).mean(axis=0) for shapedict,shape in zip(mapa.mex_manzanas_info,mapa.mex_manzanas) })

# coordinatesCity ={blockKey:list(blockCoordinate) for blockKey,blockCoordinate in allcoordinates.iteritems() if blockKey[:9] in cityLocalities} 

# with open('/Users/db/Dropbox/slums/coordinates.json', 'w') as f:
#     json.dump(coordinatesCity, f)




