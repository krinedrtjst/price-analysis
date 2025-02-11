#!/usr/bin/env python
# coding: utf-8

# # Análise sobre a Precificação de Aluguel Temporário de Imóveis  

# ## Analisando e Limpando os Dados

# In[3]:


import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
import geopandas as gpd
import folium 
from folium.plugins import HeatMap
get_ipython().system('pip install mapclassify')
get_ipython().system('pip install pandas folium')


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor  
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from time import time

from warnings import simplefilter
simplefilter("ignore")

from IPython.display import display, HTML

get_ipython().system('pip install jupyter')
get_ipython().system('pip install nbConvert')


# In[4]:


data = pd.read_csv(r"C:\Users\lost4\OneDrive\Documentos\DATA\DATA SCIENCE\lighthouse\teste_indicium_precificacao.csv", encoding='utf-8')
data.head()


# In[5]:


def create_scrollable_table(data, table_id, title):
    html = f'<h3>{title}</h3>'
    html += f'<div id="{table_id}" style="height:200px; overflow:auto;">'
    html += data.to_html()
    html += '</div>'
    return html


# In[6]:


print(f'O conjunto de dados possui {data.shape[0]} linhas e {data.shape[1]} colunas.')


# In[7]:


data.info()


# No conjunto de dados apresentados, obtemos 4 colunas com valores vazios, sendo eles, Nome e Nome do Anfitrião. Ambos, podem ser provientes de cadastrados não finalizados. 
# 
# Os valores não preenchidos nas colunas de Ultima Review e Review por Mês podem nos atualizar da sazionalidade e periodicidade das reservas. 

# In[9]:


numerical_features = data.select_dtypes(include=[np.number])
numerical_features.describe()


# In[10]:


numerical_features = data.select_dtypes(include=[np.number])
summary_stats = numerical_features.describe().T
html_numerical = create_scrollable_table(summary_stats, 'numerical_features', 'Estatísticas Numerais para as Variáveis')

display(HTML(html_numerical))


# In[11]:


categorical_features = data.select_dtypes(include=[object])
cat_summary_stats = categorical_features.describe().T
html_categorical = create_scrollable_table(cat_summary_stats, 'categorical_features', 'Sumarizando Estaticamente as Variáveis Categóricas')

display(HTML(html_categorical ))


# * Procurando valores Nulos e Duplicados 

# In[13]:


data.isnull().sum()


# In[14]:


print(f'Obtemos {data.duplicated().sum()} linhas duplicados nesse conjunto de dados.')


# In[15]:


data2 = data.copy()
data2.dropna(inplace=True)

print(f'Após remoção dos valores vazios, obtemos de {data2.host_id.nunique()} host IDs e {data2.host_name.nunique()} host names.')
print("Não foi apresenta relação concomitante entre 'host_id' and 'host_name'.")


# In[16]:


data2.drop(['id','host_name','last_review'],axis=1,inplace=True)

data2.head()


# In[17]:


print(f'Após remoção dos valores nulos, contém {data2.name.nunique()} nomes de apartamentos e de quartos{data2.host_id.nunique()} host IDs.')
print("Não encontramos relações concomitantes entre o nome dos quartos e apartes e a coluna HOST ID.")
print('Identificando assim, o perfil de alguns investidores no conjunto de dados que possuem mais de uma propriedade.')


# In[18]:


data2.rename(columns={'neighbourhood_group':'borough'},inplace=True)


# # Análise Exploratória dos Dados

# ## Apartmentos por Bairro e Regiões de Nova York

# In[21]:


print(f'Temos {data2.borough.nunique()} regiões em Nova York que contém. Que contém informações sobre cerca de {data2.neighbourhood.nunique()} bairros.')


# Sumarizaremos abaixo as regiões com os maiores montantes de apartamentos e os bairros pertencentes a ele.

# In[23]:


data2.groupby(['borough','neighbourhood'])['neighbourhood'].count().reset_index(name='apt. count').sort_values('apt. count',ascending=False).head(10)


# In[24]:


# Dados agrupados
neighbourhood_group = data2.groupby('borough')['borough'].count().reset_index(name='count').sort_values('count', ascending=False)
neighbourhood = data2.groupby('neighbourhood')['neighbourhood'].count().reset_index(name='count').sort_values('count', ascending=False)

# Criando a figura
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

#  Apartamentos por Vizinhança
sns.barplot(data=neighbourhood_group, x='borough', y='count', ax=ax1, palette='viridis')
ax1.set_title('Apartments by Borough', size=18)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=10)

# Top 10 imóveis  por bairro (Top 10)
sns.barplot(data=neighbourhood.head(10), x='neighbourhood', y='count', ax=ax2, palette='magma')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, fontsize=7)
ax2.set_title('Apartments by Neighbourhood (Top 10)', size=18)

# Título 
plt.suptitle('Apartments by NYC Area', size=25)

# Ajustar layout
plt.tight_layout()

#  gráfico
plt.show()


# In[25]:


gpd.GeoDataFrame(
    data2,geometry=gpd.points_from_xy(data2["longitude"],data2["latitude"]),crs="epsg:4386"
).explore(width=1000,height=600,name="correct")


# In[26]:


# Calculando o valor médio através da latitude e Longitude
center_lat = data2['latitude'].mean()
center_lon = data2['longitude'].mean()

# Criando o mapa
base_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)


# In[27]:


heat_data = data2[['latitude', 'longitude', 'price']].values.tolist()

HeatMap(heat_data, radius=15, max_zoom=13).add_to(base_map)


# In[28]:


# Definindo uma função para visualizar o preço
def get_color(price):
    if price <= 50:
        return 'green'
    elif 50 < price <= 150:
        return 'orange'
    else:
        return 'red'

# Adicionando círculos para delimitar as áreas 
for idx, row in data.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=get_color(row['price']),
        fill=True,
        fill_color=get_color(row['price']),
        fill_opacity=0.7,
        popup=f"Price: ${row['price']}"
    ).add_to(base_map)


# In[29]:


# Salvando o mapa
base_map.save('MapPrices.html')


# ## Regiões mais Caras e Ecônomicas

# In[31]:


avg_price_borough = data2.groupby('borough')['price'].mean().reset_index(name='mean price')
new_row = {'borough':'New York','mean price':data2['price'].mean()} 
avg_price_borough = avg_price_borough._append(new_row,ignore_index=True).sort_values('mean price',ascending=False)

median_price_borough = data2.groupby('borough')['price'].median().reset_index(name='median price')
new_row = {'borough':'New York','median price':data2['price'].median()} 
median_price_borough = median_price_borough._append(new_row,ignore_index=True).sort_values('median price',ascending=False)

avg_price_neighbourhood = data2.groupby('neighbourhood')['price'].mean().reset_index(name='mean price').sort_values('mean price',ascending=False)

median_price_neighbourhood = data2.groupby('neighbourhood')['price'].median().reset_index(name='median price').sort_values('median price',ascending=False)


bigfig = plt.figure(figsize=(12,6))

(top,bottom) = bigfig.subfigures(2,1)

### Top figures ###
top.subplots_adjust(left=.1,right=.9,wspace=.4,hspace=.4)

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,6))

ax1 = sns.barplot(data=avg_price_borough,x='borough',y='mean price',ax=ax1, palette='viridis')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45,fontsize=7)
ax1.set_title('Avg Price by Borough',size=15)

ax2 = sns.barplot(data=avg_price_neighbourhood.head(10),x='neighbourhood',y='mean price',ax=ax2, palette='magma')
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45,fontsize=7)
ax2.set_title('Most Expensive Neighbourhoods',size=15)

ax3 = sns.barplot(data=avg_price_neighbourhood.tail(10),x='neighbourhood',y='mean price',ax=ax3, palette='magma')
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=45,fontsize=7)
ax3.set_title('Least Expensive Neighbourhoods',size=15)

plt.suptitle('Average Price by NYC Borough and Neighbourhood',size=25)

plt.tight_layout()

### Bottom figures ###
bottom.subplots_adjust(left=.1,right=.9,wspace=.4,hspace=.4)

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,6))

ax1 = sns.barplot(data=median_price_borough,x='borough',y='median price',ax=ax1, palette='viridis')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45,fontsize=7)
ax1.set_title('Median Price by Borough',size=15)

ax2 = sns.barplot(data=median_price_neighbourhood.head(10),x='neighbourhood',y='median price',ax=ax2, palette='magma')
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45,fontsize=7)
ax2.set_title('Most Expensive Neighbourhoods (Median)',size=15)

ax3 = sns.violinplot(data=data2,x='borough',y='price',ax=ax3, palette='magma')
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=45,fontsize=7)
ax3.set_title('Violin Plot of Price by Borough',size=15)

plt.suptitle('Median Price by NYC Borough and Neighbourhood',size=25)

plt.tight_layout()


# Legendando os Histogramas apresentados:
# * Manhattan possui o custo mais alto para um locatário iniciar sua renda alternativa na região de Nova York
# * A diferença entre o preço médio entre o aluguel da região mais cara e mais barata é menos que a metade entre eles.
# * A mediana de preço da mais cara chega a quase $ 300,00 e a mais barata $ 60,00. Demonstrando que metade dos apartamentos mais caros estão acima de $ 300,00 e metade abaixo. Demonstrando que, entre as locações mais baratas, metade exata dos valores transitam entre $ 60,00. 

# ## Tipos de Quartos Disponíveis

# In[34]:


room_type = data2.groupby('room_type')['room_type'].count().reset_index(name='count').sort_values('count',ascending=False)

room_type_borough = data2.groupby(['room_type','borough'])['room_type'].count().reset_index(name='count').sort_values('count',ascending=False)

room_type_price = data2.groupby('room_type')['price'].mean().reset_index(name='mean price').sort_values('mean price',ascending=False)


fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,6))

ax1 = sns.barplot(data=room_type,x='room_type',y='count',ax=ax1, palette='viridis')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45,fontsize=8)
ax1.set_title('Airbnb Room Types in NYC',size=15)

ax2 = sns.histplot(data=data2,x='borough',hue='room_type',multiple='stack',ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=45,fontsize=8)
ax2.set_title('Airbnb Room Type by Borough',size=15)

ax3 = sns.barplot(data=room_type_price,x='room_type',y='mean price',ax=ax3, palette='viridis')
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=45,fontsize=8)
ax3.set_title('Avg Price by Room Type',size=15)

plt.suptitle('Room Type Distribution in NYC',size=25)

plt.tight_layout()


# Insights sobre o espaço arquitetônico:
# * O espaço arquitetônico mais procurado são casas e apartamentos. 
# * Quartos privativos são a segunda opção entre eles.
# * O preço médio pelo espaço espaço arquitetônico mais ampliando chega a $200,00, os valores dos quartos privativos e compartilhados contabilizam metade desse valor. 

# ## Número Mínimo de Estadia por Noites

# Minimum nights is an important indicator. If it is high, it means that the property is used for a long time, which is a good sign from a customer point of view.

# In[38]:


minimum_nights_borough = data2.groupby('borough')['minimum_nights'].mean().reset_index(name='avg no. of minimum nights')
new_row = {'borough':'New York','avg no. of minimum nights':data2['minimum_nights'].mean()} 
minimum_nights_borough = minimum_nights_borough._append(new_row,ignore_index=True).sort_values('avg no. of minimum nights',ascending=False)

minimum_nights_neighbourhood = data2.groupby('neighbourhood')['minimum_nights'].mean().reset_index(name='avg no. of minimum nights').sort_values('avg no. of minimum nights',ascending=False)

fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,6))

ax1 = sns.barplot(data=minimum_nights_borough,x='borough',y='avg no. of minimum nights',ax=ax1, palette='viridis')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=45,fontsize=8)
ax1.set_title('Minimum Nights by Borough',size=15)

ax2 = sns.barplot(data=minimum_nights_neighbourhood.head(10),x='neighbourhood',y='avg no. of minimum nights',ax=ax2, palette='magma',)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=75,fontsize=8)
ax2.set_title('Minimum Nights (Top10 Neighbourhoods)',size=12)

ax3 = sns.barplot(data=minimum_nights_neighbourhood.tail(10),x='neighbourhood',y='avg no. of minimum nights',ax=ax3, palette='magma')
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=75,fontsize=8)
ax3.set_title('Minimum Nights (Worst Neighbourhoods)',size=12)

plt.suptitle('Distribution of Minimum Nights',size=25)

plt.tight_layout()


# Os dias de locamento nos informam sobre o comportamental preferencial do consumidor:
# * Os locatários costumam alugar os espaços disponíveis em Manhattan por no mínimo uma semana. 
# * A região de Staten Island foi a menos procurada, apresentando uma média de até 4 noite alugadas por locador. 

# In[40]:


neighbourhood_list = neighbourhood.loc[neighbourhood['count'] > 20]['neighbourhood'].to_list()

minimum_nights_neighbourhood2 = data2.loc[data2.neighbourhood.isin(neighbourhood_list)].groupby('neighbourhood')['minimum_nights'].mean().reset_index(name='avg no. of minimum nights').sort_values('avg no. of minimum nights',ascending=False)


fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,6))

ax1 = sns.barplot(data=minimum_nights_neighbourhood2.head(10),x='neighbourhood',y='avg no. of minimum nights',ax=ax1, palette='magma')
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=75,fontsize=8)
ax1.set_title('Minimum Nights (Top10 Neighbourhoods)',size=15)

ax2 = sns.barplot(data=minimum_nights_neighbourhood2.tail(10),x='neighbourhood',y='avg no. of minimum nights',ax=ax2, palette='magma')
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=75,fontsize=8)
ax2.set_title('Minimum Nights (Worst Neighbourhoods)',size=15)

plt.suptitle('Distribution of Minimum Nights (Neighbourhoods with at least 20 Apartments)',size=20)

plt.tight_layout()


# * O número mínimo de noites locadas entre os 10 bairros procurados, nos concedem incidência de 12 dias entre elas.
# * O mínimo de noite entre os bairros menos procurados também apresenta dois números similares, com dois 2 dias nos dois bairros.

# In[42]:


data2.columns


# In[43]:


df = data2.copy()
df.head()


# In[44]:


df.columns


# In[45]:


import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go

# Encaixando uma distribuição normal no Preço dos aluguéis
mu, sigma = stats.norm.fit(df['price'])

# Criando os Histrogramas
hist_data = go.Histogram(
    x=df['price'], 
    nbinsx=50, 
    name="Histogram", 
    opacity=0.75, 
    histnorm='probability density', 
    marker=dict(color='purple')
)

# Criando a curva de distribuição normal
x_norm = np.linspace(df['price'].min(), df['price'].max(), 1000)
y_norm = stats.norm.pdf(x_norm, mu, sigma)
norm_data = go.Scatter(
    x=x_norm, 
    y=y_norm, 
    mode="lines", 
    name=f"Normal dist. (μ={mu:.2f}, σ={sigma:.2f})", 
    line=dict(color="green")
)


fig = go.Figure(data=[hist_data, norm_data])

fig.update_layout(
    title="Price Distribution",
    xaxis_title="Price",
    yaxis_title="Density",
    template="plotly",  # ou simplesmente remova esta linha
    font=dict(color='black'),  # Texto em preto para fundo claro
    plot_bgcolor='white',  # Fundo do gráfico
    paper_bgcolor='white'  # Fundo do papel
)


# Criando a Q-Q plot
qq_data = stats.probplot(df['price'], dist="norm")
qq_fig = px.scatter(x=qq_data[0][0], y=qq_data[0][1], labels={'x': 'Theoretical Quantiles', 'y': 'Ordered Values'}, color_discrete_sequence=["purple"])
qq_fig.update_layout(
    title="Q-Q plot",
    plot_bgcolor='rgba(32, 32, 32, 1)',
    paper_bgcolor='rgba(32, 32, 32, 1)',
    font=dict(color='white')
)

# Ajustando o melhor encaixe para a linha
slope, intercept, r_value, p_value, std_err = stats.linregress(qq_data[0][0], qq_data[0][1])
line_x = np.array(qq_data[0][0])
line_y = intercept + slope * line_x

# Adicionando a linha para melhor the Q-Q plot
line_data = go.Scatter(x=line_x, y=line_y, mode="lines", name="Normal Line", line=dict(color="green"))

# Atualizando the Q-Q plot with the normal line
qq_fig.add_trace(line_data)


qq_fig.update_layout(
    title="Q-Q plot",
    plot_bgcolor='white',  
    paper_bgcolor='white', 
    font=dict(color='black')
)

# Mostrando a figura
fig.show()
qq_fig.show()


# # Avaliações

# Sumarizando essa seção do Dataset. 

# In[48]:


data2.groupby(['name','neighbourhood'])['number_of_reviews'].count().reset_index(name='count').sort_values('count',ascending=False).head(10)


# In[49]:


reviews_per_borough = data2.groupby('borough')['number_of_reviews'].sum().reset_index(name='total review count')

avg_reviews_borough = data2.groupby('borough')['number_of_reviews'].mean().reset_index(name='avg reviews/apartment')

reviews_per_borough = reviews_per_borough.merge(avg_reviews_borough,on='borough').sort_values('avg reviews/apartment',ascending=False)

reviews_per_borough


# Com uma incidência nova sobre as avaliações apresentadas, fatores como média do preço e oferta de quartos e apartamentos não incidiram na estimulação do público em compartilhar sua experiência em avaliações após a estadia. 

# In[51]:


reviews_per_neighbourhood = data2.groupby('neighbourhood')['number_of_reviews'].sum().reset_index(name='count')

avg_reviews_neighbourhood = data2.groupby('neighbourhood')['number_of_reviews'].mean().reset_index(name='avg reviews/apartment')

reviews_per_neighbourhood = reviews_per_neighbourhood.merge(avg_reviews_neighbourhood,on='neighbourhood').sort_values('count',ascending=False).head(10)

reviews_per_neighbourhood


# Os pontos geográficos são fatores determinantes pelas exposição e estimulação de avaliações. 

# ## Anfritiões

# Observamos um fator de investimento e alguns locatários ofertando múltiplos imóveis.  

# In[55]:


data2.groupby('host_id')['host_id'].value_counts().reset_index(name='count').sort_values('count',ascending=False).head(10)


# O primeiro anfintrião possui 207 imóveis ofertados para localização. Valor destoante dos apresentados pelos demais locatários do aplicativo. 

# In[57]:


apartments_219517861 = data2[data2['host_id'] == 219517861].groupby(['borough','neighbourhood'])['name'].count().reset_index(name='count')

price_219517861 = data2[data2['host_id'] == 219517861].groupby(['borough','neighbourhood'])['price'].mean().reset_index(name='avg price')

apartments_219517861 = apartments_219517861.merge(price_219517861,on='neighbourhood').sort_values('count',ascending=False)

apartments_219517861


# Todos os bens dele estão localizados na região de Manhattan e o valor médio dos 170 móveis ofertados no Financial District ficam $30,00 abaixo da mediana dos preços dos demais imóveis ofertados em Nova York.  

# In[59]:


gpd.GeoDataFrame(
    data2[data2['host_id'] == 219517861],geometry=gpd.points_from_xy(data2[data2['host_id'] == 219517861]["longitude"],data2[data2['host_id'] == 219517861]["latitude"]),crs="epsg:4386"
).explore(width=1000,height=600,name="correct")


# Observando agora os dados do 2º maior locatário de imóveis. 

# In[61]:


apartments_61391963 = data2[data2['host_id'] == 61391963].groupby(['borough','neighbourhood'])['name'].count().reset_index(name='count')

price_61391963 = data2[data2['host_id'] == 61391963].groupby(['borough','neighbourhood'])['price'].mean().reset_index(name='avg price')
apartments_61391963 = apartments_61391963.merge(price_61391963,on='neighbourhood').sort_values('count',ascending=False)

apartments_61391963


# O segundo locatário oferta seus imóveis todos em Manhattan, com uma diversidade maior entre os bairros e uma média de preço similiar ao resto dos locatários presente na plataforma. 

# In[63]:


gpd.GeoDataFrame(
    data2[data2['host_id'] == 61391963],geometry=gpd.points_from_xy(data2[data2['host_id'] == 61391963]["longitude"],data2[data2['host_id'] == 61391963]["latitude"]),crs="epsg:4386"
).explore(width=1000,height=600,name="correct")


# # Outliers 

# In[65]:


data2.neighbourhood.nunique(), data2.host_id.nunique()


# In[66]:


data3 = data2.copy()

data3.drop(['neighbourhood','name','host_id'],axis=1,inplace=True)

data3.head()


# Processando os dados quantitativos apresentados

# In[68]:


num_features = list(data3.select_dtypes(include=np.number).columns.values)

print(num_features)


# I am plotting the boxplots with the outliers.

# In[70]:


for i in range(3):

    if i < 2:
        fig,(ax1,ax2,ax3) = plt.subplots(ncols=3,figsize=(12,4))
    
        ax1 = sns.boxplot(data3[num_features[i*3]],ax=ax1)
        ax1.set_title(f'Boxplot of {num_features[i*3]}',fontsize=15)
    
        ax2 = sns.boxplot(data3[num_features[i*3+1]],ax=ax2)
        ax2.set_title(f'Boxplot of {num_features[i*3+1]}',fontsize=15)
    
        ax3 = sns.boxplot(data3[num_features[i*3+2]],ax=ax3)
        ax3.set_title(f'Boxplot of {num_features[i*3+2]}',fontsize=15)
    
        fig.suptitle("Boxplots of the Outliers",fontsize=24)    
    
        plt.tight_layout()

    else:

        fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,4))
    
        ax1 = sns.boxplot(data3[num_features[i*3]],ax=ax1)
        ax1.set_title(f'Boxplot of {num_features[i*3]}',fontsize=15)
    
        ax2 = sns.boxplot(data3[num_features[i*3+1]],ax=ax2)
        ax2.set_title(f'Boxplot of {num_features[i*3+1]}',fontsize=15)
    
        fig.suptitle("Boxplots of the Outliers",fontsize=24)    
    
        plt.tight_layout()


# O Standart Deviation de apresenta um comportamento adequado pelo visualizado anteriormente, com poucas incidências de outliers, com excessões sendo raras e situacionais. 

# In[72]:


outliers_perc = []

for k,v in data3[num_features].items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    out_tuple = (k,int(perc))
    outliers_perc.append(out_tuple)
    print("Column %s outliers = %.2f%%" % (k,perc))


# Os Outliers de preço e longitude  apresentando <5% de incidência, significando uma pouca relevancia estatística. 
# Os Outliers de noites mínimas similar com o de noites mínimas, pois são valores proporcionais, e neste ponto apresentam mais situações singulares. 

# In[74]:


def outlier_capper(data,features):
    
    data_out = data.copy()
    
    for column in features:        
        
        # First define the first and third quartiles
        Q1 = data_out[column].quantile(0.25)
        Q3 = data_out[column].quantile(0.75)
        # Define the inter-quartile range
        IQR = Q3 - Q1
        # ... and the lower/higher threshold values
        lowerL  = Q1 - 1.5 * IQR
        higherL = Q3 + 1.5 * IQR
        
        # Impute 'left' outliers
        data_out.loc[data_out[column] < lowerL,column] = lowerL
        # Impute 'right' outliers
        data_out.loc[data_out[column] > higherL,column] = higherL
        
    return data_out
    

data3 = outlier_capper(data3,num_features) 


# # Engenheira de Recursos

# Escalando às variaveis numéricas 

# In[77]:


# Initialize a scaler, then apply it to the numerical features
scaler = MinMaxScaler()

# List of numerical columns
num_cols = [col for col in data3.columns if data3[col].dtypes != 'O']

# Scale the numerical columns
data3[num_cols] = scaler.fit_transform(data3[num_cols])


# In[78]:


data3 = pd.get_dummies(data3)

data3.head()


# Now, I can define X and y and then perform the train-test split.

# In[80]:


X = data3.drop('price',axis=1)
y = data3['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


# Configurando a Floresta Aleatória com 10 árvores de decisão. 

# In[82]:


rf = RandomForestRegressor(n_estimators=20)
rf.fit(X_train,y_train)


# In[83]:


# To sort the index in descending order, I multiply 'rf.feature_importances_' by -1
sorted_idx = (-rf.feature_importances_).argsort()

list_of_tuples = list(zip(X.columns[sorted_idx],
                           rf.feature_importances_[sorted_idx]))

feat_importance = pd.DataFrame(list_of_tuples,
                  columns=['feature','feature importance'])

feat_importance


# One can easily check that the resulting feature importances are normalized to one.

# In[85]:


sum = 0

for index,row in feat_importance.iterrows():
    sum += row['feature importance']

sum


# In[86]:


perm_importance = permutation_importance(rf,X_train,y_train)

sorted_idx = (-perm_importance.importances_mean).argsort()

list_of_tuples  = list(zip(X.columns[sorted_idx],
                           perm_importance.importances_mean[sorted_idx]))

perm_importance = pd.DataFrame(list_of_tuples,
                  columns=['feature','permutation importance'])

perm_importance


# Diferentemente do caso anterior, as importâncias por permutação não são normalizadas para 1.

# In[88]:


fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(12,8))

ax1 = sns.barplot(data=feat_importance,x='feature',y='feature importance',ax=ax1, palette='magma')
ax1.set_title('Feature Importance',fontsize=25)
ax1.set_xticklabels(ax1.get_xticklabels(),rotation=60,fontsize=7)

ax2 = sns.barplot(data=perm_importance,x='feature',y='permutation importance',ax=ax2,  palette='magma')
ax2.set_title('Permutation Importance',fontsize=25)
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=60,fontsize=7)
    
plt.tight_layout()


# Os dois métodos preditivos seguiram o mesmo método de aprendizado e apresentaram uma proposta similar.

# In[90]:


collin_check_list = ['latitude','longitude','minimum_nights', 
                     'number_of_reviews','reviews_per_month',
                     'calculated_host_listings_count',
                     'availability_365']

sns.pairplot(X_train[collin_check_list],plot_kws={'alpha':0.4,'size':5})


# Não é apresentado uma colinearidade que possa afetar a regressão dos resultados. 

# In[92]:


# Create correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(X_train.corr(method='pearson'),annot=True,cmap='Blues')
plt.title('Correlation heatmap',fontsize=25)

plt.tight_layout()
plt.show()


# * Cerca de 13% dos imóveis alugados em Manhattan são uma Casa ou um Apartamento inteiro.
# * Dentre os locadores da plataforma, 17% deles passam as noites mínimas em Staten Island.
# * Dentre os locatários da plataforma, 19% deles ofertam quartos privativos e 12% quartos compartilhados. 

# In[94]:


len(data2.room_type)


# In[95]:


data2.room_type.value_counts()


# As casas e os quartos privativos apresentam números inversamente proporcionais, sendo 98% de correlação negativa. 
# Os quartos compartilhados representam os 2% restante do montante de  imóveis

# # Construção do Modelo Preditivo

# In[98]:


def train_predict(learner,sample_size,X_train,y_train,X_test,y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # Calculate the training time
    results['train_time'] = end - start
        
    #  Get the predictions on the test set,
    #  then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time

    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute R^2 score on the first 300 training samples
    results['r2_train'] = r2_score(y_train[:300],predictions_train)
        
    # Compute R^2 score on test set
    results['r2_test'] = r2_score(y_test,predictions_test)
    
    # Compute MSE on the the first 300 training samples
    results['mse_train'] = mean_squared_error(y_train[:300],predictions_train)
        
    # Compute F-score on the test set
    results['mse_test'] = mean_squared_error(y_test,predictions_test)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__,sample_size))
        
    # Return the results
    return results


# In[99]:


# Initialize the three models
reg_A = RandomForestRegressor(random_state=42)
reg_B = GradientBoostingRegressor(random_state=42)
reg_C = DecisionTreeRegressor(random_state=42)

# Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1   = int(round(len(X_train) / 100))
samples_10  = int(round(len(X_train) / 10))
samples_100 = len(X_train)

# Collect results on the learners
results = {}
for reg in [reg_A,reg_B,reg_C]:
    reg_name = reg.__class__.__name__
    results[reg_name] = {}
    for i,samples in enumerate([samples_1,samples_10,samples_100]):
        results[reg_name][i] = \
        train_predict(reg,samples,X_train,y_train,X_test,y_test)


# In[100]:


# Printing out the values
for i in results.items():
    print(i[0])
    display(pd.DataFrame(i[1]).rename(columns={0:'1%',1:'10%',2:'100%'}))


# Prosseguindo com o refinamento da Floresta Aleatória. 

# In[102]:


random_forest_tuning = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100,200,300,400],
    'max_depth': [None,5,10]
}

GSCV = GridSearchCV(estimator=random_forest_tuning,param_grid=param_grid,cv=5)
GSCV.fit(X_train,y_train)

GSCV.best_params_ 


# **Model performance on the train data**

# In[ ]:


r_sq = GSCV.score(X_train,y_train)
y_pred_train = GSCV.predict(X_train)

print('R^2:',r2_score(y_train,y_pred_train))


# **Model performance on the test data**

# In[ ]:


r_sq_test = GSCV.score(X_test,y_test)
y_pred_test = GSCV.predict(X_test)

print('R^2:',r2_score(y_test,y_pred_test))


# O segundo modelo preditivo apresentou um coeficiente de determinação de  0,6668. Indicando um ajuste bom, porém passível de melhorias. 
# Onde, 33,32% da variabilidade sobre a variância não é explicada. Indicando assim, a importância de investigações mais específicas para introdução de variaveis mais significativas. o problema.

# In[103]:


df = data3.copy()
df.head()


# In[105]:


df.columns


# In[107]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Exemplo de preparação de dados
features = ['latitude', 'longitude', 'room_type_Entire home/apt', 'minimum_nights', 'number_of_reviews', 
            'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
X = df[features]
y = df['price']  # Supondo que 'preco' é a coluna alvo

# Codificar variáveis categóricas
X = pd.get_dummies(X, columns=['room_type_Entire home/apt'])

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dados do novo 
new_airbnb_df = {
    'id': 2595,
    'name': 'Skylit Midtown Castle',
    'host_id': 2845,
    'host_name': 'Jennifer',
    'borough_Manhattan': 'Manhattan',
    'borough': 'Midtown',
    'latitude': 40.75362,
    'longitude': -73.98377,
    'room_type': 'Entire home/apt',
    'minimum_nights': 1,
    'number_of_reviews': 45,
    'last_review': '2019-05-21',
    'reviews_per_month': 0.38,
    'calculated_host_listings_count': 2,
    'availability_365': 355
}

# Criar DataFrame
new_airbnb = pd.DataFrame([new_airbnb_df])

# Codificar variáveis categóricas
new_airbnb = pd.get_dummies(new_airbnb, columns=['room_type'])

# Garantir que as colunas do novo Airbnb correspondam às do conjunto de treinamento
new_airbnb = new_airbnb.reindex(columns=X.columns, fill_value=0)

# Prevendo o preço
predict_price = model.predict(new_airbnb)
print(f"Price Predict: ${predict_price[0]:.2f}")


# # Recomendações

# * Através desta análise observamos alguns pontos pertinentes para variabilidade e apresentação dos preços, sendo um deles, impacto sazonal dos feriados e datas comemorativas. 
# Esse período do ano impactaria os imóveis mais alugados como Apartamentos e Casas, já que podem confortar famílias e grupos numerosos. Ocasionando um impacto nos preços estipulados por seus anfitriões.
# 
# * Uma correlação positiva encontrada no dataset foi a longitude e sua procura por hospedagem prévias, sendo elas, um cómodo em um ponto estrátegico com aeroportos, e instituições subtâncias de emergências para população com estadia passageira. Salientando assim, que mesmo em um bairro com pouca demanda, a comodidade de localização pode impactar no preço ofertado. 
# 
# * Com uma média de preço por noite de 152,00 dólares e uma mediana de 300,00 dólares. Um Standart Deviation dos preço em 159,00 dólares. Os testes estatísticos e resultados apresentados observam esse valor médio referenciado, para uma margem de erro mais cara de  182,95 dólares e um RMSE de teste de 142,72 dólares. 
# 
# * Lugares como Manhattan possui o custo mais alto para um locatário iniciar sua renda alternativa na região de Nova York.  A diferença entre o preço médio entre o aluguel da região mais cara e mais barata é menos que a metade entre eles.
# 
# * A mediana de preço da mais cara chega a quase 300,00 dólares e a mais barata $60,00. Demonstrando que metade dos apartamentos mais caros estão acima de $300,00 e metade abaixo. Demonstrando que, entre as locações mais baratas, metade exata dos valores transitam entre $60,00.
# Insights sobre o espaço arquitetônic
# 
# * O espaço arquitetônico mais procurado são casas e artamentos disponibilizados. 
# * Quartos privativos são a segunda opção entrelee  
# * O preço médio pelo espaço espaço arquitetônico mais ampliando cheg a $20 dólares0,00, os valores dos quartos privativos e compartilhados contabilizam metade desalor.
# ana. 
# * A região de Staten Island foi a menos procurada, apresentando uma média de até 4 noite alugadas por lo * O número mínimo de noites e sua disponibilidade ao longo do ano apresentaram uma correlação positiva de 17% em Staten Island, interferindo assim positivamente em seu preço.
# 
# *  ador.
# A compra de um imóvel mais indicada é em Man
# 
# *  A predição de preço por uma noite em Skylit Midtown Castle são de 87,00 Dólares. preço.
# 
