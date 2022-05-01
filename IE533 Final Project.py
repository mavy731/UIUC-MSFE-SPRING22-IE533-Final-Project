import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import count
import random
import operator

#create test file with the first 10000 edges
# fin=open('Data/cit-Patents.txt')
# a=fin.readlines()
# fout=open('Data/test.txt','w')
# b=''.join(a[:10000])
# fout.write(b)
# fin.close()
# fout.close()

def read_txt(data):
    g = nx.read_edgelist(data, create_using=nx.DiGraph)
    return g


def degree_histogram_directed(G, in_degree=False, out_degree=False):
    # modified from networkx function networkx.degree()
    """Return a list of the frequency of each degree value.

       Parameters
       ----------
       G : Networkx graph
          A graph
       in_degree : bool
       out_degree : bool

       Returns
       -------
       hist : list
          A list of frequencies of degrees.
          The degree values are the index in the list.

       Notes
       -----
       Note: the bins are width one, hence len(list) can be large
       (Order(number_of_edges))
       """
    nodes = G.nodes()
    if in_degree:
        in_degree = dict(G.in_degree())
        degseq = [in_degree.get(k, 0) for k in nodes]
    elif out_degree:
        out_degree = dict(G.out_degree())
        degseq = [out_degree.get(k, 0) for k in nodes]
    else:
        degseq = [v for k, v in G.degree()]
    dmax = max(degseq) + 1
    freq = [0 for d in range(dmax)]
    for d in degseq:
        freq[d] += 1
    return freq

def degree_centrality(G, in_degree=False, out_degree=False):
    if in_degree:
        series = pd.Series(nx.in_degree_centrality(G))
    elif out_degree:
        series = pd.Series(nx.out_degree_centrality(G))
    else:
        series = pd.Series(nx.degree_centrality(G))
    return series

def find_specific_attribute_node(G, attr, value):
    result = []
    d = nx.get_node_attributes(G, attr)
    for key, v in d.items():
        if(v == value):
            result.append(key)
    return result

def small_world_effect(G):
    G = G.to_undirected()
    seed = random.seed()
    sigma = nx.sigma(G, niter=200, nrand=15, seed = seed)
    omega = nx.omega(G, niter=20, nrand=15, seed = seed)
    return (sigma, omega)


#import data set
pat = pd.read_csv('Data/apat63_99.txt', header = 0)
pat = pat.set_index('PATENT')
coname = pd.read_csv('Data/aconame.txt', header = 0)
coname =coname.set_index('ASSIGNEE')
#g = read_txt('Data/test.txt')
g = read_txt('Data/cit-Patents.txt')
print('finish loading')
#basic infor
nodes = list(g.nodes)
edges = list(g.edges)
n_nodes = len(nodes)

#assigne attributes

for i in range(0, n_nodes):
    id = int(nodes[i])
    if (id < 3070801):
        continue
    try:
        year = pat.loc[id,'GYEAR']
        country = pat.loc[id, 'COUNTRY']
        assignee = pat.loc[id, 'ASSIGNEE']
        asscode = pat.loc[id,'ASSCODE']
    except:
        continue
    g.add_node(nodes[i],country=country,assignee=assignee, asscode  = asscode, year = year)
print('finish adding attributes')

#citation count of country

sum_c = pat['COUNTRY'].value_counts()
plt.figure(figsize=(6, 6))
plt.axes(xscale = "log")
plt.hist(sum_c, bins =[0,10,100,1000,10000,100000,1000000,10000000],alpha=0.75)
plt.xlabel('Number of Patents(in log)')
plt.ylabel('Number of Countries')
plt.title('Patents of Countries')
plt.savefig('Graph/Patents of Countries.png')
plt.show()
#print(sum_c.head(20))

#citation count of assignee

sum_ae = pat['ASSIGNEE'].value_counts()
plt.figure(figsize=(6, 6))
plt.axes(yscale = "log")
ax = plt.gca()
ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
plt.hist(sum_ae, bins =30,alpha=0.75)
plt.xlabel('Number of Patents')
plt.ylabel('Number of Assignees(in log)')
plt.title('Patents of Assignees')
plt.savefig('Graph/Patents of Assignees.png')
plt.show()
sum_ae = sum_ae.to_frame()
for i in range(0, 20):
    id = sum_ae.index[i]
    print(id)
    sum_ae.loc[id,'Coname'] = coname.loc[id,'COMPNAME']
# print(sum_ae.head(20))

#citation count of year

sum_y = pat['GYEAR'].value_counts()
sum_y =sum_y.sort_index()
#print(sum_y)
plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
plt.plot(sum_y,marker='+',color = 'royalblue', alpha=0.75 )
plt.xlabel('Year')
plt.ylabel('Number of Patents')
plt.title('Patents of Years')
plt.savefig('Graph/Patents of Years.png')
plt.show()

#citation count of assignee category

sum_ac = pat['ASSCODE'].value_counts()
print(sum_ac)
plt.figure(figsize=(6, 6))
plt.bar(sum_ac.index,sum_ac,color = 'royalblue', alpha=0.75)
plt.xlabel('Assignee Type')
plt.ylabel('Number of Patents')
plt.title('Patents of Assignee Category')
plt.savefig('Graph/Patents of Assignee Category.png')
plt.show()


#indegree/outdegree density graph

freq_indegree = degree_histogram_directed(g, in_degree=True)
freq_outdegree = degree_histogram_directed(g, out_degree=True)
plt.figure(figsize=(12, 8))
plt.axes(yscale = "log")
plt.plot(range(len(freq_indegree)), freq_indegree, marker = '^', color='seagreen',label='in-degree', alpha=0.5)
plt.plot(range(len(freq_outdegree)), freq_outdegree, marker = '^',color='navy', label='out-degree', alpha=0.5)
plt.xlabel('Degree')
plt.ylabel('Number of Nodes(in log)')
plt.title('Degree Distribution of Patent Citations')
plt.legend()
plt.savefig('Graph/degree_distribution(in_out).png')
plt.show()

#distribution of degree centrality

in_centrality = degree_centrality(g, in_degree=True)
out_centrality = degree_centrality(g, in_degree=True)
centrality = degree_centrality(g)
plt.figure(figsize=(6, 4))
plt.axes(yscale = "log")
ax = plt.gca()
ax.ticklabel_format(style='sci',scilimits=(0,0),axis='x')
plt.hist(in_centrality, bins =20,alpha=0.75)
plt.xlabel('Degree Centrality')
plt.ylabel('Number of Nodes(in log)')
plt.title('In-degree Centrality')
plt.savefig('Graph/In-degree Centrality.png')
plt.show()
in_degree = dict(g.in_degree())
out_degree = dict(g.out_degree())
# print('Density', nx.density(g))
# print('Transitivity ', nx.transitivity(g))
print('Reciprocity ', nx.reciprocity(g))
print('Avg In-degree ', sum(in_degree.values())/len(in_degree))
print('Avd Out-degree ',sum(out_degree.values())/len(out_degree))

#ego_graph

degree = g.degree()
largest = max(dict(degree).items(), key=operator.itemgetter(1))[0]
# 4723129
# 793
print('finish sorting ego')
largest_ego = nx.ego_graph(g,largest, undirected=True)
low = [n for n, d in largest_ego.degree() if d < 8]
largest_ego.remove_nodes_from(low)
pos = nx.spring_layout(largest_ego, k=0.3)
plt.figure()
plt.title('Main egonet of the largest hub (degree> 7)')
nx.draw(largest_ego, pos, node_color='b', node_size=50,with_labels=False)
options = {"node_size": 300, "node_color": "r"}
nx.draw_networkx_nodes(largest_ego, pos, nodelist=[largest], **options)
plt.savefig('Graph/Main egonet.png')
plt.show()

#Betweenness Centrality
low = [n for n, d in g.degree() if d < 150]
g.remove_nodes_from(low)
print('finish shrink')
g1 = g.to_undirected()
components = nx.connected_components(g1)
lar = max(components, key=len)
b = g1.subgraph(lar)
centrality = nx.betweenness_centrality(b, endpoints=True)
lpc = nx.community.label_propagation_communities(b)
community_index = {n: i for i, com in enumerate(lpc) for n in com}
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(b, k=0.15, seed=random.seed())
node_color = [community_index[n] for n in b]
node_size = [v * 20000 for v in centrality.values()]
nx.draw_networkx(
    b,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Betweeness Centrality Analysis", font)
# Change font color for legend
font["color"] = "royalblue"
ax.text(
    0.80,
    0.10,
    "node color = community structure",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = betweeness centrality",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.savefig('Graph/Betweeness Centrality.png')
plt.show()

#IBM Graph

top_assignee = int(sum_ae.index[0])
top_nodes = find_specific_attribute_node(g, 'assignee',top_assignee )
top_g = nx.DiGraph(g.subgraph(top_nodes))
print(len(top_g.nodes()))
low = [n for n, d in top_g.degree() if d < 15]
top_g.remove_nodes_from(low)
largest = max(dict(top_g.degree()).items(), key=operator.itemgetter(1))[0]
plt.figure(figsize=(9, 6))
pos = nx.spring_layout(top_g, k=0.5)
plt.title('Patent citation network for topmost applicant "IBM" (degree>=15)')
nx.draw(top_g, pos, node_color='slateblue', node_size=50,with_labels=False,
        alpha = 0.5,
        edge_color="gainsboro",)
plt.savefig('Graph/network IBM.png')
plt.show()

#top 10 assignee degree20

top10_assignee = int(sum_ae.index[0])
top10_nodes = find_specific_attribute_node(g, 'assignee',top10_assignee )
for i in range(1,10):
    top10_assignee = int(sum_ae.index[i])
    top10_nodes = top10_nodes+find_specific_attribute_node(g, 'assignee',top10_assignee )
top10_g = nx.DiGraph(g.subgraph(top10_nodes))
low = [n for n, d in top10_g.degree() if d < 20]
top10_g.remove_nodes_from(low)
size = [v * 10 for v in dict(top10_g.degree).values()]
groups = set(nx.get_node_attributes(top10_g,'assignee').values())
color_mapping = dict(zip(sorted(groups),count()))
color = [color_mapping[top10_g.nodes[n]['assignee']] for n in top10_g.nodes]
pos = nx.spring_layout(top10_g, k=0.15, seed=random.seed())
fig, ax = plt.subplots(figsize=(20, 15))
nx.draw_networkx(
    top10_g,
    pos=pos,
    with_labels=False,
    node_color=color,
    node_size=size,
    edge_color="gainsboro",
    alpha=0.4,
)
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Top 10 Assignees Patent Citation Network (degree>=20)", font)
font["color"] = "royalblue"
ax.text(
    0.80,
    0.10,
    "node color = Assignee",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = node degree",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.savefig('Graph/Top10 network.png')
plt.show()

#top 10 assignee degree20

top10_assignee = int(sum_ae.index[0])
top10_nodes = find_specific_attribute_node(g, 'assignee',top10_assignee )
for i in range(1,10):
    top10_assignee = int(sum_ae.index[i])
    top10_nodes = top10_nodes+find_specific_attribute_node(g, 'assignee',top10_assignee )
top10_g = nx.DiGraph(g.subgraph(top10_nodes))
low = [n for n, d in top10_g.degree() if d < 35]
top10_g.remove_nodes_from(low)
size = [v * 300 for v in dict(top10_g.degree).values()]
groups = set(nx.get_node_attributes(top10_g,'assignee').values())
color_mapping = dict(zip(sorted(groups),count()))
color = [color_mapping[top10_g.nodes[n]['assignee']] for n in top10_g.nodes]
pos = nx.circular_layout(top10_g)
fig, ax = plt.subplots(figsize=(20, 15))
nx.draw_networkx(
    top10_g,
    pos=pos,
    with_labels=False,
    node_color=color,
    node_size=size,
    edge_color="gainsboro",
    alpha=0.4,
)
font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title("Top 10 Assignees Patent Citation Network (degree>=35 and in-circle)", font)
font["color"] = "royalblue"
ax.text(
    0.80,
    0.10,
    "node color = Assignee",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.text(
    0.80,
    0.06,
    "node size = node degree",
    horizontalalignment="center",
    transform=ax.transAxes,
    fontdict=font,
)
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.savefig('Graph/Top10 network circle.png')
plt.show()

#small-world effect
#creating network of assignee
top20_assignee = int(sum_ae.index[0])
top20_nodes = find_specific_attribute_node(g, 'assignee',top20_assignee )
for i in range(1,20):
    top20_assignee = int(sum_ae.index[i])
    top20_nodes = top20_nodes + find_specific_attribute_node(g, 'assignee', top20_assignee)
top20_g = nx.DiGraph(g.subgraph(top20_nodes))
ass = nx.get_node_attributes(top20_g, "assignee")
comp =  [int(item) for item in list(sum_ae.index[0:20])]
arr = np.zeros((20,20))
e_df = pd.DataFrame(arr, columns=comp)
e_df['new_index'] = comp
e_df = e_df.set_index('new_index')
for e in list(top20_g.edges):
    s = e[0]
    t = e[1]
    i = ass[str(s)]
    j = ass[str(t)]
    e_df.loc[i][j] = e_df.loc[i][j]+1
e_list = []
for i in comp:
    for j in comp:
        if (e_df.loc[i][j] != 0 and i!=j):
            e_list.append([i, j, e_df.loc[i][j]])
G = nx.DiGraph()
G.add_weighted_edges_from(e_list)
pos = nx.spring_layout(G, k=0.15, seed=random.seed())
size = [v * 0.1 for v in dict(G.in_degree(weight='weight')).values()]
plt.figure(figsize=(12, 8))
nx.draw_networkx(
    G,
    pos=pos,
    with_labels=True,
    node_color='royalblue',
    node_size=size,
    edge_color="dimgray",
    alpha=0.7,
)
plt.title('Citation network among top 20 assignee')
plt.tight_layout()
plt.axis("off")
plt.savefig('Graph/Top20 company network.png')
plt.show()
sigma, omega = small_world_effect(G)
print(format(sigma,'.5f' ), format(omega, '.5f'))


