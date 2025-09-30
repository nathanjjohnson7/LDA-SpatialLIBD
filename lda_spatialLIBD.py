from LDA import LDA
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cm as cm

"""
#ran R in wsl2 to obtain the data

#followed below link to install R 4.5 for ubuntu
#https://cran.r-project.org/bin/linux/ubuntu/

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
    
BiocManager::install("ExperimentHub")
BiocManager::install("SpatialExperiment")

library("ExperimentHub")
library("SpatialExperiment")
eh <- ExperimentHub()

#list datasets
query(eh, "spatialLIBD")

#load the full dataset
sp <- eh[["EH9628"]]

#store data in CSVs
write.csv(colData(sp), "spatialLIBD_coldata.csv")
write.csv(rowData(sp), "spatialLIBD_rowdata.csv")
write.csv(spatialCoords(sp), "spatialLIBD_spatialcoords.csv")
#write.csv(as.matrix(imgData(sp)), "spatialLIBD_imgdata.csv")
write.csv(imgData(sp)[, c("sample_id", "image_id", "scaleFactor")], "spatialLIBD_imgdata.csv")
write.csv(as.data.frame(as.matrix(counts(sp))), "spatialLIBD_counts.csv")
"""

#load barcode/spot metadata
coldata = pd.read_csv("spatial_libd\\spatialLIBD_coldata.csv")
coldata = coldata.drop(columns=["Unnamed: 0"])

#load gene metadata
rowdata = pd.read_csv("spatial_libd\\spatialLIBD_rowdata.csv")
rowdata = rowdata.drop(columns=["Unnamed: 0"])

#load spatial coordinates
spatial_coords = pd.read_csv("spatial_libd\\spatialLIBD_spatialcoords.csv")

#load expression counts (genes × barcodes)
counts = pd.read_csv("spatial_libd\\spatialLIBD_counts.csv", index_col=0)

#get HiRes image scalefactor
imgdata = pd.read_csv("spatial_libd\\spatialLIBD_imgdata.csv")
scalefactor = imgdata[imgdata["image_id"] == "hires"]["scaleFactor"].iloc[0]

#filter out NaNs for the labels ("layer 1", layer 2", ... "white matter")
coldata = coldata[pd.notnull(coldata["ground_truth"])]
coldata = coldata.reset_index(drop=True)

#get rid of barcodes in spatial_coords that have been filtered out of coldata
spatial_coords = spatial_coords[spatial_coords['Unnamed: 0'].isin(coldata['barcode_id'])].copy()

#only store counts for barcodes in coldata
counts = counts[coldata['barcode_id']]

#get rid of rows in counts for genes that are never expressed
row_sums = np.array(counts.sum(axis=1, numeric_only=True))
empty_rows = np.where(row_sums==0)[0]
counts = counts.drop(counts.index[empty_rows], axis=0)

#get rid of rows in rowdata for genes that are never expressed
rowdata = rowdata.drop(empty_rows, axis=0)
rowdata = rowdata.reset_index(drop=True)

#create Bag-of-Words
BoW = counts.to_numpy().T

#get the number of barcodes each gene is expressed in
gene_appearances = np.sum(np.clip(BoW, 0, 1), axis=0)

#get genes that appear in less than 5% of barcodes or in all of them
too_little_too_much = np.where((gene_appearances < BoW.shape[0]*0.05)
                               | (gene_appearances == BoW.shape[0]))[0]

#perfect = np.where((gene_appearances >= BoW.shape[0]*0.05)
#                               & (gene_appearances != BoW.shape[0]))[0]

#get rid of the rows for genes that appear too little or too much
counts = counts.drop(counts.index[too_little_too_much], axis=0)

#get rid of the rows for genes that appear too little or too much
rowdata = rowdata.drop(too_little_too_much, axis=0)

#recreate BoW
BoW = counts.to_numpy().T

#get vocab (all gene names)
vocab = np.array(rowdata["gene_name"])

#get labels
labels = np.array(coldata['ground_truth'])

#create LDA and train
lda = LDA(BoW, vocab, num_topics=9)
lda.complete_loop(max_iters=1000, tol=5)

#save params
np.savez(
    "lda_params.npz",
    alpha=lda.alpha,
    beta=lda.beta,
    phi=lda.phi
    gamma=lda.gamma
)

#to reload
# data = np.load("lda_params.npz", allow_pickle=True)
# alpha = data["alpha"]
# beta = data["beta"]
# phi = data["phi"]
# gamma = data["gamma"]

#get predictions
preds = np.argmax(lda.gamma, axis=1)
categories = np.unique(labels)

#create table to show prediction results
final_matrix = np.zeros((categories.shape[0], lda.k)) #(7,9) -> (num real topics, num latent topics)

for i, cat in enumerate(categories):
    indices = np.nonzero(labels == cat) #indices of all documents in a certain category
    label_preds = preds[indices] #get all predicted topics for those documents
    #find out how many documents (all of the same category) were predicted to be of the different topics
    unique_values, counts = np.unique(label_preds, return_counts=True)
    final_matrix[i][unique_values] = counts
    
result = final_matrix.astype(int)
np.set_printoptions(formatter={'int': '{:3d}'.format})  #4 spaces per number
for i, r in enumerate(result):
    print(r, categories[i])

print() 

#get table with percentages
np.set_printoptions(precision=2, suppress=True)
result_p = result/result.sum(axis=1, keepdims=True)
for i, r in enumerate(result_p):
    print(r, categories[i])

print() 

#print top 200 genes of each topic
for topic in range(lda.k):
    top_200_genes = np.argsort(lda.beta[topic])[::-1][:200]
    print("Topic ", topic)
    print(list(rowdata.iloc[top_200_genes]['gene_name']))
    print()

#downoad image from spatialibd website
img = Image.open("spatial_libd/151673_tissue_hires_image.png")

#coordinates of barcodes
x = np.array(spatial_coords['pxl_col_in_fullres'])*scalefactor
y = np.array(spatial_coords['pxl_row_in_fullres'])*scalefactor
topics = np.array(preds)  #dominant topic per spot
num_topics = lda.k

#colormap
cmap = cm.get_cmap('tab20', num_topics)

#plot image
plt.figure(figsize=(10,10), dpi=100)
plt.imshow(img, alpha=0) #change alpha to display the image underneath

#plot the dots for each topic
for t in range(num_topics):
    idx = np.where(topics == t)[0]
    plt.scatter(
        x[idx], y[idx],
        s=20,
        color=cmap(t),
        alpha=0.7,
        label=f"Topic {t}"
    )

plt.axis('off')
plt.legend(loc='center right', bbox_to_anchor=(1.02, 0.5))
plt.savefig("dlpfc_pred.png", dpi=300, bbox_inches="tight")
plt.show()
