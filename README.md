# Temporal-causal-modeling
The data and algorithm for paper "A Temporal Causal Signaling Model for E. coli's Aerobic-to-Anaerobic Transition"

![Workflow](Figure%201%20v2.png)

# Pathway-Intepreter
The Pathway Interpreter toolbox is envisioned as a computational tool developed to facilitate the analysis of regulatory/ metabolic pathways with system biology method (In detail, the Previous Network Analysis or Causal Analysis). Utilizing validated biological pathway information, it seeks to predict regulatory networks linked to speciﬁc biological processes or drug mechanisms of action (MOA).

# Happy Path:
  - **Demo data**: 
	  - Databases file: All the files are just **raw databases records** with necesary format convertion to reduce code complecity. 
		- The **Regulon\_DB\_Synonym.txt, Regulon\_DB\_Gene.txt, enzymes.txt, genes.txt pro.txt, transunits\_reorgnized.txt** are necessary file to convert the majority of gene symbols into EcoCyc ID
		- The **EcoCyc\_Reaction\_Reorganized#.xlsx, EcoCyc\_Regulation\_Reorganized#.xlsx, EcoCyc\_tf\_association\_converted.xlsx**, **Regulon\_DB\_Reorganized.txt** record the necessary interactions to build a PKN. 
	  - Input data:
		- The **deg\_ids.xlsx** records the EcoCyc ids of the DEGs. (Multiple groups for time-series analysis)
		- The **up\_stream\_regulator.xlsx** can be upstream regulator/ drug targets obatained from any mehod
  
  - **Step to happy path**:
	- Open the **main.ipynb** for analysis. (Only **Pathway\_Intepreter.py, Network2gpml.py, Gpml2df.py** will be used in main pipeline, other modules are individual tools that can use sepeately. (Suitable for general use) Here we use the modified version that can suit our pipeline best.)
	- Run the first block (Data input) to load the Pathway\_Intepreter and data process module. (If you want to use other databases, just modify the data process funcitons in this block). It should print the statistics of the IF score as the quality reference.
	- Run the main analysis pipeline to perform FET-based tf enrichment and path seasrching. The result paths will be stored as .txt and .gpml in demo output. For small network like the demo net, we draw the network in the ouput block. BUT YOU MAY WANT TO HIDE IT FOR HUGE NETWORKS.
	- You can use PathVisio to modify the color/arrange of the map. To provide necessay analysis for the modifed network, we have the gpml2df module in the last block.
