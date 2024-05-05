### This module is used to convert the gene id to gene symbol and vice versa
### BUT the main function have its own module, so this module is only for EcoCyc-RegulonDB id conversion
### Tool only

import pandas as pd

nodes_for_conversion = ["phnE"]

Gnet = pd.DataFrame(pd.read_table("/Volumes/Code/E.coli/Database/Convertion/genes.txt"))
Gidl = list(Gnet['UNIQUE-ID'])
Gsymboll = list(Gnet['NAME'])

Pnet =  pd.DataFrame(pd.read_table("/Volumes/Code/E.coli/Database/Convertion/pro.txt"))
Pidl = list(Pnet['UNIQUE-ID'])
Psymboll = list(Pnet['NAME'])

Ennet = pd.DataFrame(pd.read_table("/Volumes/Code/E.coli/Database/Convertion/enzymes.txt"))
Enidl = list(Ennet['UNIQUE-ID'])
Ensymboll = list(Ennet['NAME'])

Trans_net = pd.DataFrame(pd.read_table("/Volumes/Code/E.coli/Database/Convertion/transunits_reorgnized.txt"))
Transl = list(Trans_net['UNIQUE-ID'])
Transymboll = list(Trans_net['NAME'])

Extral = ["EG11283","EG10820","G8205 ","EG10821;EG11544","G0-9384","G0-9121;EG12844","EG10320","G6109;G6110","EG11304","G0-16657", "EG11249;EG10571","EG10466","G6525","G6686","G6231","G7948;EG10821","EG12834","EG10836","EG11131","EG10440","EG12766","G0-10433","G7601;G7602","EG10659","G7120","G8221","G6494;EG10164","EG10442;EG10443","G0-10445"]
ExtraSymboll = ["phnE","RcsAB","gapC",'GadE-RcsB', 'sroD', 'YefM-YoeB', 'FlhDC', 'DinJ-YafQ', 'ydfE', 'yoeG', 'MazE-MazF', 'HU', 'efeU', 'rzpR', 'ykiA', 'RcsB-BglJ', 'yhdW', 'RelB-RelE','IHF', 'agaA', 'insX', 'HigB-HigA', 'nmpC', 'yegZ', 'ilvG', 'CRP-Sxy', 'HipAB',"ymjB"]

error_count = 0
Id2symbol = dict(zip(Gidl + Enidl + Pidl + Transl + Extral, 
                      Gsymboll + Ensymboll + Psymboll + Transymboll + ExtraSymboll))

Symbol2id = dict(zip(Gsymboll + Ensymboll + Psymboll + Transymboll + ExtraSymboll,
                        Gidl + Enidl + Pidl + Transl + Extral))

def convert_symbol_2_id(node):
    try:
        return Symbol2id[node]
    except:
        return node
    
def convert_id_2_symbol(node):
    try:
        return Id2symbol[node]
    except:
        return node
