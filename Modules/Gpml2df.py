from xml.etree import ElementTree as ET
import pandas as pd

def gpml2df(xml_path):
    ns = {'ns': 'http://pathvisio.org/GPML/2013a'}
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nodes_data = []
    edges_data = []

    for node in root.findall('ns:DataNode', ns):
        nodes_data.append({
            'TextLabel': node.get('TextLabel'),
            'GraphId': node.get('GraphId'),
            'Type': node.get('Type')
        })

    for interaction in root.findall('ns:Interaction', ns):
        points = interaction.find('ns:Graphics', ns).findall('ns:Point', ns)
        if len(points) >= 2:
            from_node = points[0].get('GraphRef')
            to_node = points[1].get('GraphRef')
            interaction_effect = points[1].get('ArrowHead', 'None')
            if interaction_effect == 'mim-stimulation':
                interaction_effect = 1
            elif interaction_effect == 'mim-inhibition':
                interaction_effect = -1
            elif interaction_effect == 'Arrow':
                interaction_effect = 2
            edges_data.append({
                'From': from_node,
                'To': to_node,
                'InteractionEffect': interaction_effect
            })

    nodes_df = pd.DataFrame(nodes_data)
    edges_df = pd.DataFrame(edges_data)
    
    return nodes_df, edges_df