### Network2gpml module:
import random
from xml.etree.ElementTree import Element, SubElement, tostring

def string_hash(s):
    """Create a unique hash value for a string
    Args:
        s (str): The input string
    Returns: 
    A unique hash value for the string
    """
    hash_value = 0
    prime = 31  # A prime number to create a unique hash value

    for char in s:
        hash_value = hash_value * prime + ord(char)
    
    hash_value = "E" + str(hash_value)

    return str(hash_value)

def get_effect_of_edge(edge_from, edge_to, process_df):
    """Get the effect of an edge from one node to another
    Args:
        edge_from (str): The id of the node where the edge originates
        edge_to (str): The id of the node where the edge ends
        process_df (pd.DataFrame): The DataFrame containing the process information
    Returns:
        The effect of the edge if found, else None
    """
    # Convert symbols to IDs using the predefined dictionary
    from_id = edge_from
    to_id = edge_to

    # Validate if both IDs were found
    if from_id is None or to_id is None:
        raise ValueError(f"IDs not found for edge {edge_from} -> {edge_to}")

    # Look up the effect in the process DataFrame
    effect = process_df.loc[(process_df['Object_from'] == from_id) & (process_df['Object_to'] == to_id), 'Effect']
    
    # Return the matching effect if found, else None
    return effect.iloc[0] if not effect.empty else None

# Function to print all nodes and edges in the network
def print_network(network):
    """Print all nodes and edges in the network
    Args:
        network (Network): The network object containing nodes and edges
    Returns:
        None
    """
    # Iterate over all nodes
    for node_name, node in network.nodes.items():
        print(f"Node: {node_name}")
        
        # Printing all outgoing edges from the node
        for edge in node.out_edges:
            print(f"\tOutgoing to {edge.to_node.name} with effect {edge.effect}")
        
        # Printing all incoming edges to the node
        for edge in node.in_edges:
            print(f"\tIncoming from {edge.from_node.name} with effect {edge.effect}")

# Function to generate a random float with the same format as in the example
def random_position():
    """Generate a random float with the same format as in the example"""
    return round(random.uniform(100.0, 1000.0), 10)


def create_datanode_element(node_name,Symbol2id):
    """Create a DataNode element for a given node name
    Args:
        node_name (str): The name of the node
    Returns:
        The DataNode element for the node
    """
    ids = []
    if "[" in node_name:
        print(node_name)
        node_names = eval(node_name)
        for i in node_names:
            ids.append(Symbol2id.get(i,  string_hash(i)))
        Id = str(ids)
    else:
        Id = Symbol2id.get(node_name, string_hash(node_name))
    datanode = Element("DataNode", {
        "TextLabel": node_name,
        "GraphId": Id,  # Replace None with "default_id"
        "Type": "Protein"
    })

    graphics = SubElement(datanode, "Graphics", {
        "CenterX": str(random_position()),
        "CenterY": str(random_position()),
        "Width": "56.296055796056066",
        "Height": "37.51803751803743",
        "ZOrder": "32768",
        "FillColor": "ff6600",
        "FontWeight": "Bold",
        "FontStyle": "Italic",
        "FontSize": "14",
        "Valign": "Middle",
        "ShapeType": "RoundedRectangle",
        "Color": "ffffff"
    })

    xref = SubElement(datanode, "Xref", {
        "Database": "BioCyc",
        "ID": Id
    })

    return datanode

# Function to create an Interaction element
def create_interaction_element(from_node, to_node, effect,Symbol2id):
    """Create an Interaction element for a given edge
    Args:
        from_node (str): The name of the node where the edge originates
        to_node (str): The name of the node where the edge ends
        effect (int): The effect of the edge
    Returns:
        The Interaction element for the edge"""
    interaction = Element("Interaction", {
        "GraphId": "i" + str(random_position())
    })

    if effect == 1:
        color = "cc0033"
        arrow = "mim-stimulation" 
    elif effect == -1:
        color = "009933"
        arrow = "mim-inhibition" 
    elif effect == 2:
        color = "000099"
        arrow = "Arrow"
    else:
        color = "333333"  
        arrow = "Arrow" 
    graphics = SubElement(interaction, "Graphics", {
        "ZOrder": "32821",
        "LineThickness": "2.0",
        "Color": color
    })
    
    ids = []
    if "[" in from_node:
        node_names = eval(from_node)
        for i in node_names:
            ids.append(Symbol2id.get(i,  string_hash(i)))
        Id = str(ids)
    else:
        Id = Symbol2id.get(from_node,  string_hash(from_node))

    point1 = SubElement(graphics, "Point", {
        "X": str(random_position()),
        "Y": str(random_position()),
        "GraphRef": Id,
        "RelX": "0.5",
        "RelY": "1.0"
    })
    
    ids = []
    if "[" in to_node:
        node_names =  eval(to_node)
        for i in node_names:
            ids.append(Symbol2id.get(i,  string_hash(i)))
        Id = str(ids)
    else:
        Id = Symbol2id.get(to_node,  string_hash(to_node))
    point2 = SubElement(graphics, "Point", {
        "X": str(random_position()),
        "Y": str(random_position()),
        "GraphRef": Id,
        "RelX": "-0.5",
        "RelY": "-1.0",
        "ArrowHead": arrow
    })

    if effect == 1:
        effect_text = "Activation"
    xref = SubElement(interaction, "Xref", {
        "Database": "BioCyc",
        "ID": ""
    })

    return interaction

def Network2gpml(network, Id2symbol, Symbol2id, process_df, savepath = "/gpml/converted_pathway.gpml"):

    nodes,edges = network.get_nodes_edges()

    # Create the root element for the Pathway
    pathway = Element("Pathway", {
        "xmlns": "http://pathvisio.org/GPML/2013a",
        "Name": "Transformation_Path",
        "Version": "20231027",
        "Organism": "Escherichia coli"
    })

    # Add the Graphics element to the Pathway
    graphics = SubElement(pathway, "Graphics", {
        "BoardWidth": "1787.3212301142007",
        "BoardHeight": "1113.0655571368297"
    })

    # Create DataNode elements for each node
    for node in nodes:
        datanode_element = create_datanode_element(Id2symbol[node],Symbol2id)        
        pathway.append(datanode_element)

    # Create Interaction elements for each edge
    for (from_node, to_node, _,_,_) in edges:
        effect = get_effect_of_edge(from_node, to_node, process_df)
        interaction_element = create_interaction_element(Id2symbol[from_node], Id2symbol[to_node], effect, Symbol2id)
        pathway.append(interaction_element)

    # Convert the XML to a string
    xml_str = tostring(pathway, encoding='utf8', method='xml').decode()

    # Format the XML string with proper indentation
    import xml.dom.minidom

    dom = xml.dom.minidom.parseString(xml_str)  # Parse the XML string
    pretty_xml_as_string = dom.toprettyxml()

    # Replace the XML header with the new header, ensuring encoding is added.
    pretty_xml_as_string = pretty_xml_as_string.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>', 1)

    # Replace the closing tag with new content, maintaining indentation.
    pretty_xml_as_string = pretty_xml_as_string.replace('</Pathway>', '\t<InfoBox CenterX="0.0" CenterY="0.0" />\n\t<Biopax />\n</Pathway>', 1).replace(';', '_')

    pretty_xml_as_string = pretty_xml_as_string.replace('&quot;', "").replace("'", "").replace("[", "").replace("]", "").replace(", ", "_")
    with open(savepath, 'w') as file: 
        file.write(pretty_xml_as_string)
