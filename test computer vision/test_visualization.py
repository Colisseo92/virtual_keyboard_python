import json
import networkx as nx
import matplotlib.pyplot as plt

bigrams = None
with open("E:\simplified_sample.json", "r") as file:
    bigrams = json.load(file)

# Create a new directed graph
G = nx.DiGraph()

# Loop through the smaller dataset to add edges
for first_word, pairs in bigrams.items():
    for second_word, freq in pairs:
        # Add the edge (first_word -> second_word) with the frequency as the weight
        G.add_edge(first_word, second_word, weight=freq)

# Draw the graph
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.3)  # Positioning of nodes

# Draw nodes and edges
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', alpha=0.7)

# Draw edge labels (showing the frequency of each bigram)
edge_weights = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)

# Show the plot
plt.title("Bigram Network Graph (Smaller Dataset)")
plt.show()
