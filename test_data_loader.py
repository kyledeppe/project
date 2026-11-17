from data_loader import load_school_networks

data = load_school_networks()

print("\nNetwork dimensions:")
print(data.A1.shape)
print(data.A2.shape)

print("\nMetadata:")
print(data.metadata.head())

print("\nDay 1 edges:", data.G1.number_of_edges())
print("Day 2 edges:", data.G2.number_of_edges())