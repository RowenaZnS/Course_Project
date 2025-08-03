import pickle

with open('wdbc.pkl', 'rb') as file:
    data = pickle.load(file)

print("data type:", type(data))
print("data features:", data.columns)

print("data visulization")
print(data)
