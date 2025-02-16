import pickle

combined_data = []

# Parcours des fichiers demo_data1.pkl à demo_data11.pkl
for i in range(1, 54):
    file_path = f"/home/alexis/SuperTuxKart/stk_actor/demo_data{i}.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        combined_data.extend(data)  # Ajoute les transitions du fichier à la liste globale

# Sauvegarde des données combinées dans un nouveau fichier
save_path = "/home/alexis/SuperTuxKart/stk_actor/combined_demo_data.pkl"
with open(save_path, "wb") as f:
    pickle.dump(combined_data, f)

print(f"Enregistrement terminé : {len(combined_data)} transitions sauvegardées dans {save_path}.")
