import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import ks_2samp, chi2


# Paso 1: Definir la misma arquitectura del modelo
class GenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenerativeModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# Definir la misma estructura de modelo y hiperparámetros
input_size = 1
hidden_size = 64
output_size = 1

model = GenerativeModel(input_size, hidden_size, output_size)

# Cargar el estado del modelo entrenado desde "generative_model_number.pth"
model.load_state_dict(torch.load("generative_model_number.pth"))
model.eval()

# titulo
st.sidebar.title("Generador de números aleatorios GAN")
st.sidebar.write(
    "El siguiente programa genera números aleatorios a partir de una red neuronal artificial entrenada con numeros generados a partir del **algoritmo de congruencial multiplicativo**, con el algoritmo **GAN** (Generative Adversarial Network)."
)
st.sidebar.write(
    "Las redes generativas adversativas, también conocidas como GANs en inglés, son una clase de algoritmos de inteligencia artificial que se utilizan en el aprendizaje no supervisado"
)
# Generar 20 números aleatorios
numero = st.sidebar.slider(
    "Ingrese el número de datos a generar",
    min_value=30,
    max_value=1000,
    value=30,
    step=1,
)
num_generated = numero

generated_numbers = []

with torch.no_grad():
    for _ in range(num_generated):
        random_input = torch.randn(1, input_size)  # Entrada aleatoria
        generated_number = model(random_input)
        generated_numbers.append(generated_number.item())

# Convertir la lista de números generados a un arreglo NumPy
generated_numbers_np = np.array(generated_numbers)

# Imprimir los números generados
# print("Números generados:")
# print(generated_numbers_np)

generated_numbers = generated_numbers_np


# Prueba de independencia: Chi-cuadrado de Pearson
def independence_test(numbers):
    n = len(numbers)
    unique_numbers, counts = np.unique(numbers, return_counts=True)
    expected_frequencies = np.full_like(counts, n // len(unique_numbers))
    chi_square_statistic = np.sum(
        (counts - expected_frequencies) ** 2 / expected_frequencies
    )
    degrees_of_freedom = len(unique_numbers) - 1
    p_value = 1 - chi2.cdf(chi_square_statistic, degrees_of_freedom)
    return chi_square_statistic, p_value


# Realizar la prueba de independencia
chi_square_statistic_independence, p_value_independence = independence_test(
    generated_numbers
)

# Imprimir los resultados
# print("\nPrueba de independencia (Chi-cuadrado de Pearson):")
# print(f"Estadística de chi-cuadrado: {chi_square_statistic_independence}")
# print(f"P-valor: {p_value_independence}")

# Mostrar con Streamlit
# Numeros generados
st.write("Números generados:")
st.write(generated_numbers_np)
st.write("Prueba de independencia (Chi-cuadrado de Pearson):")
st.success(f"Estadística de chi-cuadrado: {chi_square_statistic_independence}")
st.success(f"P-valor: {p_value_independence}")
