# GAN Pseudoaleatorios
El algoritmo congruencial multiplicativo es una forma de generar números pseudoaleatorios. En este caso, vamos a usar un algoritmo congruencial multiplicativo para generar 20,000 números y guardarlos en un archivo .csv. 

**Algoritmo Congruencial Multiplicativo:**

El algoritmo se basa en la siguiente fórmula:

$$X_{n+1} = (a * X_n) mod(m)$$

Donde:
- $(X_{n+1})$ es el próximo número pseudoaleatorio.
- $(X_n)$ es el número actual en la secuencia.
- $(a)$ es el multiplicador.
- $(m)$ es el módulo.

Para empezar, necesitamos definir los valores de (a), (m), y (X_0) (la semilla inicial). Luego, iteraremos a través del algoritmo para generar los números y los guardaremos en un archivo .csv.

```python
import csv

# Definir los valores iniciales
a = 1664525
m = 2**32
X0 = 12345

# Crear una lista para almacenar los números generados
numeros_generados = []

# Generar 20,000 números pseudoaleatorios
for _ in range(20000):
    X0 = (a * X0) % m
    numeros_generados.append(X0)

# Guardar los números en un archivo .csv
with open('numeros_pseudoaleatorios.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Numero'])
    for numero in numeros_generados:
        writer.writerow([numero])

print("Se han generado y guardado 20,000 números pseudoaleatorios en 'numeros_pseudoaleatorios.csv'.")
```

**Puntos de vista:**

1. Este algoritmo genera números pseudoaleatorios utilizando la fórmula matemática proporcionada. Es ampliamente utilizado en aplicaciones de simulación y modelado.

2. Sin embargo, es importante tener en cuenta que los números generados por este algoritmo no son verdaderamente aleatorios y pueden tener patrones predecibles si no se seleccionan adecuadamente los valores de $(a)$, $(m)$, y la semilla inicial $(X_0)$. Por lo tanto, se deben tomar precauciones al utilizar este tipo de generadores en aplicaciones críticas de seguridad.

3. Una mejora en la generación de números aleatorios podría ser considerar el uso de bibliotecas de generación de números aleatorios criptográficamente seguros en lugar de este algoritmo para aplicaciones sensibles a la seguridad.

# Modelo GAN para generar números pseudoaleatorios
```python
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Paso 1: Cargar y preparar los datos
data = pd.read_csv("numeros_pseudoaleatorios.csv")
X = data["Numero"].values.reshape(-1, 1)
y = data["Numero"].values.reshape(-1, 1)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir los datos a tensores de PyTorch
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)


# Paso 2: Diseño del modelo
class GenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenerativeModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# Definir hiperparámetros
input_size = 1
hidden_size = 64
output_size = 1
learning_rate = 0.001
num_epochs = 10000

# Crear el modelo
model = GenerativeModel(input_size, hidden_size, output_size)

# Definir función de pérdida y optimizador
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Paso 3: Entrenamiento del modelo
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Época [{epoch + 1}/{num_epochs}], Pérdida: {loss.item():.4f}")

# Paso 4: Evaluación del modelo en el conjunto de prueba
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f"Pérdida en el conjunto de prueba: {test_loss.item():.4f}")

# Paso 5: Generación de números aleatorios
# Puedes generar números aleatorios alimentando el modelo con una entrada aleatoria
random_input = torch.randn(1, input_size)  # Entrada aleatoria
generated_number = model(random_input)
print(f"Número generado: {generated_number.item()}")

# Guardar el modelo entrenado
torch.save(model.state_dict(), "generative_model_number.pth")

```
Explicación del código GAN para generar números pseudoaleatorios:
Un Generative Adversarial Network (GAN) es un tipo de arquitectura de red neuronal utilizada para generar datos que se asemejan a una distribución de datos existente, en este caso, números pseudoaleatorios. El GAN consta de dos redes neuronales principales: el generador y el discriminador, que compiten entre sí en un proceso de aprendizaje adversarial.

A continuación, te explicaré los componentes clave del código GAN que has proporcionado para generar números pseudoaleatorios:

```python
import torch
import torch.nn as nn
import numpy as np
```

Estas son las importaciones típicas de PyTorch y NumPy para trabajar con redes neuronales y datos numéricos en Python.

```python
# Paso 1: Definir la arquitectura del generador y el discriminador
```

En este paso, se definen las arquitecturas de las redes del generador y el discriminador. Estas arquitecturas determinan cómo se generan y se evalúan los números pseudoaleatorios.

```python
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

El generador es una red neuronal que toma una entrada aleatoria (una semilla) y la transforma en un número pseudoaleatorio. En este caso, es una red feedforward simple con dos capas lineales. El generador apunta a aprender a generar números que se asemejan a los números pseudoaleatorios reales.

```python
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x
```

El discriminador es una red neuronal que evalúa si un número dado es real (proveniente del conjunto de datos real) o generado por el generador. También es una red feedforward simple con dos capas lineales, pero utiliza una función de activación sigmoidal en la capa final para producir una probabilidad que indica si la entrada es real o generada.

```python
# Paso 2: Crear instancias del generador y el discriminador
```

En este paso, se crean instancias de las redes del generador y el discriminador utilizando las arquitecturas definidas anteriormente.

```python
input_size = 1  # Tamaño de entrada del generador y el discriminador
hidden_size = 64  # Tamaño de la capa oculta
output_size = 1  # Tamaño de salida del generador y el discriminador

generator = Generator(input_size, hidden_size, output_size)
discriminator = Discriminator(input_size, hidden_size, output_size)
```

Aquí, `generator` es una instancia del generador y `discriminator` es una instancia del discriminador.

```python
# Paso 3: Definir la función de pérdida y el optimizador
```

En este paso, se define la función de pérdida que se utiliza para entrenar el GAN y el optimizador para actualizar los pesos de las redes durante el entrenamiento.

```python
criterion = nn.BCELoss()  # Función de pérdida de entropía cruzada binaria
optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.001)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.001)
```

La función de pérdida `nn.BCELoss()` se utiliza porque el objetivo del discriminador es clasificar si una entrada es real o falsa, lo que se reduce a un problema de clasificación binaria. Los optimizadores Adam se utilizan para actualizar los pesos de las redes durante el entrenamiento.

```python
# Paso 4: Entrenamiento del GAN
```

En este paso, se lleva a cabo el entrenamiento adversarial del GAN, donde el generador y el discriminador compiten entre sí. El objetivo del generador es generar números que engañen al discriminador, mientras que el discriminador trata de distinguir entre números reales y generados.

```python
num_epochs = 10000  # Número de épocas de entrenamiento

for epoch in range(num_epochs):
    # Paso 4.1: Entrenar al discriminador
    discriminator.zero_grad()
    
    # Generar números reales desde el conjunto de datos real
    real_data = torch.FloatTensor(np.random.choice(real_data, batch_size))
    
    # Generar números falsos con el generador
    fake_data = generator(torch.randn(batch_size, input_size))
    
    # Calcular la pérdida del discriminador para números reales
    output_real = discriminator(real_data)
    loss_real = criterion(output_real, torch.ones_like(output_real
```

# Como usar el modelo GAN para generar números pseudoaleatorios

**Paso 1: Definir la misma arquitectura del modelo**

En este paso, definimos la arquitectura del modelo generativo. El modelo es una red neuronal que consta de dos capas lineales (fully connected). Se define como una clase llamada `GenerativeModel`. La primera capa (`layer1`) tiene una función de activación ReLU y la segunda capa (`layer2`) produce la salida.

```python
class GenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenerativeModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

**Definir la misma estructura de modelo y hiperparámetros**

En este paso, definimos los hiperparámetros del modelo, como el tamaño de entrada, el tamaño de la capa oculta y el tamaño de salida. Estos hiperparámetros deben coincidir con los que se usaron durante el entrenamiento del modelo.

```python
input_size = 1
hidden_size = 64
output_size = 1

model = GenerativeModel(input_size, hidden_size, output_size)
```

**Cargar el estado del modelo entrenado**

Aquí, cargamos los pesos del modelo previamente entrenado desde el archivo "generative_model_number.pth" en el modelo que acabas de definir. Esto es esencial para que el modelo generativo tenga la capacidad de generar números pseudoaleatorios similares a los que se generaron durante el entrenamiento.

```python
model.load_state_dict(torch.load("generative_model_number.pth"))
model.eval()
```

**Generar números aleatorios**

En este paso, generas una serie de números pseudoaleatorios utilizando el modelo previamente entrenado. La variable `num_generated` define cuántos números se generarán.

```python
num_generated = 20

generated_numbers = []

with torch.no_grad():
    for _ in range(num_generated):
        random_input = torch.randn(1, input_size)  # Entrada aleatoria
        generated_number = model(random_input)
        generated_numbers.append(generated_number.item())
```

Aquí, se utiliza una distribución normal estándar (`torch.randn`) para generar una entrada aleatoria, y luego el modelo toma esta entrada para generar un número.

**Convertir y mostrar resultados**

Finalmente, conviertes la lista de números generados en un arreglo NumPy para su fácil manejo y los imprimes en la pantalla.

```python
generated_numbers_np = np.array(generated_numbers)

print("Números generados:")
print(generated_numbers_np)
```

De esta manera, obtenemos una lista de números generados por el modelo generativo a partir de entradas aleatorias, lo que te permite generar números pseudoaleatorios basados en la distribución que el modelo aprendió durante el entrenamiento.

# Prueba de independencia de los números generados

Explicación paso a paso:

1. **Cálculo del número de observaciones y conteo de ocurrencias:** Primero, se calcula el número total de observaciones en la muestra (n) y se cuenta cuántas veces ocurre cada número único en la lista de números generados.

2. **Cálculo de las frecuencias esperadas:** Se calculan las frecuencias esperadas asumiendo que los números son independientes y que se distribuyen uniformemente. En este caso, se asume que cada número único debería ocurrir aproximadamente el mismo número de veces.

3. **Cálculo de la estadística de chi-cuadrado:** Se calcula la estadística de chi-cuadrado utilizando la fórmula estándar que compara las frecuencias observadas con las frecuencias esperadas. Esto mide cuánto se desvían los números observados de lo que se esperaría bajo la hipótesis de independencia.

4. **Cálculo de los grados de libertad:** Se calcula el número de grados de libertad (df), que es igual a la cantidad de números únicos menos 1.

5. **Cálculo del p-valor:** Se utiliza la función de distribución acumulativa chi-cuadrado para calcular el p-valor. El p-valor es la probabilidad de obtener una estadística de chi-cuadrado igual o más extrema que la observada, bajo la hipótesis nula de independencia.

6. **Devolver estadísticas y p-valor:** Finalmente, la función devuelve la estadística de chi-cuadrado y el p-valor calculados.

En el código proporcionado, estos pasos se aplican a tus números generados y se imprimen los resultados. El p-valor se usa para determinar si los números generados son independientes. Si el p-valor es muy bajo (generalmente menor que un umbral, como 0.05), se rechaza la hipótesis de independencia, lo que significa que los números no son independientes.

```python
generated_numbers = generated_numbers_np
# Prueba de independencia: Chi-cuadrado de Pearson
def independence_test(numbers):
    # Paso 1: Calcular el número de observaciones y contar las ocurrencias de cada número único
    n = len(numbers)
    unique_numbers, counts = np.unique(numbers, return_counts=True)

    # Paso 2: Calcular las frecuencias esperadas
    expected_frequencies = np.full_like(counts, n // len(unique_numbers))

    # Paso 3: Calcular la estadística de chi-cuadrado
    chi_square_statistic = np.sum(
        (counts - expected_frequencies) ** 2 / expected_frequencies
    )

    # Paso 4: Calcular los grados de libertad (df)
    degrees_of_freedom = len(unique_numbers) - 1

    # Paso 5: Calcular el p-valor utilizando la función de distribución acumulativa chi-cuadrado
    p_value = 1 - chi2.cdf(chi_square_statistic, degrees_of_freedom)

    # Paso 6: Devolver la estadística de chi-cuadrado y el p-valor
    return chi_square_statistic, p_value

# Realizar la prueba de independencia
chi_square_statistic_independence, p_value_independence = independence_test(
    generated_numbers
)

# Imprimir los resultados
print("\nPrueba de independencia (Chi-cuadrado de Pearson):")
print(f"Estadística de chi-cuadrado: {chi_square_statistic_independence}")
print(f"P-valor: {p_value_independence}")
```

## Resultados
    
```cmd
jairo@Jairo2024 MINGW64 ~/OneDrive/2022/Escritorio/SS
$ python app3.py
Números generados:
[141.53794861 128.98271179 190.48973083 103.08026123 107.96403503
 108.45558167 202.97294617 104.47090149 111.03050232 170.54071045
 118.9838562  122.22854614 172.4667511   98.67279053 113.923172
 110.16133118 172.23948669 123.1424408  287.56182861 121.00236511
 136.4100647  195.31669617 134.25010681 145.96043396 111.49868774
 156.9511261  107.37231445 118.58254242 337.84677124 290.8302002 ]

Prueba de independencia (Chi-cuadrado de Pearson):
Estadística de chi-cuadrado: 0.0
P-valor: 1.0
```
# Análisis del modelo GAN para generar números pseudoaleatorios

El código que proporcionaste tiene como objetivo cargar un modelo previamente entrenado desde un archivo `.pth` y luego inspeccionar los valores de los pesos y sesgos de cada capa de ese modelo. Esto es útil cuando deseas conocer los detalles específicos de cómo se configuraron las capas de la red neuronal después del entrenamiento. Aquí tienes una explicación paso a paso:

**Paso 1: Definir la arquitectura del modelo**

En este paso, defines la arquitectura del modelo generativo utilizando PyTorch. El modelo consta de dos capas lineales (fully connected). La primera capa (`layer1`) tiene una función de activación ReLU y la segunda capa (`layer2`) produce la salida. Esto se define en la clase `GenerativeModel`.

```python
class GenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GenerativeModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
```

**Definir hiperparámetros**

Se definen los hiperparámetros del modelo, como el tamaño de entrada (`input_size`), el tamaño de la capa oculta (`hidden_size`) y el tamaño de salida (`output_size`). Estos valores deben coincidir con la arquitectura utilizada durante el entrenamiento del modelo.

```python
input_size = 1
hidden_size = 64
output_size = 1
```

**Cargar el modelo desde el archivo `.pth`**

Utilizas `torch.load` para cargar los valores de los pesos y sesgos previamente entrenados desde el archivo `"generative_model_number.pth"` y luego los asignas al modelo que has definido.

```python
model = GenerativeModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("generative_model_number.pth"))
```

**Acceder a los parámetros del modelo**

Después de cargar el modelo, puedes acceder a sus parámetros mediante `model.state_dict()`. Esto devuelve un diccionario que contiene los nombres de los parámetros como claves y los valores numéricos de los pesos y sesgos como valores.

```python
parameters = model.state_dict()
```

**Obtener los valores de los pesos y sesgos de cada capa**

Puedes extraer los valores de los pesos y sesgos de cada capa del modelo utilizando las claves correspondientes en el diccionario de parámetros.

```python
layer1_weights = parameters["layer1.weight"]
layer1_bias = parameters["layer1.bias"]
layer2_weights = parameters["layer2.weight"]
layer2_bias = parameters["layer2.bias"]
```

Estos valores son tensores de PyTorch que contienen los detalles de cómo se configuraron las capas durante el entrenamiento.

**Inspeccionar los valores numéricos de los pesos y sesgos**

Por último, puedes imprimir en la pantalla los valores numéricos de los pesos y sesgos de cada capa utilizando los tensores que obtuviste anteriormente.

```python
print("Pesos de la capa 1:")
print(layer1_weights)
print("Sesgos de la capa 1:")
print(layer1_bias)

print("Pesos de la capa 2:")
print(layer2_weights)
print("Sesgos de la capa 2:")
print(layer2_bias)
```

Esto te permite examinar en detalle cómo se configuró el modelo generativo después del entrenamiento, lo que puede ser útil para comprender su funcionamiento y realizar ajustes si es necesario.

## Resultados actuales
```cmd
$ python app4.py
Pesos de la capa 1:
tensor([[12.6321],[13.0715],[11.6713],[ 0.4325],
        [13.2417],[10.0522],[ 7.9740],[12.0076],
        [ 9.6743],[ 6.7737],[12.1319],[-0.2168],
        [12.9082],[12.3018],[10.0930],[12.9512],
        [12.9148],[ 4.9754],[13.2947],[-0.1430],
        [ 8.9387],[13.2259],[ 8.6136],[13.0866],
        [ 0.2028],[ 6.1417],[-0.4972],[13.2263],
        [12.5759],[ 9.8980],[11.8135],[10.6541],
        [12.8690],[ 0.4270],[13.2369],[ 9.8517],
        [ 9.3390],[ 9.8404],[13.0735],[11.9808],
        [ 9.6515],[11.7791],[12.8425],[12.5866],
        [13.3540],[13.1822],[ 0.2171],[11.7843],
        [13.1715],[12.8985],[13.1691],[ 5.9375],
        [ 9.6940],[12.0622],[13.1371],[13.4297],
        [12.8637],[13.2612],[13.1432],[13.2260],
        [12.9866],[11.5406],[13.2325],[ 0.1642]])
Sesgos de la capa 1:
tensor([12.9530, 12.9719, 12.9195, -0.8948, 13.5155, 12.8688, 12.5960, 13.1248,
        12.8171, 12.3941, 13.1975, -0.3777, 12.7119, 13.0241, 12.5625, 12.2762,
        12.2047, 12.4793, 13.0455, -0.5400, 12.6658, 11.8140, 12.6451, 13.4345,
        -0.4795, 12.5005, -0.8972, 13.3245, 12.5015, 12.5016, 12.4842, 13.2078,
        12.1246, -0.7875, 13.2174, 12.6271, 12.7055, 12.6559, 11.6196, 13.3851,
        12.7896, 13.3294, 11.6424, 12.7582, 12.8172, 13.3249, -0.9475, 13.3627,
        13.2132, 13.4365, 12.3471, 12.5110, 12.6042, 13.1448, 11.9900, 12.7656,
        13.0180, 12.1473, 11.8678, 12.1116, 12.7501, 13.0806, 13.3554, -0.7750])
Pesos de la capa 2:
tensor([[12.1427, 12.0210, 12.4443, -0.0833, 11.7701, 12.8168, 13.3096, 12.1904,
         12.8921, 13.6343, 12.1548, -0.0532, 12.1912, 12.1549, 13.1397, 12.3658,
         12.4148, 13.4148, 11.8779, -0.1037, 13.1614, 12.4738, 13.2114, 11.8429,
         -0.0615, 13.4387,  0.0585, 11.8952, 12.3918, 13.2609, 12.7300, 12.4076,
         12.4852, -0.0953, 11.9154, 13.0636, 13.0742, 13.0674, 12.6520, 12.0604,
         12.8947, 12.2153, 12.8149, 12.2539, 12.0458, 11.8378,  0.0153, 12.1892,
         11.9211, 11.9281, 12.2291, 13.3679, 13.1513, 12.2175, 12.4174, 12.0132,
         12.0247, 12.2980, 12.4916, 12.2905, 12.1566, 12.3088, 11.8935,  0.0868]])
Sesgos de la capa 2:
tensor([9.8833])

```

