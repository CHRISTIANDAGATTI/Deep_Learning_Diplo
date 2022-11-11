## Entregable de la materia optativa Aprendizaje Profundo de la Diplomatura en Ciencias de Datos FaMAFyC 2022.

Integrantes: Pilar Ávila, Pablo Madriaga y Christian Dagatti.
Profesores: Johanna Frau y Mauricio Mazuecos

Consigna: Predecir la categoría de un artículo de MELI a partir de sus títulos.
Ejercicio:
Implementar una red neuronal que asigne una categoría dado un título. Para este práctico se puede usar cualquier tipo de red neuronal. Les que hagan solo la primera mitad de la materia, implementarán un MLP. Quienes cursan la materia completa, deberían implementar algo más complejo, usando CNNs, RNNs o Transformers.
Algunas consideraciones a tener en cuenta para estructurar el trabajo:
1.	Hacer un preprocesamiento de los datos (¿Cómo vamos a representar los datos de entrada y las categorías?).
2.	Tener un manejador del dataset (alguna clase o función que nos divida los datos en batches).
3.	Crear una clase para el modelo que se pueda instanciar con diferentes hiperparámetros
4.	Hacer logs de entrenamiento (reportar tiempo transcurrido, iteraciones/s, loss, accuracy, etc.). Usar MLFlow.
5.	Hacer un gráfico de la función de loss a lo largo de las epochs. MLFlow también puede generar la gráfica.
6.	Reportar performance en el conjunto de test con el mejor modelo entrenado. La métrica para reportar será balanced accuracy (Macro-recall).

# Análisis exploratorio
A modo exploratorio hicimos un breve análisis y visualización de la base de datos de Meli Challenge 2019, que cuenta con tres subconjuntos de datos: entrenamiento, validación y test  del idioma spanish.
Para cada subconjunto, vimos:

•	Cantidad y nombre de columnas con sus tipos:
-	object: language, label_quality, title, category, split, tokenized_title, data
-	int64: target, n_labels, size

•	Cantidad de registros totales.

•	Cantidad de valores nulos: ninguna para cualquier conjunto

•	Cantidad de categorías distintas (n_labels): igual cantidad para todos, 632

•	Conteo de títulos por categorías.

•	Cantidad de títulos distintos: coincidían en la cantidad de registros

# Preprocesamiento y tokenización de los datos
Filtramos y modificamos algunas consideraciones en las palabras del dataset para posteriormente poder vectorizalos y tokenizar sin inconvenientes.
En primer lugar concatenamos los 3 conjuntos para lograr que el proceso en los datos asigne el mismo token a la misma palabra en los distintos subconjuntos (train/validation/ test). 
Para el preprocesamiento se utilizan módulos  de las librerías nltk y gensim que ejecutan las siguientes tareas: Transformar todas las cadenas en minúsculas; eliminar etiquetas de código del tipo; separar por un espacio de cadenas alfanuméricas; reemplazar signos de puntuación ASCII por espacios; eliminar cualquier otro carácter que no sea letras o números; remover espacios múltiples;  eliminar dígitos numéricos y descartar las cadenas de longitud menor a 3. 
Una vez generado el diccionario de palabras, se eliminan de este las palabras vacías. 
Luego, se incluyen dos tokens especiales. Uno para las palabras desconocidas (1) y otro para el relleno al ajustar el tamaño de las cadenas (0). 
Por último, se codifican las categorías con un índice, por orden de aparición. En este caso se cuenta con 632 categorías diferentes.

# PadSequences y Dataloaders
Se creó una clase PadSequences para iguales el tamaño de los datos con los que será alimentada la red.

Como en este caso trabajamos con secuencias de palabras (representadas por sus índices en un vocabulario), cuando queremos buscar un batch de datos, el DataLoader de PyTorch espera que los datos del batch tengan la misma dimensión (para poder llevarlos todos a un tensor de dimensión fija). Esto lo podemos lograr mediante el parámetro de collate_fn. En particular, esta función se encarga de tomar varios elementos de un Dataset y combinarlos de manera que puedan ser devueltos como un tensor de PyTorch. Se define un módulo PadSequences que toma un valor mínimo, opcionalmente un valor máximo y un valor de relleno (pad) y dada una lista de secuencias, devuelve un tensor con padding sobre dichas secuencias.


# DataLoaders
Se utilizaron los dataloaders de Pytorch para pasar los datos por lotes a la red.
Ya habiendo definido nuestros conjuntos de datos y nuestra collation_fn, podemos definir nuestros DataLoader, uno para entrenamiento y otro para evaluación. La diferencia fundamental está en shuffle, no queremos mezclar los valores de evaluación cada vez que evaluamos porque al evaluar mediante mini-batchs nos puede generar inconsistencias.
Creamos los dataloaders para cada conjunto haciendo uso de la clase Dataloader, que nos ayuda a entrenar con mini-batches (elegimos 128 para agilizar el tiempo de entrenamiento) al modelo para aumentar la eficiencia evitando iterar de a un elemento

# Embeddings
En todos nuestros experimentos utilizamos la primera capa de embeddings que es rellenada con los valores de word embeddings (conversión del texto a una representación por vectores) continuos preentrenados en español de SBW, de 300 dimensiones (descargado en la carpata data).

## MODELO MLP 

Experimentos

•	Modelo baseline
Se diseñó un modelo simple con una capa de emeding (300), luego dos capas ocultas (de tamaño 2048 y 1024) con función de activación relu y la capa de salida. La función de pérdida utilizada fue CrossEntropyLoss, que es apropiada para problemas de clasificación muticlase. EL learning Rate usado fue de 0.002.
Elegimos inicialmente utiilizar SGD (CON momentum=0.9) como algoritmo de optimización. 
Por una cuestión de capacidad de procesamiento todos los modelos aplicados sobre el total del dataset fueron entrenados en 5 épocas (se intentó con 20 y 10 épocas para evaluar si la función de pérdida podía mostrar signos de sobreajuste; pero esto no fue posible porque se detenía el kernel por falta de recursos. La métrica utilizada para evaluar los modelos fue balanced accuracy y en conjunto de entrenamiento fue de 0.489 y en validación de 0.524

•	Pruebas
Para la búsqueda de los mejores hiperparámetros y teniendo en cuenta que por el tamaño del dataset y los recursos disponibles lleva mucho tiempo generar pruebas de hiperparámetros sobre el dataset completo (aproximadamente entre 10 y 15 minutos cada época) se decidió reducir el conjuntos de entrenamiento y validación para disminuir el tiempo de procesamiento.
Inicialmente decidimos realizar experimentos  utilizando una muestra del dataset  de solo 5 categorías. Para esto se filtraron en todos los subconjutnos del datasets (train, validation y test) y se conservaron todos los registros de las categorias seleccionadas reduciendo considerablemente el tamaño y el tiempo de procesamiento. Aunque por recomendación de los profes aprendimos que no era una buena práctica ya que estaríamos entrenando un modelo con muy pocas categorías e intentaríamos predecir posteriormente 632 y tampoco nos permite analizar la mejor combinación de hiperparámetros ya que los resultados obtenidos en el modelo de prueba no serían “trasladables” si solo tiene 5 categorías analizadas.
Por tal motivo posteriormente decidimos disminuir el dataset (siempre el fin de reducir el tiempo de procesamiento) generando muestras (del 10%) que en train, validation y test contengan las 632 categorias del dataset.
Finalmente obtenidos la mejor combinación de hiperparámetros para nuestro problema entrenamos el modelo sobre el dataset completo.

•	Modelo Final – Conclusiones:

Modelo: optamos por utilizar un embedding pre-entrenado y luego definimos el modelo con "embedding_size": 300; "hidden1_size": 1024, "hidden2_size": 2048;  Optimizador:Adam; FnLoss: CrossEntropy; Lr: 0.001; Fn de activación: relu; Dropout = 0.1; Frezze embedding=True.


•	En relación al número de capas, en las pruebas no se notó mejora aumentando el número, razón por la cual se mantuvo en dos capas ocultas.
•	En relación al valor de las capas, fuimos variando entre los valores 2048-1024-512 para conseguir la mejor combinación.
•	En relación al learning rate, hicimos pruebas con valores como 0.001 / 0.002 y más grandes como 0.01 concluyendo que los mejores resultados se obtuvieron con el valor de 0.001 para el modelo aplicado a muestras en el que podíamos permitirnos probar con más épocas si creíamos necesario por lo cual la velocidad de aprendizaje podía ser menor.
•	Todos los experimentos fueron registrados con MLflow para poder comparar los modelos y obtener las gráficas de pérdida para entrenamiento y evaluación. 
•	Usamos funciones de activación relu en las capas ocultas, aunque también hicimos pruebas con función gelu sin obtener variaciones significativas.
•	A la salida de la red no utilizamos una función de activación ya que al usar cómo loss una cross entropy loss entendemos ya que esta ya implementa una softmax por defecto apropiada para problemas de clasificación muticlase.
•	En cuanto al optimizador probamos con SGD en primer momento, pero no obtuvimos mejores resultados comparándolo contra Adam, por eso elegimos este último.
•	Probamos regularización mediante weight decay pero no obtuvimos mejoras

Listado de experimentos sobre dataset de muestra:
 
![image](https://user-images.githubusercontent.com/102828334/200931463-d4ec5e71-53c8-4829-a9bf-9370c8b98a03.png)




Listado de experimentos sobre dataset completo:

 ![image](https://user-images.githubusercontent.com/102828334/200931707-840add21-e1fa-4068-a1f8-25490f02b0eb.png)



Frezze embedding
Al poner en False el freeze embedings, obtenemos muchos mejores resultados que utilizando los mismos parámetros y seteandolo en True. Esto nos generó la duda, de que al realizar esta modificación estemos ante un caso de overfitting.
Analizando los gráficos de función de perdida en conjunto de entrenamiento y validación nos muestran un marcado sobreajuste en el modelo.

Parámetros con mejor métrica Balanced_Accuracy en entrenamiento y validación (caso de sobreajuste):
 ![image](https://user-images.githubusercontent.com/102828334/200931734-d1144a8d-1de9-45e5-b2d4-2aacac1e20eb.png)



Función de pérdida en conjuntos de entrenamiento y validación para experimento con mejor Balanced_Accuracy (caso de sobreajuste):
![image](https://user-images.githubusercontent.com/102828334/200931795-879fb273-6f98-47f2-b005-2b3876cdd902.png)


## MODELO MLP FINAL

Función de pérdida en conjuntos de validación y test para experimento con mejor Balanced_Accuracy (Modelo elegido):
 ![image](https://user-images.githubusercontent.com/102828334/200932047-e8110e83-df6c-4753-a704-eea92058ecd4.png)


Por tal aunque la métrica BAS para el conjunto de Validación y entrenamiento es menor que al modelo anterior elegimos finalmente poner en “True” el freeze embedings evitando el sobreajuste.

Parámetros con mejor métrica Balanced_Accuracy en entrenamiento y validación
 ![image](https://user-images.githubusercontent.com/102828334/200932292-7786b520-495b-4893-9254-ae4d4ad2038c.png)

Balanced Accuracy sobre test – Modelo Final: 
 ![image](https://user-images.githubusercontent.com/102828334/200932365-6967b00d-d071-4e9a-9920-ab7003062ef3.png)


A futuro entendemos que nuestro accuracy puede mejorar, podríamos probar con modificar la arquitectura de nuestra red y algunos otros parámetros. También entendemos que podríamos aumentar el número de épocas al menos a 10 para lograr mejores predicciones teniendo en cuenta no entrar en sobre ajuste. 
