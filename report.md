## Descripcion General
Nuestro problema consiste en extraer el texto de una coleccion de las cartas del Generalisimo Maximo Gomez y realizar un analisis de sentimientos al dicho resultado.

## Caracteristicas de los datos
Los datos provistos estan conformados por fotos de cartas historicas escritas a mano por Gomez. Al ser documentos historicos, algunos estan dañados por quemaduras, desgaste del papel, rasgados, por lo que algunas presentan incompletitud en los propios datos a procesar. Tambien cabe destacar que el color del papel no es constante, hay tonos mas claros, mas oscuros, con manchas de humedad, incluso con hojas rayadas (con los renglones escritos). Ademas, existe un subconjunto de cartas mecanografiadas y algunas completadas por terceras personas. La cantidad de cartas es alrededor de 160, junto con las cartas personales enviadas a Lola Tio.

## Analisis del estado del arte
En un primer momento se realizo un analisis del estado del arte con respecto al tema: 'Reconocimiento de texto manuscrito y analisis de sentimiento'. Se encontraron diversos enfoques, desde el uso de CNN (redes neuronales de convolucion) y SVM para la extraccion de texto, hasta Naive Bayes y SVM nuevamente en el analisis de sentimiento. Estas propuestas fueron descartadas luego de intentar aplicarlas a nuestro caso especifico, por el hecho de que no teniamos potencialmente la cantidad de datos necesarios para un buen entrenamiento de estos modelos, por lo que se decide buscar formas alternativas para darle solucion a nuestra problematica, con lo que luego de realizar otro analisis se decide utilizar YOLOv11 y TrOCR para la extraccion del texto de las imagenes y un modelo basador en RoBERTa, para el analisis de sentimientos. Tanto para el TrOCR como para RoBERTa se utilizaron modelos preentrenados.

## Extraccion de texto manuscrito
### Seleccion de caracteristicas
Dada la varianza de los datos antes descrita, se decide tomar las imagenes y convertirlas a escala de grises, para minimizar el sesgo que puede traer un color de fondo diferente, un color de tinta diferente, etc. Se decidio reducir la complejidad del problema original, tomando las palabras de las cartas como entrada de nuestro algoritmo. Para ello se entreno un modelo YOLOv11, tomando un subconjunto de los datos como dataset, se utilizo la plataforma roboflow para facilitar la extraccion de las bounding boxes anotadas, de donde como todas las palabras tenian igual peso para nuestro problema, decidimos anotar solamente con un label "word" y entrenar el modelo para detectar solamente los bloques contiguos de escritura, que representan en nuestro caso palabras o segmentos de palabras. 

### Evaluacion del modelo YOLOv11
Para evaluar el modelo YOLO (entrenado en Roboflow), se utilizaron las metricas mAP, Accuracy, Precision y Recall. La metrica mAP se utiliza mucho en los modelos de deteccion de objetos para evaluar que tan bien el modelo se ajusta tanto a la deteccion de los bounding boxes como a la clasificacion de las mismas. Precision representa la razon de Verdaderos Positivos (True Positives) con respecto al total de positivos inferidos por el modelo (Verdaderos + Falsos), se puede formular como: "Del total de predicciones, cuantas fueros correctas?". Recall representa la razon de los Verdaderos Positivos con respecto a los Verdaderos Positivos y Falsos Negativos, de lo cual se puede formular: "Del total de positivos, cuantos el modelo fue capaz de identificar correctamente?".
Los resultados fueron los siguientes:

- mAP: 0.94
- Precision: 0.89
- Recall: 0.942

Dichos valores reflejan un buen rendimiento de nuestro modelo, en la deteccion de bloques de letras manuscritos como los objetos, ademas de luego de varias iteraciones y pruebas se decidio ajustar los hiperparametros del modelo a 20% de confianza, o sea, como cota inferior para el valor de deteccion del objeto en cuestion y un 45% de solapamiento entre objetos, lo cual nos permite tener mejor diferenciacion entre imagenes con saltos de linea mas estrechos, asi como letras con trazos mas alargados.


### Extraccion del texto de las palabras
Luego de tener las imagenes de las palabras de una carta, se utiliza un modelo TrOCR, especificamente 'microsoft/trocr-small-handwritten' de HuggingFace. Dicho modelo esta basado en transformers, con un encoder de imagenes y un decoder de texto, lo que permite extraer el texto de las imagenes, este modelo especificamente texto manuscrito. Sobre ello, se decide hacer fine tuning, preparando un dataset con las imagenes extraidas con YOLO y el texto contenido. El cual no fue posible de realizar por falta de procesamiento en los equipos de los integrantes, y falta de tiempo para configurar un entorno de Google Collab para realizar la tarea, por lo que se provee la evaluacion de los resultados utilizando el modelo preentrenado base.

### Evaluacion del modelo TrOCR
Para evaluar el modelo TrOCR (version preentrenada base), tambien se utilizaron las metricas Precision, Recall, ademas de agregar CER (Character Error Rate), WER (Word Error Rate), F1 y Accuracy. La primera representa el porcentaje de error con respecto a los valores esperados con respecto a los caracteres. La segunda de manera similar pero con respecto a las palabras. La puntuacion F1 utiliza una media armonica para balancear los resultados de Precision y Recall, bastante util cuando existe un desbalance entre estas o cuando es necesario tener especialmente en cuenta tanto los Verdaderos Positivos como los Falsos Negativos. Accuracy representa el porciento de aciertos del modelo con respecto a los valores correctos anotados en el dataset
Los resultados fueron los siguientes:

- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1: 0.0000
- CER: 14.4978
- WER: 9.7353

Dichas métricas nos dan a entender que el modelo es pésimo ajustándose a nuestros datos, pues no pudo acertar en ningún caso, de ahi las scores de 0, aparte que se espera que los valores de CER y WER estren entre 0 y 1 (representando un porciento), pero al ser mayores que 1, implica que la tasa de error no es ni siquiera valida, pues los fallos son enormes. Esperamos que si se pudiera lograr hacer fine tuning estos resultados mejoren o al menos logren un mejor ajuste a nuestros datos, lo cual como se explico previamente no fue posible realizar.

## Analisis de sentimientos
Dado que nuestro conjunto de datos es bastante reducido, optamos por el uso de un modelo preentrenado, basado en RoBERTa, una version de BERT optimizada. Dicho modelo fue entrenado utilizando varios datasets en español, principalmente conteniendo tweets, reviews de productos, comentarios de peliculas, etc. Es sabido que el lenguaje moderno difiere bastante tanto gramatica como contextualmente del usado en documentos historicos, pero a falta de algo mas ajustado, se decidio aceptar dicho sesgo. Ademas, dichos datasets no son publicos y a pesar de solicitar acceso a ellos, hasta la fecha no hemos recibido respuesta.

Luego se realizaron pruebas con un modelo no supervisado de clusters basado en K-Means. Se realizaron diferentes pruebas con diferentes cantidades de clusters, de 2 a 7, para comparar los resultados obtenidos y analizar si se pueden extraer datos extra a los retornados por pysentimiento y realizar comparaciones.

### Preparacion y analisis de los datos:
Se obtuvo una colección de cartas de José Martí a distintos remitentes, las cuales se mezclaron con las cartas de Máximo Gómez que teníamos previamente. Dicho epistolario se encuentra en formato PDF, se realizó la transcripción de un subconjunto de las cartas de este PDF. Pudimos observar que las cartas de nuestro apóstol tienen un marcado carácter pesimista, lo cual influyó en el análisis de sentimientos, pudimos apreciar que a pesar de en ciertos casos no estar refiriéndose a nada particularmente malo, el apóstol mantenía una forma de escritura formal y nostálgica, lo cual afecta el rendimiento del modelo mencionado anteriormente. Esto no nos imposibilitó sacar conclusiones y establecer métricas que veremos adelante.

Al extraer las cartas del Epistolario, se realizó la anotación manual de aproximadamente 50 cartas. Luego, comparamos nuestras anotaciones con diferentes modelos de lenguaje disponibles en la web para tener una referencia de los datos, además de nuestra interpretación de estos. Luego de tener anotados los datos, ejecutamos el modelo preentrenado para ver sus resultados.

Para codificar el texto de las cartas se utilizo un embedding, utilizando un modelo BERT multilenguaje: "paraphrase-multilingual-MiniLM-L12-v2".Luego de tener los vectores de características retornado por el embedder, se le aplica PCA a dicho vector, para reducir las dimensiones de nuestro problema y eliminar componentes potencialmente colineales y con alta correlación, lo que mejora el resultado de nuestro algoritmo de clusters.

### Evaluación del modelo de analisis de sentimientos
En la evaluación del modelo de análisis de sentimientos se utilizaron nuevamente las métricas de Accuracy, Precision, Recall y F1, ya que las clases son discretas, aunque el modelo también devuelve resultados continuos, pero decidimos solamente utilizar las clases discretas, puesto que una anotación manual de los sentimientos de las cartas de forma continua es una tarea bastante tediosa y abierta a introducir sesgos innecesarios. 
Los resultados fueron:

Overall Metrics:
Accuracy: 0.6667

Per-class Metrics:

NEG:
Precision: 0.7143
Recall: 0.7895
F1-score: 0.7500

NEU:
Precision: 0.3529
Recall: 0.6667
F1-score: 0.4615

POS:
Precision: 0.8947
Recall: 0.5862
F1-score: 0.7083

Estos resultados son bastante buenos, dados los mencionados anteriormente, lo que nos permite determinar que la elección de modelo fue correcta. Se nota una falta de precisión en la categoría NEU, esto se explica con el marcado carácter pesimista que describimos al inicio, a pesar de no estar hablando de nada especialmente negativo (NEG). El modelo los identifica como tales y esto afecta grandemente la métrica de la clase NEG, la cual no tiene una puntuación tan baja dada la cantidad de cartas con carácter negativo en la muestra. Se necesita hacer fine tuning si se encuentra una fuente de textos históricos más grande y que sea posible anotar, para obtener resultados mas certeros.

Para evaluar las iteraciones del modelo no supervisado se utilizaron las métricas Coeficiente de Silueta, Homogeneidad, Completitud y Media de Silueta, cuyos resultados para cada tamaño de cluster están provistos en el Jupyter Notebook relacionado con el tema de Análisis de Sentimientos.


### Conclusiones
A pesar de que, debido a no poder realizar el fine tuning del último paso de la extracción de texto, y por tanto, no se pudo obtener resultados finales, las métricas de los modelos utilizados con respecto a los datos provistos de entrada fueron alentadoras. Esto nos da una idea de que con un poco más de tiempo y mejor equipo de cómputo el problema puede ser resuelto con un buen resultado final. Además, esperamos que nuestra experiencia sea de utilidad en dominios similares, especialmente de ayuda a la Oficina del Historiador de la Ciudad, que puede poseer colecciones de cartas o manuscritos históricos, de distintos autores, que por falta de una herramienta que solucione decentemente el problema, solamente tienen en su mano las imágenes de dichos documentos. Cabe destacar también que el resultado no va a ser perfecto, por lo que se sugiere también el uso de alguna herramienta basada en Grandes Modelos de Lenguaje (LLMs) para ayudar a corregir la salida y mejorar la calidad de la transcripción.