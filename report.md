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

Dichas metricas nos dan a entender que el modelo es pesimo ajustandose a nuestros datos, pues no pudo acertar en ningun caso, de ahi las scores de 0, aparte que se espera que los valores de CER y WER estren entre 0 y 1 (representando un porciento), pero al ser mayores que 1, implica que la tasa de error no es ni siquiera valida, pues los fallos son enormes. Esperamos que si se pudiera lograr hacer fine tuning estos resultados mejoren o al menos logren un mejor ajuste a nuestros datos, lo cual como se explico previamente no fue posible realizar.

## Analisis de sentimientos
Dado que nuestro conjunto de datos es bastante reducido, optamos por el uso de un modelo preentrenado, basado en RoBERTa, una version de BERT optimizada. Dicho modelo fue entrenado utilizando varios datasets en español, principalmente conteniendo tweets, reviews de productos, comentarios de peliculas, etc. Es sabido que el lenguaje moderno difiere bastante tanto gramatica como contextualmente del usado en documentos historicos, pero a falta de algo mas ajustado, se decidio aceptar dicho sesgo. Ademas, dichos datasets no son publicos y a pesar de solicitar acceso a ellos, hasta la fecha no hemos recibido respuesta.

### Evaluacion del modelo de analisis de sentimientos
En la evaluacion del modelo de analisis de sentimientos se utilizaron nuevamente las metricas de Accuracy, Precision, Recall y F1, ya que las clases son discretas, aunque el modelo tambien devuelve resultados continuos, pero decidimos solamente utilizar las clases discretas, puesto que una anotacion manual de los sentimientos de las cartas de forma continua es una tarea bastante tediosa y abierta a introducir sesgos innecesarios. 
Los resultados fueron:

- Accuracy: 0.7857
- Precision: 0.8190
- Recall: 0.7857
- F1: 0.7852

Estos resultados son bastante buenos, lo que nos permite determinar que la eleccion de modelo fue correcta, aunque se podria intentar hacer fine tuning si se encuentra una fuente de textos historicos mas grande y que sea posible anotar.

### Conclusiones
A pesar de que debido a no poder realizar el fine tuning del ultimo paso de la extraccion de texto y por tanto, no se pudo obtener resultados finales, las metricas de los modelos utilizados con respecto a los datos provistos de entrada fueron alentadoras. Esto nos da una idea de que con un poco de mas tiempo y mejor equipo de computo el problema puede ser resuelto con un buen resultado final. Ademas, esperamos que nuestra experiencia sea de utilidad en dominios similares, especialmente de ayuda a la Oficina del Historiador de la Ciudad que puede poseer colecciones de cartas o manuscritos historicos, de distintos autores, que por falta de una herramienta que solucione decentemente el problema, solamente tienen en su mano las imagenes de dichos documentos. Cabe destacar tambien que el resultado no va a ser perfecto, por lo que se sugiere tambien el uso de alguna herramienta basada en Grandes Modelos de Lenguaje (LLMs) para ayudar a corregir la salida y mejorar la calidad de la transcripcion.