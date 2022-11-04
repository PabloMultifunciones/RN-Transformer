# RN-Transformer
Redes Neuronales - Transformer
### Introduccion ###

El Transformador fue propuesto en el artículo La atención es todo lo que necesitas. Una implementación de TensorFlow está disponible como parte del paquete Tensor2Tensor. El grupo de PNL de Harvard creó una guía que anota el documento con la implementación de PyTorch. En esta publicación, intentaremos simplificar un poco las cosas e introducir los conceptos uno por uno para que sea más fácil de entender para las personas sin un conocimiento profundo del tema.

### Una mirada de alto nivel ###

Comencemos mirando el modelo como una sola caja negra. En una aplicación de traducción automática, tomaría una oración en un idioma y generaría su traducción en otro.  

![the_transformer_3](https://user-images.githubusercontent.com/95035101/200037826-765d9bc7-9e14-4a0e-aa2b-e36d26bc45da.png)

Al abrir esa bondad de Optimus Prime, vemos un componente de codificación, un componente de decodificación y conexiones entre ellos.  

![The_transformer_encoders_decoders](https://user-images.githubusercontent.com/95035101/200037877-b7091dcd-c4a6-45d1-8691-4a0967389dac.png)

El componente de codificación es una pila de codificadores (el papel apila seis de ellos uno encima del otro; no hay nada mágico en el número seis, definitivamente se puede experimentar con otros arreglos). El componente de decodificación es una pila de decodificadores del mismo número.

![The_transformer_encoder_decoder_stack](https://user-images.githubusercontent.com/95035101/200037936-38ffa81c-87df-40b2-87a0-926a941e13c7.png)

Los codificadores son todos idénticos en estructura (pero no comparten pesos). Cada uno se divide en dos subcapas:

![Transformer_encoder](https://user-images.githubusercontent.com/95035101/200037991-1c9b6a92-e100-4883-ab4c-61d0a38fba06.png)

Las entradas del codificador fluyen primero a través de una capa de autoatención, una capa que ayuda al codificador a ver otras palabras en la oración de entrada mientras codifica una palabra específica. Veremos más de cerca la autoatención más adelante en la publicación.  

Las salidas de la capa de autoatención se envían a una red neuronal de avance. La misma red de feed-forward se aplica de forma independiente a cada posición.  

El decodificador tiene ambas capas, pero entre ellas hay una capa de atención que ayuda al decodificador a enfocarse en partes relevantes de la oración de entrada (similar a lo que hace la atención en los modelos seq2seq).  

![Transformer_decoder](https://user-images.githubusercontent.com/95035101/200038101-d31c35fb-2ee0-4535-8d11-f1e68c1685c8.png)

### Traer los tensores a la imagen ###

Ahora que hemos visto los componentes principales del modelo, comencemos a ver los diversos vectores/tensores y cómo fluyen entre estos componentes para convertir la entrada de un modelo entrenado en una salida.  

Como es el caso en las aplicaciones NLP en general, comenzamos convirtiendo cada palabra de entrada en un vector utilizando un algoritmo de incrustación.  

![embeddings](https://user-images.githubusercontent.com/95035101/200038757-6ef03b88-f4fe-4b33-813a-2af3b608fcda.png)

La incrustación solo ocurre en el codificador de más abajo. La abstracción que es común a todos los codificadores es que reciben una lista de vectores, cada uno de tamaño 512: en el codificador inferior, sería la palabra incrustaciones, pero en otros codificadores, sería la salida del codificador que está directamente debajo. . El tamaño de esta lista es un hiperparámetro que podemos configurar; básicamente, sería la longitud de la oración más larga en nuestro conjunto de datos de entrenamiento.  

Después de incrustar las palabras en nuestra secuencia de entrada, cada una de ellas fluye a través de cada una de las dos capas del codificador.  

![encoder_with_tensors](https://user-images.githubusercontent.com/95035101/200038822-a7455c20-3f4d-478e-96bb-ba017d118eeb.png)

Aquí comenzamos a ver una propiedad clave del Transformador, que es que la palabra en cada posición fluye a través de su propio camino en el codificador. Existen dependencias entre estas rutas en la capa de autoatención. Sin embargo, la capa de avance no tiene esas dependencias y, por lo tanto, las diversas rutas se pueden ejecutar en paralelo mientras fluye a través de la capa de avance.

A continuación, cambiaremos el ejemplo a una oración más corta y veremos qué sucede en cada subcapa del codificador.

### ¡Ahora estamos codificando! ###

Como ya hemos mencionado, un codificador recibe una lista de vectores como entrada. Procesa esta lista pasando estos vectores a una capa de "autoatención", luego a una red neuronal de avance, luego envía la salida hacia arriba al siguiente codificador.  

![encoder_with_tensors_2](https://user-images.githubusercontent.com/95035101/200039624-a81a3e2e-9431-41f4-9b8c-717f88fb6438.png)

### Autoatención a un alto nivel ### 

No se deje engañar por la palabra "autoatención" como si fuera un concepto con el que todos deberían estar familiarizados. Personalmente, nunca me había topado con el concepto hasta que leí el documento Attention is All You Need. Vamos a destilar cómo funciona.  

Digamos que la siguiente oración es una oración de entrada que queremos traducir:  

”El animal no cruzó la calle porque estaba demasiado cansado”  

¿A qué se refiere “eso” en esta oración? ¿Se refiere a la calle o al animal? Es una pregunta simple para un ser humano, pero no tan simple para un algoritmo.  

Cuando el modelo está procesando la palabra “eso”, la autoatención le permite asociar “eso” con “animal”.  

A medida que el modelo procesa cada palabra (cada posición en la secuencia de entrada), la atención propia le permite observar otras posiciones en la secuencia de entrada en busca de pistas que puedan ayudar a codificar mejor esta palabra.  

Si está familiarizado con los RNN, piense en cómo mantener un estado oculto permite que un RNN incorpore su representación de palabras/vectores anteriores que ha procesado con el actual que está procesando. La autoatención es el método que utiliza el Transformador para incorporar la "comprensión" de otras palabras relevantes a la que estamos procesando actualmente.  

![transformer_self-attention_visualization](https://user-images.githubusercontent.com/95035101/200039926-59c11668-1c54-453c-8634-2e12bdbbf204.png)

### Autoatención en detalle ###

Primero veamos cómo calcular la autoatención usando vectores, luego procedamos a ver cómo se implementa realmente, usando matrices.  

El primer paso para calcular la autoatención es crear tres vectores a partir de cada uno de los vectores de entrada del codificador (en este caso, la incrustación de cada palabra). Entonces, para cada palabra, creamos un vector de consulta, un vector clave y un vector de valor. Estos vectores se crean multiplicando la incrustación por tres matrices que entrenamos durante el proceso de entrenamiento.  

Tenga en cuenta que estos nuevos vectores son más pequeños en dimensión que el vector incrustado. Su dimensionalidad es 64, mientras que los vectores de entrada/salida del codificador y el incrustado tienen una dimensionalidad de 512. No TIENEN QUE ser más pequeños, esta es una opción de arquitectura para hacer que el cálculo de la atención de múltiples cabezas (en su mayoría) sea constante.  

![transformer_self_attention_vectors](https://user-images.githubusercontent.com/95035101/200040240-1a45b682-3090-407b-9e3f-5af4fe1bb9ae.png)

¿Qué son los vectores de "consulta", "clave" y "valor"?  

Son abstracciones útiles para calcular y pensar la atención. Una vez que continúe leyendo cómo se calcula la atención a continuación, sabrá prácticamente todo lo que necesita saber sobre el papel que juega cada uno de estos vectores.  

El segundo paso para calcular la autoatención es calcular una puntuación. Digamos que estamos calculando la autoatención de la primera palabra en este ejemplo, "Pensando". Necesitamos puntuar cada palabra de la oración de entrada contra esta palabra. La puntuación determina cuánto enfoque colocar en otras partes de la oración de entrada a medida que codificamos una palabra en una posición determinada.  

La puntuación se calcula tomando el producto escalar del vector de consulta con el vector clave de la palabra respectiva que estamos puntuando. Entonces, si estamos procesando la autoatención de la palabra en la posición n. ° 1, el primer puntaje sería el producto escalar de q1 y k1. La segunda puntuación sería el producto escalar de q1 y k2.  

![transformer_self_attention_score](https://user-images.githubusercontent.com/95035101/200041767-03606776-dd8d-4d3c-9ee1-e41ba5c211e7.png)

Los pasos tercero y cuarto consisten en dividir las puntuaciones por 8 (la raíz cuadrada de la dimensión de los vectores clave utilizados en el documento: 64). Esto lleva a tener gradientes más estables. Podría haber otros valores posibles aquí, pero este es el predeterminado), luego pase el resultado a través de una operación softmax. Softmax normaliza las puntuaciones para que todas sean positivas y sumen 1.  

![self-attention_softmax](https://user-images.githubusercontent.com/95035101/200041927-61423d42-b0fb-4831-ab40-59236d64803c.png)

Esta puntuación softmax determina cuánto se expresará cada palabra en esta posición. Claramente, la palabra en esta posición tendrá la puntuación de softmax más alta, pero a veces es útil prestar atención a otra palabra que sea relevante para la palabra actual.  

El quinto paso es multiplicar cada vector de valor por la puntuación softmax (en preparación para resumirlos). La intuición aquí es mantener intactos los valores de las palabras en las que queremos centrarnos y ahogar las palabras irrelevantes (multiplicándolas por números pequeños como 0,001, por ejemplo).  

El sexto paso es sumar los vectores de valores ponderados. Esto produce la salida de la capa de autoatención en esta posición (para la primera palabra).  

![self-attention-output](https://user-images.githubusercontent.com/95035101/200042143-ffe68b1d-0f4d-4ab4-9e7a-00850906dd1c.png)

Eso concluye el cálculo de la autoatención. El vector resultante es uno que podemos enviar a la red neuronal de avance. Sin embargo, en la implementación real, este cálculo se realiza en forma de matriz para un procesamiento más rápido. Así que veamos eso ahora que hemos visto la intuición del cálculo a nivel de palabra.

### Cálculo matricial de la autoatención ### 

The first step is to calculate the Query, Key, and Value matrices. We do that by packing our embeddings into a matrix X, and multiplying it by the weight matrices we’ve trained (WQ, WK, WV).  

![self-attention-matrix-calculation](https://user-images.githubusercontent.com/95035101/200042342-f99112fb-02a1-4a37-af77-8e0bd851d460.png)

Finalmente, dado que estamos tratando con matrices, podemos condensar los pasos dos a seis en una fórmula para calcular los resultados de la capa de autoatención.  

![self-attention-matrix-calculation-2](https://user-images.githubusercontent.com/95035101/200042413-55ad5b9b-876a-4e96-8e74-54d9dc2744c1.png)

### La bestia con muchas cabezas ###

El documento refinó aún más la capa de autoatención al agregar un mecanismo llamado atención de "múltiples cabezas". Esto mejora el rendimiento de la capa de atención de dos maneras:  

1. Expande la capacidad del modelo para enfocarse en diferentes posiciones. Sí, en el ejemplo anterior, z1 contiene un poco de cualquier otra codificación, pero podría estar dominada por la propia palabra. Si estamos traduciendo una oración como “El animal no cruzó la calle porque estaba demasiado cansado”, sería útil saber a qué palabra se refiere.  

2. Da a la capa de atención múltiples “subespacios de representación”. Como veremos a continuación, con la atención de varios cabezales no solo tenemos uno, sino varios conjuntos de matrices de ponderación de Consulta/Clave/Valor (el Transformador utiliza ocho cabezales de atención, por lo que terminamos con ocho conjuntos para cada codificador/decodificador) . Cada uno de estos conjuntos se inicializa aleatoriamente. Luego, después del entrenamiento, cada conjunto se usa para proyectar las incrustaciones de entrada (o vectores de codificadores/decodificadores inferiores) en un subespacio de representación diferente.  

![transformer_attention_heads_qkv](https://user-images.githubusercontent.com/95035101/200042690-f6c070fc-5e1e-4a44-83dc-7674b955305c.png)

Si hacemos el mismo cálculo de autoatención que describimos anteriormente, solo ocho veces diferentes con diferentes matrices de peso, terminamos con ocho matrices Z diferentes.  

![transformer_attention_heads_z](https://user-images.githubusercontent.com/95035101/200042761-1dc420ce-51db-4adb-a4a9-5a42d239f014.png)

Esto nos deja con un pequeño desafío. La capa de avance no espera ocho matrices, espera una sola matriz (un vector para cada palabra). Así que necesitamos una forma de condensar estos ocho en una sola matriz.  

¿Como hacemos eso? Concatenamos las matrices y luego las multiplicamos por una matriz de pesos adicional WO.  

![transformer_attention_heads_weight_matrix_o](https://user-images.githubusercontent.com/95035101/200042911-c73c4b8a-7a17-40da-af96-5786e9ff032f.png)

Eso es prácticamente todo lo que hay para la autoatención de múltiples cabezas. Es un buen puñado de matrices, me doy cuenta. Permítanme tratar de ponerlos todos en una imagen para que podamos verlos en un solo lugar.

![transformer_multi-headed_self-attention-recap](https://user-images.githubusercontent.com/95035101/200043008-a678af4c-486d-43c8-8381-232e446823f3.png)

Ahora que hemos tocado las cabezas de atención, revisemos nuestro ejemplo anterior para ver dónde se enfocan las diferentes cabezas de atención a medida que codificamos la palabra "eso" en nuestra oración de ejemplo:

![transformer_self-attention_visualization_2](https://user-images.githubusercontent.com/95035101/200043067-31e16138-4147-4ba4-95c6-8ffa8c886398.png)

Sin embargo, si agregamos todas las cabezas de atención a la imagen, las cosas pueden ser más difíciles de interpretar:

![transformer_self-attention_visualization_3](https://user-images.githubusercontent.com/95035101/200043136-ea036df9-2d3a-4349-852c-e0f82911ddfe.png)

