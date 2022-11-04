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
