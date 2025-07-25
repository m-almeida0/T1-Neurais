# Projeto de Redes Neurais - entrega 1

## Apresentação
Este projeto tem como objetivo gerar automaticamente legendas descritivas para imagens utilizando técnicas de Visão Computacional e Processamento de Linguagem Natural com Redes Neurais. Dessa forma, a ideia do nosso projeto principal é criar uma pipeline simples de Image Captioning, combinando uma CNN para extração de imagem e uma LSTM para gerar texto, com o objetivo de compreender melhor a integração entre redes neurais convolucionais e recorrentes em uma tarefa multimodal (imagem + texto). 

Para o projeto, utilizamos o dataset [flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k), que compõe-se de cerca de 8 mil imagens com uma ou mais descrições textuais. Porém, como essa entrega foca em fazer a resolução de uma tarefa sem utilizar redes multimodais, alteramos o propósito do dataset ao atribuir rótulos aos dados. Fizemos isso através de uma análise das palavras mais comuns nos dados e escolhemos 10 delas (para referência, essa etapa começa no título '*Parte 2: Configuração e Análise do Dataset*' no código).

A partir daí, fizemos uma comparação entre dois modelos: uma rede CNN não treinada e a backbone da RESNET50 pretreinada, ambos plugados em um classificador (no código, você pode conferir os modelos em '*Parte 1/Classes*', vendo a SimpleCNN e a ResnetCNN). Para a análise, separamos os dados e as predições em suas respectivas classes e ao mesmo tempo que verificamos a acurácia por classe, também utilizamos o [Class Activation Map](https://github.com/frgfm/torch-cam) para auxiliar a entender o comportamento das redes.

Para concluir, foi possível observar o comportamento/organização da ativação dos filtros em uma rede não treinada e uma pretreinada, sendo que embora a acurácia da rede pretreinada não estivesse ótima, suas predições eram justificadas dado o comportamento dos nossos dados não estarem sendo tratados como multilabel.

## O projeto
por enquanto, modularizamos o projeto em 1 notebook *entrega1.ipynb* e 2 pastas principais:
- **data**: contém os dados utilizados no projeto, separados entre processados e crus.
- **models**: contém os modelos utilizados.