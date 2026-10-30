# BookLog Scan

> Transforme uma foto do seu livro em uma capa pronta para catalogação.

Projeto 2 (Opção 4 — aplicação prática) da disciplina de **Computação Visual** — Turma 07G, Universidade Presbiteriana Mackenzie. Professor: André Kishimoto.

## Integrantes

- Gabriel Nottoli Buck — RA 10425384
- Julia Andrade — RA 10427828

## Contexto

O **BookLog** é um aplicativo Android/iOS para catalogação e organização de livros. O **BookLog Scan** é um **protótipo web, independente do aplicativo**, de uma possível funcionalidade futura: digitalizar a capa de um livro físico a partir de uma fotografia. Ele **não está integrado** ao BookLog; o trabalho avaliado é o processamento de imagens.

## Problema

Ao cadastrar um livro físico, a foto costuma ter perspectiva inclinada, fundo aparecendo, ruído, iluminação irregular e dimensões inadequadas para uma biblioteca de capas.

## Solução

A aplicação recebe a foto e executa um pipeline de Computação Visual que localiza a capa, corrige a perspectiva, melhora o contraste da luminância e gera uma imagem padronizada (400×600 px por padrão), exibindo cada etapa intermediária. Se a capa não for identificada com confiança, o usuário marca os quatro cantos e o restante do pipeline segue normalmente.

## Pipeline

```
Entrada → grayscale → filtragem → bordas → contornos → perspectiva → intensidade/histograma → interpolação → resultado
```

A imagem colorida original nunca é modificada: as etapas 2 a 5 trabalham numa cópia reduzida (maior lado = 1000 px) e a perspectiva é aplicada à imagem colorida em resolução total.

| # | Etapa | O que o programa faz | Módulo |
|---|---|---|---|
| 1 | Original | Decodifica JPG/PNG (Pillow), aplica a orientação EXIF, compõe PNG transparente sobre branco, reduz fotos > 3000 px | `preprocessing.py` |
| 2 | Grayscale | `cv2.cvtColor(BGR2GRAY)`: Y = 0,299R + 0,587G + 0,114B | `preprocessing.py` |
| 3 | Filtragem espacial | `cv2.GaussianBlur` com núcleo 5×5 (ajustável; σ calculado pelo OpenCV a partir do tamanho do núcleo) | `preprocessing.py` |
| 4 | Bordas | `cv2.Canny` (limiares 20/50) na imagem em cinza suavizada **e** em cada canal B, G e R suavizado do mesmo modo; as 4 saídas são unidas por OU lógico. Depois: dilatação 3×3 e fechamento morfológico (2 iterações) para fechar lacunas | `edge_detection.py` |
| 5 | Contornos e validação | `findContours` sobre as bordas; os 12 maiores com ≥ 10% da área da foto viram fecho convexo e são simplificados a 4 vértices (`approxPolyDP`); validação geométrica e pontuação (abaixo) | `cover_detection.py` |
| 6 | Ordenação dos cantos | Ordena os 4 pontos pelo ângulo em torno do centroide e começa pelo mais próximo da origem: TL, TR, BR, BL | `perspective.py` |
| 7 | Perspectiva | `getPerspectiveTransform` + `warpPerspective` (`INTER_CUBIC`) sobre a imagem colorida original; destino = lados opostos mais longos | `perspective.py` |
| 8 | Intensidade | Sobre o canal **L do LAB**: CLAHE (padrão, `clipLimit` 2,0, blocos 8×8), equalização global, alongamento linear (percentis 1–99) ou gama; mistura opcional com a original (intensidade) | `enhancement.py` |
| 9 | Histogramas | Histograma (`calcHist`) e distribuição acumulada do canal L antes e depois, na mesma escala; histogramas R/G/B sobrepostos; média, desvio padrão e faixa dinâmica (p1–p99) | `histogram.py` |
| 10 | Interpolação | `cv2.resize` para o tamanho de saída. Automático: **`INTER_AREA`** se a capa tem largura e altura ≥ às de saída (redução); **`INTER_CUBIC`** caso contrário (ampliação, ou redução em apenas um eixo). O usuário pode escolher outra interpolação | `resize.py` |

## Relação com o conteúdo da disciplina

| Conteúdo | Onde aparece | O que de fato é feito |
|---|---|---|
| Filtragem espacial | Etapa 3 | Convolução com núcleo gaussiano 5×5 (passa-baixa) antes do gradiente, para reduzir ruído e bordas falsas |
| Detecção de bordas | Etapa 4 | Canny (gradiente de Sobel, supressão de não-máximos, histerese 20/50) em cinza e em B/G/R, com união dos mapas |
| Processamento por cor | Etapa 4 | Bordas de cada canal combinadas com as do cinza; tratamento de intensidade só no canal L (LAB) para preservar a cor |
| Região de interesse | Etapa 5 | Contornos, fecho convexo, aproximação poligonal e critérios geométricos |
| Transformação geométrica | Etapas 6–7 | Homografia 3×3 determinada por 4 pares de pontos; a matriz é exibida na interface |
| Transformações de intensidade | Etapa 8 | CLAHE, equalização, alongamento linear de contraste e gama `s = 255·(r/255)^γ` |
| Histograma | Etapa 9 | Distribuição da luminância antes/depois do tratamento, com curva acumulada |
| Interpolação | Etapas 7 e 10 | `INTER_CUBIC` na perspectiva; `INTER_AREA` (redução) ou `INTER_CUBIC` (ampliação) no redimensionamento; as demais são opcionais |

## Modo automático

O programa procura a capa nas bordas e só a aceita se ela passar em todos os critérios. Um contorno vira candidato se:

- virar quadrilátero convexo (aproximação poligonal com tolerância de 1,5% a 8% do perímetro);
- ocupar entre 10% e 98,5% da foto e **não ser a moldura da foto** (os 4 cantos a menos de 2% das bordas da imagem);
- tiver solidez (área do contorno / área do fecho convexo) ≥ 0,85;
- ser **bem descrito por 4 vértices**: área do quadrilátero / área do fecho convexo ≥ 0,90 (rejeita, por exemplo, uma mesa redonda, que viraria um quadrilátero inscrito no círculo);
- tiver ângulos internos entre 55° e 125°;
- tiver lados opostos com razão ≥ 0,35;
- tiver proporção largura/altura entre 0,40 e 2,50;
- ter **suporte de borda**: pelo menos 50% do perímetro do quadrilátero a até 4 px de uma borda do Canny.

A pontuação é: área relativa (reduzida pela metade se > 95%) × solidez × retangularidade × 1,0 (proporção típica de capa) ou 0,7 × suporte de borda. Vence a maior. Se o Canny não gerar nenhum candidato, tenta-se um **plano B**: limiarização de Otsu (nas duas polaridades, com fechamento e abertura morfológicos) e a mesma busca e validação.

O Canny também roda em cada canal de cor porque duas cores diferentes podem ter o mesmo brilho em cinza (por exemplo, capa vermelha sobre mesa cinza): nesse caso a borda só existe em algum canal.

A prioridade do detector é **evitar falsos positivos**: se nenhum candidato passar nos critérios, o app não escolhe uma região "mais ou menos" e pede o ajuste manual.

## Ajuste manual

Quando a capa não é identificada com confiança, o app informa isso e mostra o painel de ajuste, sem interromper o fluxo:

1. posicione os quatro cantos (superior esquerdo, superior direito, inferior direito, inferior esquerdo) com os controles deslizantes, em % da largura e da altura da foto;
2. confira a região selecionada na pré-visualização;
3. clique em **Aplicar correção**;
4. o pipeline continua normalmente: perspectiva, tratamento, histogramas, redimensionamento e download.

O mesmo painel está disponível, como opção, quando a detecção automática funciona; há um botão para voltar à detecção automática.

## Funcionalidades

- Upload de JPG, JPEG e PNG; imagem original exibida; botão **Processar capa**; exemplos sintéticos selecionáveis.
- 10 etapas visíveis, cada uma com a explicação do que foi feito e os parâmetros usados.
- Tamanho de saída 400×600, 600×900 ou 800×1200; interpolação, orientação e tratamento de intensidade configuráveis.
- Download do resultado em PNG ou JPEG.
- Erros (arquivo inválido, imagem pequena, falha no processamento) viram mensagens claras, sem traceback.

## Tecnologias

Python 3.10–3.12 (desenvolvido com 3.11), OpenCV (`opencv-python-headless`), NumPy, Streamlit, Pillow e Matplotlib (somente para desenhar os histogramas). Testes: pytest.

## Estrutura do repositório

```
booklog-scan/
├── app.py                          # interface Streamlit
├── src/
│   ├── preprocessing.py            # leitura, cinza, filtro gaussiano
│   ├── edge_detection.py           # Canny (cinza + B/G/R) e fechamento morfológico
│   ├── cover_detection.py          # contornos, validação, pontuação, plano B (Otsu)
│   ├── perspective.py              # ordenação de pontos, homografia, orientação
│   ├── enhancement.py              # CLAHE, equalização, alongamento, gama
│   ├── histogram.py                # histogramas antes/depois
│   ├── resize.py                   # interpolação e tamanho final
│   └── pipeline.py                 # orquestra as etapas
├── tests/                          # testes automatizados
├── examples/
│   ├── generate_examples.py        # gera as imagens SINTÉTICAS
│   ├── sintetico_*.jpg             # imagens sintéticas (não são fotos reais)
│   ├── validate_real.py            # tabela-resumo para fotografias
│   └── real/README.md              # validação com fotografias reais externas
├── .streamlit/config.toml
├── requirements.txt                # dependências da aplicação
├── requirements-dev.txt            # + pytest
├── integrantes_referencias_build.txt
└── README.md
```

## Instalação e execução

Python recomendado: 3.10 a 3.12.

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

A aplicação abre em http://localhost:8501.

## Como usar

1. Envie uma foto com a capa inteira visível, ou escolha um dos exemplos sintéticos.
2. Confira a imagem original e clique em **Processar capa**.
3. Se a capa for identificada, veja as etapas e o resultado. Se não, ajuste os quatro cantos e clique em **Aplicar correção**.
4. Alterar tamanho, interpolação ou tratamento na barra lateral recalcula tudo.
5. Baixe a capa em PNG ou JPEG.

Dicas: capa inteira no quadro, contraste entre a capa e o fundo, luz uniforme e câmera aproximadamente de frente.

## Testes

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Cobrem leitura e validação de arquivos (inválido, pequeno, grande, PNG transparente), ordenação dos pontos, homografia, redimensionamento e escolha de interpolação, tratamentos de intensidade, histogramas, critérios de rejeição (moldura da foto, suporte de borda, objeto redondo), detecção em cenas sintéticas, falha controlada sem livro, quadrilátero manual e o script de validação.

## Validação com fotografias reais

O pipeline foi experimentado em 8 cenas reais variadas (capas vistas de cima, em perspectiva, em pé, e cenas sem capa frontal). Algumas foram detectadas automaticamente; as demais não atenderam aos critérios mínimos de confiança e, nesses casos, o sistema direcionou o usuário ao ajuste manual. Essa decisão é intencional: evita apresentar como válido um recorte incorreto.

Esse conjunto é pequeno e não controlado, **não constitui benchmark e não permite estimar taxa de acerto**. Exemplos de situações que limitam a detecção automática: bordas fragmentadas, baixo contraste entre capa e fundo, reflexos, capa parcialmente visível, lombada ou página interna no lugar da capa frontal e objetos concorrentes na cena.

As fotografias externas usadas não fazem parte do repositório nem são necessárias para executá-lo; a descrição dos casos e os resultados observados estão em [`examples/real/README.md`](examples/real/README.md). Para validar com fotos próprias: `python examples/validate_real.py`.

## Publicação (Streamlit Community Cloud)

1. Publique o repositório no GitHub (público).
2. Em https://share.streamlit.io, crie um app apontando para o repositório, a branch principal e o arquivo `app.py`.
3. Em *Advanced settings*, escolha Python 3.11 ou 3.12. Não há segredos nem variáveis de ambiente.

## Limitações conhecidas

- A detecção automática é conservadora: bordas fragmentadas, baixo contraste, reflexos, capa parcialmente visível, lombada ou página interna em vez da capa frontal e objetos concorrentes levam ao ajuste manual.
- Livros curvados (distorção não projetiva) e fotos com vários livros podem falhar ou escolher a região errada.
- Uma capa que ocupe a foto inteira (cantos a menos de 2% das bordas) é tratada como moldura da foto e rejeitada.
- Se a capa tiver moldura interna bem marcada e a borda externa não for detectada, a moldura pode ser escolhida.
- Capa deitada é girada 90° no sentido horário (modo automático); se o texto ficar de lado, mude a orientação na barra lateral.
- O ajuste manual usa controles deslizantes, não arrastar pontos sobre a foto.
- Fotos acima de 3000 px no maior lado são reduzidas; arquivos acima de 30 MB são recusados.
- Os limiares foram definidos com imagens sintéticas e ajustados com poucas fotografias reais; novas fotos podem pedir novos ajustes.

## Referências

- GONZALEZ, Rafael C.; WOODS, Richard E. *Processamento Digital de Imagens*. 3ª ed. Pearson Prentice Hall, 2010.
- MAJUMDER, Aditi; GOPI, M. *Introduction to Visual Computing: Core Concepts in Computer Vision, Graphics, and Image Processing*. CRC Press, 2018.
- CANNY, John. A Computational Approach to Edge Detection. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, v. PAMI-8, n. 6, p. 679–698, 1986.
- Documentação oficial do OpenCV: https://docs.opencv.org/
- Documentação do Streamlit: https://docs.streamlit.io/

A lista completa, com as funções do OpenCV utilizadas, está em `integrantes_referencias_build.txt`.
