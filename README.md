# Trabalho de FIA: Raciocínio Espacial Neuro-Simbólico com LTNtorch

**Disciplina:** Fundamentos de Inteligência Artificial
**Ferramenta:** Logic Tensor Networks (LTNtorch)

### 👨‍💻 Integrantes do Grupo

* Gabriel Menezes Rodrigues
* Joshua Tadeu Bastos Pereira
* Romário Roberto de Pádua Andrade
<!-- adicione aqui os demais integrantes -->

---

## 1. Descrição Teórica

### 1.1. Inteligência Artificial Neuro-Simbólica (NeSy)
A **IA Neuro-Simbólica (NeSy)** combina o aprendizado de redes neurais (Deep Learning) com o raciocínio e a explicabilidade da lógica simbólica. As redes neurais lidam bem com dados numéricos e com incerteza. A lógica fornece a estrutura para raciocinar sobre esses dados, o que permite sistemas que aprendem com menos exemplos e respeitam regras de conhecimento prévio.

### 1.2. Logic Tensor Networks (LTN)
O projeto usa **Logic Tensor Networks (LTN)**, uma estrutura que integra lógica de primeira ordem a redes profundas por meio da **Lógica Real** (Real Logic):
* **Constantes e Variáveis** (`ltn.Constant`, `ltn.Variable`) são tensores: uma entidade específica ou um conjunto de entidades quantificável.
* **Predicados** (ex: `is_circle(x)`) são redes neurais que retornam um grau de verdade em $[0,1]$.
* **Conectivos e Quantificadores** são operadores fuzzy diferenciáveis:

| Símbolo | Operador LTNtorch | Fórmula |
| :--- | :--- | :--- |
| $\neg$ | `NotStandard` | $1 - u$ |
| $\wedge$ | `AndProd` | $u \cdot v$ |
| $\vee$ | `OrProbSum` | $u + v - u \cdot v$ |
| $\Rightarrow$ | `ImpliesReichenbach` | $1 - u + u \cdot v$ |
| $\forall$ | `AggregPMeanError(p=2)` | $1 - \left(\frac{1}{n}\sum(1-u_i)^2\right)^{1/2}$ |
| $\exists$ | `AggregPMean(p=2)` | $\left(\frac{1}{n}\sum u_i^2\right)^{1/2}$ |

* **Axiomas** são fórmulas lógicas que entram na função de perda: $\text{Loss} = 1 - \text{SatAgg}(\text{KB})$. O treinamento busca maximizar a satisfatibilidade da base de conhecimento.

### 1.3. Representação dos Dados
O notebook trabalha com três ambientes:
* **Objetos coloridos:** vetores RGB de 3 posições (ex: bola vermelha `[1, 0, 0]`, cubo azul `[0, 0, 1]`).
* **Objetos para igualdade:** vetores RGB, incluindo dois vermelhos próximos (`[1, 0, 0]` e `[0.95, 0.05, 0]`) e um azul.
* **Figuras geométricas 2D:** vetores de 2 posições com o centro $(x, y)$. Cada forma é sorteada em uma região própria do plano:

| Forma | Predicado | Região x | Região y |
| :--- | :--- | :--- | :--- |
| Círculo | `is_circle` | [0, 4] | [0, 10] |
| Quadrado | `is_square` | [6, 10] | [0, 10] |
| Triângulo | `is_triangle` | [0, 10] | [6, 10] |
| Pentágono | `is_pentagon` | [11, 14] | [0, 10] |
| Hexágono | `is_hexagon` | [0, 10] | [-4, -1] |

---

## 2. Experimentos

O sistema foi avaliado em **5 execuções independentes** do notebook (sementes 0 a 4). Os pontos das figuras geométricas são gerados aleatoriamente a cada rodada.

### 2.1. Base de Conhecimento Básica
Predicados `is_red`, `is_blue` e `is_round` (MLPs 3→10→1) treinados por 100 épocas com os axiomas:
1.  **Fatos:** $IsRed(bola\_vermelha)$, $IsBlue(cubo\_azul)$, $IsRound(bola\_vermelha)$, $\neg IsRound(cubo\_azul)$
2.  **Regra:** $\forall x\,IsRound(x)$, com $x \in$ bolas
3.  **Regra:** $\forall x\,(IsRed(x) \Rightarrow IsRound(x))$

### 2.2. Formas de Igualdade
Quatro predicados de igualdade $Eq(x,y)$, avaliados com os axiomas de reflexividade $\forall x\,Eq(x,x)$, simetria $Eq(a,b)$ e $Eq(b,a)$ entre os dois vermelhos, e $\neg Eq(vermelho, azul)$:

| Igualdade | Definição | Parâmetro treinável |
| :--- | :--- | :--- |
| **Diagonal** | $\frac{1}{2}(\cos(x,y) + 1)$ | — |
| **Cosseno** | $\sigma\left(T \cdot \frac{\cos(x,y)+1}{2}\right)$ | temperatura $T$ |
| **Euclidiana** | $e^{-\gamma \lVert x-y \rVert^2}$ | $\gamma$ |
| **Aprendível** | MLP sobre $[x, y]$ (6→64→32→1) | todos os pesos |

### 2.3. Relações Geométricas em um Plano 2D
Predicados de forma (`ShapeClassifier`, MLP 2→16→1) e relações espaciais aprendíveis (`LearnableSpatialRelation`), treinados por 300 épocas:
* `LeftOf(x,y)`: usa a diferença $x_y - x_x$
* `Above(x,y)`: usa a diferença $y_x - y_y$
* `Overlaps(x,y)`: usa a distância $\lVert x - y \rVert$

**Axiomas da base de conhecimento:**

0.  **Pertinência:** $\forall x\,Forma(x)$ e $\forall x\,\neg OutraForma(x)$ para cada uma das 5 formas (25 axiomas)
1.  $\forall x,y\,(Circle(x) \wedge Square(y) \Rightarrow LeftOf(x,y))$
2.  $\forall x,y\,(Triangle(x) \wedge Circle(y) \Rightarrow Above(x,y))$
3.  $\forall x,y\,(Overlaps(x,y) \Rightarrow Overlaps(y,x))$
    * *Significado:* a sobreposição é simétrica.
4.  $\forall x\,\neg Overlaps(x,x)$
    * *Significado:* nenhuma figura se sobrepõe a si mesma.
5.  $\forall x,y\,(Square(x) \wedge Pentagon(y) \Rightarrow LeftOf(x,y))$
6.  $\forall x,y\,(Circle(x) \wedge Hexagon(y) \Rightarrow Above(x,y))$
7.  $\forall x,y\,(Pentagon(x) \wedge Hexagon(y) \Rightarrow \neg Overlaps(x,y))$
8.  $\forall x\,(Circle(x) \Rightarrow \exists y\,(Pentagon(y) \wedge LeftOf(x,y)))$
    * *Significado:* todo círculo tem algum pentágono à sua direita.

**Consultas em dados novos** (3 figuras de cada forma, não vistas no treino):

| Consulta | Pergunta | Esperado |
| :--- | :--- | :--- |
| Q1 | $\forall x,y\,(Circle(x) \wedge Square(y) \Rightarrow LeftOf(x,y))$ | ≈ 1 |
| Q2 | $Overlaps(c_1, s_1)$ | ≈ 0 |
| Q3 | $Circle(c_1)$ | ≈ 1 |
| Q4 | $Pentagon(p_1)$ | ≈ 1 |
| Q5 | $Hexagon(p_1)$ | ≈ 0 |
| Q6 | $Hexagon(h_1)$ | ≈ 1 |
| Q7 | $LeftOf(s_1, p_1)$ | ≈ 1 |
| Q8 | $Above(c_1, h_1)$ | ≈ 1 |

---

## 3. Resultados Consolidados

Médias e desvios padrão das 5 execuções.

### 3.1. Base de Conhecimento Básica

| Métrica | Média | Desvio Padrão ($\sigma$) | Análise |
| :--- | :--- | :--- | :--- |
| **Satisfação inicial** | 0.5174 | 0.0144 | Predicados sem treino respondem perto de 0.5. |
| **Satisfação final** | **0.9309** | 0.0294 | **Alta.** 100 épocas bastam para satisfazer fatos e regras. |
| A bola vermelha é vermelha? | **0.9381** | 0.0145 | Fato aprendido. |
| O cubo azul é redondo? | 0.0790 | 0.0219 | Negação aprendida corretamente. |
| Todas as bolas são redondas? | **0.9425** | 0.0104 | Quantificador universal satisfeito. |

### 3.2. Formas de Igualdade

| Igualdade | Satisfação da KB | Desvio Padrão ($\sigma$) | Análise |
| :--- | :--- | :--- | :--- |
| **Diagonal** | 0.7764 | 0.0000 | Sem parâmetros. Vermelho e azul são ortogonais, então $\cos = 0$ e $Eq = 0.5$, o que impede satisfazer $\neg Eq(vermelho, azul)$. |
| **Cosseno** | 0.5793 | 0.0000 | **Pior caso.** A sigmoide sobre $[0,1]$ nunca chega perto de 0, e a temperatura sozinha não separa vermelho de azul (0.69). |
| **Euclidiana** | **0.9917** | 0.0000 | **Melhor caso.** $\gamma$ cresce até ≈ 2.4: vermelhos iguais (0.9879) e vermelho ≠ azul (0.0076). |
| **Aprendível** | 0.7113 | 0.0000 | Separa os pares avaliados (1.0 e 0.0), mas não garante reflexividade para todos os objetos. |

O desvio padrão zero é esperado: os objetos são fixos, as inicializações de $T$ e $\gamma$ são constantes, e a rede aprendível satura nas mesmas respostas em todas as rodadas.

### 3.3. Relações Geométricas

| Métrica | Média | Desvio Padrão ($\sigma$) | Análise |
| :--- | :--- | :--- | :--- |
| **Satisfação final da KB** | **0.8729** | 0.0316 | **Alta**, considerando 37 axiomas simultâneos. |
| **Q1** (∀ círculos à esquerda de quadrados) | **0.9995** | 0.0005 | Regra universal generaliza para dados novos. |
| **Q2** (sobreposição círculo/quadrado) | 0.0003 | 0.0007 | Figuras distantes não se sobrepõem. |
| **Q3** (círculo é círculo) | 0.7080 | **0.3311** | **Instável.** A faixa dos círculos se sobrepõe à dos triângulos (y entre 6 e 10). Um círculo sorteado ali pode ser confundido com triângulo. |
| **Q4** (pentágono é pentágono) | **0.9622** | 0.0369 | Região exclusiva: classificação confiável. |
| **Q5** (pentágono é hexágono) | 0.0069 | 0.0130 | Exclusão mútua aprendida. |
| **Q6** (hexágono é hexágono) | **0.9992** | 0.0016 | Região exclusiva: classificação confiável. |
| **Q7** (quadrado à esquerda do pentágono) | **0.9975** | 0.0034 | Axioma 5 generaliza. |
| **Q8** (círculo acima do hexágono) | **1.0000** | 0.0000 | Axioma 6 generaliza. |

### 3.4. Explicação do Raciocínio
O experimento mostra que o LTN é capaz de:
1.  Realizar **Grounding Simbólico**: os predicados de forma reconhecem pentágonos e hexágonos com valor de verdade acima de 0.96 em dados novos.
2.  Incorporar **Conhecimento Abstrato**: as relações espaciais (Q1, Q7 e Q8 acima de 0.99) vêm dos axiomas, e não de rótulos de pares.
3.  Usar **Quantificadores Existenciais** (axioma 8) junto com universais na mesma base de conhecimento.
4.  Expor **limites dos dados**: a instabilidade de Q3 mostra que, quando duas classes ocupam a mesma região do plano, a posição $(x, y)$ sozinha não basta para separá-las.

Também foi necessário incluir os **axiomas de pertinência**. Sem eles, regras como $Circle(x) \wedge Square(y) \Rightarrow LeftOf(x,y)$ ficam satisfeitas de forma trivial com $Circle(x) \approx 0$ para todo ponto, e o modelo nunca aprende o que é cada forma.

---

## 4. Como Executar

1. Criar o ambiente e instalar as dependências: `pip install LTNtorch torch numpy matplotlib jupyter`
2. Abrir o notebook `Trabalho_FIA_LTN.ipynb` e executar todas as células, em ordem.
3. A última célula mostra o gráfico das figuras de treino e das regiões aprendidas por cada predicado de forma.

O treinamento roda em CPU em poucos segundos. Como os dados das figuras são aleatórios, os valores variam um pouco a cada execução.
