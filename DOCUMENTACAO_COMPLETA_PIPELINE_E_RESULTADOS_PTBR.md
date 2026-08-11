# Documentação completa do pipeline, dos resultados, das figuras e das tabelas

## OpenStreetMap mapping maturity em Petrópolis, Brasil

Este documento apresenta, em português do Brasil, uma explicação detalhada e integrada do pipeline computacional desenvolvido no repositório `nathandamas/sotm-br-2026`. O objetivo é registrar de forma técnica, reprodutível e interpretável o papel de cada notebook, as decisões metodológicas adotadas, os resultados parciais produzidos em cada etapa e a relação entre esses resultados e os produtos finais utilizados na análise científica.

A estrutura analítica foi organizada em treze notebooks sequenciais. O fluxo parte da validação dos dados espaciais do IBGE, passa pela agregação da grade de 200 m para 1 km, preparação das geometrias para consulta à ohsome API, obtenção e consolidação do histórico do OpenStreetMap, construção das séries temporais, derivação de indicadores intrínsecos de maturidade e dinâmica de contribuição, análise multivariada, construção e interpretação de uma tipologia espacial e, finalmente, geração de mapas e tabelas prontos para uso em artigo científico.

Os resultados executados dos notebooks encontram-se materializados em `data/output/notebooks/`, separados por etapa. As figuras finais estão em `data/output/figures/` e as tabelas em `data/output/tables/`.

---

# 1. Visão geral do problema científico

O pipeline foi desenvolvido para investigar como a história de contribuição do OpenStreetMap pode ser utilizada para caracterizar diferentes estágios e regimes de maturidade do mapeamento em escala intraurbana. A unidade espacial de análise é uma grade regular de 1 km × 1 km derivada da grade estatística do IBGE e recortada espacialmente para o município de Petrópolis, Rio de Janeiro.

O ponto central da abordagem é que a situação atual do mapa, observada em um único instante, não representa toda a informação disponível no histórico do OpenStreetMap. Duas células podem possuir atualmente comprimentos de vias semelhantes, mas trajetórias temporais muito diferentes: uma pode ter atingido seu estado atual rapidamente, em um curto pulso de expansão; outra pode ter acumulado geometrias lentamente ao longo de anos; uma terceira pode ter atingido um plateau e continuar recebendo manutenção recorrente. Essas diferenças constituem sinais intrínsecos da dinâmica de produção do dado voluntário.

Por essa razão, o pipeline evita tratar “maturidade”, “saturação” ou “plateau” como sinônimos de completude externa. As métricas construídas descrevem propriedades da própria trajetória temporal do OSM. Elas não demonstram que o mapa represente toda a infraestrutura real existente, nem constituem medidas diretas de acurácia posicional, acurácia temática ou completude em relação a uma base de referência externa.

---

# 2. Estrutura geral do pipeline

O fluxo metodológico pode ser resumido em seis grandes blocos:

1. **Preparação espacial:** Notebooks 01–03.
2. **Consulta e validação da ohsome API:** Notebooks 04–05.
3. **Construção de séries temporais e indicadores:** Notebooks 06–07.
4. **Análise multivariada e construção da tipologia:** Notebooks 08–09.
5. **Interpretação substantiva e consolidação da tipologia:** Notebooks 10–11.
6. **Produção de figuras e tabelas para o artigo:** Notebooks 12–13.

A sequência é deliberadamente cumulativa. Cada notebook consome produtos validados das etapas anteriores. Assim, os notebooks não devem ser interpretados como scripts independentes, mas como módulos de um pipeline científico único e rastreável.

---

# 3. Notebook 01 — validação dos dados de entrada do IBGE

## 3.1 Objetivo

O Notebook 01, `01_validate_ibge_input_data.ipynb`, é a etapa de diagnóstico dos arquivos espaciais utilizados na construção da grade analítica. Sua função é verificar se os dados de entrada possuem estrutura, sistema de referência, geometrias e atributos compatíveis com a agregação posterior da grade original de 200 m para células de 1 km.

A decisão metodológica mais importante desta etapa foi não realizar uma agregação espacial puramente geométrica se o próprio arquivo do IBGE já contivesse um identificador oficial capaz de associar cada célula de 200 m à respectiva célula de 1 km. A inspeção do esquema mostrou que a estrutura do dado permitia utilizar o identificador oficial `nome_1km` como chave de agregação.

## 3.2 Por que essa validação é importante

Uma agregação geométrica realizada apenas por coordenadas ou arredondamento pode introduzir inconsistências nas bordas, erros de precisão e divergências em relação à estrutura oficial do IBGE. Ao utilizar `nome_1km`, a agregação preserva a chave oficial da célula de 1 km e reduz a dependência de inferências geométricas desnecessárias.

O Notebook 01 também estabelece o princípio utilizado no restante do pipeline: os dados-fonte são tratados como imutáveis. As transformações são realizadas em arquivos derivados e os produtos intermediários e finais são salvos em novos diretórios.

## 3.3 Resultado parcial

A execução validou a possibilidade de realizar a agregação 200 m → 1 km por meio do identificador oficial do IBGE e forneceu o diagnóstico que fundamentou o Notebook 02. O notebook executado e seu relatório HTML estão materializados em `data/output/notebooks/01/`.

---

# 4. Notebook 02 — construção da grade analítica de 1 km

## 4.1 Objetivo

O Notebook 02, `02_build_ibge_1km_grid.ipynb`, transforma a grade de 200 m do IBGE em uma grade analítica de 1 km, preservando o identificador oficial `nome_1km` como referência estrutural e criando uma chave operacional `grid_1km_id` para as etapas seguintes.

## 4.2 Agregação

As células de 200 m pertencentes ao mesmo `nome_1km` são agrupadas. Para cada grupo são realizados controles de integridade, incluindo:

- número de células de 200 m presentes no grupo;
- flag de completude do grupo;
- validação das geometrias;
- controle de área em CRS métrico adequado;
- preservação da geometria original consolidada da célula de 1 km.

Como uma célula de 1 km corresponde teoricamente a uma matriz de 5 × 5 células de 200 m, um grupo completo deve conter 25 subcélulas. A flag de completude permite identificar grupos incompletos, especialmente em regiões onde o dado de origem foi recortado ou apresenta limites espaciais específicos.

## 4.3 Relação espacial com Petrópolis

Após a agregação, cada célula de 1 km é comparada com o limite municipal de Petrópolis. O notebook calcula a relação espacial e a fração de interseção com o município, permitindo distinguir células totalmente internas de células parcialmente interceptadas pelo limite municipal.

A seleção adotada para o estudo mantém todas as células de 1 km que intersectam Petrópolis, preservando a geometria integral da célula, em vez de recortar cada quadrado pelo limite municipal. Essa decisão é importante para manter unidades espaciais regulares e comparáveis.

## 4.4 Resultado parcial

O resultado consolidado é uma grade de **895 células de 1 km** que intersectam Petrópolis. O principal produto é:

`data/processed/Petropolis/petropolis_grid_1km.parquet`

Uma cópia do resultado executado encontra-se também em `data/output/notebooks/02/`.

Esse conjunto de 895 células torna-se a referência espacial para todas as análises posteriores.

---

# 5. Notebook 03 — preparação dos boundaries para a ohsome API

## 5.1 Objetivo

O Notebook 03, `03_prepare_ohsome_boundaries.ipynb`, converte a grade de 1 km em uma estrutura adequada aos endpoints `groupBy/boundary` da ohsome API.

## 5.2 Conversão de CRS

O GeoParquet analítico permanece em CRS projetado apropriado às operações métricas. Entretanto, as geometrias submetidas à API são transformadas para **EPSG:4326**, pois a interface da ohsome opera com coordenadas geográficas longitude/latitude.

A geometria utilizada na consulta é uma cópia reprojetada; a geometria original das células não é substituída.

## 5.3 Identificação das geometrias

Cada boundary é associado ao `grid_1km_id`. Esse identificador é crítico para que os resultados da API possam ser associados novamente à célula correta sem depender da ordem de retorno das respostas.

Também são realizados controles de validade geométrica e de tamanho das geometrias enviadas à API.

## 5.4 Conjunto de teste

Antes da consulta em escala completa, o notebook seleciona um subconjunto de **10 células**. Esse conjunto foi utilizado para validar os endpoints, a codificação dos boundaries e o parser das respostas no Notebook 04.

## 5.5 Resultado parcial

Os principais produtos incluem GeoJSONs de boundaries completos e de teste, preservados entre os resultados materializados em `data/output/notebooks/03/`.

---

# 6. Notebook 04 — teste real da ohsome API em 10 células

## 6.1 Objetivo

O Notebook 04, `04_query_ohsome_test_cells.ipynb`, foi o primeiro teste real da estrutura atual de respostas da ohsome API. Em vez de escalar diretamente para todas as 895 células, o procedimento foi validado em apenas 10 unidades espaciais.

## 6.2 Consulta ao endpoint `/metadata`

Antes das consultas analíticas, o notebook consulta `/metadata` para registrar a versão disponível da API e a extensão temporal da OSHDB. Na execução de produção, a versão observada foi **ohsome API 1.10.4**, com extensão temporal aproximada de **2007-10-08 a 2026-07-27**.

Essa consulta é importante porque a disponibilidade histórica da OSHDB evolui com o tempo. Registrar a extensão temporal efetivamente utilizada torna a análise mais reprodutível.

## 6.3 Quatro métricas iniciais

O teste utiliza o filtro:

`highway=* and geometry:line`

para representar geometrias lineares de vias mapeadas no OpenStreetMap.

Quatro endpoints são avaliados inicialmente:

- comprimento de elementos viários;
- número de elementos;
- número de usuários/contribuidores;
- número de contribuições.

Todos são agrupados por `grid_1km_id` por meio de `groupBy/boundary`.

## 6.4 Validação do parser

O objetivo central não era ainda produzir resultados científicos definitivos, mas verificar:

- formato real do JSON retornado;
- nomes de campos;
- estrutura dos grupos;
- tratamento dos IDs das células;
- tipos numéricos;
- timestamps;
- códigos HTTP e eventuais mensagens de erro.

O teste confirmou a estrutura necessária para escalar as consultas. As respostas brutas foram preservadas, permitindo auditoria posterior.

## 6.5 Resultado parcial

A etapa demonstrou que a estratégia de consulta por boundaries era operacional para as 10 células e forneceu a base segura para o Notebook 05.

---

# 7. Notebook 05 — consulta completa das 895 células

## 7.1 Objetivo

O Notebook 05, `05_query_ohsome_all_895_cells.ipynb`, escala a consulta validada do Notebook 04 para a totalidade das 895 células.

## 7.2 Estratégia de processamento

Para evitar requisições excessivamente grandes, as células foram processadas em **36 lotes de 25 células**, sendo o último lote composto pelo número residual de células.

Foram incorporados mecanismos de:

- batching;
- persistência das respostas brutas;
- retry/backoff;
- checkpoints;
- retomada sem repetir lotes já concluídos;
- validação por endpoint.

Essa arquitetura reduz o risco de perder toda a execução por falhas temporárias de rede ou do serviço externo.

## 7.3 Cobertura

Os quatro indicadores foram obtidos para **895/895 células**, sem IDs ausentes ou duplicados.

Os controles finais ficaram em `PASS` para:

- número de linhas;
- unicidade de IDs;
- correspondência com a grade;
- completude das métricas;
- ausência de valores negativos impossíveis.

## 7.4 Resultados descritivos iniciais

Os resultados agregados das 895 células mostraram:

- comprimento médio de vias por célula: aproximadamente **2.934,20 m**;
- mediana: aproximadamente **2.056,78 m**;
- máximo: aproximadamente **17.137,68 m**;
- **114 células** com comprimento zero;
- contagem média de elementos viários: aproximadamente **12,88**;
- média de contribuidores por célula: aproximadamente **15,36**;
- média de contribuições: aproximadamente **90,93**.

O comprimento total somado pelas células ficou em torno de **2.626 km de vias mapeadas**.

É importante observar que somar “contribuidores por célula” não produz o número de usuários únicos do município, porque o mesmo usuário pode contribuir em várias células.

## 7.5 Produtos principais

Entre os produtos centrais encontram-se:

- `petropolis_ohsome_895_cells.parquet`;
- `petropolis_grid_1km_with_ohsome_metrics.parquet`.

O segundo associa diretamente as métricas às geometrias da grade.

---

# 8. Notebook 06 — construção das séries temporais mensais

## 8.1 Objetivo

O Notebook 06, `06_build_monthly_osm_timeseries.ipynb`, deixa de trabalhar apenas com agregados atuais e reconstrói a trajetória mensal do mapeamento em cada célula.

## 8.2 Intervalo temporal

Foram utilizados apenas meses-calendário completos dentro da extensão disponível da OSHDB. A série resultante cobre **224 meses completos**, de **novembro de 2007 a junho de 2026**.

Com 895 células × 224 meses, o painel contém exatamente **200.480 registros célula-mês**.

## 8.3 Métricas mensais

O notebook consolida oito métricas primárias:

- `road_length_m`;
- `road_element_count`;
- `contributors_count`;
- `contributions_count`;
- `creation_contributions_count`;
- `deletion_contributions_count`;
- `tag_change_contributions_count`;
- `geometry_change_contributions_count`.

As quatro categorias de tipo de contribuição não são tratadas como mutuamente exclusivas. Uma edição pode envolver simultaneamente mudança de tags e mudança geométrica.

## 8.4 Variáveis derivadas

Também são calculadas variáveis auxiliares, como:

- delta mensal do comprimento de vias;
- delta de elementos;
- densidade viária;
- mês ativo/inativo;
- contribuições por contribuidor;
- contribuições acumuladas;
- proporções dos tipos de contribuição.

## 8.5 Resultado parcial

A execução apresentou cobertura completa das 895 células e das oito métricas primárias, sem valores negativos ou ausências inesperadas. Os controles de qualidade ficaram em `PASS`.

O painel temporal em formato long torna-se a base para a construção dos indicadores de maturidade do Notebook 07.

---

# 9. Notebook 07 — construção dos indicadores de crescimento, maturidade, persistência e contribuição

## 9.1 Objetivo

O Notebook 07, `07_build_osm_maturity_indicators.ipynb`, transforma as séries mensais em indicadores sintéticos interpretáveis por célula.

O objetivo não é criar um único “score” opaco de qualidade, mas preservar componentes independentes que descrevam diferentes dimensões da trajetória do OSM.

## 9.2 Indicadores de crescimento

Entre os indicadores derivados encontram-se:

- primeiro mês com vias mapeadas;
- idade do mapeamento;
- comprimento atual de vias;
- mês do maior crescimento;
- magnitude do maior crescimento;
- número de meses com crescimento positivo;
- número de meses com redução;
- taxa inicial de crescimento;
- taxa recente de crescimento;
- desaceleração do crescimento.

Durante o desenvolvimento, a primeira formulação baseada em mediana de deltas mensais se mostrou inadequada para séries muito esparsas, produzindo variáveis degeneradas. A metodologia foi corrigida para utilizar uma inclinação líquida robusta em janela temporal, evitando que muitos meses sem alteração forçassem artificialmente a taxa para zero.

## 9.3 Maturidade e saturação intrínseca

Foram calculados indicadores como:

- `time_to_50pct`;
- `time_to_80pct`;
- `time_to_90pct`;
- `time_to_95pct`;
- meses desde atingir 90%;
- início do plateau;
- duração do plateau;
- razão de saturação.

A referência de saturação é **intrínseca**, baseada na própria trajetória da célula e em uma referência recente robusta. Portanto, atingir 90% significa aproximar-se de seu próprio nível recente, não atingir 90% da infraestrutura real existente.

## 9.4 Persistência

São avaliados:

- número de meses ativos;
- proporção de meses ativos;
- maior sequência ativa;
- maior sequência inativa;
- meses desde a última atividade;
- atividade pós-plateau;
- atividade recente;
- índice de persistência.

## 9.5 Dinâmica de contribuição

Incluem-se:

- total, média, mediana e pico de contribuições;
- contribuidores;
- contribuições por contribuidor;
- composição por criação, deleção, mudança de tags e geometria;
- concentração temporal das contribuições;
- atividade recente versus histórica;
- proxy de volatilidade de contribuidores;
- balanço entre expansão e manutenção.

O proxy final de volatilidade dos contribuidores utiliza o coeficiente de variação dos meses ativos, evitando que meses sem atividade dominem a medida.

## 9.6 Resultados gerais

Dos 895 quadrados:

- **781** possuem vias atualmente mapeadas;
- **795** apresentam alguma contribuição no histórico considerado;
- **749** tiveram plateau intrínseco detectado.

Os controles de qualidade ficaram em `PASS` após a correção dos indicadores degenerados.

---

# 10. Notebook 08 — análise exploratória multivariada e espacial

## 10.1 Objetivo

O Notebook 08, `08_explore_osm_indicator_structure.ipynb`, investiga como os indicadores do Notebook 07 se relacionam entre si antes da construção da tipologia.

## 10.2 Redundância

A matriz de correlação de Spearman identificou **74 pares** com `|ρ| ≥ 0,85`. Esse resultado demonstra que vários indicadores capturam dimensões semelhantes e que utilizar todos simultaneamente em um clustering produziria redundância e pesos implícitos indesejados.

## 10.3 PCA

A PCA foi utilizada como ferramenta diagnóstica, não como índice de maturidade e nem como única base para o clustering.

Após correção do pré-processamento, a sequência adotada foi:

1. imputação pela mediana;
2. winsorization entre os percentis 1 e 99;
3. `StandardScaler`;
4. PCA.

A variância explicada foi aproximadamente:

- PC1: **43,64%**;
- PC2: **11,65%**;
- PC3: **9,73%**.

Os três primeiros componentes somam cerca de **65,02%** da variância.

Nenhum loading individual passou a dominar o PC1, o que corrigiu a versão anterior em que uma variável de desaceleração concentrava quase toda a variância.

## 10.4 Autocorrelação espacial

A análise de Moran’s I global mostrou organização espacial significativa dos indicadores. Exemplos:

- contribuições totais: Moran’s I ≈ **0,696**;
- razão de meses ativos: ≈ **0,631**;
- comprimento atual de vias: ≈ **0,628**;
- idade do mapeamento: ≈ **0,539**;
- índice de persistência: ≈ **0,504**.

Os valores apresentaram `p_sim = 0,001` nas análises destacadas.

Isso indica que os padrões de mapeamento não estão distribuídos aleatoriamente no espaço de Petrópolis.

---

# 11. Notebook 09 — construção e validação da tipologia de células

## 11.1 Objetivo

O Notebook 09, `09_build_osm_cell_typology.ipynb`, transforma a estrutura multivariada em uma tipologia de regimes de histórico de mapeamento.

## 11.2 Classe estrutural sem vias

As **114 células sem comprimento viário atualmente mapeado** foram mantidas como uma classe estrutural separada. Elas não foram submetidas ao clustering, porque diversos indicadores de trajetória viária são estruturalmente indefinidos nessas células.

Essa decisão evita imputar artificialmente idade, plateau ou tempo até 90% para células onde a trajetória de vias atual não existe.

## 11.3 Células submetidas ao clustering

O clustering foi realizado nas **781 células com vias presentes**.

Foram selecionados dez indicadores não redundantes e substantivamente interpretáveis, incluindo:

- comprimento atual de vias;
- idade do mapeamento;
- desaceleração do crescimento;
- tempo até 90%;
- duração do plateau;
- razão de meses ativos;
- atividade recente;
- concentração temporal;
- proxy de expansão;
- volatilidade de contribuidores.

## 11.4 Métodos comparados

Foram comparados:

- k-means;
- clustering hierárquico Ward;
- valores de `k` entre 2 e 8.

As soluções foram avaliadas por:

- silhouette;
- Calinski–Harabasz;
- Davies–Bouldin;
- tamanho mínimo de cluster;
- estabilidade por reamostragem;
- coerência espacial posterior.

## 11.5 Solução selecionada

A solução escolhida foi **k-means, k = 2** para as células com vias.

Métricas principais:

- Silhouette: **0,8371**;
- Calinski–Harabasz: **1.592,42**;
- Davies–Bouldin: **0,4334**;
- ARI médio em 50 reamostragens: **0,9169**;
- percentil 5% do ARI: **0,8394**;
- menor cluster: **63 células**, aproximadamente **8,07%** das 781 células com vias;
- concordância de vizinhança Queen: aproximadamente **0,8801**;
- sensibilidade PCA-3 versus solução original: **ARI = 1,000**.

## 11.6 Composição inicial

A classificação operacional ficou composta por:

- classe estrutural sem vias: 114 células;
- cluster provisório 1: 718 células;
- cluster provisório 2: 63 células.

Os nomes substantivos só foram atribuídos posteriormente, após análise estatística e espacial nos Notebooks 10 e 11.

---

# 12. Notebook 10 — interpretação estatística e espacial da tipologia

## 12.1 Objetivo

O Notebook 10, `10_interpret_osm_typology.ipynb`, testa formalmente as diferenças entre as classes e analisa sua estrutura espacial.

## 12.2 Testes estatísticos

São utilizados:

- Kruskal–Wallis para diferenças globais;
- correção Benjamini–Hochberg/FDR;
- Mann–Whitney U em comparações entre os dois regimes com vias;
- Cliff’s delta como tamanho de efeito;
- Fisher exact test para variáveis binárias selecionadas.

A combinação de significância estatística e tamanho de efeito evita interpretar diferenças pequenas apenas porque o número de células é elevado.

## 12.3 Resultados substantivos dos dois regimes com vias

O cluster de maior intensidade apresentou, em relação ao cluster de menor intensidade:

- maior comprimento mediano de vias: **4,75 km versus 2,27 km**;
- mais contribuições: **96 versus 28**;
- maior razão de meses ativos: **0,06 versus 0,04**;
- mais contributor-months: **19 versus 11**;
- maior volatilidade de contribuidores: **0,60 versus 0,33**;
- maior pico mensal de expansão: aproximadamente **2.137 m versus 1.401 m**;
- mais meses com crescimento positivo: **7 versus 4**;
- taxa inicial mediana de crescimento: aproximadamente **95,21 m/mês versus 0**;
- forte desaceleração posterior: aproximadamente **93,82 m/mês versus 0**.

A duração do plateau é maior no regime de menor intensidade: **70 meses versus 59 meses**.

## 12.4 Coerência espacial

A tipologia apresenta organização espacial significativa.

Foram observadas **2.959 arestas de vizinhança Queen**. Dessas, **74,28%** conectam células da mesma classe. Sob 999 permutações aleatórias dos rótulos, a expectativa média é **66,41%**, com desvio-padrão de aproximadamente 0,71 ponto percentual.

O valor-p empírico é **0,001**.

Essa análise é posterior ao clustering: a contiguidade espacial não foi utilizada para forçar a solução.

---

# 13. Notebook 11 — consolidação final dos nomes e interpretação

## 13.1 Objetivo

O Notebook 11, `11_finalize_osm_typology.ipynb`, transforma os rótulos provisórios em descrições substantivas conservadoras, preservando simultaneamente as classes originais para rastreabilidade.

## 13.2 Tipologia final

A composição final é:

### Classe 0 — `No currently mapped OSM road lines`

- 114 células;
- **12,74%** da grade.

Significa que a célula não possui atualmente geometria linear `highway=*` mapeada no snapshot analisado.

Não significa ausência física de vias e não deve ser interpretada como classe de incompletude.

### Classe 1 — `Lower-intensity, longer-plateau mapping regime`

- 718 células;
- **80,22%** da grade.

Caracteriza o regime majoritário, com menor intensidade histórica de contribuição e menor extensão viária mediana, combinado a plateau intrínseco relativamente mais longo.

### Classe 2 — `Higher-intensity, expansion-pulse mapping regime`

- 63 células;
- **7,04%** da grade.

Apresenta maior intensidade de contribuição, maior extensão de vias mapeadas e pulsos mais intensos de expansão, seguidos por desaceleração pronunciada.

## 13.3 Resultado conceitual

A principal conclusão metodológica da tipologia é que a história de mapeamento do OSM em Petrópolis não pode ser representada adequadamente por uma simples escala linear “menos completo → mais completo”. O que emerge é uma combinação de estados estruturais e regimes temporais distintos.

---

# 14. Notebook 12 — produção dos mapas e figuras do artigo

## 14.1 Objetivo

O Notebook 12 gera a cartografia final a partir dos produtos validados dos notebooks anteriores.

Os arquivos finais estão em:

`data/output/figures/`

Cada figura é exportada em PNG de alta resolução e PDF.

## 14.2 Padrão cartográfico

Os mapas foram padronizados com:

- grade de células de 1 km;
- paleta **viridis**;
- base cartográfica **CARTO Positron** quando disponível;
- legenda;
- escala gráfica;
- seta de norte;
- grade de coordenadas geográficas;
- identificação de CRS/SRC e EPSG;
- fonte dos dados;
- atribuição cartográfica.

O CRS cartográfico utilizado nas análises métricas é SIRGAS 2000 / UTM zone 23S, EPSG:31983, enquanto a grade de coordenadas é representada em longitude/latitude.

## 14.3 Figura 01 — tipologia final

`figure_01_final_typology.png/pdf`

É o mapa síntese do estudo. Cada célula é colorida de acordo com uma das três classes finais. Sua função é mostrar a distribuição espacial dos regimes e revelar agrupamentos, transições e áreas estruturalmente sem vias OSM atuais.

Por ser uma variável categórica, utiliza três classes discretas derivadas da paleta viridis.

## 14.4 Figura 02 — comprimento atual de vias

`figure_02_current_road_length.png/pdf`

Representa `current_road_length_km` por célula. Mostra a intensidade espacial da rede viária atualmente mapeada no OSM.

O mapa não deve ser interpretado diretamente como densidade real de infraestrutura, porque depende tanto da rede física quanto do estado do mapeamento.

## 14.5 Figura 03 — idade do mapeamento

`figure_03_mapping_age.png/pdf`

Representa o tempo decorrido desde a presença inicial de vias mapeadas na célula. Valores maiores indicam células cuja história de vias no OSM começou mais cedo dentro da janela observada.

A variável ajuda a diferenciar células antigas de células incorporadas mais recentemente ao histórico do mapa.

## 14.6 Figura 04 — proporção de meses ativos

`figure_04_active_month_ratio.png/pdf`

Mostra a fração de meses em que houve atividade de contribuição na célula.

Valores elevados representam persistência temporal de atividade, enquanto valores baixos indicam trajetórias mais episódicas.

## 14.7 Figura 05 — contribuições totais

`figure_05_total_contributions.png/pdf`

Representa o volume acumulado de contribuições OSM ao longo da série histórica.

A distribuição é fortemente assimétrica; por isso, a normalização cartográfica utiliza escala logarítmica para preservar contraste entre células com níveis baixos, médios e elevados de atividade.

## 14.8 Figura 06 — concentração temporal das contribuições

`figure_06_temporal_concentration.png/pdf`

Representa um índice HHI de concentração temporal das contribuições.

Valores maiores significam que uma parcela maior da atividade total está concentrada em poucos períodos. Valores menores sugerem distribuição mais uniforme ao longo do tempo.

A interpretação deve ser feita em conjunto com o volume total de contribuições: concentração elevada com baixo volume não é equivalente a concentração elevada em uma célula extremamente ativa.

## 14.9 Figura 07 — tempo até 90% da referência intrínseca

`figure_07_time_to_90pct.png/pdf`

Mostra quantos meses a célula levou para atingir 90% de sua referência recente intrínseca de comprimento viário.

A palavra “intrínseca” é fundamental: 90% não significa 90% das vias reais, mas 90% do nível de referência derivado da própria série temporal da célula.

## 14.10 Figura 08 — duração do plateau

`figure_08_plateau_duration.png/pdf`

Representa por quanto tempo a trajetória de comprimento viário permaneceu em um estado de plateau segundo os critérios do pipeline.

Plateau prolongado descreve estabilidade da trajetória mapeada. Não demonstra que nenhum elemento real esteja faltando.

## 14.11 Manifesto das figuras

`data/output/figures/figure_manifest.csv` registra os 16 arquivos finais, correspondentes a oito mapas em dois formatos.

---

# 15. Notebook 13 — produção das tabelas do artigo

## 15.1 Objetivo

O Notebook 13 converte os principais resultados em tabelas editoriais padronizadas.

Os arquivos encontram-se em:

`data/output/tables/`

Cada tabela principal é disponibilizada em:

- CSV;
- Markdown;
- LaTeX.

Essa multiplicidade de formatos permite análise, leitura humana e inserção direta em manuscritos científicos.

---

# 16. Tabela 01 — composição da grade e da tipologia final

`table_01_study_grid_typology_composition.*`

A tabela apresenta a estrutura global da amostra espacial:

| Classe | Células | Participação |
|---|---:|---:|
| Sem vias OSM atualmente mapeadas | 114 | 12,74% |
| Regime de menor intensidade e plateau mais longo | 718 | 80,22% |
| Regime de maior intensidade e pulso de expansão | 63 | 7,04% |
| Total | 895 | 100% |

A principal função dessa tabela é estabelecer imediatamente o tamanho da amostra e demonstrar que a tipologia é fortemente assimétrica: o regime de menor intensidade é majoritário, enquanto o regime de pulso de expansão é minoritário.

---

# 17. Tabela 02 — perfis substantivos das três classes

`table_02_substantive_class_profiles.*`

A tabela utiliza mediana e intervalo interquartil `[Q1–Q3]` para descrever cada classe nas escalas originais dos indicadores.

Entre os contrastes mais importantes:

- comprimento atual de vias: 0,00 km na classe estrutural, 2,27 km no regime de menor intensidade e 4,75 km no regime de maior intensidade;
- proporção de meses ativos: 0,00, 0,04 e 0,06;
- contribuições acumuladas: 0, 28 e 96;
- contributor-months: 0, 11 e 19;
- volatilidade de contribuidores: aproximadamente 0,00, 0,33 e 0,60;
- plateau intrínseco: estruturalmente indefinido para a classe sem vias, 70 meses para o regime de menor intensidade e 59 meses para o regime de maior intensidade.

A tabela fornece o perfil interpretativo básico das classes sem depender de valores padronizados.

---

# 18. Tabela 03 — comparação inferencial dos dois regimes com vias

`table_03_road_present_regime_comparison.*`

Essa é a tabela inferencial principal. A classe estrutural sem vias é excluída de comparações em que os indicadores dependem da existência de uma trajetória de vias.

São apresentados:

- mediana do regime de menor intensidade;
- mediana do regime de maior intensidade;
- Cliff’s delta;
- estatística Mann–Whitney U;
- valor-p bruto;
- valor-p ajustado por Benjamini–Hochberg.

Resultados particularmente fortes incluem:

- crescimento inicial: Cliff’s delta = **−1,00**, `p < 0,001`;
- desaceleração do crescimento: Cliff’s delta = **−1,00**, `p < 0,001`;
- comprimento atual de vias: delta ≈ **−0,512**, `p ajustado < 0,001`;
- meses de crescimento positivo: delta ≈ **−0,423**, `p ajustado < 0,001`;
- pico de crescimento: delta ≈ **−0,405**, `p ajustado < 0,001`;
- contribuições acumuladas: delta ≈ **−0,401**, `p ajustado < 0,001`;
- volatilidade de contribuidores: delta ≈ **−0,399**, `p ajustado < 0,001`.

O sinal negativo ocorre porque o delta foi orientado como “regime de menor intensidade menos regime de maior intensidade”. Portanto, valores negativos indicam valores maiores no regime de pulso de expansão.

Nem todos os indicadores apresentam diferença significativa. Por exemplo, `time_to_90pct` e concentração temporal HHI não mostram evidência forte de diferença entre os regimes após ajuste. Isso é importante porque demonstra que a tipologia não separa simplesmente todas as variáveis em valores baixos e altos.

---

# 19. Tabela 04 — coerência espacial da tipologia

`table_04_spatial_coherence_summary.*`

A tabela resume a organização espacial das classes.

Principais valores:

- 2.959 arestas Queen;
- 74,28% das adjacências observadas ligam células da mesma classe;
- expectativa sob permutação aleatória: 66,41%;
- desvio-padrão da distribuição nula: 0,71 ponto percentual;
- valor-p empírico: 0,001;
- 999 permutações;
- 8 ilhas Queen.

A tabela também informa o número de componentes conectados e a participação do maior componente de cada classe.

Esses resultados mostram que a tipologia possui coerência espacial superior ao esperado por acaso, embora a configuração dos clusters tenha sido definida sem incorporar restrições espaciais.

---

# 20. Tabela 05 — definições, interpretação e limitações

`table_05_class_definitions_and_limitations.*`

Essa tabela tem função conceitual e editorial. Para cada classe, apresenta:

- definição operacional;
- interpretação substantiva;
- principal limitação.

Ela é especialmente importante para impedir extrapolações indevidas.

A classe “No currently mapped OSM road lines” não é classe de incompletude. As duas classes com vias são regimes relativos da história intrínseca do OSM e não categorias de qualidade absoluta.

---

# 21. Relação entre os notebooks e os produtos finais

A lógica geral pode ser representada como:

```text
Dados IBGE + limite municipal
        ↓
Notebook 01 — validação
        ↓
Notebook 02 — grade de 1 km, 895 células
        ↓
Notebook 03 — boundaries EPSG:4326
        ↓
Notebook 04 — teste da ohsome API em 10 células
        ↓
Notebook 05 — consulta completa das 895 células
        ↓
Notebook 06 — painel mensal de 200.480 registros
        ↓
Notebook 07 — indicadores por célula
        ↓
Notebook 08 — redundância, PCA e autocorrelação espacial
        ↓
Notebook 09 — tipologia k-means + classe estrutural
        ↓
Notebook 10 — testes estatísticos e coerência espacial
        ↓
Notebook 11 — nomenclatura substantiva final
        ↓
Notebook 12 — mapas do artigo
        ↓
Notebook 13 — tabelas do artigo
```

---

# 22. Principais resultados consolidados

O pipeline permite resumir o caso de Petrópolis pelos seguintes resultados centrais:

- 895 células de 1 km compõem a grade analítica;
- 781 células possuem vias OSM atualmente mapeadas;
- 114 células não possuem geometria linear `highway=*` atualmente mapeada;
- 224 meses completos compõem a série temporal;
- o painel mensal contém 200.480 registros;
- 795 células apresentam alguma atividade de contribuição;
- 749 apresentam plateau intrínseco detectável;
- a análise multivariada revela forte redundância entre parte dos indicadores, justificando seleção de variáveis;
- a tipologia de células com vias é melhor representada, dentro dos critérios adotados, por dois regimes k-means;
- a solução apresenta silhouette elevada e alta estabilidade por reamostragem;
- as classes exibem coerência espacial significativamente maior que o esperado por permutação aleatória;
- o regime minoritário de maior intensidade é caracterizado por pulsos de expansão mais fortes e maior atividade histórica;
- o regime majoritário de menor intensidade apresenta menor intensidade de contribuição e plateau mais longo;
- as classes são regimes intrínsecos de histórico OSM, não classes de completude ou acurácia externa.

---

# 23. Como interpretar corretamente “maturidade”, “saturação” e “plateau” neste projeto

Esses conceitos são usados de maneira estritamente operacional.

**Maturidade** representa a combinação de características temporais do histórico do OSM, como idade do mapeamento, atividade, persistência, crescimento e estabilização.

**Saturação intrínseca** indica proximidade da trajetória de uma célula em relação à sua própria referência recente. Não representa comparação com uma verdade externa.

**Plateau** representa um intervalo de estabilidade na trajetória de comprimento viário, segundo limiares e janelas definidos no pipeline.

Uma célula em plateau pode continuar recebendo edições de manutenção, tags, geometria ou atributos. Portanto, plateau geométrico não implica inatividade editorial.

---

# 24. Limitações metodológicas

A interpretação dos resultados deve considerar algumas limitações fundamentais.

Primeiro, `highway=* and geometry:line` representa uma definição operacional da rede viária mapeada. Elementos OSM fora desse filtro não são capturados pela métrica de comprimento de vias.

Segundo, a quantidade de vias mapeadas é resultado tanto da infraestrutura real quanto do comportamento dos mapeadores. Não se pode interpretar diretamente um comprimento maior como evidência de uma rede física mais densa sem informação externa.

Terceiro, a classe sem vias atuais não implica ausência de vias no mundo real.

Quarto, as métricas de contribuidor calculadas por célula não devem ser somadas para inferir usuários únicos municipais.

Quinto, mudanças na OSHDB, na ohsome API ou no próprio OSM podem produzir resultados ligeiramente diferentes em execuções futuras. Por isso, o repositório preserva resultados executados e artefatos canônicos.

Sexto, a tipologia é relativa ao espaço e ao período analisados. A aplicação a outra cidade pode resultar em estrutura multivariada e número de regimes diferentes.

---

# 25. Organização dos resultados no repositório

A estrutura final de saída é:

```text
data/output/
├── figures/
│   ├── figure_01_final_typology.png
│   ├── figure_01_final_typology.pdf
│   ├── ...
│   └── figure_manifest.csv
├── tables/
│   ├── table_01_study_grid_typology_composition.csv
│   ├── table_01_study_grid_typology_composition.md
│   ├── table_01_study_grid_typology_composition.tex
│   ├── ...
│   ├── table_quality_checks.csv
│   └── table_manifest.csv
└── notebooks/
    ├── 01/
    ├── 02/
    ├── 03/
    ├── 04/
    ├── 05/
    ├── 06/
    ├── 07/
    ├── 08/
    ├── 09/
    ├── 10/
    ├── 11/
    ├── 12/
    ├── 13/
    └── manifest.csv
```

`data/output/notebooks/manifest.csv` indexa os arquivos efetivamente materializados dos resultados dos notebooks.

---

# 26. Uso desta documentação na elaboração do artigo

Este documento pode ser utilizado como memória metodológica para a redação das seções de Métodos, Resultados e Discussão.

Para **Métodos**, os Notebooks 01–09 descrevem preparação dos dados, consultas à API, indicadores e clustering.

Para **Resultados**, os Notebooks 10–13 concentram a interpretação estatística, espacial e editorial.

Para **Discussão**, é recomendável enfatizar a distinção entre maturidade intrínseca e completude externa, além da existência de diferentes regimes históricos de mapeamento.

As figuras 01–08 fornecem a expressão espacial dos principais resultados, enquanto as tabelas 01–05 organizam composição, perfis, diferenças estatísticas, coerência espacial e limitações conceituais.

---

# 27. Síntese final

O pipeline desenvolvido demonstra uma forma reprodutível de transformar o histórico de contribuições do OpenStreetMap em uma caracterização espaço-temporal de regimes de mapeamento. Em vez de reduzir a análise a uma única medida de cobertura atual, o método incorpora crescimento, atividade, persistência, contribuição, estabilização e estrutura espacial.

No caso de Petrópolis, a análise identifica uma classe estrutural sem vias atualmente mapeadas e dois regimes principais entre as células com vias: um regime majoritário de menor intensidade e plateau mais longo e um regime minoritário de maior intensidade caracterizado por pulsos de expansão mais fortes.

Esses resultados fornecem uma base para discutir maturidade da informação geográfica voluntária, mas não substituem validações externas de completude ou acurácia. A contribuição metodológica está justamente em separar essas dimensões e utilizar a história do OSM como fonte complementar de evidência sobre o processo de construção e manutenção do mapa.

---

## Arquivos de referência no repositório

- Código-fonte dos notebooks: `notebooks/`
- Scripts auxiliares: `scripts/`
- Resultados executados por notebook: `data/output/notebooks/`
- Figuras finais: `data/output/figures/`
- Tabelas finais: `data/output/tables/`
- README geral: `README.md`
- Licença: `LICENSE`

Esta documentação deve ser atualizada caso novos notebooks, indicadores, mapas ou tabelas sejam incorporados ao pipeline.