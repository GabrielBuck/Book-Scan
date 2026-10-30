# Fotografias reais

Esta pasta é reservada para **fotografias reais** de livros (JPG, JPEG ou PNG). O app lista no seletor de exemplos tudo o que for colocado aqui, e `python examples/validate_real.py` gera uma tabela-resumo para essas fotos.

**Nenhuma fotografia faz parte do repositório, e nenhuma é necessária para executar o projeto.** A aplicação, os testes e os exemplos sintéticos (`examples/sintetico_*.jpg`, gerados por `examples/generate_examples.py`) funcionam sem elas. O grupo poderá adicionar aqui fotografias próprias.

## Fotografias externas usadas na validação

Durante o desenvolvimento, o pipeline foi experimentado em 8 fotografias externas (não tiradas pelo grupo), com cenas variadas. As fotografias externas utilizadas durante a validação foram mantidas apenas localmente e não fazem parte deste repositório público.

| Caso (nome local usado na validação) | Descrição da cena | Resultado observado |
|---|---|---|
| `foto_01_capa_vinho_papelao` | Capa de tecido vinho vista de cima, sobre superfície de papelão | Capa detectada automaticamente (Canny); lombada fora do recorte |
| `foto_02_capa_branca_mesa_madeira` | Capa branca vista de cima, sobre mesa redonda de madeira | Capa detectada automaticamente (Canny) |
| `foto_03_capa_cinza_toalha` | Capa cinza com perspectiva, sobre toalha cinza, luz lateral | Critérios de confiança não atendidos; ajuste manual |
| `foto_04_livro_em_pe_capa_clara` | Livro em pé, capa clara em perspectiva lateral, fundo desfocado | Detectada via Otsu, região aproximada (leve inclinação em um lado) |
| `foto_05_pagina_interna_sem_capa` | Página interna de um livro aberto | Não há capa frontal; ajuste manual |
| `foto_06_capa_vermelha_perspectiva` | Capa vermelha com título grande, perspectiva forte | Bordas da capa fragmentadas; ajuste manual |
| `foto_07_lombada_sem_capa_frontal` | Lombada de um livro antigo numa estante | Não há capa frontal; ajuste manual |
| `foto_08_livro_preto_laterais` | Livro preto sobre superfície preta, com as páginas laterais visíveis | Baixo contraste; ajuste manual |

"Detectada automaticamente" significa que um candidato passou nos critérios de validação e a região escolhida foi conferida visualmente. Nos demais casos, o sistema encaminhou o usuário ao ajuste manual dos quatro cantos, que é o comportamento pretendido quando não há confiança suficiente. Esta amostra é pequena e não controlada: não é um benchmark.

## Como validar com fotos próprias

```bash
python examples/validate_real.py
```

Grava em `validation_output/` (ignorada pelo Git) a tabela `resultados.md` e `.csv`, as capas geradas e as fotos com a detecção desenhada. "Detecção automática = sim" indica que um candidato passou na validação; confira visualmente em `validation_output/deteccao/`.

Antes de publicar fotos próprias, confirme que não há pessoas, endereços ou outros dados pessoais, e que você tem o direito de distribuí-las.
