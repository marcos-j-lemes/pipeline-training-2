# Pipeline de treinamento

A configuracao principal do projeto fica em `config.yaml`.

Os arquivos `treino/checkpoints/config.json` e `re-treino/entrada/config.json`, quando existirem, devem ser tratados como snapshots antigos ou artefatos de compatibilidade. Para mudar arquitetura, caminhos, batch size, epochs ou vocabulario, edite `config.yaml`.

## Ordem da pipeline

1. Scraper

```bash
python3 wiki_scraper/main.py --config config.yaml
```

Por padrao, ele le os artigos em `wiki_scraper/list.txt` e salva `.txt` em `wiki_scraper/dataset`.

2. Treinar BPE e tokenizar dados do modelo

```bash
python3 tokenizador/main.py --config config.yaml --stage all
```

O dataset usado para treinar o BPE pode ser diferente do dataset usado para treinar o modelo:

- `tokenizer.bpe_dataset_path`: texto usado para aprender o vocabulario BPE.
- `tokenizer.model_dataset_path`: texto transformado em `train.bin` e `val.bin`.

3. Treinar modelo

```bash
python3 treino/main.py --config config.yaml
```

O treino usa `tokenizer.train_bin_path`, monta o modelo com a secao `model` e salva em `treino/checkpoints`.

4. Inferencia

```bash
python3 inferencia/main.py --config config.yaml
```

Para uma resposta unica:

```bash
python3 inferencia/main.py --config config.yaml --once "Ola"
```

5. Re-treino

Coloque em `re-treino/entrada`:

- `encoder.pth`: pesos ja treinados.
- `train.bin`: dados tokenizados para continuar o treino.
- `vocab.json`: vocabulario usado nesses dados.

Depois execute:

```bash
python3 re-treino/main.py --config config.yaml
```

O re-treino usa a arquitetura e hiperparametros do `config.yaml`, carrega os pesos de `re-treino/entrada/encoder.pth` e salva novos checkpoints em `re-treino/checkpoints`.

## Observacao sobre retomada

Se o arquivo `.pth` tiver apenas pesos do modelo, o treinamento continua a partir desses pesos, mas o otimizador comeca do zero.

Os scripts novos salvam tambem `checkpoint_completo.pt`, que inclui pesos, otimizador e epoch. Esse formato e melhor para retomar o treino de forma mais fiel.
