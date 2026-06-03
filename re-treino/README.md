# Re-treino

Coloque os arquivos usados para continuar o treinamento em `re-treino/entrada`:

- `encoder.pth`: pesos do modelo antigo, copiado de `treino/checkpoints/encoder.pth`.
- `vocab.json`: vocabulario usado para gerar o `train.bin`.
- `train.bin`: dados ja tokenizados com o mesmo vocabulario.

Os parametros principais ficam no `config.yaml` da raiz. A arquitetura configurada ali precisa bater com o `.pth` carregado.

Execute:

```bash
python3 re-treino/main.py --config config.yaml
```

Ou, usando o ambiente virtual do projeto:

```bash
ml_env/bin/python re-treino/main.py --config config.yaml
```

O script salva os novos arquivos em `re-treino/checkpoints`.

## Como a retomada funciona

O `encoder.pth` atual guarda apenas os pesos do modelo. Isso permite continuar treinando, porque o script recria a arquitetura a partir da `config.json` e depois carrega esses pesos antes de iniciar novas epochs.

O detalhe e que o otimizador `AdamW` nao foi salvo no checkpoint antigo. Por isso, nesta primeira retomada, o otimizador comeca do zero. Os pesos continuam treinados, mas as estatisticas internas do AdamW, como medias dos gradientes, nao existem mais.

Depois do re-treino, o script tambem salva `checkpoint_completo.pt`, que inclui:

- pesos do modelo;
- estado do otimizador;
- ultima epoch;
- config usada.

Se voce quiser retomar de novo de forma mais fiel, copie/renomeie `checkpoint_completo.pt` para `re-treino/entrada/encoder.pth` antes da proxima execucao. O script entende tanto o `.pth` antigo simples quanto esse checkpoint completo novo.
