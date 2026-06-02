# Wiki Scraper

Script simples para baixar artigos da Wikipedia, limpar o HTML e salvar o texto em arquivos `.txt`.

Por padrão ele usa a Wikipedia em português e salva os arquivos na pasta `dataset`.

## Uso básico

Baixar um artigo:

```powershell
python wiki_scraper.py Amendoeira
```

Baixar vários artigos:

```powershell
python wiki_scraper.py Amendoeira Laranja Pinheiro
```

Usar outra língua:

```powershell
python wiki_scraper.py --lang en Photosynthesis
```

Escolher a pasta de saída:

```powershell
python wiki_scraper.py --output-dir artigos Amendoeira
```

## Usando uma lista

Crie um arquivo `.txt` com um artigo por linha:

```text
Amendoeira
Laranja
Maçã
Pinheiro
```

Depois rode:

```powershell
python wiki_scraper.py --file lista.txt
```

Linhas vazias são ignoradas. Linhas começando com `#` também são ignoradas.

## Opções úteis

`--file lista.txt`: lê os nomes dos artigos de um arquivo.

`--output-dir dataset`: define onde os `.txt` serão salvos.

`--lang pt`: define o idioma da Wikipedia.

`--filename`: usa o nome passado no comando como nome do arquivo, em vez do título retornado pela Wikipedia.

## Exemplo comum

```powershell
python wiki_scraper.py --file lista.txt --output-dir dataset --filename
```

Isso lê os artigos de `lista.txt`, baixa cada página, limpa o conteúdo e salva tudo em `dataset`.
