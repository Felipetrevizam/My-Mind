# Clonar sites com `wget`

## 1. Instale o Homebrew no seu computador

Siga as instruções do [site oficial do Homebrew](https://brew.sh/) para instalar o Homebrew.

## 2. Instale o `wget`

Execute o seguinte comando no terminal:

```bash
brew install wget
```

## 3. Acesse, pelo terminal, a pasta onde você deseja salvar o site

Use o comando cd para navegar até a pasta desejada:

```bash
cd caminho/para/a/pasta
```

## 4. Utilize o seguinte comando para clonar o site

Execute o comando abaixo, substituindo http://www.site.com pelo endereço do site que você deseja clonar:

```bash
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent http://www.site.com
```

## 5. Visualize o site no VSCode

Instale a extensão HTML Preview no VSCode.
No terminal do VSCode, use o atalho Cmd (ou Ctrl) + Shift + P, digite "HTML Preview" e selecione a extensão para visualizar o site clonado.
