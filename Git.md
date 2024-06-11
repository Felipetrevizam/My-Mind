# Criando um Repositório

## Criar um Novo Repositório

1. **Crie um repositório com o mesmo nome que sua pasta principal.**

2. **Utilize os seguintes comandos no VS Code:**

   ```bash
   git init
   git add README.md
   git commit -m "Primeiro Commit"
   git branch -M main
   git remote add origin LinkDoRepositorio
   git push -u origin main
   ```

## Realizar Novas Alterações

1. **Utilize os seguintes comandos no VS Code:**

   ```bash
   git add .
   git commit -m "Mensagem"
   git push
   ```

## Clonar um Repositório

1. **Vá até o repositório que deseja clonar, clique em "Code" e copie o link.**

2. **Utilize os seguintes comandos no VS Code:**

   ```bash
   git clone LinkDoRepositorio
   ```

## Remover um Repositório Iniciado com `git init`

1. **Utilize os seguintes comandos no VS Code:**

   ```bash
   rm -rf .git
   ```
