# Desafio MBA Engenharia de Software com IA - Full Cycle

## 🎯 Visão Geral

Este projeto implementa um sistema de **Ingestão e Busca Inteligente de Documentos** usando conceitos de RAG (Retrieval-Augmented Generation). O objetivo é permitir a ingestão de um PDF (ou outros documentos), indexação semântica e respostas automáticas a perguntas usando mecanismos de busca vetorial e modelos de linguagem natural.



**Exemplo prático:**
```
💬 Pergunta: "Quantas empresas existem no documento?"
🤖 Resposta: "Existem 84 empresas no documento."

💬 Pergunta: "Qual empresa tem o maior valor?"
🤖 Resposta: "Aliança Esportes ME R$ 4.485.320.049,16"

💬 Pergunta: "Qual é a capital do Brasil?" (fora do contexto)
🤖 Resposta: "Não tenho informações necessárias para responder sua pergunta."
```
---


## 🏗️ Arquitetura

- **� Embeddings**: OpenAi (`text-embedding-3-small`)
- **💬 Chat**: Open Ai (`gpt-5-mini"`)  
- **🗄️ Banco Vetorial**: PostgreSQL + pgVector
- **⚡ Framework**: LangChain + Python
- **� Infraestrutura**: Docker Compose


### Passo a passo rápido

1. **Clone e configure o ambiente**
```bash
git clone <seu-repositorio>
cd mba-ia-desafio-ingestao-busca
   ```
Criar ambiente virtual
```bash
   python3 -m venv venv
   ```

Ativar ambiente
```bash
   source venv/bin/activate  # No Windows: venv\Scripts\activate
   ```

Instalar dependências
```bash
  pip install -r requirements.txt
   ```

2. **Configure a API da OpenAi**
- Acesse o site  [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)
- Após o login, clique em "API Keys" no menu lateral esquerdo.
   - Clique no botão "Create new secret key".
   - Dê um nome para a chave que a identifique facilmente.
   - Clique em "Create secret key".
- Configure o arquivo `.env`:
```env
OPENAI_API_KEY=<sua-chave-aqui>
OPENAI_EMBEDDING_MODEL='text-embedding-3-small'
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=pdf-document
PDF_PATH=../document.pdf
```
- Para mais detalhes, consulte o tutorial completo: [Como Gerar uma API Key na OpenAI?](https://hub.asimov.academy/tutorial/como-gerar-uma-api-key-na-openai/)


3. **Suba o banco de dados**
```bash
docker compose up -d
```

4. **Coloque seu PDF**
- Adicione seu arquivo PDF como `document.pdf` na raiz do projeto

5. **Processe o PDF (uma vez apenas)**
```bash
# Windows PowerShell
$env:PDF_PATH="document.pdf"; $env:DATABASE_URL="postgresql://postgres:postgres@localhost:5432/rag"; $env:PG_VECTOR_COLLECTION_NAME="pdf_documents"; venv\Scripts\python.exe src\ingest.py
```

6. **Execute o chat**
```bash
# Windows PowerShell  
$env:PDF_PATH="document.pdf"; $env:DATABASE_URL="postgresql://postgres:postgres@localhost:5432/rag"; $env:PG_VECTOR_COLLECTION_NAME="pdf_documents"; venv\Scripts\python.exe src\chat.py
```



## 📁 Estrutura do Projeto

```
mba-ia-desafio-ingestao-busca/
├── src/
│   ├── chat.py              # Interface de chat interativa (Q&A)
│   ├── ingest.py            # Ingestão e indexação de documentos
│   ├── responder.py         # Gerador de respostas com LLM
│   ├── retriever.py         # Recuperação baseada em embeddings
│   └── search.py            # Funcionalidade de busca estrutural
├── .env.example             # Exemplo de variáveis ambiente
├── .gitignore               # Ignora arquivos temporários/sensíveis
├── docker-compose.yml       # Configuração Docker (serviços e banco vetorial)
├── document.pdf             # Documento de referência/indexação
├── requirements.txt         # Dependências Python
└── README.md                # Este arquivo
```



## 🛑 Como parar a aplicação

```bash
# 1. Sair do chat (Ctrl+C ou digitar 'sair, exit ou close')
# 2. Parar o banco de dados
docker compose down
# 3. Desativar ambiente virtual  
deactivate
```

## 📝 Licença

Este projeto foi desenvolvido para o MBA de Engenharia de Software com IA - Full Cycle.

---

**🚀 Desenvolvido por [Luís Henrique Machado](https://github.com/lhfam97)**
