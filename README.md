# Guia de Replicação do Projeto

Este arquivo `.md` descreve **todo o passo a passo** para que qualquer pessoa consiga replicar o projeto completo: preparação do ambiente, organização do dataset, geração do `dataset.json`, treinamento da U-Net e execução dos métodos XAI individualmente (Grad-CAM, Grad-CAM++, IG e DeepLift).

---

## 1. Visão Geral

O projeto realiza **segmentação de imagens médicas (MRI)** utilizando o **nnU-Net** e técnicas de **Explainable AI (XAI)**. O objetivo é permitir que qualquer pesquisador ou aluno reproduza o pipeline completo, incluindo treinamento, inferência e análise explicável.

---

## 2. Requisitos

### Softwares e versões recomendadas

* Python 3.10+
* PyTorch (preferencialmente com CUDA)
* nnU-Net V1
* MONAI
* Captum (para Integrated Gradients e DeepLift)
* Nibabel
* NumPy, SciPy, Matplotlib
* GPU com 8GB+ (recomendado)

### Instalação do ambiente

Crie um ambiente virtual:

```
python -m venv nnunet-venv
```

Ative:

* **Windows:**

```
nnunet-venv\Scripts\activate
```

* **Linux/macOS:**

```
source nnunet-venv/bin/activate
```

Instale dependências:

```
pip install -r requirements.txt
```

Se estiver usando CUDA, instale o PyTorch correspondente conforme o site oficial.

---

## 3. Estrutura do Dataset (BraTS 2020)

A estrutura esperada é:

```
Dataset505_BraTS2020/
├── imagesTr/
│   ├── BraTS20_Training_001_0000.nii
│   ├── BraTS20_Training_001_0001.nii
│   └── ...
├── labelsTr/
│   ├── BraTS20_Training_001.nii
│   └── ...
```

### Passos para organizar:

1. Baixe o conjunto **BraTS 2020 Training**.
2. Extraia todos os casos.
3. Coloque as modalidades dentro de `imagesTr`.
4. Coloque os arquivos de máscara (`seg.nii`) em `labelsTr`.
5. (Opcional) Gere um subconjunto de 20% usando o script disponível.

---

## 4. Gerar o arquivo `dataset.json`

Use o script incluído no projeto:

```
python generate_dataset_json.py
```

Isso criará automaticamente um `dataset.json` no diretório do dataset.

---

## 5. Treinamento da U-Net (nnU-Net)

Para iniciar o treinamento:

```
python train.py
```

O script realiza:

* Carregamento do dataset
* Definição do modelo U-Net
* Treinamento completo ou continuação via checkpoint
* Salvamento de logs e pesos

Os resultados ficam em `outputs/models/`.

---

## 6. Execução dos Métodos XAI (individualmente)

Cada método é executado **separadamente** para evitar sobrecarga da GPU.

### 6.1 Grad-CAM

```
python run_gradcam.py
```

Gera:

* NIfTI 3D do Grad-CAM
* PNGs das fatias
* Métricas em CSV

---

### 6.2 Grad-CAM++

```
python run_gradcam_pp.py
```

Gera os mesmos arquivos, utilizando o método Grad-CAM++.

---

### 6.3 Integrated Gradients (IG)

```
python run_ig.py
```

Baseado no Captum.

---

### 6.4 DeepLift

```
python run_deeplift.py
```

Também baseado no Captum.

---

## 7. Resultados e Estrutura de Saída

A pasta `outputs/` contém:

```
outputs/
├── nii/       # volumes 3D gerados pelos métodos XAI
├── png/       # visualizações em 2D
├── csv/       # métricas por amostra
├── logs/      # logs de treinamento e XAI
└── models/    # pesos da U-Net
```

---

## 8. Solução de Problemas

### Erro de GPU (CUDA Out of Memory)

* Reduza o batch size
* Execute XAI em menos amostras
* Feche programas que usam GPU (ex.: navegadores)

### Erros com NIfTI

* Confirme se os nomes seguem o padrão nnU-Net
* Verifique se há arquivos `.nii` corrompidos

### Problemas no treinamento

* Verifique se o dataset está corretamente estruturado
* Confirme se o PyTorch está usando CUDA

---

## 9. Citação

Se utilizar o projeto, cite:

* BraTS 2020
* nnU-Net (Isensee et al.)

---

## 10. Contato

Caso tenha dúvidas, sugestões ou queira contribuir, abra uma **issue** no repositório do GitHub.

---
