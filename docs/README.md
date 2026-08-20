# 🚨 DIY LLM V4 Clone — AI Setara DeepSeek V4

> Proyek **open-source** untuk **membangun AI bahasa** setara dengan **DeepSeek-V4**, dilatih untuk konteks **hukum, kepolisian, dan bahasa Indonesia**.

---

## 🧠 Spesifikasi Model Target

| Fitur | DeepSeek V4 Equivalent |
|-------|-------------------------|
| **Arsitektur** | Transformer Decoder |
| **Jumlah Parameter** | 8B – 15B |
| **Token Context** | 128K |
| **Bahasa** | Multi-bahasa, fokus Indonesia |
| **Training Data** | 2T token + domain hukum/DIY |
| **Inferensi** | OpenAI-compatible API |
| **UI** | Gradio / Chat UI |
| **Monitoring** | Prometheus + Grafana |

---

## 📁 Struktur Proyek

```
diy-llm-v4-clone/
├── models/                       # Folder untuk checkpoint model
├── training/
│   ├── dataset/                  # Dataset JSON / JSONL
│   ├── scripts/                  # Fine-tuning scripts
│   └── configs/                  # Konfigurasi training
├── inference/
│   ├── api/                      # API (FastAPI / OpenAI-compatible)
│   └── ui/                       # UI (Gradio)
├── docker/                       # Docker image build context
│   ├── Dockerfile
│   └── entrypoint.sh
├── docs/
│   └── README.md                 # Tutorial ini
└── README.md
```

---

## 🚀 Panduan Setup (3 Menit)

### 1. Clone Repository
```bash
git clone https://github.com/vanderstark/diy-llm-v4-clone.git
cd diy-llm-v4-clone
```

### 2. Jalankan dengan Docker Compose
```bash
docker compose up --build -d
```

### 3. Akses
- API: `http://localhost:8000/docs`
- UI: `http://localhost:7860`
- Grafana: `http://localhost:3000` (admin/admin)

---

## 🧪 Membangun Model dari Awal

Gunakan script berikut untuk memulai training/fine-tuning:

```bash
cd training/scripts
python train.py --config ../configs/deepseek_clone.yaml
```

### Contoh konfigurasi (`configs/deepseek_clone.yaml`)
```yaml
model_name: "deepseek-ai/deepseek-v4-tiny-clone"
tokenizer: "deepseek-ai/DeepSeek-V4-BPE"
dataset_path: "../dataset/diy_training_data.jsonl"
output_dir: "/app/models/checkpoints"
per_device_train_batch_size: 8
gradient_accumulation_steps: 4
learning_rate: 2e-5
num_train_epochs: 3
fp16: true
deepspeed: "../configs/deepspeed_config.json"
logging_steps: 50
eval_steps: 500
save_steps: 1000
push_to_hub: false
```

---

## 🧬 Fine-Tuning untuk Domain Khusus

Gunakan LoRA / QLoRA agar tidak perlu training full model:

```bash
python lora_finetune.py \
  --base_model "deepseek-ai/deepseek-v4-tiny" \
  --dataset "../dataset/diy_qa.jsonl" \
  --output_dir "../checkpoints/lora-diy-v4" \
  --per_device_train_batch_size 4 \
  --lora_r 64 \
  --lora_alpha 16 \
  --bits 4
```

---

## 🔐 Catatan Keamanan

- Semua model harus disimpan di folder terisolasi (`models/`)
- API Key hANYA disimpan di `.env`, **JANGAN commit ke repo**
- Aktifkan auto-redact untuk output yang mengandung NIK / data sensitif
- RBAC di layer API untuk kontrol akses unit ke unit

---

## 📦 Deploy ke Hugging Face Spaces

1. Buat space baru di [huggingface.co/spaces](https://huggingface.co/spaces)
2. Upload seluruh folder `inference/ui`
3. Atur secrets di Settings → Secrets:
   - `HF_TOKEN`: token HF Anda
4. Space akan otomatis build & deploy

---

## © 2025 DIY LLM V4 Clone

## 🏋️ Training di Server GPU

1. Install Unsloth:
   `pip install unsloth`

2. Jalankan training:
   `python training/scripts/train_unsloth.py`
