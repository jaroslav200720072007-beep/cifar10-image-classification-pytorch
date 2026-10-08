<div align="center">

# 🖼️ CIFAR-10 Image Classifier на PyTorch

**Багатокласова класифікація зображень повнозв'язною нейромережею (MLP).**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N9UVmZGwLWI8OM0VzSg-8PC65kypWcuh?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📖 Про проєкт

Навчальний проєкт з нейромереж: з нуля реалізовано повний цикл розв'язання задачі **мультикласової класифікації зображень** на датасеті **CIFAR-10** (60 000 кольорових зображень 32×32 у 10 класах).

У проєкті розібрано основні речі, які потрібні для роботи з PyTorch:

- підготовка даних через `Dataset` і `DataLoader`;
- побудова власної архітектури на основі `nn.Module`;
- написання власних циклів навчання та валідації;
- збереження й завантаження ваг моделі;
- візуалізація передбачень.

## ✨ Можливості

- ✅ Аугментація тренувальних даних (віддзеркалення, випадкові зсуви)
- ✅ Багатошарова нейромережа з `BatchNorm1d` та `ReLU`
- ✅ Автоматичний вибір пристрою (`cuda` / `cpu`)
- ✅ Прогрес-бар навчання через `tqdm`
- ✅ Метрики loss та accuracy на train і validation після кожної епохи
- ✅ Збереження навченої моделі у `.pth`
- ✅ Візуалізація передбачень на валідаційних зображеннях

## 🧠 Архітектура моделі

Повнозв'язна мережа (MLP). Зображення розгортається у вектор із 3 × 32 × 32 = 3072 ознак.

```
Input (3072)
   │
   ├─ Linear(3072 → 256) → BatchNorm1d → ReLU
   ├─ Linear(256  → 512) → BatchNorm1d → ReLU
   ├─ Linear(512  → 256) → BatchNorm1d → ReLU
   └─ Linear(256  → 10)                        → логіти 10 класів
```

Загалом близько **1,05 млн** параметрів.

## ⚙️ Параметри навчання

| Параметр | Значення |
|----------|----------|
| Датасет | CIFAR-10 (50 000 train / 10 000 validation) |
| Функція втрат | `CrossEntropyLoss` |
| Оптимізатор | Adam, `lr = 1e-3` |
| Розмір батчу | 64 |
| Кількість епох | 20 |
| Аугментація | `RandomHorizontalFlip`, `RandomCrop(32, padding=4)` |
| Пристрій | GPU (CUDA), якщо доступний, інакше CPU |

## 🏷 Класи

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

## 🛠 Технології

| Категорія | Інструменти |
|-----------|-------------|
| Мова | Python |
| Deep Learning | PyTorch, torchvision |
| Візуалізація | Matplotlib |
| Утиліти | tqdm |
| Середовище | Google Colab |

## ⚡ Швидкий старт

### Варіант 1. Google Colab (без встановлення)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N9UVmZGwLWI8OM0VzSg-8PC65kypWcuh?usp=sharing)

Для швидшого навчання увімкніть GPU: *Runtime → Change runtime type → T4 GPU*. Далі виконайте клітинки по черзі.

## 🔍 Як це працює

1. **Завантаження даних** — CIFAR-10 через `torchvision.datasets`. Тренувальна вибірка проходить аугментацію, валідаційна лише `ToTensor`.
2. **DataLoader** — батчі по 64 зображення, перемішування, `drop_last=True`.
3. **Модель** — клас `Net(nn.Module)` із методом `forward`, що розгортає зображення у вектор.
4. **Навчання** — функція `train()`: `zero_grad → forward → loss → backward → step`, плюс підрахунок accuracy.
5. **Валідація** — функція `evalute()` в режимі `model.eval()` та `torch.no_grad()`.
6. **Збереження** — `torch.save(model.state_dict(), "cifar_model.pth")`.
7. **Інференс** — завантаження ваг і передбачення класів для 8 валідаційних зображень.

## 💾 Збереження та завантаження моделі

```python
# Збереження ваг
torch.save(model.state_dict(), "cifar_model.pth")

# Завантаження
model = Net(input_size=3 * 32 * 32, hidden_size=256, output_size=10).to(device)
model.load_state_dict(torch.load("cifar_model.pth", map_location="cpu"))
model.eval()
```

## 📈 Результати

| Метрика | Train | Validation |
|---------|-------|------------|
| Loss | _XX_ | _XX_ |
| Accuracy | _XX %_ | _XX %_ |



## 🗺 Плани розвитку

- [ ] Замінити MLP на згорткову мережу (CNN), що зазвичай значно покращує якість на зображеннях
- [ ] Додати Dropout і learning rate scheduler
- [ ] Побудувати графіки loss/accuracy по епохах
- [ ] Додати confusion matrix
- [ ] Спробувати transfer learning (ResNet тощо)

