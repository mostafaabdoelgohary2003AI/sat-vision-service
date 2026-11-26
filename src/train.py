"""Fine-tune ResNet18 on EuroSAT RGB. Checkpoint after every epoch (resumable)."""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from .data import load_arrays_cached, stratified_split, CLASSES

CKPT = "results/model.pt"
METRICS = "results/metrics.json"
EPOCHS = int(os.environ.get("EPOCHS", "2"))
PHASE = os.environ.get("PHASE", "A")  # A: head-only; B: +layer4; C: +layer3

def norm(X):
    x = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std

def build_model(fresh=True):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, len(CLASSES))
    return m

def backbone_forward(m, x):
    with torch.no_grad():
        x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
        x = m.layer1(x); x = m.layer2(x); x = m.layer3(x)
    return x

def head_forward(m, f):
    x = m.layer4(f); x = m.avgpool(x); x = torch.flatten(x, 1)
    return m.fc(x)

def main():
    os.makedirs("results", exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    X, y = load_arrays_cached()
    tr_idx, te_idx = stratified_split(y)
    Xtr, ytr = norm(X[tr_idx]), torch.from_numpy(y[tr_idx])
    Xte, yte = norm(X[te_idx]), torch.from_numpy(y[te_idx])
    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    model = build_model()
    for p in model.parameters():
        p.requires_grad = False
    groups = [{"params": model.fc.parameters(), "lr": 1e-3}]
    if PHASE in ("B", "C"):
        groups = [{"params": model.fc.parameters(), "lr": 1e-4},
                  {"params": model.layer4.parameters(), "lr": 2e-5}]
        for p in model.layer4.parameters():
            p.requires_grad = True
    if PHASE == "C":
        groups.append({"params": model.layer3.parameters(), "lr": 5e-6})
        for p in model.layer3.parameters():
            p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(groups)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05 if PHASE != "A" else 0.0)
    import torchvision.transforms.v2 as T
    aug = T.RandomHorizontalFlip()
    ckpt = os.environ.get("CKPT_FILE", CKPT)
    metrics = ckpt.replace("model", "metrics").replace(".pt", ".json")
    if os.environ.get("INIT_FROM") and os.path.exists(os.environ["INIT_FROM"]):
        sd = torch.load(os.environ["INIT_FROM"], map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        print("initialized from", os.environ["INIT_FROM"], "best_acc", sd.get("best_acc"))
    start_epoch, hist, best = 0, [], 0.0
    if os.path.exists(ckpt) and os.path.exists(metrics):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        try:
            opt.load_state_dict(sd["opt"])
        except ValueError:
            print("optimizer state incompatible, starting fresh optimizer")
        start_epoch = sd["epoch"] + 1
        hist = json.load(open(metrics))["history"]
        best = max(h["test_acc"] for h in hist) if hist else 0.0
    for epoch in range(start_epoch, EPOCHS):
        model.train(); t0 = time.time(); tot, correct, loss_sum = 0, 0, 0.0
        model.eval()
        for xb, yb in train_dl:
            opt.zero_grad()
            feats = backbone_forward(model, aug(xb))
            out = head_forward(model, feats)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item(); tot += len(yb)
        model.eval()
        with torch.no_grad():
            preds = torch.cat([head_forward(model, backbone_forward(model, xb)).argmax(1)
                               for xb, _ in DataLoader(TensorDataset(Xte, yte), batch_size=256)])
        acc = (preds == yte).float().mean().item()
        hist.append({"epoch": epoch, "train_loss": round(loss_sum / tot, 4),
                     "train_acc": round(correct / tot, 4), "test_acc": round(acc, 4),
                     "epoch_seconds": round(time.time() - t0, 1)})
        best = max(best, acc)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "classes": CLASSES, "best_acc": best, "phase": PHASE}, ckpt)
        if acc >= best:
            torch.save({"model": model.state_dict(), "epoch": epoch, "classes": CLASSES,
                        "test_acc": acc, "phase": PHASE}, ckpt.replace(".pt", "_best.pt"))
        json.dump({"history": hist, "train_images": len(tr_idx), "test_images": len(te_idx),
                   "classes": CLASSES, "phase": PHASE, "best_acc": best}, open(metrics, "w"), indent=2)
        print(hist[-1], flush=True)
    # final per-class metrics + confusion matrix
    model.eval()
    with torch.no_grad():
        preds = torch.cat([head_forward(model, backbone_forward(model, xb)).argmax(1)
                           for xb, _ in DataLoader(TensorDataset(Xte, yte), batch_size=256)])
    cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
    for t, p in zip(yte, preds):
        cm[t, p] += 1
    per_class = {CLASSES[i]: round((cm[i, i].item() / max(cm[i].sum().item(), 1)), 4) for i in range(len(CLASSES))}
    m = json.load(open(metrics))
    m["final_test_acc"] = round((preds == yte).float().mean().item(), 4)
    m["per_class_recall"] = per_class
    m["confusion_matrix"] = cm.tolist()
    json.dump(m, open(metrics, "w"), indent=2)
    print("final_test_acc", m["final_test_acc"])

if __name__ == "__main__":
    main()
