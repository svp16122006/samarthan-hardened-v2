from pathlib import Path
import json
from app.ml_core import TfidfVectorizerLite, LogisticRegressionLite, TextRiskModel, save_model

def train_test_split_local(texts, labels, test_size=0.25, seed=42):
    # deterministic stratified split without external ML dependencies
    groups={}
    for i,y in enumerate(labels): groups.setdefault(y,[]).append(i)
    rng=__import__('random').Random(seed); train=[]; test=[]
    for y,idxs in groups.items():
        rng.shuffle(idxs); ntest=max(1,round(len(idxs)*test_size)); test += idxs[:ntest]; train += idxs[ntest:]
    return [texts[i] for i in train],[texts[i] for i in test],[labels[i] for i in train],[labels[i] for i in test]

def metrics_local(y_true,y_pred,labels):
    cm=[[0 for _ in labels] for _ in labels]; li={v:i for i,v in enumerate(labels)}
    for a,b in zip(y_true,y_pred): cm[li[a]][li[b]]+=1
    vals=[]
    for i in range(len(labels)):
        tp=cm[i][i]; fp=sum(cm[r][i] for r in range(len(labels)))-tp; fn=sum(cm[i][c] for c in range(len(labels)))-tp
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0; f=2*pr*re/(pr+re) if pr+re else 0
        vals.append((pr,re,f))
    acc=sum(a==b for a,b in zip(y_true,y_pred))/len(y_true)
    return acc,sum(x[0] for x in vals)/len(vals),sum(x[1] for x in vals)/len(vals),sum(x[2] for x in vals)/len(vals),cm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

seed_phrases = {
    'stable': [
        'I feel safe today', 'things are manageable', 'I am calm and able to continue my routine',
        'I slept well and have support', 'I can manage the hearing', 'I know who to contact if needed',
        'I feel supported', 'my stress is low', 'I can focus on work', 'I feel in control',
        'I have no immediate concerns', 'I am prepared for the next appointment', 'my mood is okay',
        'I am safe at home', 'things are under control', 'I am comfortable with the next step',
        'I have been sleeping normally', 'I can continue normally', 'I feel steady', 'no urgent issue right now'
    ],
    'elevated': [
        'I am worried about the next hearing', 'I have been tense', 'sleep has been a little difficult',
        'I feel stressed and distracted', 'I feel anxious about case updates', 'I have been avoiding some calls',
        'my stress is higher this week', 'I feel uneasy about returning to the area', 'I am having trouble concentrating',
        'I feel pressured and need someone to talk to', 'the delay is making me worried', 'I am restless at night',
        'I feel more anxious than usual', 'I am struggling a bit with routine', 'I am concerned about testimony',
        'I feel tense when I think about the case', 'my family is worried', 'I need more support than before',
        'I feel unsettled but I am still functioning', 'my sleep and focus have worsened slightly'
    ],
    'high': [
        'I cannot sleep and feel overwhelmed', 'I am scared about the hearing and do not feel safe',
        'threats are making it hard to function', 'I keep panicking when I think about the accused',
        'I feel isolated and unable to cope', 'I am extremely anxious and need support soon',
        'I have stopped my normal activities because of fear', 'I feel exhausted and constantly on edge',
        'I am afraid to leave home', 'the delays are causing severe distress', 'I cannot concentrate and feel trapped',
        'I am having frequent panic and fear', 'I feel unsafe and my sleep is very poor', 'I am struggling to cope with testimony',
        'I am withdrawing from everyone and feel very distressed', 'I feel threatened and overwhelmed',
        'I am unable to manage daily life because of stress', 'I am terrified about what may happen next',
        'my distress is getting much worse', 'I need a counsellor urgently because I cannot cope'
    ],
    'critical': [
        'I am in immediate danger and need help now', 'someone is threatening me right now',
        'I do not feel safe where I am and need urgent protection', 'there is an immediate threat to my family',
        'I am being followed and need urgent assistance', 'the threat is happening now and I need emergency help',
        'I cannot safely attend the hearing because of an immediate threat', 'I am extremely distressed and need an urgent human response',
        'someone has arrived near my home and I am afraid', 'I need protection immediately',
        'there is a serious immediate safety concern', 'I need a counsellor or official urgently',
        'I cannot remain here safely', 'I need emergency relocation support',
        'the situation has escalated and I need immediate help', 'I feel unsafe right now',
        'I need someone to contact me urgently about my safety', 'the threats have escalated today',
        'I am afraid something bad will happen now', 'please arrange urgent support'
    ]
}
contexts = [
    'about the case', 'because of the upcoming hearing', 'after the latest update', 'during the investigation',
    'after another delay', 'before testimony', 'at home today', 'when thinking about the next step',
    'after receiving a message', 'while dealing with the current situation'
]
modifiers = ['today', 'this week', 'lately', 'right now', 'since the last hearing', 'after the recent update']
texts, labels = [], []
# Controlled synthetic augmentation: combine class-specific seed language with neutral contexts.
for label, rows in seed_phrases.items():
    for i, phrase in enumerate(rows):
        for j in range(4):
            sentence = f"{phrase} {contexts[(i+j)%len(contexts)]} {modifiers[(i*2+j)%len(modifiers)]}."
            texts.append(sentence); labels.append(label)

X_train, X_test, y_train, y_test = train_test_split_local(texts, labels, test_size=0.25, seed=42)
pipe = TextRiskModel(TfidfVectorizerLite(ngram_range=(1, 2), min_df=1), LogisticRegressionLite(list(seed_phrases), max_iter=900, lr=0.45, l2=0.002))
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
acc, precision, recall, f1, cm = metrics_local(y_test, pred, list(seed_phrases))
save_model(pipe, MODEL_DIR / 'model.pkl')
meta = {
    'labels': list(seed_phrases), 'n_total': len(texts), 'n_train': len(X_train), 'n_test': len(X_test),
    'accuracy': round(float(acc), 4), 'macro_precision': round(float(precision), 4),
    'macro_recall': round(float(recall), 4), 'macro_f1': round(float(f1), 4),
    'confusion_matrix': cm, 'data_type': 'synthetic engineering examples',
    'note': 'Not clinical validation; do not use for real-world care decisions.', 'model_type': 'Dependency-light TF-IDF + multinomial Logistic Regression (pure Python)', 'split': 'Stratified deterministic 75/25 hold-out'
}
(MODEL_DIR / 'metrics.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print(json.dumps(meta, indent=2))
