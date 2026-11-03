"""Quick start: statistical audit of a trained MLP (generalises the original Iris/IDC notebooks)."""
import warnings; warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import annstat as A

d = load_iris()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.2, random_state=42, stratify=d.target)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
net = MLPClassifier(hidden_layer_sizes=(10,), max_iter=3000, random_state=0).fit(Xtr, ytr)

report = A.StatisticalTuner(net).analyze(Xtr, ytr, Xte, yte, reduction_ks=[3, 5, 7],
                                         feature_names=d.feature_names, class_names=d.target_names)
print(report.to_markdown())

# --- or use the building blocks directly -----------------------------------------------------
a = A.SklearnMLPAdapter(net)
print(A.binomial_ci(int((a.predict(Xte) == yte).sum()), len(yte)))        # Wilson interval
H, (W, b) = a.hidden(Xte), a.head()
print(A.contribution_tests(H, W, paired=False)["pairwise"].head())      # the notebooks' Mann-Whitney tests, now Holm-adjusted
small = A.compress_lsc(a.hidden(Xtr), W, b, A.select_qr(a.hidden(Xtr), 5), "classification", a.classes_)
print("5-neuron compensated network accuracy:", (small.predict(H) == yte).mean())
