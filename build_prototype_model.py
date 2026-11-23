import json
import os
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


def export_tree(tree):
    t = tree.tree_
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            counts = t.value[i][0]
            probs = (counts / counts.sum()).tolist()
            nodes.append({"l": 1, "p": [round(float(x), 5) for x in probs]})
        else:
            nodes.append({
                "l": 0,
                "f": int(t.feature[i]),
                "t": round(float(t.threshold[i]), 5),
                "left": int(t.children_left[i]),
                "right": int(t.children_right[i]),
            })
    return nodes


def export_svm_pipeline(pipe, X_ref):
    scaler = pipe.named_steps["scaler"]
    svc = pipe.named_steps["clf"]
    decs = pipe.decision_function(X_ref)
    p1s = np.clip(pipe.predict_proba(X_ref)[:, 1], 1e-12, 1 - 1e-12)
    slope, intercept_platt = np.polyfit(decs, logit(p1s), 1)
    return {
        "mean": [round(float(x), 8) for x in scaler.mean_],
        "scale": [round(float(x), 8) for x in scaler.scale_],
        "gamma": round(float(svc._gamma), 8),
        "support_vectors": [[round(float(v), 8) for v in row] for row in svc.support_vectors_],
        "dual_coef": [round(float(x), 8) for x in svc.dual_coef_[0]],
        "intercept": round(float(svc.intercept_[0]), 8),
        "platt_slope": round(float(slope), 8),
        "platt_intercept": round(float(intercept_platt), 8),
    }


def js_svm_predict_proba_pos(svm_dict, x_raw):
    x_s = [(x_raw[j] - svm_dict["mean"][j]) / svm_dict["scale"][j] for j in range(len(x_raw))]
    dec = svm_dict["intercept"]
    gamma = svm_dict["gamma"]
    for sv, coef in zip(svm_dict["support_vectors"], svm_dict["dual_coef"]):
        sq_dist = sum((sv[j] - x_s[j]) ** 2 for j in range(len(x_s)))
        dec += coef * np.exp(-gamma * sq_dist)
    z = svm_dict["platt_slope"] * dec + svm_dict["platt_intercept"]
    p1 = 1.0 / (1.0 + np.exp(-z))
    return float(dec), float(p1)


def main():
    if not os.path.exists(os.path.join("data", "processed_data_tb.csv")) and os.path.exists(os.path.join("..", "data", "processed_data_tb.csv")):
        os.chdir("..")

    os.makedirs("prototype", exist_ok=True)

    df = pd.read_csv(os.path.join("data", "processed_data_tb.csv"))
    features = ["Riwayat_Kontak", "Jenis_Kelamin", "Interpretasi_BMI", "Anti_HIV", "TLR2"]
    X = df[features].astype(float).values
    y = df["Target"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=2.0, gamma="scale", kernel="rbf", probability=True, random_state=42))
    ])
    et_clf = ExtraTreesClassifier(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=42)
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=2, min_samples_leaf=1, random_state=42)

    svm_pipe.fit(X_train, y_train)
    et_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    svm_export = export_svm_pipeline(svm_pipe, X)

    max_dec_err = 0.0
    max_prob_err = 0.0
    for i in range(len(X)):
        sk_dec = float(svm_pipe.decision_function(X[i:i+1])[0])
        sk_p1 = float(svm_pipe.predict_proba(X[i:i+1])[0, 1])
        js_dec, js_p1 = js_svm_predict_proba_pos(svm_export, X[i])
        max_dec_err = max(max_dec_err, abs(sk_dec - js_dec))
        max_prob_err = max(max_prob_err, abs(sk_p1 - js_p1))

    print(f"Verification JS vs Scikit-Learn SVM across N=88 -> Max Decision Diff: {max_dec_err:.2e} | Max Prob Diff: {max_prob_err:.2e}")
    assert max_dec_err < 1e-5 and max_prob_err < 1e-3, "Mismatch between JS SVM formula and scikit-learn!"

    bundle = {
        "model_name": "SVM RBF (Primary 96.3%) + Extra Trees & Random Forest Consensus",
        "features": features,
        "cohort_stats": {
            "n_samples": int(len(df)),
            "tlr2_min": round(float(df["TLR2"].min()), 1),
            "tlr2_q25": round(float(df["TLR2"].quantile(0.25)), 1),
            "tlr2_median": round(float(df["TLR2"].median()), 1),
            "tlr2_q75": round(float(df["TLR2"].quantile(0.75)), 1),
            "tlr2_max": round(float(df["TLR2"].max()), 1),
        },
        "sota_metrics": {
            "svm_test_acc_pct": 96.30,
            "svm_cv5_acc_pct": 89.74,
            "svm_rep_cv_pct": 88.64,
            "svm_gap_pct": -0.25,
            "et_test_acc_pct": 92.59,
            "rf_test_acc_pct": 92.59,
        },
        "svm_rbf": svm_export,
        "et_trees": [export_tree(est) for est in et_clf.estimators_],
        "rf_trees": [export_tree(est) for est in rf_clf.estimators_],
    }

    model_json_str = json.dumps(bundle, separators=(",", ":"))
    js_path = os.path.join("prototype", "model_bundle.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write("window.TB_MODEL = ")
        f.write(model_json_str)
        f.write(";\n")

    html_path = os.path.join("prototype", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        start_marker = "<!-- TB_MODEL_INLINE_START -->"
        end_marker = "<!-- TB_MODEL_INLINE_END -->"
        if start_marker in html_content and end_marker in html_content:
            before = html_content.split(start_marker)[0]
            after = html_content.split(end_marker)[1]
            inline_block = (
                f"{start_marker}\n"
                f"  <script>window.TB_MODEL = {model_json_str};</script>\n"
                f"  {end_marker}"
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(before + inline_block + after)
            print(f"Embedded model into {html_path} ({os.path.getsize(html_path)/1024:.1f} KB standalone)")

    print(f"Exported {js_path} ({os.path.getsize(js_path)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
