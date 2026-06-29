def clean_baf_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop unused / dead features
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    # Standardize categorical text
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            if col == "device_os":
                df[col] = df[col].astype(str).str.lower().str.strip()
            else:
                df[col] = df[col].astype(str).str.upper().str.strip()

    # Target
    df[TARGET] = df[TARGET].astype(int)

    # Binary columns
    for col in BINARY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Add missing flags before replacing sentinel values
    for col in SENTINEL_COLS:
        df[f"{col}_missing"] = (df[col] == -1).astype("Int64")

    # Replace sentinel -1 with NaN
    for col in SENTINEL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] == -1, col] = np.nan

    # Numeric conversion
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log1p for heavy-tailed count-like features
    for col in LOG1P_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


def build_preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_features = NUMERIC_COLS + [f"{c}_missing" for c in SENTINEL_COLS]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("bin", binary_pipeline, BINARY_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


# ============================================================
# Training / evaluation
# ============================================================
def choose_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)

    # thresholds has length n-1; align by dropping last f1 point
    best_idx = np.nanargmax(f1[:-1])
    return thresholds[best_idx], f1[best_idx]


def main():
    # Load and clean
    df = pd.read_csv(DATA_PATH)
    df = clean_baf_dataframe(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Train / valid / test split with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Preprocess
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_valid_t = preprocessor.transform(X_valid)
    X_test_t = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_t = pd.DataFrame(X_train_t, columns=feature_names, index=X_train.index)
    X_valid_t = pd.DataFrame(X_valid_t, columns=feature_names, index=X_valid.index)
    X_test_t = pd.DataFrame(X_test_t, columns=feature_names, index=X_test.index)

    # Imbalance ratio for XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    print(f"Train negatives: {neg}")
    print(f"Train positives: {pos}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # XGBoost model
    model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train_t,
        y_train,
        eval_set=[(X_valid_t, y_valid)],
        verbose=100,
        early_stopping_rounds=100,
    )

    # Validation evaluation
    valid_prob = model.predict_proba(X_valid_t)[:, 1]
    valid_roc = roc_auc_score(y_valid, valid_prob)
    valid_pr = average_precision_score(y_valid, valid_prob)
    best_thr, best_f1 = choose_best_threshold(y_valid, valid_prob)

    print("\nValidation metrics")
    print(f"ROC-AUC:  {valid_roc:.6f}")
    print(f"PR-AUC:   {valid_pr:.6f}")
    print(f"Best threshold by F1: {best_thr:.6f}")
    print(f"Best F1 on validation: {best_f1:.6f}")

    # Final test evaluation using validation-selected threshold
    test_prob = model.predict_proba(X_test_t)[:, 1]
    test_pred = (test_prob >= best_thr).astype(int)

    test_roc = roc_auc_score(y_test, test_prob)
    test_pr = average_precision_score(y_test, test_prob)

    print("\nTest metrics")
    print(f"ROC-AUC:  {test_roc:.6f}")
    print(f"PR-AUC:   {test_pr:.6f}")
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, test_pred, digits=6))

    # Optional: save artifacts
    X_train_t.to_csv("X_train_transformed.csv", index=False)
    X_valid_t.to_csv("X_valid_transformed.csv", index=False)
    X_test_t.to_csv("X_test_transformed.csv", index=False)
    pd.Series(y_train, name=TARGET).to_csv("y_train.csv", index=False)
    pd.Series(y_valid, name=TARGET).to_csv("y_valid.csv", index=False)
    pd.Series(y_test, name=TARGET).to_csv("y_test.csv", index=False)

    print("\nSaved transformed datasets.")


if __name__ == "__main__":
    main()
