from src.problems.titanic import TitanicProblem

if __name__ == "__main__":
    p = TitanicProblem(csv_path="data/train.csv", seed=42)
    data = p.get_data()

    print("n_features:", p.n_features)
    print("feature_names:", data["feature_names"])
    print("train:", data["X_train"].shape, data["y_train"].shape)
    print("val:", data["X_val"].shape, data["y_val"].shape)
    print("test:", data["X_test"].shape, data["y_test"].shape)
    print("y_train counts:", {0: int((data["y_train"]==0).sum()), 1: int((data["y_train"]==1).sum())})
