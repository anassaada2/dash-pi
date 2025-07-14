import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier ,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
import numpy as np


# Configuration de la page
st.set_page_config(page_title="ML Model Selection and Evaluation", layout="wide")

# Titre principal
st.title("Machine Learning Model Selection and Evaluation")
st.write("Choose a dataset and a model, train it, evaluate its performance, and make predictions.")

# Sidebar : configuration
st.sidebar.header("Configuration")
model_name = st.sidebar.selectbox("Select Model", ["Random Forest", "Logistic Regression", "SVM" ,"Gradient Boosting Classifier"])
test_size = st.sidebar.slider("Test Size (proportion)", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", value=42)

# Chargement du dataset
st.title("Chargement de dataset")
file1 = st.file_uploader("Choisissez un fichier CSV (Dataset 1)", type=["csv"], key="file1")
file2 = st.file_uploader("Ajouter un deuxième dataset pour fusion (optionnel)", type=["csv"], key="file2")

df1, df2 = None, None
if file1:
    df1 = pd.read_csv(file1)
    st.success("✅ Dataset 1 chargé")

if file2:
    df2 = pd.read_csv(file2)
    st.success("✅ Dataset 2 chargé")

if df1 is not None and df2 is not None:
    df = pd.merge(df1, df2, how="outer")
    st.success("✅ Fusion réalisée avec succès")
else:
    df = df1

if df is not None:

    st.dataframe(df.head(100))
    st.write("Dimensions du DataFrame :", df.shape)

    # Sélection de la colonne cible
    target_col = st.selectbox("Choisissez la colonne cible :", df.columns)

    # Slider : seuil de suppression des colonnes avec trop de NaN
    threshold = st.slider(
        "Seuil maximum de valeurs manquantes par colonne (en %)",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Les colonnes avec plus de ce pourcentage de valeurs manquantes seront supprimées."
    )

    if target_col:
        st.write("Distribution brute de la variable cible :")
        st.write(df[target_col].value_counts())
        st.write("Valeurs manquantes dans la cible :")
        st.write(df[target_col].isna().value_counts())

        # Suppression des colonnes avec trop de NaN
        missing_ratio = (df.isna().sum() / df.shape[0])
        df = df[df.columns[missing_ratio < threshold]]
        st.write("Taux de valeurs manquantes par colonne :")
        st.write(missing_ratio.sort_values(ascending=False))

        st.success(f"Colonnes conservées (moins de {int(threshold*100)}% de NaN) : {df.shape[1]}")
        st.subheader("Distribution de la variable cible (après nettoyage)")
        st.write(df[target_col].value_counts(normalize=True))

        if st.button("Afficher les plots"):
            st.subheader("Heatmap des valeurs manquantes")
            fig = plt.figure(figsize=(20, 10))
            sns.heatmap(df.isna(), cbar=False, cmap="viridis")
            st.pyplot(fig)

        if st.button("Afficher les plots des variables numériques"):
            st.subheader("Distribution des variables numériques")
            for col in df.select_dtypes(include='float'):
                fig = plt.figure(figsize=(8, 4))
                sns.histplot(df[col], kde=True, bins=30)
                plt.title(f'Distribution de {col}')
                plt.xlabel(col)
                plt.ylabel('Fréquence')
                plt.grid(True)
                st.pyplot(fig)

        st.subheader("Les variables des types object de votre dataset")
        for col in df.select_dtypes('object'):
            st.write(f'{col :-<50} {df[col].unique()}')

        st.subheader("Distribution des variables catégorielles")
        for col in df.select_dtypes(include='object'):
            fig = plt.figure(figsize=(8, 4))
            sns.countplot(data=df, x=col, order=df[col].value_counts().index)
            plt.title(f'Distribution de {col}')
            plt.xlabel(col)
            plt.ylabel('Fréquence')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Tableau croisé (crosstab) avec la variable cible")
        cross_col = st.selectbox("Choisissez une colonne pour le crosstab avec la cible :", [col for col in df.columns if col != target_col])
        if cross_col:
            crosstab_result = pd.crosstab(df[target_col], df[cross_col])
            st.write(f"Tableau croisé entre **{target_col}** et **{cross_col}** :")
            st.dataframe(crosstab_result)

        # Séparation success/failure
        try:
            unique_vals = df[target_col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                failure_df = df[df[target_col] == 0]
                success_df = df[df[target_col] == 1]

                st.subheader("Création des ensembles positifs (réussites) et négatifs (échecs)")
                st.write(f"✅ Nombre de succès ({target_col} == 1) : {success_df.shape[0]}")
                st.write(f"❌ Nombre d'échecs ({target_col} == 0) : {failure_df.shape[0]}")

                with st.expander("Voir un aperçu des cas de succès"):
                    st.dataframe(success_df.head())
                with st.expander("Voir un aperçu des cas d'échec"):
                    st.dataframe(failure_df.head())
            else:
                st.warning(f"La colonne cible '{target_col}' ne contient pas uniquement des 0 et 1.")
        except Exception as e:
            st.error(f"Erreur lors de la séparation des classes : {e}")

        # Sous-ensembles selon mots-clés
        keywords_input = st.text_input("Entrez les mots-clés pour diviser les colonnes et afficher les plots (séparés par des virgules)", placeholder="ex: software, interfaces, network")

        if keywords_input:
            keywords = list(dict.fromkeys([kw.strip().lower() for kw in keywords_input.split(',') if kw.strip()]))
            subsets = {}
            for keyword in keywords:
                matching_cols = [col for col in df.columns if keyword in col.lower()]
                if matching_cols:
                    subsets[keyword] = df[matching_cols]

            if subsets:
                for name, sub_df in subsets.items():
                    st.subheader(f"Sous-ensemble : {name} ({sub_df.shape[1]} colonnes)")
                    st.dataframe(sub_df.head())
            else:
                st.warning("Aucune colonne ne correspond aux mots-clés saisis.")

            if st.button("Afficher les plots de distribution des sous-ensembles créés pour success et failure"):
                for keyword, sub_df in subsets.items():
                    matching_cols = sub_df.columns.tolist()
                    st.subheader(f"Distribution des variables pour le mot-clé : **{keyword}**")
                    for col in matching_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.histplot(success_df[col], label='Success', kde=True, stat="density", color='green', alpha=0.6, ax=ax)
                            sns.histplot(failure_df[col], label='Failure', kde=True, stat="density", color='red', alpha=0.6, ax=ax)
                            ax.set_title(f"Distribution de {col}")
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                        else:
                            st.info(f"La colonne **{col}** n'est pas numérique et a été ignorée pour les plots de distribution.")

            if st.button("Afficher les plots avec relation à la cible (Success vs Failure) pour chaque sous-ensemble"):
                for keyword, sub_df in subsets.items():
                    st.subheader(f"Distribution de la cible pour le sous-ensemble : **{keyword}**")
                    matching_cols = sub_df.columns.tolist()
                    for col in matching_cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.histplot(success_df[col], label='Success (label 1)', color='blue', kde=True, stat='density', alpha=0.5, ax=ax)
                            sns.histplot(failure_df[col], label='Failure (label 0)', color='orange', kde=True, stat='density', alpha=0.5, ax=ax)
                            ax.set_title(f'Distribution de {col} selon la cible ({target_col})')
                            ax.legend()
                            ax.grid(True)
                            st.pyplot(fig)
                        else:
                            st.info(f"La colonne **{col}** n'est pas numérique, ignorée.")

        # Analyse d'une colonne catégorielle vs cible
        st.subheader("Relation entre la variable cible et une variable catégorielle (ex: zone, région...)")
        cat_col = st.selectbox(
            "Choisissez une colonne catégorielle à analyser par rapport à la cible :",
            [col for col in df.select_dtypes(include='object').columns if col != target_col]
        )

        if cat_col:
            try:
                success_rate = df.groupby(cat_col)[target_col].mean().sort_values(ascending=False)
                st.write("### ✅ Taux de succès moyen par catégorie :")
                st.dataframe(success_rate)

                failure_counts = df[df[target_col] == 0][cat_col].value_counts()
                st.write("### ❌ Nombre d'échecs (label = 0) par catégorie :")
                st.dataframe(failure_counts)

             

            except Exception as e:
                st.error(f"Erreur lors de l'analyse : {e}")


if st.button("Afficher la matrice de corrélation (variables numériques)"):
    st.subheader("🔍 Matrice de corrélation des variables numériques")

    numeric_df = df.select_dtypes(include=['float', 'int'])

    if numeric_df.shape[1] < 2:
        st.warning("Pas assez de colonnes numériques pour afficher une matrice de corrélation.")
    else:
        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(72, 72))
        sns.heatmap(
            corr_matrix,
            cmap='coolwarm',
            annot=True,
            fmt=".2f",
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .6},
            ax=ax
        )
        ax.set_title("Matrice de corrélation des variables numériques")
        st.pyplot(fig)
st.title(" TrainTest - Nettoyage - Encodage")
if df is not None :
    trainset, testset = train_test_split(df, test_size=test_size, random_state=random_state)
    st.write("Les valeurs de variable target du trainset",trainset[target_col].value_counts())
    st.write("les valeurs de variable target du testset",testset[target_col].value_counts())



# ---- ✅ Encodage amélioré ----
def encodage(df, afficher_mapping=False):
    mapping_dict = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        unique_vals = df[col].dropna().unique()
        val_to_int = {val: idx for idx, val in enumerate(unique_vals)}
        df[col] = df[col].map(val_to_int)
        mapping_dict[col] = val_to_int

    if afficher_mapping:
        st.subheader("🔢 Dictionnaires d'encodage :")
        for col, mapping in mapping_dict.items():
            st.markdown(f"**Colonne : `{col}`**")
            st.json(mapping)

    return df

st.subheader("⚙️ Choisissez une méthode d'imputation pour les valeurs manquantes")
selected_imputation = st.selectbox(
    "Méthode d'imputation :",
    ["supprimer", "moyenne", "max", "min"],
    index=1,
    help="Choisissez comment traiter les valeurs manquantes dans les colonnes numériques."
)

# ---- ✅ Imputation améliorée ----
def imputation(df, methode='moyenne'):
    if methode == 'supprimer':
        df = df.dropna()
    elif methode == 'moyenne':
        df = df.fillna(df.mean(numeric_only=True))
    elif methode == 'max':
        df = df.fillna(df.max(numeric_only=True))
    elif methode == 'min':
        df = df.fillna(df.min(numeric_only=True))
    else:
        st.warning(f"❗ Méthode d'imputation inconnue : {methode}")
    return df


# ---- ✅ Fonction de prétraitement complète ----
def preprocessing(df, afficher_mapping=True):
    df = encodage(df, afficher_mapping=afficher_mapping)
    df = imputation(df, methode=selected_imputation)
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return X, y
if st.button("Prétraitement"):
    X_train, y_train = preprocessing(trainset)
    X_test, y_test = preprocessing(testset)
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.preprocessed = True

    st.write("X_train :", X_train.shape)
    st.write("y_train distribution :", y_train.value_counts())
    st.success("✅ Prétraitement terminé avec succès.")

    


def evaluation(model):
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)

    print("=== Matrice de confusion ===")
    print(confusion_matrix(y_test, ypred))
    print("\n=== Rapport de classification ===")
    print(classification_report(y_test, ypred))

    # Courbe d'apprentissage
    N, train_score, val_score = learning_curve(
        model, X_train, y_train,
        cv=3, scoring='f1',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1, 5)
    )

    plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.xlabel("Taille de l’échantillon d’entraînement")
    plt.ylabel("Score F1")
    plt.title("Courbe d'apprentissage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Mapping du nom vers le modèle
def get_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(random_state=random_state)
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif name == "SVM":
        return SVC(probability=True, random_state=random_state)
    elif name =="Gradient Boosting Classifier":
        return GradientBoostingClassifier(n_estimators=31, random_state=random_state)
    else:
        return None
    

      # --- Bouton d'évaluation (dans la sidebar) ---
st.sidebar.markdown("---")
      
if st.sidebar.button("✅ Évaluer ce modèle"):
    if not st.session_state.preprocessed:
        st.warning("⚠️ Veuillez d'abord effectuer le prétraitement.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        model = get_model(model_name)
        model.fit(X_train, y_train)
        ypred = model.predict(X_test)

        st.subheader("📊 Matrice de confusion")
        st.write(confusion_matrix(y_test, ypred))

        st.subheader("🧾 Rapport de classification")
        st.text(classification_report(y_test, ypred))

        st.subheader("📈 Courbe d'apprentissage")
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=3, scoring='f1',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(train_sizes, train_scores.mean(axis=1), label='Entraînement', marker='o')
        ax.plot(train_sizes, val_scores.mean(axis=1), label='Validation', marker='s')
        ax.set_xlabel("Taille de l'échantillon")
        ax.set_ylabel("Score F1")
        ax.set_title("Courbe d'apprentissage")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)