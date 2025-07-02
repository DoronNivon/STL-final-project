from IPython.core.display_functions import display

if __name__ == '__main__':
    #Import Libraries
    import pandas as pd
    import matplotlib
    import numpy as np
    matplotlib.use('TkAgg')      # this sets pycharm to show plots without problems.
    import concurrent.futures
    import requests
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import time
    import seaborn as sns
    # metrics to be used for comparison between the models.
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss)
    # Import necessary libraries for classification modeling and preprocessing
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
    from sklearn.preprocessing import StandardScaler
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.ensemble import RandomForestClassifier

    # Read the data CSV - ----Make sure to replace the path to the one in your computer----
    df = pd.read_csv("/Users/doron/Documents/warehouse_data.csv")

    #df = pd.read_csv("/Applications/Documents/UHaifa/Semester2/Statistical_Learning/warehouse_data.csv")

    # -------------------------------   basic EDA ----------------------------------------------------------
    print(df.dtypes)
    print(df['label'].unique())
    print(len(df['label'].unique()))
    print(df['label'].value_counts())

    # ---------------------- Original Label Distribution (Multiclass) --------------------

    label_counts = df['label'].value_counts()

    plt.figure(figsize=(8, 5))
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Original Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(block=False)

    print(df.isnull().sum())  # for each column, you get how many missing values the column has.
    nan_percentage = df.isna().mean() * 100
    print(nan_percentage)

    # ----------------------------   filtering the data to Cardboard box group and non-Cardboard box group --------------------------------

    # Create a subset of the data containing only rows labeled as 'Cardboard Box'
    df_cardboardbox = df[df['label'] == 'Cardboard Box']

    # Create a subset of the data containing all other labels (not 'Cardboard Box'), and drop any missing rows
    df_not_cardboardbox = df[df['label'] != 'Cardboard Box'].dropna()

    # Randomly sample the same number of rows from the non-cardboard group as there are in the cardboard group
    # This ensures class balance for binary classification
    df_not_cardboardbox_sampled = df_not_cardboardbox.sample(n=len(df_cardboardbox), random_state=1)

    # Concatenate the two groups into one balanced DataFrame (equal number of 'Cardboard Box' and 'not Cardboard Box')
    df_balanced = pd.concat([df_cardboardbox, df_not_cardboardbox_sampled], ignore_index=True)

    # Add a new binary target column: 1 for 'Cardboard Box', 0 for anything else
    df_balanced['label_binary'] = (df_balanced['label'] == 'Cardboard Box').astype(int)

    # ---------- ------------------ Balanced Label Distribution (Binary) --------------------------
    binary_counts = df_balanced['label_binary'].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    binary_counts.plot(kind='bar', color=['lightcoral', 'lightblue'], edgecolor='black')
    plt.title('Balanced Label Distribution (Binary)')
    plt.xlabel('Label (0 = Other, 1 = Cardboard Box)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Other', 'Cardboard Box'], rotation=0)
    plt.tight_layout()
    plt.show(block=False)

    # -------------------------------   Upload Images to Python (for the balanced dataset only) -------------------------------------------------
    # Function to download a single image from a URL
    def download_single_image(index_url_tuple):
        idx, url = index_url_tuple
        try:
            # Send HTTP GET request to the image URL with headers and timeout
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            # Check if the request was successful (status code 200)
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}")
            # Check if the response content type is an image
            if not response.headers.get('Content-Type', '').startswith('image'):
                raise Exception(f"Invalid content-type: {response.headers.get('Content-Type')}")
            # Open the image, convert it to RGB format, and return it with its index
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return idx, img
        # Catch any issues in downloading or reading the image
        except (UnidentifiedImageError, Exception) as e:
            print(f"Error at index {idx}: {e}")
            return idx, None


    # Function to download all images from a DataFrame in batches
    def load_images(df, url_column='asset_preview', pause_every=150, max_workers=1):
        images_dict = {}
        # Extract a list of (index, url) tuples from the DataFrame
        tasks = list(df[url_column].items())  # list of (index, url)
        # Process tasks in chunks to avoid overwhelming the server
        for i in range(0, len(tasks), pause_every):
            chunk = tasks[i:i + pause_every]
            # Use ThreadPoolExecutor to download images in parallel (multi-threading)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(executor.map(download_single_image, chunk),
                                    total=len(chunk),
                                    desc=f"Downloading batch {i // pause_every + 1}"))
            # Store results in a dictionary using the DataFrame index as the key
            for idx, img in results:
                images_dict[idx] = img
            # Pause between batches to reduce the risk of being rate-limited or blocked
            if i + pause_every < len(tasks):
                print("⏸ Pausing for 15 seconds to avoid getting blocked...")
                time.sleep(15)

        return images_dict

    print("loading images...")

    # Load images for the balanced dataset using the previously defined loader function.
    # It fetches images from the 'asset_preview' column in batches of 151, using a single thread.
    images_dict = load_images(df_balanced, url_column='asset_preview', pause_every=151, max_workers=1)

    # Map the downloaded images back into the DataFrame, creating a new 'image' column.
    # Each image is retrieved from the dictionary using its index.
    df_balanced['image'] = df_balanced.index.map(images_dict.get)

    # Randomly select 4 indices from the balanced DataFrame without replacement (no duplicates).
    random_df_balanced_indices = np.random.choice(df_balanced.index, size=4, replace=False)

    # Function to display multiple images (up to 4) from the DataFrame
    def visual_images(indices, df_):
        # Extract the images for the given indices from the 'image' column
        images = df_.loc[indices, 'image']  # These should be PIL Image objects
        # Extract titles for each image (e.g., asset name, label, or any relevant identifier)
        titles = df_.loc[indices, 'asset_name']  # You can change this to 'label' or 'asset_preview' if preferred
        # Set up the overall figure size for the plot grid
        plt.figure(figsize=(10, 8))
        # Loop through each image and corresponding title
        for i, (img, title) in enumerate(zip(images, titles)):
            if img is not None:
                # Create a subplot (2 rows × 2 columns layout)
                plt.subplot(2, 2, i + 1)
                # Display the image
                plt.imshow(img)
                # Set the title above the image
                plt.title(title, fontsize=10)
                # Hide the axis ticks for a cleaner view
                plt.axis('off')
        # Adjust spacing to prevent overlap between subplots
        plt.tight_layout()
        # Display the final figure in non-blocking mode
        plt.show(block=False)

    # Call the function with randomly selected indices to display 4 example images from the DataFrame
    visual_images(random_df_balanced_indices, df_balanced)

    # ------------------------------ resize image to 128 x 128  --------------------------------------------------
    # Function to resize an image and normalize its pixel values
    def resize_and_convert(img, size=(128, 128)):
        # If the image is missing (None), return None to avoid processing errors
        if img is None:
            return None
        # Resize the image to the specified size (default: 128x128 pixels)
        resized_img = img.resize(size)
        # Convert the image to a NumPy array and normalize pixel values to the [0, 1] range
        normalized_img = np.array(resized_img) / 255.0  # Normalize pixel values
        # Return the processed image array
        return normalized_img


    # Apply the resize_and_convert function to each image in the DataFrame
    # The result is stored in a new column 'image_array_resized' as normalized NumPy arrays
    df_balanced['image_array_resized'] = df_balanced['image'].apply(lambda img: resize_and_convert(img, size=(128, 128)))


    # ----------------------------------------------------------------------------------------------------
    # ------------------------------------       Train/Test Split    ---------------------------------
    # ----------------------------------------------------------------------------------------------------

    # Split the data into training and testing sets before the extracting features
    train_df, test_df = train_test_split(
        df_balanced,
        test_size=0.2,    # 20% of data goes to the test set
        stratify=df_balanced['label_binary'],   # Ensure class balance in both sets
        random_state=1)  # Set seed for reproducibility


    # ----------------------------------------------------------------------------------------------------
    # ------------------------------------       FEATURES ENGINEERING    ---------------------------------
    # ----------------------------------------------------------------------------------------------------

    # -------------------------------------    features by Image Brightness --------------------------------

    # Function to extract the average brightness from an image
    def extract_brightness(img):
        # If the image is missing (None), return a Series with a single None value labeled as 'brightness'
        if img is None:
            return pd.Series([None], index=["brightness"])
        # Convert the image (already a normalized NumPy array) into a NumPy array, if not already
        arr = np.array(img)
        # Calculate the mean pixel value across all channels and pixels — this represents overall brightness
        brightness = arr.mean()
        # Return the brightness as a pandas Series (for compatibility with apply + DataFrame column assignment)
        return pd.Series([brightness], index=["brightness"])


    # Apply the extract_brightness function to each resized image in the DataFrame
    # The result is stored in a new column called 'brightness'

    # Train data
    train_df[['brightness']] = train_df['image_array_resized'].apply(extract_brightness).apply(pd.Series)

    # Test data
    test_df[['brightness']] = test_df['image_array_resized'].apply(extract_brightness).apply(pd.Series)


    # ----------------------------------  Features by pixels  ------------------------------------------

    # Function to extract mean RGB values from an image divided into a grid
    def extract_rgb_grid_features(img, grid_size=(4, 4)):
        # If the image is missing, return a list of None values (one for each expected feature)
        if img is None:
            return [None] * (grid_size[0] * grid_size[1] * 3)

        # Assume the image is already resized to 128x128 and normalized; convert to a NumPy array
        arr = np.array(img)

        # Calculate height and width of each grid cell
        h_step = 128 // grid_size[0]
        w_step = 128 // grid_size[1]
        features = []

        # Loop over the image in grid blocks
        for i in range(0, 128, h_step):   # vertical steps
            for j in range(0, 128, w_step):     # horizontal steps
                patch = arr[i:i + h_step, j:j + w_step, :]    # Extract image patch for this grid cell
            # Calculate the mean of R, G, B channels in the patch
                r_mean = patch[:, :, 0].mean()
                g_mean = patch[:, :, 1].mean()
                b_mean = patch[:, :, 2].mean()
                # Append the mean values to the features list
                features.extend([r_mean, g_mean, b_mean])

        return features


    # Apply the RGB grid feature extractor to each image (produces a list of 4x4x3 = 48 features per image)

    # Train data
    train_grid_features = train_df['image_array_resized'].apply(
        lambda img: extract_rgb_grid_features(img, grid_size=(4, 4)))
    # list of columns

    # Convert the list of features per image into a DataFrame with one column per feature
    train_grid_df = pd.DataFrame(train_grid_features.tolist(), index=train_df.index)

    # Name the columns descriptively: grid_{i}_{C} for each grid cell and color channel
    train_grid_df.columns = [f'grid_{i}_{c}' for i in range(16) for c in ['R', 'G', 'B']]

    # Concatenate the new features back to the original DataFrame
    train_df = pd.concat([train_df, train_grid_df], axis=1)  # combine to the original data

    # Test data
    test_grid_features = test_df['image_array_resized'].apply(
        lambda img: extract_rgb_grid_features(img, grid_size=(4, 4)))
    # list of columns

    # Convert the list of features per image into a DataFrame with one column per feature
    test_grid_df = pd.DataFrame(test_grid_features.tolist(), index=test_df.index)

    # Name the columns descriptively: grid_{i}_{C} for each grid cell and color channel
    test_grid_df.columns = [f'grid_{i}_{c}' for i in range(16) for c in ['R', 'G', 'B']]

    # Concatenate the new features back to the original DataFrame
    test_df = pd.concat([test_df, test_grid_df], axis=1)  # combine to the original data


    # ------------------     Display an image with a 4x4 grid overlay     ------------------
    # Function to plot an image and overlay a 4x4 grid on top of it
    def plot_image_grid(num, df_):
        # Select the image array at the specified index
        img_grid = df_.loc[num]
        # Convert the image (NumPy array format) for plotting
        arr = np.array(img_grid)
        # Set grid dimensions (4x4)
        grid_rows, grid_cols = 4, 4
        # Calculate the height and width of each grid cell
        cell_height = arr.shape[0] // grid_rows
        cell_width = arr.shape[1] // grid_cols

        # Create a matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))
        # Show the image on the axis
        ax.imshow(arr)

        # Draw horizontal grid lines
        for i in range(1, grid_rows):
            ax.axhline(i * cell_height, color='red', linestyle='--', linewidth=1)
        # Draw vertical grid lines
        for j in range(1, grid_cols):
            ax.axvline(j * cell_width, color='red', linestyle='--', linewidth=1)

        # Add numbering to top-left corner of each grid cell
        for i in range(grid_rows):
            for j in range(grid_cols):
                grid_index = i * grid_cols + j
                x = j * cell_width + 3  # Small padding from left
                y = i * cell_height + 2  # Small padding from top
                ax.text(x, y, str(grid_index), color='white', fontsize=10,
                        ha='left', va='top', bbox=dict(facecolor='black', alpha=1.0, boxstyle='round,pad=0.1'))

        # Add a title and remove axes for a cleaner look
        ax.set_title("4x4 RGB Grid Overlay")
        plt.axis('off')
        plt.tight_layout()

        # Show the plot (non-blocking mode for interactive environments)
        plt.show(block=False)


    # Randomly choose one index from the DataFrame to visualize
    random_indices_grid = np.random.choice(train_df.index, size=1)[0]

    # Call the plotting function to display the selected image with a 4x4 grid
    plot_image_grid(random_indices_grid, train_df['image_array_resized'])


    # ------------------------------------        METHODS      ---------------------------------
    # ------------------------------------------------------------------------------------------

    # ----------------------------   Feature Preparation   ----------------------------
    # Identify all features related to the 4x4 RGB grid (48 features total)
    grid_cols = [col for col in train_df.columns if col.startswith("grid_")]
    # Combine grid features with the 'brightness' feature into the final feature list
    features = ['brightness'] + grid_cols

    # Step 2: Grouped statistics
    grouped_stats = train_df.groupby('label_binary')[features].agg(['mean', 'std'])

    # Step 3: Rename columns using level names
    grouped_stats.columns = [f"{feature}_{stat}" for feature, stat in grouped_stats.columns]

    # Step 4: Separate stats for each class
    stats_class_0 = grouped_stats.loc[0]
    stats_class_1 = grouped_stats.loc[1]

    # Step 5: Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'mean_class_0': stats_class_0[[col for col in stats_class_0.index if col.endswith('_mean')]].values,
        'mean_class_1': stats_class_1[[col for col in stats_class_1.index if col.endswith('_mean')]].values,
        'std_class_0': stats_class_0[[col for col in stats_class_0.index if col.endswith('_std')]].values,
        'std_class_1': stats_class_1[[col for col in stats_class_1.index if col.endswith('_std')]].values,
    }, index=[col.replace('_mean', '') for col in stats_class_0.index if col.endswith('_mean')])

    # Step 6: Add absolute mean difference
    comparison_df['abs_mean_diff'] = abs(comparison_df['mean_class_0'] - comparison_df['mean_class_1'])

    # Step 7: Sort by most discriminative features
    feature_stats_sorted = comparison_df.sort_values(by='abs_mean_diff', ascending=False).round(3)

    # Display top features
    display(feature_stats_sorted.head(49))

    # Number of top features to show
    top_n = 10

    # Select top N features by absolute mean difference
    top_features = feature_stats_sorted.head(top_n).index.tolist()

    # Step 1: Create a mapping for clearer labels
    train_df['label_name'] = train_df['label_binary'].map({0: 'Other', 1: 'Cardboard Box'})

    # Step 2: Melt the dataframe
    df_melted = train_df[['label_name'] + top_features].melt(id_vars='label_name',
                                                                var_name='Feature',
                                                                value_name='Value')

    # Step 3: Plot with correct legend labels
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x='Feature', y='Value', hue='label_name', palette='Set2')
    plt.title(f'Top {len(top_features)} Features by Absolute Mean Difference')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.legend(title='Class')  # cleaner title
    plt.xticks(rotation=45)
    plt.legend(title='Class', loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2)

    plt.tight_layout()
    plt.show(block=False)


    X_all_features = train_df[features]  # Features matrix: all columns except the target

    # ----------------------------   X, Y division   ----------------------------
    X_train = train_df[features]
    y_train = train_df['label_binary']

    X_test = test_df[features]
    y_test = test_df['label_binary']

    # ----------------------------   Standardization of Features   ----------------------------
    # Initialize a standard scaler to normalize features (zero mean, unit variance)
    scaler_full = StandardScaler()

    # Fit the scaler on the training data and transform both training and test sets
    X_train_scaled = scaler_full.fit_transform(X_train)
    X_test_scaled = scaler_full.transform(X_test)

    # Convert scaled arrays back to DataFrames to preserve column names and indices
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


    # ------------------------------------------------------------------------------------------
    #                                   Logistic Regression
    # ------------------------------------------------------------------------------------------
    # ----------------------------   Base Model Definition   ----------------------------
    # Initialize a basic logistic regression lasso with high max_iter to ensure convergence
    base_model = LogisticRegression(max_iter=10000,random_state=42)

    # ----------------------------   Feature Selection: Forward Selection   ----------------------------
    # Perform forward feature selection to choose the most predictive features
    # 'best' means the number of selected features will be chosen based on highest CV accuracy
    # cv=5 implies 5-fold cross-validation for evaluating each feature subset

    forward_selector = SFS(base_model,
                           k_features='best',  # Select best-performing subset of features
                           forward=True,  # Add features one at a time (forward selection)
                           floating=False,  # Disable backward steps (no floating selection)
                           scoring='accuracy',  # Use accuracy as the evaluation metric
                           cv=5,  # Cross-validation folds
                           n_jobs=1)  # Run in a single process

    # Fit the feature selector on the training data
    forward_selector = forward_selector.fit(X_train_scaled, y_train)

    # Get the indices and names of the selected features
    selected_feature_indices = list(forward_selector.k_feature_idx_)
    selected_feature_names_lr = [features[i] for i in selected_feature_indices]
    print("Selected features forward selection:", selected_feature_names_lr)

    len(selected_feature_names_lr)
    # ----------------------------   Final Model Training with Selected Features   ----------------------------

    # Filter training and test sets to include only the selected features
    X_train_selected = X_train_scaled[selected_feature_names_lr]
    X_test_selected = X_test_scaled[selected_feature_names_lr]

    # Initialize and train the final logistic regression lasso
    final_model = LogisticRegression(max_iter=1000, random_state=42)
    final_model.fit(X_train_selected, y_train)

    # ----------------------------   Feature Importance Reporting   ----------------------------
    # Extract the coefficients from the trained lasso and associate them with feature names
    coefficients = final_model.coef_[0]
    feature_importance = pd.Series(coefficients, index=selected_feature_names_lr).sort_values()
    print("Feature weights (Forward Selection):")
    print(feature_importance)

    # ----------------------------   Model Evaluation on Test Set   ----------------------------
    # Predict the labels for the test set
    y_test_pred = final_model.predict(X_test_selected)

    # Print confusion matrix and classification metrics (precision, recall, f1-score)
    print("Logistic Regression (Forward Selection) Results:")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

    # ----------------------------    Feature Importance plot   ----------------------------
    top_n = 7

    # Sort by absolute value but keep the original signed values
    top_features_forward = feature_importance.reindex(
        feature_importance.abs().sort_values(ascending=False).head(top_n).index
    )

    # Plot a horizontal bar chart of the top N feature importances (real values)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features_forward, y=top_features_forward.index, hue=top_features_forward.head(top_n),palette="viridis", legend = False)
    plt.title(f'Top {top_n} Strongest Feature Coefficients - Logistic Regression (Forward Selection)')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show(block=False)

    # ----------------------------   # Metrics for comparison #   ----------------------------
    accuracy_logis = accuracy_score(y_test, y_test_pred)
    preci_logis = precision_score(y_test, y_test_pred)
    recall_logis = recall_score(y_test, y_test_pred)
    f1_logis = f1_score(y_test, y_test_pred)
    y_proba_logis = final_model.predict_proba(X_test_selected)[:, 1] # so the number of features match with the fitting of the model and no error is made
    auc_logis = roc_auc_score(y_test, y_proba_logis)


    # ------------------------------------------------------------------------------------------
    #                                    Lasso
    # ------------------------------------------------------------------------------------------

    # Define a grid of inverse regularization strengths: C = 1 / λ
    # Smaller C = stronger regularization, larger C = weaker regularization
    Cs = np.logspace(-3, 1, 10)  # Generates 10 values from 0.001 to 10 on a log scale

    # Initialize a LogisticRegressionCV lasso with:
    # - L1 regularization (Lasso)
    # - 'liblinear' solver (supports L1)
    # - 5-fold cross-validation
    # - Accuracy as the scoring metric
    lasso = LogisticRegressionCV(
        Cs=Cs,
        cv=5,
        penalty='l1',
        solver='liblinear',  #  Must use liblinear for L1 penalty
        scoring='accuracy',
        random_state=42,
        max_iter=1000)

    # Fit the lasso using the scaled training data
    lasso.fit(X_train_scaled, y_train)

    # Print the best value of C (i.e., 1 / λ) chosen via cross-validation
    print(f"Best C (1/λ) from cross-validation: {lasso.C_[0]}")

    # ----------------------------   Evaluation on Test Set   ----------------------------
    # Predict labels for the test set
    y_pred = lasso.predict(X_test_scaled)

    # Print evaluation metrics
    print("Logistic Regression (Lassso) Evaluation Metrics:")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ----------------------------    Final Model Refit Using Selected C   ----------------------------
    # Extract the best C from CV
    final_lambda = lasso.C_[0]

    # Re-initialize LogisticRegressionCV using the selected value of C
    lasso_final = LogisticRegressionCV(
        Cs=[final_lambda],   # Use only the best C
        cv=5,
        penalty='l1',
        solver='liblinear',  # liblinear supports L1
        scoring='accuracy',
        random_state=42,
        max_iter=1000
    )

    # Fit the final model on the full training data
    lasso_final.fit(X_train_scaled, y_train)

    # ----------------------------   Analyze Feature Coefficients   ----------------------------

    # Extract the model coefficients (one per feature)
    coefs = lasso_final.coef_[0]

    # Create a Pandas Series for easier interpretation
    all_features_lasso = pd.Series(coefs, index=X_train.columns).sort_values()

    # Filter out features with non-zero coefficients (selected by Lasso)
    selected_features_lasso = all_features_lasso[all_features_lasso != 0]

    # Display selected features and their corresponding weights
    print("Selected features (Lasso) with weights:")
    print(selected_features_lasso)
    len(selected_features_lasso)

    # Predict using the refit model
    y_pred_final = lasso_final.predict(X_test_scaled)

    # Final performance metrics
    print("Logistic Regression (Lassso) Results Metrics:")
    print(confusion_matrix(y_test, y_pred_final))
    print(classification_report(y_test, y_pred_final))

    # ----------------------------    Feature Importance plot   ----------------------------

    top_n = 7

    # Sort by absolute value but keep the sign
    top_features_lasso = selected_features_lasso.reindex(
        selected_features_lasso.abs().sort_values(ascending=False).head(top_n).index
    )

    # Plot a horizontal bar chart of the top N feature importances (real values)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features_lasso, y=top_features_lasso.index, hue=top_features_lasso.head(top_n),palette="viridis", legend = False)
    plt.title(f'Top {top_n} Strongest Feature Coefficients - Lasso Logistic Regression')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show(block=False)


    # ----------------------------   # Metrics for comparison #   ----------------------------
    accuracy_logis_lasso = accuracy_score(y_test, y_pred_final)
    preci_logis_lasso = precision_score(y_test, y_pred_final)
    recal_logis_lasso = recall_score(y_test, y_pred_final)
    f1_logis_lasso = f1_score(y_test, y_pred_final)
    y_proba_logis_lasso = lasso_final.predict_proba(X_test)[:, 1]
    auc_logis_lasso = roc_auc_score(y_test, y_proba_logis_lasso)


    # ------------------------------------------------------------------------------------------
    #                                    Random Forest
    # ------------------------------------------------------------------------------------------

    # Initialize a Random Forest classifier with:
    # - 100 decision trees (n_estimators)
    # - a fixed random seed for reproducibility
    # - a fixed random seed for reproducibility
    rf = RandomForestClassifier(n_estimators=100,    # 100 decision trees (n_estimators)
                                criterion='gini',    # measure of total variance across the K classes
                                max_features='sqrt', # leads to a reduction in both test error and OOB(out of bags) error over bagging
                                min_samples_split=5, # as mentioned in tibshirani's book.
                                bootstrap=True,      # so we use the bagging (boostrap aggregation) style but taking into account sqrt(p).
                                random_state=42)
    # Train the model using the original (unscaled) training data
    rf.fit(X_train, y_train)
    # Predict class labels for the test set
    y_pred_rforest = rf.predict(X_test)

    # Calculate and print the overall accuracy
    print("Random Forest Results Metrics:")

    print("Accuracy:", accuracy_score(y_test, y_pred_rforest))

    # Print detailed classification metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred_rforest))

    # Feature importance for Random Forest
    importances_rf = rf.feature_importances_
    rf_feature_importance = pd.Series(importances_rf, index=X_train.columns).sort_values(ascending=False)
    print("Feature importance random forest:")
    print(rf_feature_importance)

    # Choose how many top features to display
    top_n = 7

    # Plot a horizontal bar chart of the top N feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rf_feature_importance.head(top_n), y=rf_feature_importance.head(top_n).index,hue = rf_feature_importance.head(top_n), palette="viridis", legend = False)
    plt.title(f'Top {top_n} Feature Importances - Random Forest')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show(block=False)

    # ----------------------------   # Metrics for comparison #   ----------------------------
    accuracy_rf = accuracy_score(y_test, y_pred_rforest)
    preci_rf = precision_score(y_test, y_pred_rforest)
    recal_rf = recall_score(y_test, y_pred_rforest)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    f1_rf = f1_score(y_test, y_pred_rforest)
    auc_rf = roc_auc_score(y_test, y_proba_rf)


    # ------------------------------------------------------------------------------------------
    #                                    GradientBoostingClassifier
    # ------------------------------------------------------------------------------------------

    from sklearn.metrics import accuracy_score, roc_auc_score

    from sklearn.ensemble import GradientBoostingClassifier
    # 3. Fit Gradient Boosting Model
    clf = GradientBoostingClassifier(n_estimators=100,  # 100 decision trees (n_estimators)
                                     learning_rate=0.1, # shrinkage parameter - This controls the rate at which boosting learns
                                     max_depth=3,       # Maximum depth of the individual regression estimators.
                                     random_state=42)
    clf.fit(X_train, y_train)

    # 4. Predict and evaluate
    y_pred_clf = clf.predict(X_test)
    y_proba_clf = clf.predict_proba(X_test)[:, 1]

    print("Gradient Boosting Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_clf))
    print("AUC:", roc_auc_score(y_test, y_proba_clf))

    # Feature importance for Gradient Boosting
    importances_clf = clf.feature_importances_
    gb_feature_importance = pd.Series(importances_clf, index=X_train.columns).sort_values(ascending=False)
    print("Feature importance Gradient Boosting:")
    print(gb_feature_importance)

    # Choose how many top features to display
    top_n = 7

    # Plot a horizontal bar chart of the top N feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x=gb_feature_importance.head(top_n), y=gb_feature_importance.head(top_n).index,hue=gb_feature_importance.head(top_n), palette="viridis", legend = False)
    plt.title(f'Top {top_n} Feature Importances - Gradient Boosting')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show(block=False)

    # ----------------------------   # Metrics for comparison #   ----------------------------
    accuracy_clf = accuracy_score(y_test, y_pred_clf)
    preci_clf = precision_score(y_test, y_pred_clf)
    recal_clf = recall_score(y_test, y_pred_clf)
    f1_clf = f1_score(y_test, y_pred_clf)
    y_proba_clf = clf.predict_proba(X_test)[:, 1]
    auc_clf = roc_auc_score(y_test, y_proba_clf)

    # ------------------------------------------------------------------------------------------
    #                                    Support Vector Machine
    # ------------------------------------------------------------------------------------------

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, roc_auc_score

    # 3. Train SVM classifier (with probability estimates)
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm.fit(X_train, y_train)

    # 4. Predict and evaluate
    y_pred_svm = svm.predict(X_test)

    # permutation_importance
    from sklearn.inspection import permutation_importance

    result_svm = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42)
    svm_importance = pd.Series(result_svm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("Feature importance SVM:")
    print(svm_importance)

    # Choose how many top features to display
    top_n = 3

    # Plot a horizontal bar chart of the top N feature importances
    plt.figure(figsize=(10, 2))
    sns.barplot(x=svm_importance.head(top_n), y=svm_importance.head(top_n).index,hue=svm_importance.head(top_n).index ,palette="viridis",legend = False)
    plt.title(f'Top {top_n} Feature Importances - SVM')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show(block=False)

    # ----------------------------   # Metrics for comparison #   ----------------------------
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    preci_svm = precision_score(y_test, y_pred_svm)
    recal_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    y_proba_svm = svm.predict_proba(X_test)[:, 1]
    auc_svm = roc_auc_score(y_test, y_proba_svm)

# ------------------------------------------------------------------------------------------
#                                   Comparison Plots
# ------------------------------------------------------------------------------------------
# Model names
models = ['Logistic\nRegression', 'Lasso\nRegression', 'Random\nForest',
          'Gradient\nBoosting', 'Support\nVector']

# Metric values for each model
accuracy = [accuracy_logis, accuracy_logis_lasso, accuracy_rf, accuracy_clf, accuracy_svm]
precision = [preci_logis, preci_logis_lasso, preci_rf,preci_clf, preci_svm]
recall = [recall_logis, recal_logis_lasso,recal_rf, recal_clf, recal_svm]
f1 = [f1_logis, f1_logis_lasso, f1_rf, f1_clf,f1_svm]
auc = [auc_logis, auc_logis_lasso, auc_rf, auc_clf, auc_svm]

# Set width of bars and positions
bar_width = 0.15
x = np.arange(len(models))
colors = plt.get_cmap('tab20').colors

# Plot grouped bar chart
plt.figure(figsize=(12, 6))
plt.bar(x - 2*bar_width, accuracy, width=bar_width, label='Accuracy', color=colors[0])
plt.bar(x - bar_width, precision, width=bar_width, label='Precision', color=colors[1])
plt.bar(x, recall, width=bar_width, label='Recall',color=colors[2])
plt.bar(x + bar_width, f1, width=bar_width, label='F1 Score',color=colors[3])
plt.bar(x + 2*bar_width, auc, width=bar_width, label='AUC', color=colors[4])

# Chart formatting
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Comparison of Classification Models by Performance Metrics')
plt.xticks(x, models)
plt.ylim(0.85, 1.05)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=5)
plt.grid(axis='y', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show(block=False)
plt.pause(60)
