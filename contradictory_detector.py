import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import os

hparams = {
    # Task specific constants
    "TASK_PROP__NUM_CLASSES": 3,

    # Dataset preprocessing
    "VOCAB_SIZE": 10000,
    "EMBEDDING_DIM": 128,
    "MAX_LENGTH": 32,
    "TRAINING_SPLIT": 0.9,
    "BATCH_SIZE": 128,

    # Model params
    "OPTIMIZER_TYPE": 'adam',
    "LOSS_FUNCTION": 'sparse_categorical_crossentropy',
    "EMBEDDING_MODEL": "https://tfhub.dev/google/nnlm-en-dim50/2",
    "LSTM_LAYER": 64,

    # Training
    "LEARNING_RATE": 0.0001,
    "EARLY_STOP_PATIENCE": 15,
    "REDUCE_LR_PATIENCE": 5,
    "REDUCE_LR_FACTOR": 0.2,
    "REDUCE_LR_MIN_LR": 0.00001,
    "EPOCHS": 100
}

def read_dataset_from_csv(csv_path, training_data):
    df = pd.read_csv(csv_path)
    df.head()

    # Standardize labels so they have 0 for negative and 1 for positive
    if training_data:
        labels = df['label']#.apply(lambda x: x.to_numpy())

    # Since the original dataset does not provide headers you need to index the columns by their index
    premise = df['premise'].to_numpy()
    hypothesis = df['hypothesis'].to_numpy()
    lang_abv = df['lang_abv'].to_numpy()

    unique_languages = df['language'].unique()
    print(f"Unique languages found: {unique_languages}")
    # Create a vocabulary mapping each language to an integer
    language_vocab = {language: i for i, language in enumerate(unique_languages)}
    print(f"Language vocabulary: {language_vocab}")
    # Create the new 'languages' column by mapping the vocabulary
    language = df['language'].map(language_vocab)

    # Create the dataset
    if training_data:
        dataset = tf.data.Dataset.from_tensor_slices(((premise,hypothesis,lang_abv), labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices(((premise,hypothesis,lang_abv)))

    # Get the first 5 elements of the dataset
    examples = list(dataset.take(5))

    dataset_type_str = "Training" if training_data else "Test"

    print(f"{dataset_type_str} dataset contains {len(dataset)} examples\n")

    if training_data:
        print(f"Text of second example look like this: {examples[1][0]}\n")
    else:
        print(f"Text of second example look like this: {examples[1][0].numpy().decode('utf-8')}\n")
    #print(f"Labels of first 5 examples look like this: {[x[1].numpy() for x in examples]}")

    return dataset

def create_pretrained_embedding_layer(embedding_file_path, text_vectorizer, vocab_size, embedding_dim):
    """
    Creates a Keras Embedding layer initialized with pre-trained weights.

    Args:
        embedding_file_path (str): The full path to the pre-trained embeddings file (e.g., GloVe, Word2Vec).
        text_vectorizer (tf.keras.layers.TextVectorization): The adapted text vectorizer layer.
        vocab_size (int): The size of the vocabulary.
        embedding_dim (int): The dimension of the embeddings.

    Returns:
        tf.keras.layers.Embedding: A non-trainable Keras Embedding layer.
    """
    # --- 1. Load Pre-trained Embeddings into a Dictionary ---
    print(f"Loading pre-trained embeddings from {embedding_file_path}...")
    pretrained_embeddings = {}
    try:
        with open(embedding_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                pretrained_embeddings[word] = coefs
    except FileNotFoundError:
        print(f"Error: Embedding file not found at {embedding_file_path}")
        return None
    print(f"Loaded {len(pretrained_embeddings)} word vectors.")

    # --- 2. Create Word Index from the Vectorizer ---
    word_index = {word: i for i, word in enumerate(text_vectorizer.get_vocabulary())}

    # --- 3. Create the Embedding Matrix ---
    print("Creating embedding matrix...")
    embeddings_matrix = np.zeros((vocab_size, embedding_dim))
    
    # Iterate through the vocabulary and populate the matrix
    for word, i in word_index.items():
        embedding_vector = pretrained_embeddings.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all-zeros.
            embeddings_matrix[i] = embedding_vector
    
    print("Embedding matrix created successfully.")

    # --- 4. Create the Keras Embedding Layer ---
    embedding_layer = tf.keras.layers.Embedding(
        vocab_size,
        embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(embeddings_matrix),
        trainable=False  # Freeze the layer to keep the pre-trained weights
    )

    return embedding_layer

def create_and_compile_model(vocab_size, embedding_dim, max_length, embedding_file_path, lstm_layer, learning_rate):

    """
    Builds a Siamese-like model with two parallel Bidirectional LSTMs.

    Args:
        vocab_size (int): The size of the vocabulary for the embedding layer.
        embedding_dim (int): The dimension of the embedding vectors.
        max_length (int): The maximum length of the input sequences.
        num_classes (int): The number of output classes for categorization.

    Returns:
        tf.keras.Model: A compiled Keras model.
    """
    # --- 1. Define the two input layers for the two phrases ---
    # Each input will be a sequence of integers of shape (max_length,)
    print(f"max_length {max_length}")

    input_a = tf.keras.layers.Input(shape=(max_length,), name='input_phrase_1')
    input_b = tf.keras.layers.Input(shape=(max_length,), name='input_phrase_2')

    # --- 2. Create Shared Layers ---
    # It's crucial to use shared layers so that both phrases are processed
    # in the exact same way and their resulting vectors are comparable.
    #shared_embedding = create_pretrained_embedding_layer(embedding_file_path, vectorizer, VOCAB_SIZE, EMBEDDING_DIM)
    embedding_url = embedding_file_path

    # 2. Create the embedding layer
    # This layer takes string inputs directly, so you don't need a TextVectorization layer beforehand.
    hub_embedding_layer = hub.KerasLayer(
        embedding_url,
        input_shape=[], # The layer expects a 1D tensor of strings
        dtype=tf.string,
        trainable=True # Set to True to fine-tune the embeddings for your specific task
    )
    shared_embedding = hub_embedding_layer

    shared_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_layer))

    # --- 3. First LSTM Branch ---
    # Pass the first input through the shared layers
    encoded_a = shared_embedding(input_a)
    encoded_a = shared_lstm(encoded_a)

    # --- 4. Second LSTM Branch ---
    # Pass the second input through the same shared layers
    encoded_b = shared_embedding(input_b)
    encoded_b = shared_lstm(encoded_b)

    # --- 5. Concatenate the Outputs ---
    # Combine the outputs of both LSTM branches to be fed into the classifier
    concatenated = tf.keras.layers.concatenate([encoded_a, encoded_b], name='concatenated_layer')

    # --- 6. Add the Classifier (Dense Layers) ---
    # One or two dense layers to learn the relationship between the two phrases
    dense_1 = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(concatenated)
    dropout = tf.keras.layers.Dropout(0.5, name='dropout')(dense_1)
    
    # --- 7. Define the Final Output Layer ---
    # For multi-class classification (e.g., contradiction, neutral, entailment)
    output = tf.keras.layers.Dense(TASK_PROP__NUM_CLASSES, activation='softmax', name='output')(dropout)

    # --- 8. Build and Compile the Final Model ---
    model = tf.keras.Model(inputs=[input_a, input_b], outputs=output)
  
    if hparams['OPTIMIZER_TYPE'] == 'adam':
        optimizer =tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                        #beta_1=0.9,
                                        #beta_2=0.999,
                                        #epsilon=1e-07,
                                        #amsgrad=False,
                                        #weight_decay=None,
                                        #clipnorm=None,
                                        #clipvalue=None,
                                        #global_clipnorm=None,
                                        #use_ema=False,
                                        #ema_momentum=0.99,
                                        #ema_overwrite_frequency=None,
                                        #loss_scale_factor=None,
                                        #gradient_accumulation_steps=None,
                                        name='adam',
                                        #**kwargs
                                    ) 
        
    if hparams['OPTIMIZER_TYPE'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss=hparams['LOSS_FUNCTION'],
                  #loss_weights=None,
                  metrics=['accuracy'],
                  #weighted_metrics=None,
                  #run_eagerly=False,
                  #steps_per_execution=1,
                  #jit_compile='auto',
                  #auto_scale_loss=True
                  )
    
    return model
    

def plot_training_charts(training_history, json_log_path, output_filename):
    """
    Saves the Keras training history to JSON and plots the loss and
    accuracy metrics as an interactive HTML chart. This version displays
    only curves and includes the best values in the legend.

    Args:
        training_history: A Keras History object from model.fit().
        json_log_path (str): Path to save the history log.
        output_filename (str): Path to save the interactive HTML chart.
    """
    history_dict = training_history.history
    serializable_history = {key: [float(value) for value in values] for key, values in history_dict.items()}

    # --- 1. Save the history dictionary to a JSON file ---
    print(f"Saving training history log to {json_log_path}...")
    try:
        with open(json_log_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print("History log saved successfully.")
    except IOError as e:
        print(f"Error saving history log: {e}")

    # --- 2. Create the interactive plot ---
    print(f"\nVisualizing training history and saving to {output_filename}...")
    
    if not history_dict or 'loss' not in history_dict or 'accuracy' not in history_dict:
        print("Warning: History is missing 'loss' or 'accuracy' keys. Cannot plot chart.")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model Accuracy'))
    epochs = list(range(1, len(history_dict['loss']) + 1))

    # --- Plot Loss (Training vs. Validation) ---
    min_loss = min(history_dict['loss'])
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=history_dict['loss'], 
        name=f'Training Loss (Min: {min_loss:.4f})', 
        mode='lines'  # Changed from 'lines+markers' to 'lines'
    ), row=1, col=1)
    
    if 'val_loss' in history_dict:
        min_val_loss = min(history_dict['val_loss'])
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=history_dict['val_loss'], 
            name=f'Validation Loss (Min: {min_val_loss:.4f})', 
            mode='lines'
        ), row=1, col=1)

    # --- Plot Accuracy (Training vs. Validation) ---
    max_accuracy = max(history_dict['accuracy'])
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=history_dict['accuracy'], 
        name=f'Training Accuracy (Max: {max_accuracy:.4f})', 
        mode='lines'
    ), row=1, col=2)
    
    if 'val_accuracy' in history_dict:
        max_val_accuracy = max(history_dict['val_accuracy'])
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=history_dict['val_accuracy'], 
            name=f'Validation Accuracy (Max: {max_val_accuracy:.4f})', 
            mode='lines'
        ), row=1, col=2)

    # --- Update layout, titles, and legend ---
    fig.update_layout(
        title_text="Model Training History",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # --- 3. Save the figure to an HTML file ---
    try:
        fig.write_html(output_filename)
        print(f"Interactive chart saved successfully to {output_filename}")
    except IOError as e:
        print(f"Error saving chart: {e}")

def save_predictions_to_csv(predictions, output_path):
    """
    Saves the model's predictions to a CSV file in the Kaggle submission format.

    Args:
        predictions (np.ndarray): The output from the model.predict() method. 
                                  This is expected to be a 2D array where each
                                  row contains the probabilities for each class.
        output_path (str): The file path where the submission CSV will be saved.
    """
    # --- 1. Get the predicted label for each prediction ---
    # The model outputs probabilities for each of the 10 digits.
    # np.argmax finds the index of the highest probability for each row (axis=1).
    # This index corresponds to the predicted digit (0-9).
    predicted_labels = np.argmax(predictions, axis=1)

    # --- 2. Create the ImageId column ---
    # Kaggle submissions typically have an 'ImageId' that starts from 1.
    num_predictions = len(predicted_labels)
    image_ids = np.arange(1, num_predictions + 1)

    # --- 3. Create a pandas DataFrame ---
    # The DataFrame will have the required 'ImageId' and 'Label' columns.
    submission_df = pd.DataFrame({
        'ImageId': image_ids,
        'Label': predicted_labels
    })

    # --- 4. Save the DataFrame to a CSV file ---
    # index=False is crucial to prevent pandas from writing the DataFrame
    # index as an extra column in the file.
    try:
        submission_df.to_csv(output_path, index=False)
        print(f"Successfully saved {num_predictions} predictions to {output_path}")
    except IOError as e:
        print(f"Error saving file: {e}")

def fit_vectorizer(dataset):
    """
    Adapts the TextVectorization layer on the training sentences
    
    Args:
        dataset (tf.data.Dataset): Tensorflow dataset with training sentences.
    
    Returns:
        tf.keras.layers.TextVectorization: an instance of the TextVectorization class adapted to the training sentences.
    """    
    
    # Instantiate the TextVectorization class, defining the necessary arguments alongside their corresponding values
    vectorizer = tf.keras.layers.TextVectorization( 
        standardize='lower_and_strip_punctuation',
        output_sequence_length=MAX_LENGTH
    ) 
    
    # Fit the tokenizer to the training sentences
    vectorizer.adapt(dataset)
    
    return vectorizer

if __name__ == '__main__':

    try:
        current_dir = os.getcwd()
        print(f"Current dir: {current_dir}")

        TRAIN_DATA_CSV_PATH = os.path.join(current_dir, "train.csv")
        TEST_DATA_CSV_PATH = os.path.join(current_dir, "test.csv")
        
        print(f"Training dataset csv path: {TRAIN_DATA_CSV_PATH}")
        print(f"Test dataset csv path: {TEST_DATA_CSV_PATH}")

        print()

        print("Decode test dataset")
        test_dataset = read_dataset_from_csv(TEST_DATA_CSV_PATH, training_data = 0)
        print("Decode train dataset")
        train_dataset_csv = read_dataset_from_csv(TRAIN_DATA_CSV_PATH, training_data = 1)

        # Calculate the number of elements for the training set
        train_size = int(hparams['TRAINING_SPLIT'] * len(train_dataset_csv))

        # Create the training dataset by taking the first 'train_size' elements
        train_dataset = train_dataset_csv.take(train_size)

        # Create the validation dataset by skipping the first 'train_size' elements
        validation_dataset = train_dataset_csv.skip(train_size)

        SHUFFLE_BUFFER_SIZE = 1000 #TODO this may be tuned
        PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
        train_dataset_final = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(hparams['BATCH_SIZE'])
        validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(hparams['BATCH_SIZE'])

        print(f"Buffered {SHUFFLE_BUFFER_SIZE} elements for the training dataset.")

        print()

        print(f"There are {len(train_dataset_final)} batches for a total of {hparams['BATCH_SIZE']*len(train_dataset_final)} elements for training.\n")
        print(f"There are {len(validation_dataset_final)} batches for a total of {hparams['BATCH_SIZE']*len(validation_dataset_final)} elements for validation.\n")

        print()

        print(f"Create and compile model")
        nn_model = create_and_compile_model(hparams['VOCAB_SIZE'], hparams['EMBEDDING_DIM'], hparams['MAX_LENGTH'], hparams['EMBEDDING_MODEL'], hparams['LSTM_LAYER'], hparams['LEARNING_RATE'])

        nn_model.summary()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',    # Monitor the validation loss
            patience=hparams['EARLY_STOP_PATIENCE'],            # Stop if val_loss doesn't improve for 5 epochs
            restore_best_weights=True # Restore model weights from the epoch with the best val_loss
        )

        learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(   monitor='val_accuracy',
                                                factor=hparams['REDUCE_LR_FACTOR'],
                                                patience=hparams['REDUCE_LR_PATIENCE'],
                                                #verbose=0,
                                                #mode='auto',
                                                #min_delta=0.0001,
                                                #cooldown=0,
                                                min_lr=hparams['REDUCE_LR_MIN_LR'],
                                                #**kwargs
                                            )

        training_history = nn_model.fit(x=train_dataset_final,
                                        #y=None,
                                        #batch_size=None,
                                        epochs=hparams['EPOCHS'],
                                        #verbose='auto',
                                        callbacks=[early_stopping_callback, learning_rate_scheduler],
                                        #validation_split=0.0,
                                        validation_data=validation_dataset_final,
                                        #shuffle=True,
                                        #class_weight=None,
                                        #sample_weight=None,
                                        #initial_epoch=0,
                                        #steps_per_epoch=None,
                                        #validation_steps=None,
                                        #validation_batch_size=None,
                                        #validation_freq=1
                                        ) 
        
        print()

        history_log_path = os.path.join(current_dir, "training_history.json")
        loss_acc_chart_path = os.path.join(current_dir, "training_loss_acc_chart.html")
        plot_training_charts(training_history, history_log_path, loss_acc_chart_path)

        SAVED_MODEL_PATH = os.path.join(current_dir, "trained_model_complete.h5")
        nn_model.save(SAVED_MODEL_PATH)
        print("\nModel saved to {SAVED_MODEL_PATH}")

        test_dataset = create_test_dataset(test_data_folder_path)
        predictions = nn_model.predict(test_dataset, #x
                                       #batch_size=None, 
                                       verbose=False, 
                                       #steps=None, 
                                       #callbacks=None
                                       )

        prediction_csv_path = os.path.join(current_dir, "submission.csv")
        save_predictions_to_csv(predictions, prediction_csv_path)

    except ValueError as e:
        print(f"\nError: {e}")