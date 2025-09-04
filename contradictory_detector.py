import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_text as text
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

hparams = {
    # Task specific constants
    "TASK_PROP__NUM_CLASSES": 3,

    # Dataset preprocessing
    #"VOCAB_SIZE": 10000,
    #"EMBEDDING_DIM": 128,
    #"MAX_LENGTH": 32,
    "TRAINING_SPLIT": 0.9,
    "BATCH_SIZE": 16,

    # Model params
    "OPTIMIZER_TYPE": 'adam',
    "LOSS_FUNCTION": 'sparse_categorical_crossentropy',
    "EMBEDDING_MODEL_URL": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
    "DENSE_LAYERS": 2,
    "DENSE_UNITS_1": 256,
    "DENSE_UNITS_2": 128,
    #"LSTM_LAYER": 32,
    "L2_REG_RATE": 0.005,
    "DROPOUT": 0.6,
    #"KERNEL_INITIALIZER": 'glorot_uniform', # Added weight initializer. #'he_normal'
    #"BIAS_INITIALIZER": 'zeros',         # Added bias initializer

    # Training
    "LEARNING_RATE": 0.0001,
    "EARLY_STOP_PATIENCE": 20,
    "REDUCE_LR_PATIENCE": 5,
    "REDUCE_LR_FACTOR": 0.2,
    "REDUCE_LR_MIN_LR": 0.00001,
    "EPOCHS": 100
}

def read_data_from_csv(csv_path, is_training_data=True):
    """
    Reads a CSV file and returns its columns as numpy arrays.
    """
    df = pd.read_csv(csv_path)
    
    premise = df['premise'].to_numpy()
    hypothesis = df['hypothesis'].to_numpy()
    lang_abv = df['lang_abv'].to_numpy()

    if is_training_data:
        labels = df['label'].to_numpy()
        return premise, hypothesis, lang_abv, labels
    else:
        # For test data, also return the 'id' column for the submission file
        ids = df['id'].to_numpy()
        return premise, hypothesis, lang_abv, ids

    
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

def create_and_compile_model(embedding_url, dense_layers, dense_units, dense_units_2, learning_rate, l2_rate, dropout_param):
    """
    Builds a Siamese-like model using a pre-trained multilingual sentence encoder.
    This model is simpler and more powerful for multi-language tasks.
    """
    # --- 1. Define the two input layers for raw text strings ---
    input_premise = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input_premise')
    input_hypothesis = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input_hypothesis')

    # --- 2. Create the Shared Multilingual Embedding Layer from TensorFlow Hub ---
    # This layer handles tokenization and embedding for 16 languages.
    shared_embedding_layer = hub.KerasLayer(embedding_url, trainable=False, name='universal_sentence_encoder')

    # --- 3. Process Both Inputs with the Shared Layer ---
    embedded_premise = shared_embedding_layer(input_premise)
    embedded_hypothesis = shared_embedding_layer(input_hypothesis)

    # --- 4. Concatenate the Sentence Embeddings ---
    concatenated = tf.keras.layers.concatenate(
        [embedded_premise, embedded_hypothesis], name='concatenated_layer'
    )

    # --- 5. Add the Classifier (Dense Layers) ---
    dense_1 = tf.keras.layers.Dense(dense_units, 
                                     activation='relu', 
                                     #kernel_initializer=kernel_initializer,
                                     #bias_initializer=bias_initializer,
                                     name='dense_1', 
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(concatenated)
    dropout_1 = tf.keras.layers.Dropout(dropout_param, name='dropout_1')(dense_1)

    if dense_layers>1:
        dense_2 = tf.keras.layers.Dense(dense_units_2, 
                                     activation='relu',
                                     #kernel_initializer=kernel_initializer,
                                     #bias_initializer=bias_initializer,
                                     name='dense_2', 
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(dropout_param, name='dropout_2')(dense_2)
    
    # --- 6. Define the Final Output Layer ---
    output = tf.keras.layers.Dense(hparams['TASK_PROP__NUM_CLASSES'], activation='softmax', name='output')(dropout_2 if dense_layers>1 else dropout)

    # --- 7. Build and Compile the Final Model ---
    model = tf.keras.Model(inputs=[input_premise, input_hypothesis], outputs=output)
 
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
        
    elif hparams['OPTIMIZER_TYPE'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    else:
        print(f"Warning: Optimizer type '{hparams['OPTIMIZER_TYPE']}' not recognized. Defaulting to Adam.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
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

def save_predictions_to_csv(predictions, ids, output_path):
    """
    Saves the model's predictions to a CSV file in the required submission format.

    Args:
        predictions (np.ndarray): The output from the model.predict() method.
        ids (np.ndarray): The array of ids corresponding to the predictions.
        output_path (str): The file path where the submission CSV will be saved.
    """
    # Get the predicted label for each prediction
    predicted_labels = np.argmax(predictions, axis=1)

    # Create a pandas DataFrame with the specified column names
    submission_df = pd.DataFrame({
        'id': ids,
        'prediction': predicted_labels
    })

    # Save the DataFrame to a CSV file
    try:
        submission_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(predicted_labels)} predictions to {output_path}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':

    try:
        current_dir = os.getcwd()
        print(f"Current dir: {current_dir}")

        TRAIN_DATA_CSV_PATH = os.path.join(current_dir, "train.csv")
        TEST_DATA_CSV_PATH = os.path.join(current_dir, "test.csv")
        
        # Read the data from CSV files once
        premise_train, hypothesis_train, lang_abv_train, labels_train = read_data_from_csv(TRAIN_DATA_CSV_PATH, is_training_data=True)
        premise_test, hypothesis_test, lang_abv_test, ids_test = read_data_from_csv(TEST_DATA_CSV_PATH, is_training_data=False)
        
        # Create TensorFlow datasets using dictionaries to map inputs to layer names
        train_inputs = {
            'input_premise': premise_train,
            'input_hypothesis': hypothesis_train,
            'input_language': lang_abv_train
        }
        train_dataset_csv = tf.data.Dataset.from_tensor_slices((train_inputs, labels_train))

        test_inputs = {
            'input_premise': premise_test,
            'input_hypothesis': hypothesis_test,
            'input_language': lang_abv_test
        }
        test_dataset_raw = tf.data.Dataset.from_tensor_slices(test_inputs)


        print(f"Training dataset contains {len(train_dataset_csv)} examples\n")
        print(f"Test dataset contains {len(test_dataset_raw)} examples\n")

        # Calculate the number of elements for the training set
        train_size = int(hparams['TRAINING_SPLIT'] * len(train_dataset_csv))

        # Create the training dataset by taking the first 'train_size' elements
        train_dataset = train_dataset_csv.take(train_size)

        # Create the validation dataset by skipping the first 'train_size' elements
        validation_dataset = train_dataset_csv.skip(train_size)

        SHUFFLE_BUFFER_SIZE = 1000
        PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
        train_dataset_final = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(hparams['BATCH_SIZE'])
        validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(hparams['BATCH_SIZE'])

        print(f"Buffered {SHUFFLE_BUFFER_SIZE} elements for the training dataset.")

        print()

        print(f"There are {len(train_dataset_final)} batches for a total of {hparams['BATCH_SIZE']*len(train_dataset_final)} elements for training.\n")
        print(f"There are {len(validation_dataset_final)} batches for a total of {hparams['BATCH_SIZE']*len(validation_dataset_final)} elements for validation.\n")

        print()

        print(f"Create and compile model")
        nn_model = create_and_compile_model(
            hparams['EMBEDDING_MODEL_URL'],
            hparams['DENSE_LAYERS'], 
            hparams['DENSE_UNITS_1'],
            hparams['DENSE_UNITS_2'], 
            hparams['LEARNING_RATE'],
            hparams['L2_REG_RATE'],
            hparams['DROPOUT']
        )
        nn_model.summary()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',    # Monitor the validation loss
            #min_delta = 0,
            patience=hparams['EARLY_STOP_PATIENCE'],            # Stop if val_loss doesn't improve for 5 epochs
            verbose=1,
            #mode='auto',
            #baseline=None,
            restore_best_weights=True # Restore model weights from the epoch with the best val_loss
            #start_from_epoch=0
        )

        learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(   monitor='val_accuracy',
                                                factor=hparams['REDUCE_LR_FACTOR'],
                                                patience=hparams['REDUCE_LR_PATIENCE'],
                                                verbose=1,
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

        SAVED_MODEL_PATH = os.path.join(current_dir, "trained_model_complete.tf")
        nn_model.save(SAVED_MODEL_PATH, save_format='tf')
        print("\nModel saved to {SAVED_MODEL_PATH}")

        predictions = nn_model.predict(test_dataset_raw.batch(hparams['BATCH_SIZE']), #x
                                       #batch_size=None, 
                                       verbose=False, 
                                       #steps=None, 
                                       #callbacks=None
                                       )

        prediction_csv_path = os.path.join(current_dir, "submission.csv")
        save_predictions_to_csv(predictions, ids_test, prediction_csv_path)

        # Print the hyperparameter table
        print("-" * 50)
        print("Hyperparameters for this run:")
        for key, value in hparams.items():
            print(f"{key:<25}: {value}")
        print("-" * 50)

    except ValueError as e:
        print(f"\nError: {e}")