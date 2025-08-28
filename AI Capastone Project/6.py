# --- Task 1: What does preds > 0.5 do? ---
# Converts predicted probabilities to binary (0 or 1) by thresholding at 0.5.

# --- Task 2: Print Keras model metrics ---
print_metrics(y_true, y_pred)  # assume function implemented elsewhere

# --- Task 3: Significance of the F1-score ---
# The F1-score balances precision and recall, giving a single metric that captures both false positives and false negatives.

# --- Task 4: Print PyTorch model metrics ---
print_metrics(all_labels, all_preds)

# --- Task 5: Count false negatives in confusion matrix ---
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)
false_negatives = cm[1,0]
print("False negatives:", false_negatives)

