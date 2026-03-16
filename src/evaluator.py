from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(boxA, boxB):
    """Compute Intersection over Union between two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union_area = areaA + areaB - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def evaluate_detection(pred_bboxes, pred_classes, true_bboxes, true_classes, iou_th=0.5):
    """Compute Precision, Recall, and F1 by matching predictions to ground truths via IoU."""
    tp = np.zeros(len(pred_bboxes))
    matched_gt = set()  # Track which ground-truth boxes have been matched

    for i, (pred_box, pred_cls) in enumerate(zip(pred_bboxes, pred_classes)):
        best_iou = 0.0
        best_gt_idx = -1

        for j, (true_box, true_cls) in enumerate(zip(true_bboxes, true_classes)):
            if j in matched_gt:
                continue  # Already matched to another prediction
            if pred_cls != true_cls:
                continue  # Class mismatch — not a true positive
            iou = compute_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_iou >= iou_th and best_gt_idx >= 0:
            tp[i] = 1
            matched_gt.add(best_gt_idx)  # Mark GT as used

    num_tp = np.sum(tp)
    precision = num_tp / len(pred_bboxes) if len(pred_bboxes) > 0 else 0.0
    recall    = num_tp / len(true_bboxes) if len(true_bboxes) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot and save a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)

    # Annotate each cell with its count
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("✅ Confusion matrix saved to confusion_matrix.png")
    plt.close()
