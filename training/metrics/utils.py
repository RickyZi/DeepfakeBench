from sklearn import metrics
# from metrics import RocCurveDisplay, roc_curve, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve, precision_score
import numpy as np
# import torch
import os

def parse_metric_for_print(metric_dict):
    # print(metric_dict)
    if metric_dict is None:
        print("metric dict is none!")
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def check_graph_name(graph_path):
    # check if the graph_name already exists, 
    # if it does, add a number to the end of the graph_name
    # e.g., graph_name = 'loss' -> 'loss_1', 'loss_2', 'loss_3', ...

    if os.path.exists(graph_path):
        # graph_path = '/home/rz/rz-test/bceWLL_test/roc-curve/' + model.name + "_" + tags + '_roc_curve_auc.png'
        # get file name without the file extension
        # 
        # if the graph_name already exists, add a number to the end of the graph_name
        # e.g., graph_name = 'loss' -> 'loss_1', 'loss_2', 'loss_3', ...
        name = graph_path.split('.')[0] # remove the file extension -> /home/rz/rz-test/bceWLL_test/roc-curve/' + model.name + "_" + tags + '_roc_curve_auc
        i = 1
        while True:
            new_graph_name = f'{name}_{i}.png'
            if not os.path.exists(new_graph_name):
                graph_path = new_graph_name
                break
            i += 1
    return graph_path

def get_test_metrics(y_pred, y_true, img_names, tags=''): # model, dataset, 
    # compute video-level auc for the frame-level methods.
    # img_names: list of image paths (list of tuples)
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])  
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr


        # Check for NaN values in fnr and fpr
        if np.isnan(fnr).all() or np.isnan(fpr).all():
            # raise ValueError("fnr or fpr contains only NaN values")

            # handle NaN values by filling them with a default value
            fnr = np.nan_to_num(fnr, nan=0.0)
            fpr = np.nan_to_num(fpr, nan=0.0)

        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer

        # v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # return v_auc, v_eer

    # -------------------------------------------------------------------- #
    # keep track of the number of correct predictions for each class
    correct_original, correct_simswap, correct_ghost, correct_facedancer = 0, 0, 0, 0
    total_original, total_simswap, total_ghost, total_facedancer = 0, 0, 0, 0 
    # -------------------------------------------------------------------- #
    # y_pred = predicted scores (probabilities)
    # y_true = true binary labels (ground truth, 0 or 1)
    # prediction_class = predicted class labels (0 or 1)
    # -------------------------------------------------------------------- #
    y_pred = y_pred.squeeze() # remove the extra dimension (batch_size, 1) -> (batch_size, )
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1

    # print("len(y_true)", len(y_true)) # 2400 -> # test images
    # print("len(y_pred)", len(y_pred)) # 2400 -> # test images
    # print("len(img_names)", len(img_names)) # 2400 -> # test images
    # breakpoint()
    # -------------------------------------------------------------------- #
    # acc
    prediction_class = (y_pred > 0.5).astype(int) # convert prediction to class labels 
    # print("len(prediction_class)", len(prediction_class)) # 2400
    # -------------------------------------------------------------------- #
    # print("type(y_pred)", type(y_pred)) # ndarray
    # print("len(y_pred)", len(y_pred)) # 4800 -> # test images
    # print("y_pred[:10]", y_pred[:10]) 
    # print("type(y_true)", type(y_true)) # ndarray
    # print("len(y_true)", len(y_true)) # 4800 -> # test images
    # print("y_true[:10]", y_true[:10])
    
    # print("len(prediciton_class)", len(prediction_class)) # 4800 -> # test images
    # print("prediction_class[:10]", prediction_class[:10]) # print the first 10 predictions
    # print("type(prediction_class)", type(prediction_class)) # numpy.ndarray
    # print("len(prediction_class)", len(prediction_class)) # 4800
    # # breakpoint()
    # -------------------------------------------------------------------- #
    # type(y_pred) <class 'numpy.ndarray'>
    # len(y_pred) 4800
    # y_pred[:10] [0.20893824 0.07067125 0.33554134 0.87064254 0.87171334 0.7668541
    # 0.75242585 0.2539114  0.80467933 0.8326783 ]
    # type(y_true) <class 'numpy.ndarray'>
    # len(y_true) 4800
    # y_true[:10] [1 1 0 1 1 1 1 1 1 1]
    # prediction_class[:10] [0 0 0 1 1 1 1 0 1 1]
    # type(prediction_class) <class 'numpy.ndarray'>
    # len(prediction_class) 4800
    # -------------------------------------------------------------------- #


    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item() 
    # correct: number of correct predictions -> sum of correct predictions
    # np.clip(y_true, a_min=0, a_max=1): clip the values in the array to be between 0 and 1
    acc = correct / len(prediction_class) * 100

    # -------------------------------------------------------------------- #
    # compute accuracy for each algo: ghost, simswap, facedancer, original
    # print("type(image_names)", type(img_names)) # tuple
    # print("len(image_names)", len(img_names)) # 2
    # print("img_names[0]", img_names[0]) # label
    # print("img_names[1]", img_names[1]) # imgs_path
    # breakpoint() 

    for i in range(len(img_names)):
        if 'original' in img_names[i]:
            total_original += 1
            if prediction_class[i] == y_true[i]:
                correct_original += 1
        elif 'simswap' in img_names[i]:
            total_simswap += 1
            if prediction_class[i] == y_true[i]:
                correct_simswap += 1
        elif 'ghost' in img_names[i]:
            total_ghost += 1
            if prediction_class[i] == y_true[i]:
                correct_ghost += 1
        elif 'facedancer' in img_names[i]:
            total_facedancer += 1
            if prediction_class[i] == y_true[i]:
                correct_facedancer += 1

    acc_original = (correct_original / (total_original * 100)) if total_original != 0 else 0
    acc_ghost = (correct_ghost / total_ghost * 100) if total_ghost != 0 else 0
    acc_simswap = (correct_simswap / total_simswap * 100) if total_simswap != 0 else 0
    acc_facedancer = (correct_facedancer / total_facedancer * 100) if total_facedancer != 0 else 0

    # print("acc_original", acc_original)
    # print("acc_ghost", acc_ghost)
    # print("acc_simswap", acc_simswap)
    # print("acc_facedancer", acc_facedancer)
    # -------------------------------------------------------------------- #
    # balanced accuracy
    balanced_acc = metrics.balanced_accuracy_score(y_true, prediction_class) 

    # Ensure the directory exists
    output_dir = '/home/rz/DeepfakeBench/training/results/'+tags +'/testing/graphs/' #'/home/rz/DeepfakeBench/training/metrics/graphs/'+model+'/dfb_'+dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # -------------------------------------------------------------------- #
    # convert to numpy array (tensor -> numpy array), then give it to roc_curve
    # labels_list = y_true.cpu().numpy() # convert to numpy array -> true binary labels (ground truth, 0 or 1)
    # pred_list = y_pred.cpu().numpy() # convert to numpy array -> predicted scores (probabilities)
    # pred_list = torch.cat(y_pred, dim=0).cpu().numpy() 
    # compute the confusion matrix, save it as a figure
    cm = metrics.confusion_matrix(y_true, prediction_class)
    # cm = confusion_matrix(labels_list, pred_list)
    tn, fp, fn, tp = cm.ravel()
    cm_display = metrics.ConfusionMatrixDisplay(cm).plot()
    cm_path = check_graph_name(os.path.join(output_dir, 'confusion_matrix.png'))
    cm_display.figure_.savefig(cm_path)
    # confusion matrix tells us how many true positives, true negatives, false positives, and false negatives we have

    # compute TPR and TNR
    TPR = tp / (tp + fn) # sensitivity, recall -> from all the positive classes, how many were correctly predicted
    # recall should be as high as possible -> we want to avoid false negatives -> max recall = 1
    TNR = tn / (tn + fp) # specificity -> from all the negative classes, how many were correctly predicted. Ideally, we want to avoid false positives -> max specificity = 1
    # -------------------------------------------------------------------- #
    # auc
    fpr, tpr, _ = metrics.roc_curve(y_true, prediction_class, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # check if auc is not NaN, in case set it to 0
    auc = metrics.auc(fpr, tpr) if not np.isnan(metrics.auc(fpr, tpr)) else 0
    # auc = metrics.roc_auc_score(y_true, y_pred) if not np.isnan(metrics.roc_auc_score(y_true, y_pred)) else 0
    # -------------------------------------------------------------------- #
    # display the ROC curve
    display_roc = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display_roc.plot()
    roc_path = check_graph_name(os.path.join(output_dir, 'roc_curve.png'))
    display_roc.figure_.savefig(roc_path)
    # ROC-AUC curve tells how much the model is capable of distinguishing between classes
    # the higher the AUC, the better the model is at predicting 0s as 0s and 1s as 1s
    # AUC = 1 -> perfect model, AUC = 0.5 -> random model
    # ROC curve is a plot of TPR vs FPR at different classification thresholds
    # -------------------------------------------------------------------- #
    # eer -> Equal Error Rate: the poi nt where fnr = fpr. The lower the EER, the better the model is at distinguishing between classes
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] if not np.isnan(fpr).all() else 0 # np.isnan(fpr).all() -> check if fpr contains only NaN values
    # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # Equal Error Rate is computed as the point where fnr = fpr -> in general the lower the EER, the higher the model's accuracy
    # ap -> Average Precision: the higher the AP, the better the model is at distinguishing between classes.
    ap = metrics.average_precision_score(y_true, y_pred)
    # pos_label: 1 (default) -> the class to report if average='binary' and pos_label is not specified 
    # in our case, pos_label = 0 (real) and neg_label = 1 (fake) -> but we are interested in the ability of the model in detecting fake images -> so we set pos_label = 1 

    # -------------------------------------------------------------------- #
    # compute precision and recall
    # y_pred_binary = (y_pred > 0.5).astype(int) # convert prediction to binary labels
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    # compute the precision-recall curve and save it as a figure
    display_pr = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=ap)
    display_pr.plot()
    pr_path = check_graph_name(os.path.join(output_dir, 'pr_curve.png'))
    display_pr.figure_.savefig(pr_path)
    # precision-recall curve is a plot of precision vs recall at different classification thresholds -> it shows the trade-off between precision and recall 
    # ideally, we want both precision and recall to be as high as possible -> max precision = 1, max recall = 1 -> perfect model 
    # pr-curve summarized by the AP score -> the higher the AP score, the better the model is at distinguishing between classes
    # Average Precision (AP) summarizes such a plot as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight
    # ideally we want the AP score to be as high as possible -> max AP = 1 -> curve that passes through (1,1)

    # compute the precision score
    # precision_scr = metrics.precision_score(y_true, prediction_class) 
    # precision_scr -> from all the classes we have predicted as positive, how many are actually positive.
    # precision should be as high as possible -> we want to avoid false positives -> max precision = 1
    # -------------------------------------------------------------------- #

    

    # # compute video-level auc for the frame-level methods.
    # if type(img_names[0]) is not list: 
    #     # calculate video-level auc for the frame-level methods.
    #     v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    # else:
    #     # video-level methods
    #     v_auc=auc

    # return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}
    return {
        'acc': acc, # Accuracy
        'balanced_acc': balanced_acc,
        # Accuracy for each algo: ghost, simswap, facedancer, original
        'test_accuracy_original': acc_original,
        'test_accuracy_simswap': acc_simswap,
        'test_accuracy_ghost': acc_ghost,
        'test_accuracy_facedancer': acc_facedancer,
        'auc': auc, # Area Under the ROC Curve
        'ap': ap, # Average Precision 
        # 'video_auc': v_auc, # keep it? 
        'TPR': TPR, # True Positive Rate (Sensitivity or Recall)
        'TNR': TNR, # True Negative Rate (Specificity)
        'eer': eer, # Equal Error Rate 
        # -------------------------------------------------------------------- #
        # 'precision': precision_scr, # Precision
        # 'recall': list(recall),
        # 'pred': list(y_pred), # list of predictionù
        # 'label': list(y_true), # list of labels
        # 'img_path_collection': img_names # list of image paths
    }

# ----------------------------------------------------------------------------------------------------------------------------- #
# TODO: 
# - compute accuracy for each algo: ghost, simswap, facedancer, original
# - compute confusion matrix (labels, predictions)
# - compute TRP, TNR
# - compute ROC curve and AUC score -> save the ROC curve 
# - compute AP score
# - compute precision and recall -> save the precision-recall curve

#return test_accuracy, test_accuracy_original, test_accuracy_simswap, test_accuracy_ghost, test_accuracy_facedancer, 
# TPR, TNR, fpr_roc, tpr_roc, auc_score, ap_score, precision, recall, list(labels_list), list(probs_list), img_path_collection
# ----------------------------------------------------------------------------------------------------------------------------- #
