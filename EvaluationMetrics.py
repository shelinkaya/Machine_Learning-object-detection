#The order of the bounding boxes are different. In the below code we can find the overlaped order and calculate the iou score
#calculation of IOU 

ious = np.zeros((100, 100))
for i in range (0,100):
    for j in range(0,100):
        pxmin,pymin,pxmax,pymax=pred['boxes'][i]
        gtxmin,gtymin,gtxmax,gtymax=gt[j]
        ixmin=max(pxmin,gtxmin)
        ixmax=min(pxmax,gtxmax)
        iymin=max(pymin,gtymin)
        iymax=min(pymax,gtymax)
        parea=(pxmax-pxmin)*(pymax-pymin)
        gtarea=(gtxmax-gtxmin)*(gtymax-gtymin)
        if (ixmax - ixmin) < 0 and (iymax - iymin) < 0:
            iarea = 0
        else:
            iarea = (ixmax - ixmin) * (iymax - iymin)
        IOU=iarea/(parea+gtarea-iarea)
        
        if 1>IOU>0.6:
            ious[i, j] = IOU
print(np.amax(ious, axis=1).tolist())
            
#Plotting gt and predicted bounding boxes here/to demonstrate

import imageio
import matplotlib.pyplot
import matplotlib.patches

im = Image.open('fiber23.png')
fig,ax=matplotlib.pyplot.subplots(1)
ax.imshow(im)

for i in range(0,100):
    gt_box = gt[i]
    pred_box = pred['boxes'][i]




    gt_rect = matplotlib.patches.Rectangle((gt_box[0], gt_box[1]),
                                       (gt_box[2]-gt_box[0]),
                                       (gt_box[3]-gt_box[1]),
                                       linewidth=2,
                                       edgecolor='purple',
                                       facecolor='none')

    pred_rect = matplotlib.patches.Rectangle((pred_box[0], pred_box[1]),
                                         (pred_box[2]-pred_box[0]),
                                         (pred_box[3]-pred_box[1]),
                                         linewidth=2,
                                         edgecolor='yellow',
                                         facecolor='none')
    ax.add_patch(gt_rect)
    ax.add_patch(pred_rect)
    
import sklearn.metrics 
from sklearn.metrics import confusion_matrix
import matplotlib

#evaluation of precision,recall, confusion matrix

##label y_true and y_label as positive and negative
import numpy
#assign confidence score and predicted score to label as positive and negative
conf_scores=pred['scores']
pred_scores=np.amax(ious, axis=1).tolist()



thresholde=0.90
y_pred=["positive"if score>=thresholde else "negative" for score in pred_scores]
y_true=["positive"if score>=thresholde else "negative" for score in conf_scores]

#evaluate confusion matrix, precision and recall

r=numpy.flip(sklearn.metrics.confusion_matrix(y_true,y_pred))
print(r)
precision=sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
print(precision)
recall=sklearn.metrics.recall_score(y_true=y_true,y_pred=y_pred,pos_label="positive")
print(recall)

##precision recall evaluation
 # define thresholds from 0.5 to 0.9 with step size of 0.05
import sklearn.metrics
import numpy
thresholds = numpy.arange(start=0.5, stop=0.95, step=0.05)

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls
  
##plotting precision recall curve

precisions, recalls = precision_recall_curve(y_true=y_true, 
                                             pred_scores=pred_scores,
                                             thresholds=thresholds)

matplotlib.pyplot.plot(recalls, precisions, linewidth=4, color="green")
matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
matplotlib.pyplot.show()

# evaluation of f1 score 

f1 = 2 * ((numpy.array(precisions) * numpy.array(recalls)) / (numpy.array(precisions) + numpy.array(recalls)))

## place a  point which shows best balance between recall and precsion

matplotlib.pyplot.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
matplotlib.pyplot.scatter(recalls[5], precisions[5], zorder=1, linewidth=6)

matplotlib.pyplot.xlabel("Recall", fontsize=12, fontweight='bold')
matplotlib.pyplot.ylabel("Precision", fontsize=12, fontweight='bold')
matplotlib.pyplot.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
matplotlib.pyplot.show()

## average precision evaluation


def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls


thresholds=numpy.arange(start=0.5, stop=0.95, step=0.05)

precisions, recalls = precision_recall_curve(y_true=y_true, 
                                             pred_scores=pred_scores, 
                                             thresholds=thresholds)

precisions.append(1)
recalls.append(0)

precisions = numpy.array(precisions)
recalls = numpy.array(recalls)

AP = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
print(AP)

