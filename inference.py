import cv2
import numpy as np
import onnxruntime as ort

# 1. 加载模型
yolo_session = ort.InferenceSession("./assets/best.onnx")
ghostnet_session = ort.InferenceSession("./assets/ghostnet_model_epoch19.onnx")
roa_session = ort.InferenceSession("./assets/best_roa_model.onnx")

def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    return img

# 2. YOLO预处理（保持不变）
def yolo_preprocess(img, input_size=960):
    h, w = img.shape[:2]
    scale = input_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = resized
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]
    return input_tensor, scale, nw, nh

# 3. YOLO后处理（保持不变）
def yolo_postprocess(outputs, conf_threshold=0.5, iou_threshold=0.5):
    preds = outputs[0]
    preds = np.transpose(preds, (0, 2, 1))
    preds = preds[0]

    boxes, scores = [], []
    for det in preds:
        conf = det[4]
        if conf < conf_threshold:
            continue
        cx, cy, w, h = det[:4]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        boxes.append([x1, y1, x2, y2])
        scores.append(conf)

    boxes = np.array(boxes)
    scores = np.array(scores)

    if len(boxes) == 0:
        return np.array([]), np.array([])

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        boxes = boxes[indices]
        scores = scores[indices]
    else:
        boxes = np.array([])
        scores = np.array([])

    return boxes, scores

# 4. GhostNet预处理（保持不变）
def ghostnet_preprocess(cropped_img):
    img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_norm = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_norm - mean) / std
    input_tensor = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]
    return input_tensor

# 5. roa模型预处理，复用GhostNet预处理（假设输入格式相同）
def roa_preprocess(cropped_img):
    # 这里与ghostnet_preprocess一样，方便直接调用
    return ghostnet_preprocess(cropped_img)

# 6. Softmax函数（保持不变）
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# 7. 分类标签（保持不变）
damage_classes = ['abrasion_pressure', 'healthy', 'pitting', 'spalling']

def run_inference(image_path):
    gamma = 0.1
    orig_img = imread_unicode(image_path)
    input_tensor, scale, nw, nh = yolo_preprocess(orig_img)

    # YOLO推理
    yolo_outputs = yolo_session.run(None, {yolo_session.get_inputs()[0].name: input_tensor})
    boxes, scores = yolo_postprocess(yolo_outputs, conf_threshold=0.5, iou_threshold=0.5)

    h, w = orig_img.shape[:2]

    if len(boxes) == 0:
        print("未检测到目标")
        return

    # 找置信度最高的bbox，作为整体分类和整体roa来源
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]
    x1_b, y1_b, x2_b, y2_b = best_box
    x1_b = int((x1_b / nw) * w)
    x2_b = int((x2_b / nw) * w)
    y1_b = int((y1_b / nh) * h)
    y2_b = int((y2_b / nh) * h)
    roi_best = orig_img[y1_b:y2_b, x1_b:x2_b]

    # 整体分类
    gn_input_best = ghostnet_preprocess(roi_best)
    gn_outputs_best = ghostnet_session.run(None, {ghostnet_session.get_inputs()[0].name: gn_input_best})
    gn_preds_best = np.array(gn_outputs_best[0])
    probs_best = softmax(gn_preds_best)
    class_id_best = np.argmax(probs_best)
    class_name_best = damage_classes[class_id_best]

    # 整体 roa
    roa_input_best = roa_preprocess(roi_best)
    roa_outputs_best = roa_session.run(None, {roa_session.get_inputs()[0].name: roa_input_best})
    roa_pred_best = gamma * float(roa_outputs_best[0][0][0])  # 假设输出形状 [1,1]

    # 左上角显示整体分类
    label_overall = f"Gear Class: {class_name_best}"
    color_overall = (0, 255, 0) if class_name_best == 'healthy' else (0, 0, 255)
    cv2.putText(orig_img, label_overall, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color_overall, 4)

    # 每个bbox单独显示分类+roa
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 = int((x1 / nw) * w)
        x2 = int((x2 / nw) * w)
        y1 = int((y1 / nh) * h)
        y2 = int((y2 / nh) * h)

        roi = orig_img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 分类
        gn_input = ghostnet_preprocess(roi)
        gn_outputs = ghostnet_session.run(None, {ghostnet_session.get_inputs()[0].name: gn_input})
        gn_preds = np.array(gn_outputs[0])
        probs = softmax(gn_preds)
        class_id = np.argmax(probs)
        class_name = damage_classes[class_id]
        score = probs[0, class_id]

        # roa
        roa_input = roa_preprocess(roi)
        roa_outputs = roa_session.run(None, {roa_session.get_inputs()[0].name: roa_input})
        roa_pred = gamma * float(roa_outputs[0][0][0])

        color = (0, 255, 0) if class_name == 'healthy' else (0, 0, 255)
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)

        label_cls = f"{class_name} {score:.2f}"
        label_roa = f"Ratio of Demaged Areas: {roa_pred:.2f}"

        text_x = x1
        text_y1 = max(y1 - 35, 20)
        text_y2 = max(y1 - 10, 20)

        cv2.putText(orig_img, label_cls, (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        cv2.putText(orig_img, label_roa, (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

    return orig_img, {"overall_class": class_name_best, "overall_roa": roa_pred_best}