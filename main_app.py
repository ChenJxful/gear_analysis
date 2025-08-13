# main_app.py
import sys
import os
import cv2
import numpy as np
from openai import OpenAI
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame,
    QSizePolicy, QPushButton, QFileDialog, QMessageBox, QLineEdit,
    QStackedLayout, QTextEdit, QComboBox, QGridLayout, QDialog, QDialogButtonBox
)
from PyQt5.QtGui import QFont, QPalette, QColor, QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from inference import run_inference, imread_unicode


# ---------- 非线性寿命估算 ----------
import onnxruntime as ort
from scipy.stats import norm

def nonlinear_life(image_path: str,
                   detect_model_path: str = "./assets/detect.onnx",
                   seg_model_path: str   = "./assets/yolov8seg_best.onnx",
                   rpm: int              = 10211,
                   max_depth_mm: float   = 2.0,
                   baseline_life: float  = 8000,
                   alpha: float          = 2.0,
                   beta: float           = 1.5,
                   k_decay: float        = 3.0) -> dict:
    original_img = imread_unicode(image_path)
    if original_img is None:
        raise FileNotFoundError(image_path)
    image = cv2.resize(original_img, (960, 960))
    input_tensor = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    sess_det = ort.InferenceSession(detect_model_path, providers=['CPUExecutionProvider'])
    outs = sess_det.run(None, {sess_det.get_inputs()[0].name: input_tensor})
    boxes = outs[0][0]
    gear_boxes = [b for b in boxes if b[4] > 0.7]
    if not gear_boxes:
        raise ValueError("未检测到置信度>0.7的齿面区域")
    x1, y1, x2, y2 = map(int, gear_boxes[0][:4])
    gear_roi = original_img[y1:y2, x1:x2]
    gear_roi = cv2.resize(gear_roi, (960, 960))

    sess_seg = ort.InferenceSession(seg_model_path, providers=['CPUExecutionProvider'])
    seg_tensor = gear_roi.transpose(2, 0, 1).astype(np.float32) / 255.0
    seg_tensor = np.expand_dims(seg_tensor, axis=0)
    masks = sess_seg.run(None, {sess_seg.get_inputs()[0].name: seg_tensor})[1][0]

    stress_dist = norm.pdf(np.arange(960), loc=480, scale=180)
    stress_dist /= stress_dist.max()

    min_life = float('inf')
    min_idx = -1
    details_lines = []

    for idx, mask in enumerate(masks):
        mask_bin = cv2.resize((mask > 0.7).astype(np.uint8), (960, 960), interpolation=cv2.INTER_NEAREST)
        if np.sum(mask_bin) == 0:
            continue
        gray = cv2.cvtColor(gear_roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        masked = cv2.bitwise_and(enhanced, enhanced, mask=mask_bin)
        depth_mm = ((255 - masked.astype(np.float32)) / 255) * max_depth_mm
        depth_mm[mask_bin == 0] = 0
        y_idx, x_idx = np.where(mask_bin > 0)
        stress_vals = stress_dist[x_idx]
        depth_vals = depth_mm[y_idx, x_idx]
        dmg = np.mean((stress_vals ** alpha) * (depth_vals ** beta))
        life = baseline_life * np.exp(-k_decay * dmg) * (10211 / rpm)
        details_lines.append(
            f"区域{idx+1}: 深度均值={np.mean(depth_vals):.4f} mm, "
            f"损伤因子={dmg:.4f}, 寿命={life:.2f} h"
        )
        if life < min_life:
            min_life, min_idx = life, idx

    if min_idx == -1:
        min_life = baseline_life * (10211 / rpm)
        details_lines.append("未检测到损伤，按健康状态估算。")
    return {"min_life_hours": min_life, "danger_idx": min_idx, "details": "<br>".join(details_lines)}

# ---------- OpenAI / 物理模型 ----------
try:
    client = OpenAI(api_key="sk-5e2dcab1e8764df5be0fc6604d507953", base_url="https://api.deepseek.com")
except Exception as e:
    print(f"客户端未能成功初始化: {e}")
    client = None

class OpenAIStreamThread(QThread):
    token_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    def __init__(self, prompt): super().__init__(); self.prompt = prompt
    def run(self):
        if not client:
            self.token_signal.emit("\n[AI分析失败: 客户端未能成功初始化]\n")
            self.finished_signal.emit()
            return
        try:
            stream = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role":"system","content":"你是一个专业机械工程师..."},
                          {"role":"user","content":self.prompt}], stream=True, temperature=0.7)
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    self.token_signal.emit(chunk.choices[0].delta.content)
            self.finished_signal.emit()
        except Exception as e:
            self.token_signal.emit(f"\n[AI分析失败: {str(e)}]\n")
            self.finished_signal.emit()

class PhysicsRULModel:
    def __init__(self):
        self.material_db = {
            "16Cr3NiWMoVNbE": {"p": 8.7, "C_H": 1600},
            "20CrMnTi": {"p": 8.7, "C_H": 1200},
            "12Cr2Ni4A": {"p": 8.5, "C_H": 1450},
            "42CrMo": {"p": 7.5, "C_H": 1100},
        }

    def _lookup_material_properties(self, material_name):
        return self.material_db.get(material_name)

    def _map_level_to_base_torque(self, level):
        return {"轻载": 30, "中载": 60, "重载": 100}.get(level, 60)

    def _map_level_to_equivalent_factor(self, level):
        return {"轻载": 0.8, "中载": 0.9, "重载": 1.0}.get(level, 0.9)

    def estimate_rul(self, damage_type, area_ratio, geometry, material_name, load):
        props = self._lookup_material_properties(material_name)
        if not props:
            raise ValueError(f"材料 '{material_name}' 未在数据库中找到。")
        T, n = load["torque"], load["rpm"]
        d1 = geometry["module"] * geometry["teeth"]
        b = geometry.get("face_width", geometry["module"] * 10)
        F_t = 2000 * T / d1
        T_eq_factor = self._map_level_to_equivalent_factor(load["level"])
        T_eq = T * T_eq_factor
        base_sigma_H = 2.5 * 189.8 * 0.9 * np.sqrt((F_t / (d1 * b)) * 2)
        sigma_H = base_sigma_H * (1.0 + 0.5 * (area_ratio ** 0.5) if area_ratio > 0 else 1.0) * 1.5
        L10 = ((props['C_H'] / max(sigma_H, 1e-6)) ** props['p']) * (1e6 / (60 * n)) * 0.7
        if damage_type.lower() == 'healthy' or area_ratio == 0:
            rul_hours = L10
        else:
            rul_hours = L10 / (1 + 15 * (area_ratio ** 1.7)) * (T_eq_factor ** 3)
        return {"method": "工程修正后ISO模型", "sigma_H": sigma_H, "L10_hours": L10, "rul_hours": rul_hours,
                "details": {"基础扭矩 T": T, "等效扭矩 Teq": T_eq, "切向力 Ft": F_t, "基础接触应力 σH_base": base_sigma_H}}

# ---------- 弹窗 ----------
class ParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("非线性寿命预测参数设置")
        self.setFixedSize(320, 220)
        grid = QGridLayout(self)
        grid.setSpacing(10)

        self.inputs = {}
        labels = ["最大深度(mm):", "基准寿命(h):", "应力指数α:", "深度指数β:", "衰减系数k:"]
        keys   = ["max_depth_mm", "baseline_life", "alpha", "beta", "k_decay"]
        defaults = [parent.nonlin_params[k] for k in keys]

        for row, (lab, key, val) in enumerate(zip(labels, keys, defaults)):
            grid.addWidget(QLabel(lab), row, 0)
            le = QLineEdit(str(val))
            le.setValidator(QDoubleValidator(0.01, 999999, 3))
            grid.addWidget(le, row, 1)
            self.inputs[key] = le

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        grid.addWidget(buttons, len(keys), 0, 1, 2)

    def get_values(self):
        return {k: float(v.text()) for k, v in self.inputs.items()}

# ---------- 主界面 ----------
class GearAnalysisUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("齿轮损伤智能分析平台")
        self.setGeometry(100, 100, 1400, 900)
        self.uploaded_image_path = None
        self.ai_thread = None
        self.expanded_index = -1
        self.last_result_info = None
        self.last_rul_results = None
        self.last_ai_analysis_text = ""
        self.last_nonlinear_result = None
        # 默认非线性参数
        self.nonlin_params = {
            "max_depth_mm": 2.0,
            "baseline_life": 8000,
            "alpha": 2.0,
            "beta": 1.5,
            "k_decay": 3.0,
        }
        self.panel_titles = [
            "主要损伤类型分析",
            "损伤区域可视化",
            "剩余使用寿命估计",
            "AI辅助分析与建议",
            "非线性寿命预测",
        ]
        self.physics_model = PhysicsRULModel()
        self.current_unscaled_pixmap = None
        self.last_scaled_size = QSize(0, 0)
        self.damage_map = {
            'pitting': '点蚀',
            'spalling': '剥落',
            'crack': '裂纹',
            'wear': '磨损',
            'abrasion_pressure': '擦伤压伤',
            'healthy': '健康',
        }
        self.init_ui()

    def init_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(0, 102, 204))
        self.setAutoFillBackground(True)
        self.setPalette(palette)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(0)

        box_frame = QFrame()
        box_frame.setStyleSheet("background-color: #003366; border-radius: 10px;")
        box_layout = QVBoxLayout(box_frame)
        box_layout.setContentsMargins(20, 20, 20, 20)
        box_layout.setSpacing(20)

        title_label = QLabel("齿轮损伤智能分析平台")
        title_label.setFont(QFont("SimHei", 28, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        title_label.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        box_layout.addWidget(title_label)

        white_frame_layout = QHBoxLayout()
        white_frame_layout.setSpacing(20)

        left_frame = QFrame()
        left_frame.setStyleSheet("background-color: white; border-radius: 15px;")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(15, 15, 15, 15)
        left_layout.setSpacing(15)

        self.image_label = QLabel("请上传一张齿轮图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(QFont("SimHei", 16))
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #aaa; border-radius: 10px;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumHeight(400)
        left_layout.addWidget(self.image_label, 1)

        input_form_frame = QFrame()
        input_form_frame.setStyleSheet("background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;")
        form_layout = QGridLayout(input_form_frame)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)

        self.module_input = QLineEdit("2.5")
        self.teeth_input = QLineEdit("30")
        self.material_combo = QComboBox()
        self.material_combo.addItems(self.physics_model.material_db.keys())
        self.torque_combo = QComboBox()
        self.torque_combo.addItems(["轻载", "中载", "重载"])
        self.rpm_input = QLineEdit("10000")

        widgets = [
            (QLabel("模数 (mm):"), self.module_input),
            (QLabel("齿数:"), self.teeth_input),
            (QLabel("材料牌号:"), self.material_combo),
            (QLabel("工况等级:"), self.torque_combo),
            (QLabel("转速 (RPM):"), self.rpm_input),
        ]
        for i, (label, widget) in enumerate(widgets):
            label.setFont(QFont("SimHei", 11))
            widget.setFont(QFont("Arial", 11))
            widget.setMinimumHeight(30)
            form_layout.addWidget(label, i, 0)
            form_layout.addWidget(widget, i, 1)
        left_layout.addWidget(input_form_frame)

        button_layout = QHBoxLayout()
        btn_upload = QPushButton("上传图像")
        self.btn_analyze = QPushButton("分析结果")
        btn_param = QPushButton("非线性寿命预测参数设置")
        for btn in (btn_upload, self.btn_analyze, btn_param):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.setMinimumHeight(45)
            btn.setFont(QFont("SimHei", 14, QFont.Bold))
            btn.setStyleSheet("""
                QPushButton { background-color: #007bff; color: white; border: none; border-radius: 12px; padding: 8px 16px; }
                QPushButton:hover { background-color: #0056b3; }
                QPushButton:pressed { background-color: #004085; }
                QPushButton:disabled { background-color: #999; }
            """)
        btn_upload.clicked.connect(self.on_upload_clicked)
        self.btn_analyze.clicked.connect(self.on_analysis_clicked)
        btn_param.clicked.connect(self.open_param_dialog)
        button_layout.addWidget(btn_upload)
        button_layout.addWidget(self.btn_analyze)
        button_layout.addWidget(btn_param)
        left_layout.addLayout(button_layout)

        right_frame = QFrame()
        right_frame.setStyleSheet("background-color: white; border-radius: 15px;")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(20, 20, 20, 20)
        self.stack_layout = QStackedLayout()

        self.default_view = QWidget()
        default_layout = QVBoxLayout(self.default_view)
        default_layout.setContentsMargins(0, 0, 0, 0)
        default_layout.setSpacing(15)
        self.interaction_boxes = []
        for i, title in enumerate(self.panel_titles):
            box = QFrame()
            box.setCursor(Qt.PointingHandCursor)
            box.setFixedHeight(80)
            box.setStyleSheet("QFrame { background-color: #f5f5f5; border-radius: 12px; border: 1px solid #ccc; } QFrame:hover { border: 2px solid #007bff; }")
            label = QLabel(title, box)
            label.setAlignment(Qt.AlignCenter)
            label.setFont(QFont("SimHei", 16))
            QVBoxLayout(box).addWidget(label)
            box.mousePressEvent = lambda event, idx=i: self.expand_box(idx)
            self.interaction_boxes.append(box)
            default_layout.addWidget(box)
        default_layout.addStretch()

        self.expanded_view = QWidget()
        self.expanded_layout = QVBoxLayout(self.expanded_view)
        self.expanded_layout.setContentsMargins(0, 0, 0, 0)
        self.expanded_layout.setSpacing(10)
        self.stack_layout.addWidget(self.default_view)
        self.stack_layout.addWidget(self.expanded_view)
        right_layout.addLayout(self.stack_layout)
        white_frame_layout.addWidget(left_frame, 55)
        white_frame_layout.addWidget(right_frame, 45)
        box_layout.addLayout(white_frame_layout)
        main_layout.addWidget(box_frame)

    # ---------- 新增弹窗 ----------
    def open_param_dialog(self):
        dlg = ParamDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.nonlin_params = dlg.get_values()

    # ---------- 其余方法 ----------
    def on_upload_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.png *.jpg *.jpeg *.bmp)")
        if filename:
            self.last_result_info = None
            self.last_rul_results = None
            self.last_ai_analysis_text = ""
            self.last_nonlinear_result = None
            self.collapse_view()
            self.uploaded_image_path = filename
            self.current_unscaled_pixmap = QPixmap(filename)
            self.last_scaled_size = QSize(0, 0)
            if self.current_unscaled_pixmap.isNull():
                self.image_label.setText("图片加载失败！")
                QMessageBox.warning(self, "错误", "加载图片失败！")
            else:
                self.image_label.setText("")
                self.update_image_display()

    def on_analysis_clicked(self):
        if not self.uploaded_image_path:
            QMessageBox.warning(self, "提示", "请先上传一张图片！")
            return
        try:
            torque_level_str = self.torque_combo.currentText()
            base_torque = self.physics_model._map_level_to_base_torque(torque_level_str)
            load = {"rpm": int(self.rpm_input.text()), "torque": base_torque, "level": torque_level_str}
            geometry = {"module": float(self.module_input.text()), "teeth": int(self.teeth_input.text())}
            material_name = self.material_combo.currentText()
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请在左侧输入有效的物理参数（模数、齿数、转速必须为数字）。")
            return

        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setText("分析中...")
        QApplication.processEvents()
        try:
            processed_img, result_info = run_inference(self.uploaded_image_path)
            self.last_result_info = result_info
            damage_type = result_info.get('overall_class', 'unknown')
            area_ratio = result_info.get('overall_roa', 0.0)
            self.last_rul_results = self.physics_model.estimate_rul(damage_type, area_ratio, geometry, material_name, load)

            rpm_user = int(self.rpm_input.text())
            self.last_nonlinear_result = nonlinear_life(
                self.uploaded_image_path,
                rpm=rpm_user,
                **self.nonlin_params
            )

            img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.current_unscaled_pixmap = QPixmap.fromImage(q_img)
            self.last_scaled_size = QSize(0, 0)
            self.update_image_display()
            self.expand_box(2)

            prompt = (
                f"针对一个齿轮进行分析。物理参数：模数={geometry['module']}mm, 齿数={geometry['teeth']}, "
                f"材料={material_name}, 工况扭矩={load['torque']}N·m, 转速={load['rpm']}RPM。\n"
                f"模型识别结果：损伤类型='{self.damage_map.get(damage_type, damage_type)}', 损伤面积比={area_ratio:.2%}。\n"
                f"寿命评估结果：预计总寿命={self.last_rul_results['L10_hours']:.1f}小时, "
                f"预计剩余寿命={self.last_rul_results['rul_hours']:.1f}小时。\n"
                f"请以专业机械工程师口吻，整合所有信息，每点结束后换行：\n"
                f"1. 结论先行：根据分析结果，直接给出最核心的维修或更换建议。\n"
                f"2. 状态解读：简要说明当前损伤的严重程度。\n"
                f"3. 预防建议：提供1-2条预防性维护建议。"
            )
            self.ai_thread = OpenAIStreamThread(prompt)
            self.ai_thread.token_signal.connect(self.update_ai_response)
            self.ai_thread.finished_signal.connect(self.analysis_done)
            self.ai_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "分析失败", f"执行分析时发生错误: {e}")
            self.analysis_done()

    def update_ai_response(self, text):
        self.last_ai_analysis_text += text
        if self.expanded_index == 3:
            ai_panel_content_widget = self.expanded_view.findChild(QTextEdit)
            if ai_panel_content_widget:
                ai_panel_content_widget.insertPlainText(text)
                ai_panel_content_widget.moveCursor(ai_panel_content_widget.textCursor().End)

    def analysis_done(self):
        if self.expanded_index == 3:
            ai_panel_content_widget = self.expanded_view.findChild(QTextEdit)
            if ai_panel_content_widget:
                ai_panel_content_widget.append("\n\n[分析完成]")
        self.btn_analyze.setEnabled(True)
        self.btn_analyze.setText("分析结果")

    def collapse_view(self):
        self.expanded_index = -1
        self.stack_layout.setCurrentIndex(0)

    def expand_box(self, index):
        self.expanded_index = index
        for i in reversed(range(self.expanded_layout.count())):
            widget = self.expanded_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        expanded_box = QFrame()
        expanded_box.setStyleSheet("background-color: #f8f9fa; border-radius: 12px; border: 1px solid #ccc;")
        expanded_layout_inner = QVBoxLayout(expanded_box)
        expanded_layout_inner.setContentsMargins(15, 10, 15, 15)

        title_label = QLabel(self.panel_titles[index])
        title_label.setFont(QFont("SimHei", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333; padding-bottom: 10px; border-bottom: 2px solid #eee;")
        expanded_layout_inner.addWidget(title_label)

        content_widget = QTextEdit()
        content_widget.setReadOnly(True)
        content_widget.setFont(QFont("SimHei", 12))
        content_widget.setStyleSheet("background-color: #ffffff; border: none; padding: 5px;")
        if not self.last_result_info:
            content_widget.setText("请先点击“分析结果”按钮进行分析。")
        else:
            if index == 0:
                cls_name_raw = self.last_result_info.get('overall_class', '未知')
                cls_name_mapped = self.damage_map.get(cls_name_raw, cls_name_raw.upper())
                area_ratio = self.last_result_info.get('overall_roa', 0.0)

                # 每种损伤类型的详细介绍（约120~150字）
                damage_desc = {
                    '点蚀': (
                        "点蚀是齿轮齿面在长期循环接触应力作用下产生的细小坑洼，多出现在高负荷或润滑不足的情况下。"
                        "随着时间推移，点蚀会不断扩展、相互连结，导致齿面粗糙度上升，啮合精度下降，并引发振动和噪声增加。"
                        "早期点蚀若能及时处理，可有效延长齿轮使用寿命。"
                    ),
                    '剥落': (
                        "剥落是材料表层在疲劳作用或冲击载荷下大面积脱落的现象，通常伴随深度凹陷和金属碎屑脱落。"
                        "这种损伤会显著削弱齿轮的承载能力，并加剧传动系统的振动与噪声。"
                        "剥落一旦形成，往往发展迅速，应尽快进行更换或修复处理。"
                    ),
                    '裂纹': (
                        "裂纹通常产生在齿根或齿面高应力集中区域，可能由过载、材料缺陷或疲劳累积引起。"
                        "裂纹初期肉眼不易察觉，但会在运行中不断扩展，最终导致齿轮断裂。"
                        "若检测到裂纹，应立即停止使用并进行无损检测，防止灾难性失效。"
                    ),
                    '磨损': (
                        "磨损是齿轮在长期摩擦接触中齿面材料逐渐损耗的过程，常见于润滑不足、异物进入或材料硬度不足的情况。"
                        "磨损会导致啮合间隙增大、传动效率降低，并可能引发振动和噪声问题。"
                        "定期检查和更换润滑油可有效减缓磨损速度。"
                    ),
                    '擦伤压伤': (
                        "擦伤和压伤多由异物夹入啮合区或过载造成，表现为齿面局部划痕、压痕或表层金属塑性变形。"
                        "这些损伤会破坏润滑膜的连续性，使齿轮局部承受更高应力，进而加速疲劳破坏。"
                        "及时清除异物并保持良好润滑是避免此类问题的关键。"
                    ),
                    '健康': (
                        "检测结果显示该齿轮表面无明显损伤，啮合状态良好，运行平稳。"
                        "在正常工作条件下，其可继续长期使用。"
                        "建议保持稳定负荷、适当润滑和定期维护，以延缓磨损和疲劳，确保齿轮系统的高效运行。"
                    )
                }

                extra_info = damage_desc.get(cls_name_mapped, "暂无该类型的详细说明。")

                content_widget.setHtml(
                    f"""
                    <div style="text-align: center; font-size: 18px; line-height: 1.8; padding: 30px 20px;">
                        <h3 style="font-size: 26px; margin-bottom: 20px;">检测结果</h3>
                        <p>通过深度学习模型分析，图像中齿轮的主要损伤类型被识别为：</p>
                        <p style="font-size: 28px; color: #dc3545; font-weight: bold; margin: 15px 0;">
                            {cls_name_mapped}
                        </p>
                        <p style="font-size: 20px;">损伤面积占齿面比例估计为：
                            <b>{area_ratio:.2%}</b>
                        </p>
                        <div style="margin-top: 40px; text-align: justify; font-size: 22px; color: #444;">
                            {extra_info}
                        </div>
                    </div>
                    """
                )

            elif index == 1:
                content_widget.setHtml(
                    """
                    <div style="text-align: center; font-size: 25px; line-height: 1.8; padding-top: 20px;">
                        <h3 style="font-size: 30px; margin-bottom: 15px;">可视化说明</h3>
                        <p>损伤的具体位置和范围已在左侧的图像中通过高亮矩形框进行标注。</p>
                        <ul style="display: inline-block; text-align: left; margin-top: 15px; font-size: 25px;">
                            <li><b style="color: red;">红色框</b>: 表示检测到的损伤区域。</li>
                            <li><b style="color: green;">绿色框</b>: 表示健康的齿轮区域。</li>
                        </ul>
                    </div>
                    """
                )
            elif index == 2:
                if not self.last_rul_results:
                    content_widget.setText("RUL结果计算失败，请检查输入参数。")
                else:
                    res = self.last_rul_results
                    details = res['details']
                    html = (
                        f"<h3>物理模型寿命评估</h3><p><b>计算方法:</b> {res['method']}</p>"
                        f"<p style='font-size: 20px;'>预计剩余寿命 (RUL): <b style='color:#dc3545;'>{res['rul_hours']:.1f} 小时</b></p>"
                        f"<p>预计总寿命 (L10): <b>{res['L10_hours']:.1f} 小时</b></p><hr>"
                        f"<h4>关键计算参数:</h4><ul>"
                        f"<li>有效接触应力 (σH): {res['sigma_H']:.2f} MPa</li>"
                        f"<li>基础扭矩 (T): {details['基础扭矩 T']:.1f} N·m</li>"
                        f"<li>等效扭矩 (Teq): {details['等效扭矩 Teq']:.1f} N·m</li>"
                        f"<li>切向力 (Ft): {details['切向力 Ft']:.1f} N</li></ul>"
                    )
                    content_widget.setHtml(html)
            elif index == 3:
                content_widget.setText(self.last_ai_analysis_text)
                if not (self.ai_thread and self.ai_thread.isRunning()):
                    content_widget.setText(self.last_ai_analysis_text if self.last_ai_analysis_text else "正在等待AI生成分析报告...")
            elif index == 4:
                try:
                    rpm_user = int(self.rpm_input.text())
                    res = self.last_nonlinear_result
                    html = (
                        f"<h3>非线性寿命估算结果（用户转速 {rpm_user} rpm）</h3>"
                        f"<p style='font-size:18px;color:#dc3545;'>"
                        f"最短剩余寿命：<b>{res['min_life_hours']:.2f} 小时</b></p>"
                        f"<p>最危险区域编号：<b>{res['danger_idx']+1 if res['danger_idx']!=-1 else 'N/A'}</b></p>"
                        "<hr><h4>各区域明细：</h4>"
                        f"<p>{res['details']}</p>"
                    )
                except Exception as e:
                    html = f"<h3>非线性寿命估算失败</h3><p>{e}</p>"
                content_widget.setHtml(html)

        expanded_layout_inner.addWidget(content_widget)
        collapse_btn = QPushButton("返回")
        collapse_btn.setMinimumHeight(40)
        collapse_btn.setFont(QFont("SimHei", 14))
        collapse_btn.setStyleSheet("""
            QPushButton { background-color: #6c757d; color: white; border: none; border-radius: 10px; }
            QPushButton:hover { background-color: #5a6268; }
        """)
        collapse_btn.clicked.connect(self.collapse_view)

        self.expanded_layout.addWidget(expanded_box)
        self.expanded_layout.addWidget(collapse_btn)
        self.stack_layout.setCurrentIndex(1)

    # ---------- 修复：缺失的方法 ----------
    def update_image_display(self):
        if self.current_unscaled_pixmap is None or self.current_unscaled_pixmap.isNull():
            return
        current_label_size = self.image_label.size()
        if self.last_scaled_size == current_label_size:
            return
        scaled_pixmap = self.current_unscaled_pixmap.scaled(
            current_label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.last_scaled_size = current_label_size

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image_display()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GearAnalysisUI()
    window.show()
    sys.exit(app.exec_())